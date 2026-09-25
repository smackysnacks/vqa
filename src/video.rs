//! Frame assembly: turning the codebook, palette, and vector-pointer chunks
//! nested inside VQFR (and VQFL) chunks into pixel frames.
//!
//! A VQA frame is a mosaic of `block_width` x `block_height` blocks. The
//! codebook is the lookup table of block pixel data; the pointer chunks say
//! which codebook entry every screen block uses. 8-bit movies redraw every
//! block each frame from a VPT? table; HiColor movies update the previous
//! frame differentially with a VPTR/VPRZ command stream.

use crate::error::Error;
use crate::lcw;
use crate::parser::{RawChunk, VQAHeader, VQAVersion, raw_chunk};
use crate::rgb;

/// Sanity limit on the pixels in one frame and on codebook bytes, so a
/// malformed header cannot demand gigabyte allocations.
const MAX_FRAME_PIXELS: usize = 1 << 24;

/// One decoded video frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// The pixel data, row-major.
    pub pixels: FramePixels,
}

/// A frame's pixel data, row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FramePixels {
    /// 8-bit palette indices plus the palette in effect, already scaled from
    /// VGA 6-bit to full 8-bit range.
    Indexed {
        /// One palette index per pixel.
        pixels: Vec<u8>,
        /// The RGB palette the indices point into.
        palette: Vec<[u8; 3]>,
    },
    /// 15-bit `0rrrrrgg gggbbbbb` pixels (5 bits per channel).
    HiColor {
        /// One packed 15-bit value per pixel.
        pixels: Vec<u16>,
    },
}

/// A decoded frame borrowed from the [`FrameDecoder`] that produced it:
/// a [`Frame`] without the copy, valid until the decoder moves on to the
/// next frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRef<'a> {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// The pixel data, row-major.
    pub pixels: FramePixelsRef<'a>,
}

/// A borrowed frame's pixel data, row-major; see [`FramePixels`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FramePixelsRef<'a> {
    /// 8-bit palette indices plus the palette in effect, already scaled from
    /// VGA 6-bit to full 8-bit range.
    Indexed {
        /// One palette index per pixel.
        pixels: &'a [u8],
        /// The RGB palette the indices point into.
        palette: &'a [[u8; 3]],
    },
    /// 15-bit `0rrrrrgg gggbbbbb` pixels (5 bits per channel).
    HiColor {
        /// One packed 15-bit value per pixel.
        pixels: &'a [u16],
    },
}

impl Frame {
    /// Borrow the frame as a [`FrameRef`].
    pub fn view(&self) -> FrameRef<'_> {
        FrameRef {
            width: self.width,
            height: self.height,
            pixels: match &self.pixels {
                FramePixels::Indexed { pixels, palette } => {
                    FramePixelsRef::Indexed { pixels, palette }
                }
                FramePixels::HiColor { pixels } => FramePixelsRef::HiColor { pixels },
            },
        }
    }

    /// Convert the frame to packed RGB888 bytes, row-major. Indexed pixels
    /// with no palette entry come out black.
    pub fn to_rgb888(&self) -> Vec<u8> {
        self.view().to_rgb888()
    }

    /// Like [`Frame::to_rgb888`], but writes into `out`, so one buffer can
    /// be reused across frames.
    ///
    /// # Panics
    ///
    /// If `out` isn't exactly three bytes per pixel long.
    pub fn write_rgb888(&self, out: &mut [u8]) {
        self.view().write_rgb888(out);
    }
}

impl FrameRef<'_> {
    /// Copy the pixels out into an owned [`Frame`].
    pub fn to_frame(&self) -> Frame {
        Frame {
            width: self.width,
            height: self.height,
            pixels: match self.pixels {
                FramePixelsRef::Indexed { pixels, palette } => FramePixels::Indexed {
                    pixels: pixels.to_vec(),
                    palette: palette.to_vec(),
                },
                FramePixelsRef::HiColor { pixels } => FramePixels::HiColor {
                    pixels: pixels.to_vec(),
                },
            },
        }
    }

    /// Convert the frame to packed RGB888 bytes, row-major. Indexed pixels
    /// with no palette entry come out black.
    pub fn to_rgb888(&self) -> Vec<u8> {
        let mut out = vec![0; self.pixel_count() * 3];
        self.write_rgb888(&mut out);
        out
    }

    /// Like [`FrameRef::to_rgb888`], but writes into `out`, so one buffer
    /// can be reused across frames.
    ///
    /// # Panics
    ///
    /// If `out` isn't exactly three bytes per pixel long.
    pub fn write_rgb888(&self, out: &mut [u8]) {
        assert_eq!(
            out.len(),
            self.pixel_count() * 3,
            "RGB888 output must hold three bytes per pixel"
        );
        match self.pixels {
            FramePixelsRef::Indexed { pixels, palette } => {
                rgb::indexed_to_rgb888(pixels, palette, out);
            }
            FramePixelsRef::HiColor { pixels } => rgb::hicolor_to_rgb888(pixels, out),
        }
    }

    /// The number of pixels in `pixels`.
    fn pixel_count(&self) -> usize {
        match self.pixels {
            FramePixelsRef::Indexed { pixels, .. } => pixels.len(),
            FramePixelsRef::HiColor { pixels } => pixels.len(),
        }
    }
}

/// Stateful decoder for a movie's video stream.
///
/// Feed it every VQFL chunk ([`FrameDecoder::process_vqfl`]) and VQFR chunk
/// ([`FrameDecoder::decode_frame`]) in file order; it maintains the codebook
/// (including accumulation of partial codebooks across `cbparts` frames),
/// the palette, and the previous frame that HiColor movies update
/// differentially. Cloning it saves that state, e.g. to resume decoding
/// from a checkpoint after seeking.
#[derive(Clone)]
pub struct FrameDecoder {
    version: VQAVersion,
    hicolor: bool,
    width: usize,
    height: usize,
    block_w: usize,
    block_h: usize,
    blocks_x: usize,
    blocks_y: usize,
    /// the HiVal byte marking a solid-color block in a v2 pointer table
    fill_sentinel: u8,
    max_codebook_bytes: usize,
    /// parts making up one full codebook (0 = full codebooks only)
    cbparts: usize,
    /// current codebook - 8-bit movies store palette indices...
    codebook8: Vec<u8>,
    /// ...HiColor movies 15-bit pixels
    codebook16: Vec<u16>,
    /// staged partial codebook data and how many parts are in
    parts: Vec<u8>,
    parts_count: usize,
    parts_compressed: bool,
    palette: Vec<[u8; 3]>,
    frame8: Vec<u8>,
    frame16: Vec<u16>,
}

impl FrameDecoder {
    /// Build a decoder sized from the header. Fails on a zero block size or
    /// on frame dimensions past the sanity cap.
    pub fn new(header: &VQAHeader) -> Result<FrameDecoder, Error> {
        let block_w = usize::from(header.block_width);
        let block_h = usize::from(header.block_height);
        if block_w == 0 || block_h == 0 {
            return Err(Error::Video("block size is zero"));
        }

        let width = usize::from(header.width);
        let height = usize::from(header.height);
        if width * height > MAX_FRAME_PIXELS {
            return Err(Error::TooLarge("frame dimensions"));
        }

        let hicolor = header.is_hicolor();
        let max_blocks = match header.maxblocks {
            0 => 0xff00,
            n => usize::from(n),
        };
        let entry_bytes = block_w * block_h * if hicolor { 2 } else { 1 };
        let max_codebook_bytes = (max_blocks * entry_bytes).min(MAX_FRAME_PIXELS);

        Ok(FrameDecoder {
            version: header.version,
            hicolor,
            width,
            height,
            block_w,
            block_h,
            blocks_x: width / block_w,
            blocks_y: height / block_h,
            // normal movies hold at most 0x0f00 codebook entries, so 0x0f
            // can flag a fill; the hi-res movies use 0xff instead
            fill_sentinel: if header.maxblocks > 0x0f00 {
                0xff
            } else {
                0x0f
            },
            max_codebook_bytes,
            cbparts: usize::from(header.cbparts),
            codebook8: Vec::new(),
            codebook16: Vec::new(),
            parts: Vec::new(),
            parts_count: 0,
            parts_compressed: false,
            palette: Vec::new(),
            frame8: if hicolor {
                Vec::new()
            } else {
                vec![0; width * height]
            },
            frame16: if hicolor {
                vec![0; width * height]
            } else {
                Vec::new()
            },
        })
    }

    /// Process a VQFL chunk's payload: codebook (and palette) sub-chunks
    /// that apply to the following frames.
    pub fn process_vqfl(&mut self, mut data: &[u8]) -> Result<(), Error> {
        while !data.is_empty() {
            let (rest, chunk) = raw_chunk(data).map_err(|_| Error::Parse)?;
            data = rest;
            self.side_chunk(&chunk)?;
        }
        Ok(())
    }

    /// Decode one VQFR chunk's payload into the next frame.
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<Frame, Error> {
        Ok(self.decode_frame_ref(data)?.to_frame())
    }

    /// Like [`FrameDecoder::decode_frame`], but borrows the frame from the
    /// decoder instead of copying it out.
    pub fn decode_frame_ref(&mut self, mut data: &[u8]) -> Result<FrameRef<'_>, Error> {
        while !data.is_empty() {
            let (rest, chunk) = raw_chunk(data).map_err(|_| Error::Parse)?;
            data = rest;
            match &chunk.id {
                b"VPT0" => self.render_vpt(chunk.data)?,
                b"VPTZ" => {
                    let table = lcw::decompress(chunk.data, self.pointer_table_len())?;
                    self.render_vpt(&table)?;
                }
                b"VPTR" => self.render_vptr(chunk.data)?,
                b"VPRZ" => {
                    // a command stream has no fixed size; bound it generously
                    let cap = self.blocks_x * self.blocks_y * 8 + 256;
                    let stream = lcw::decompress(chunk.data, cap)?;
                    self.render_vptr(&stream)?;
                }
                _ => self.side_chunk(&chunk)?,
            }
        }

        // a codebook completed by this frame's part takes effect only after
        // the frame is drawn
        self.finish_codebook_parts()?;

        Ok(self.frame())
    }

    /// Handle the non-pointer sub-chunks: codebooks, codebook parts, and
    /// palettes. Anything unrecognized is skipped.
    fn side_chunk(&mut self, chunk: &RawChunk<'_>) -> Result<(), Error> {
        match &chunk.id {
            b"CBF0" => self.set_codebook(chunk.data.to_vec()),
            b"CBFZ" => {
                let data = lcw::decompress(chunk.data, self.max_codebook_bytes)?;
                self.set_codebook(data)
            }
            b"CBP0" | b"CBPZ" => self.stage_codebook_part(chunk),
            b"CPL0" => self.set_palette(chunk.data),
            b"CPLZ" => {
                let data = lcw::decompress(chunk.data, 256 * 3)?;
                self.set_palette(&data)
            }
            _ => Ok(()),
        }
    }

    fn entry_len(&self) -> usize {
        self.block_w * self.block_h
    }

    fn pointer_table_len(&self) -> usize {
        self.blocks_x * self.blocks_y * 2
    }

    fn set_codebook(&mut self, mut bytes: Vec<u8>) -> Result<(), Error> {
        if bytes.len() > self.max_codebook_bytes {
            return Err(Error::TooLarge("codebook"));
        }
        // Westwood's compressor sometimes leaves 1-2 stray bytes after the
        // last entry (CBFZ chunks in retail HiColor movies); drop the partial
        // entry instead of rejecting the codebook, like the original players
        let entry_bytes = self.entry_len() * if self.hicolor { 2 } else { 1 };
        bytes.truncate(bytes.len() - bytes.len() % entry_bytes);
        if self.hicolor {
            self.codebook16 = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&p| u16::from_le_bytes(p))
                .collect();
        } else {
            self.codebook8 = bytes;
        }
        Ok(())
    }

    fn stage_codebook_part(&mut self, chunk: &RawChunk<'_>) -> Result<(), Error> {
        let compressed = chunk.id[3] == b'Z';
        if self.parts_count == 0 {
            self.parts.clear();
            self.parts_compressed = compressed;
        } else if compressed != self.parts_compressed {
            return Err(Error::Video("mixed CBP0/CBPZ parts"));
        }
        if self.parts.len() + chunk.data.len() > self.max_codebook_bytes {
            return Err(Error::TooLarge("codebook parts"));
        }
        self.parts.extend_from_slice(chunk.data);
        self.parts_count += 1;
        Ok(())
    }

    /// Swap in the codebook accumulated from CBP? parts once `cbparts`
    /// frames have contributed one part each.
    fn finish_codebook_parts(&mut self) -> Result<(), Error> {
        if self.cbparts == 0 || self.parts_count < self.cbparts {
            return Ok(());
        }
        let staged = std::mem::take(&mut self.parts);
        self.parts_count = 0;
        let bytes = if self.parts_compressed {
            lcw::decompress(&staged, self.max_codebook_bytes)?
        } else {
            staged
        };
        self.set_codebook(bytes)
    }

    fn set_palette(&mut self, data: &[u8]) -> Result<(), Error> {
        if !data.len().is_multiple_of(3) || data.len() > 256 * 3 {
            return Err(Error::Video("palette size"));
        }
        self.palette = data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|&[r, g, b]| {
                // scale VGA 6-bit values to full 8-bit range
                let scale = |v: u8| (v & 0x3f) << 2 | (v & 0x3f) >> 4;
                [scale(r), scale(g), scale(b)]
            })
            .collect();
        Ok(())
    }

    /// The block size: `BW` x `BH` when those are nonzero, else the
    /// header's. The render paths are monomorphized for the common sizes so
    /// every block row becomes a fixed-size copy.
    #[inline(always)]
    fn block_size<const BW: usize, const BH: usize>(&self) -> (usize, usize) {
        if BW == 0 {
            (self.block_w, self.block_h)
        } else {
            debug_assert_eq!((BW, BH), (self.block_w, self.block_h));
            (BW, BH)
        }
    }

    /// Draw a full 8-bit frame from a (decompressed) VPT? pointer table.
    fn render_vpt(&mut self, table: &[u8]) -> Result<(), Error> {
        if self.hicolor {
            return Err(Error::Video("VPT? pointer table in a HiColor movie"));
        }
        if table.len() != self.blocks_x * self.blocks_y * 2 {
            return Err(Error::Video("pointer table size mismatch"));
        }
        // 8-bit movies use 4x2 blocks; 4x4 covers the rest seen in the wild
        match (self.block_w, self.block_h) {
            (4, 2) => self.render_vpt_with::<4, 2>(table),
            (4, 4) => self.render_vpt_with::<4, 4>(table),
            _ => self.render_vpt_with::<0, 0>(table),
        }
    }

    #[inline(always)]
    fn render_vpt_with<const BW: usize, const BH: usize>(
        &mut self,
        table: &[u8],
    ) -> Result<(), Error> {
        let blocks = self.blocks_x * self.blocks_y;
        for by in 0..self.blocks_y {
            for bx in 0..self.blocks_x {
                let i = by * self.blocks_x + bx;
                match self.version {
                    // v1: interleaved 16-bit entries; 0xff flags a fill with
                    // color 255-LoVal, indexes are stored premultiplied by 8
                    VQAVersion::One => {
                        let (lo, hi) = (table[i * 2], table[i * 2 + 1]);
                        if hi == 0xff {
                            self.fill_block::<BW, BH>(bx, by, 255 - lo);
                        } else {
                            let index = (usize::from(hi) << 8 | usize::from(lo)) / 8;
                            self.copy_block8::<BW, BH>(bx, by, index)?;
                        }
                    }
                    // v2: table split into a LoVal half and a HiVal half
                    _ => {
                        let (lo, hi) = (table[i], table[blocks + i]);
                        if hi == self.fill_sentinel {
                            self.fill_block::<BW, BH>(bx, by, lo);
                        } else {
                            let index = usize::from(hi) << 8 | usize::from(lo);
                            self.copy_block8::<BW, BH>(bx, by, index)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Apply a (decompressed) HiColor VPTR command stream to the previous
    /// frame. Commands walk the frame's blocks row-major.
    fn render_vptr(&mut self, stream: &[u8]) -> Result<(), Error> {
        if !self.hicolor {
            return Err(Error::Video("VPTR pointer stream in an 8-bit movie"));
        }
        // every known HiColor movie uses 4x2 or 4x4 blocks
        match (self.block_w, self.block_h) {
            (4, 2) => self.render_vptr_with::<4, 2>(stream),
            (4, 4) => self.render_vptr_with::<4, 4>(stream),
            _ => self.render_vptr_with::<0, 0>(stream),
        }
    }

    #[inline(always)]
    fn render_vptr_with<const BW: usize, const BH: usize>(
        &mut self,
        stream: &[u8],
    ) -> Result<(), Error> {
        let mut pos = 0; // current block, row-major
        let mut sp = 0;
        while sp < stream.len() {
            let val = match stream.get(sp..sp + 2) {
                Some(b) => u16::from_le_bytes([b[0], b[1]]),
                None => return Err(Error::Video("dangling byte in pointer stream")),
            };
            sp += 2;

            let run_count = usize::from(val >> 8 & 0x1f) + 1;
            match val >> 13 {
                // skip blocks (leave them unchanged)
                0b000 => pos += usize::from(val & 0x1fff),
                // write one of the first 256 blocks 2*(run+1) times
                0b001 => {
                    for _ in 0..run_count * 2 {
                        self.write_block16::<BW, BH>(&mut pos, usize::from(val & 0xff), false)?;
                    }
                }
                // write a block, then 2*(run+1) more indexed by stream bytes
                0b010 => {
                    self.write_block16::<BW, BH>(&mut pos, usize::from(val & 0xff), false)?;
                    for _ in 0..run_count * 2 {
                        let index = *stream
                            .get(sp)
                            .ok_or(Error::Video("truncated pointer stream"))?;
                        sp += 1;
                        self.write_block16::<BW, BH>(&mut pos, usize::from(index), false)?;
                    }
                }
                // write a single block, optionally skipping alpha pixels
                0b011 => {
                    self.write_block16::<BW, BH>(&mut pos, usize::from(val & 0x1fff), false)?
                }
                0b100 => self.write_block16::<BW, BH>(&mut pos, usize::from(val & 0x1fff), true)?,
                // write a block N times, N from the next stream byte,
                // optionally skipping alpha pixels
                0b101 | 0b110 => {
                    let count = *stream
                        .get(sp)
                        .ok_or(Error::Video("truncated pointer stream"))?;
                    sp += 1;
                    let alpha = val >> 13 == 0b110;
                    for _ in 0..count {
                        self.write_block16::<BW, BH>(&mut pos, usize::from(val & 0x1fff), alpha)?;
                    }
                }
                _ => return Err(Error::Video("unknown pointer stream command")),
            }
        }
        Ok(())
    }

    /// Write codebook entry `index` at block position `pos` (advancing it).
    /// With `alpha_skip`, pixels whose alpha bit is set keep their previous
    /// value (Blade Runner overlay movies).
    #[inline(always)]
    fn write_block16<const BW: usize, const BH: usize>(
        &mut self,
        pos: &mut usize,
        index: usize,
        alpha_skip: bool,
    ) -> Result<(), Error> {
        let (bw, bh) = self.block_size::<BW, BH>();
        if *pos >= self.blocks_x * self.blocks_y {
            return Err(Error::Video("pointer stream writes past the frame"));
        }
        let entry = bw * bh;
        let Some(src) = self.codebook16.get(index * entry..(index + 1) * entry) else {
            // retail movies contain the occasional stray command word whose
            // index points past the codebook (the original players read out
            // of bounds and drew garbage); keep the block's previous pixels
            *pos += 1;
            return Ok(());
        };
        let (bx, by) = (*pos % self.blocks_x, *pos / self.blocks_x);
        let dst = by * bh * self.width + bx * bw;
        for (row, src) in src.chunks_exact(bw).enumerate() {
            let at = dst + row * self.width;
            let dst = &mut self.frame16[at..at + bw];
            if alpha_skip {
                for (pixel, &new) in dst.iter_mut().zip(src) {
                    if new & 0x8000 == 0 {
                        *pixel = new;
                    }
                }
            } else {
                dst.copy_from_slice(src);
            }
        }
        *pos += 1;
        Ok(())
    }

    /// Copy codebook entry `index` into the 8-bit frame at block (bx, by).
    #[inline(always)]
    fn copy_block8<const BW: usize, const BH: usize>(
        &mut self,
        bx: usize,
        by: usize,
        index: usize,
    ) -> Result<(), Error> {
        let (bw, bh) = self.block_size::<BW, BH>();
        let entry = bw * bh;
        let src = self
            .codebook8
            .get(index * entry..(index + 1) * entry)
            .ok_or(Error::Video("block index outside the codebook"))?;
        let dst = by * bh * self.width + bx * bw;
        for (row, src) in src.chunks_exact(bw).enumerate() {
            let at = dst + row * self.width;
            self.frame8[at..at + bw].copy_from_slice(src);
        }
        Ok(())
    }

    /// Fill block (bx, by) of the 8-bit frame with a solid color.
    #[inline(always)]
    fn fill_block<const BW: usize, const BH: usize>(&mut self, bx: usize, by: usize, color: u8) {
        let (bw, bh) = self.block_size::<BW, BH>();
        let dst = by * bh * self.width + bx * bw;
        for row in 0..bh {
            let at = dst + row * self.width;
            self.frame8[at..at + bw].fill(color);
        }
    }

    /// The current frame, as the last VQFR chunk left it.
    fn frame(&self) -> FrameRef<'_> {
        FrameRef {
            width: self.width,
            height: self.height,
            pixels: if self.hicolor {
                FramePixelsRef::HiColor {
                    pixels: &self.frame16,
                }
            } else {
                FramePixelsRef::Indexed {
                    pixels: &self.frame8,
                    palette: &self.palette,
                }
            },
        }
    }
}

#[cfg(test)]
// binary literals below group digits by field (prefix_run_index), not by four
#[allow(clippy::unusual_byte_groupings)]
mod tests {
    use super::*;

    /// An 8x4 v2 movie with 4x2 blocks: 2x2 = 4 blocks per frame.
    fn v2_header() -> VQAHeader {
        VQAHeader {
            version: VQAVersion::Two,
            flags: 0,
            num_frames: 3,
            width: 8,
            height: 4,
            block_width: 4,
            block_height: 2,
            frame_rate: 15,
            cbparts: 0,
            colors: 256,
            maxblocks: 0x0f00,
            unk1: 0,
            unk2: 0,
            freq: 22050,
            channels: 1,
            bits: 16,
            unk3: 0,
            unk4: 0,
            max_cbfz_size: 0,
            unk5: 0,
        }
    }

    fn hicolor_header() -> VQAHeader {
        VQAHeader {
            version: VQAVersion::Three,
            colors: 0,
            channels: 2,
            ..v2_header()
        }
    }

    /// Wrap `data` in a chunk header with the given ID.
    fn chunk(id: &str, data: &[u8]) -> Vec<u8> {
        let mut out = id.as_bytes().to_vec();
        out.extend((data.len() as u32).to_be_bytes());
        out.extend(data);
        if data.len() % 2 == 1 {
            out.push(0);
        }
        out
    }

    #[test]
    fn renders_v2_frame_from_codebook_palette_and_pointers() {
        let mut decoder = FrameDecoder::new(&v2_header()).unwrap();

        // two codebook entries: blocks of pixel values 0..8 and 10..18
        let codebook: Vec<u8> = (0..8).chain(10..18).collect();
        // two colors; raw VGA 6-bit values
        let palette = [0x3f, 0, 0, 0, 0x20, 0];
        // blocks: entry 0, entry 1, fill with color 7, entry 0
        let table = [0u8, 1, 7, 0, /* hi half */ 0, 0, 0x0f, 0];

        let mut vqfr = chunk("CBF0", &codebook);
        vqfr.extend(chunk("CPL0", &palette));
        vqfr.extend(chunk("VPT0", &table));

        let frame = decoder.decode_frame(&vqfr).unwrap();
        match &frame.pixels {
            FramePixels::Indexed { pixels, palette } => {
                #[rustfmt::skip]
                assert_eq!(pixels, &vec![
                    0,  1,  2,  3,   10, 11, 12, 13,
                    4,  5,  6,  7,   14, 15, 16, 17,
                    7,  7,  7,  7,    0,  1,  2,  3,
                    7,  7,  7,  7,    4,  5,  6,  7,
                ]);
                assert_eq!(palette[0], [0xff, 0, 0]);
                assert_eq!(palette[1], [0, 0x82, 0]);
            }
            _ => panic!("expected an indexed frame"),
        }
    }

    #[test]
    fn accumulates_codebook_parts_and_swaps_after_the_carrying_frame() {
        let mut header = v2_header();
        header.cbparts = 2;
        let mut decoder = FrameDecoder::new(&header).unwrap();

        let old_codebook: Vec<u8> = vec![1; 8];
        let new_first: Vec<u8> = vec![2; 8];
        let new_second: Vec<u8> = vec![3; 8];
        let table = [0u8, 0, 0, 0, 0, 0, 0, 0]; // every block uses entry 0

        // frame 1: full codebook + first part of the next one
        let mut vqfr = chunk("CBF0", &old_codebook);
        vqfr.extend(chunk("CBP0", &new_first));
        vqfr.extend(chunk("VPT0", &table));
        let frame = decoder.decode_frame(&vqfr).unwrap();
        assert!(matches!(&frame.pixels, FramePixels::Indexed { pixels, .. } if pixels[0] == 1));

        // frame 2 carries the last part; it still draws with the old codebook
        let mut vqfr = chunk("CBP0", &new_second);
        vqfr.extend(chunk("VPT0", &table));
        let frame = decoder.decode_frame(&vqfr).unwrap();
        assert!(matches!(&frame.pixels, FramePixels::Indexed { pixels, .. } if pixels[0] == 1));

        // frame 3 uses the swapped-in codebook (entry 0 comes from part one)
        let frame = decoder.decode_frame(&chunk("VPT0", &table)).unwrap();
        assert!(matches!(&frame.pixels, FramePixels::Indexed { pixels, .. } if pixels[0] == 2));
    }

    #[test]
    fn decompresses_vptz_pointer_tables() {
        let mut decoder = FrameDecoder::new(&v2_header()).unwrap();

        let codebook: Vec<u8> = (0..8).collect();
        let table = [0u8, 0, 0, 0, 0, 0, 0, 0];
        // LCW: one literal run with the whole table, then the end marker
        let mut compressed = vec![0x80 | table.len() as u8];
        compressed.extend(table);
        compressed.push(0x80);

        let mut vqfr = chunk("CBF0", &codebook);
        vqfr.extend(chunk("VPTZ", &compressed));
        let frame = decoder.decode_frame(&vqfr).unwrap();
        // every block draws entry 0; pixel (7, 1) is column 3, row 1 of the
        // second block in the top row
        assert!(matches!(&frame.pixels, FramePixels::Indexed { pixels, .. }
            if pixels[7] == 3 && pixels[15] == 7));
    }

    #[test]
    fn renders_hicolor_vptr_commands_differentially() {
        let mut decoder = FrameDecoder::new(&hicolor_header()).unwrap();

        // two entries of 15-bit pixels, as little-endian bytes
        let mut codebook = Vec::new();
        for pixel in [0x7fffu16; 8].iter().chain([0x0300u16; 8].iter()) {
            codebook.extend(&pixel.to_le_bytes());
        }

        // frame 1: write block 1, then block 0 three times (prefix 101)
        let mut stream = Vec::new();
        stream.extend(&(0b011_0000000000001u16).to_le_bytes());
        stream.extend(&(0b101_0000000000000u16).to_le_bytes());
        stream.push(3);

        let mut vqfr = chunk("CBF0", &codebook);
        vqfr.extend(chunk("VPTR", &stream));
        let frame = decoder.decode_frame(&vqfr).unwrap();
        let FramePixels::HiColor { pixels } = &frame.pixels else {
            panic!("expected a hicolor frame");
        };
        assert_eq!(pixels[0], 0x0300);
        assert_eq!(pixels[4], 0x7fff);
        assert_eq!(pixels[7], 0x7fff);

        // frame 2: skip 3 blocks, rewrite only the last with block 1;
        // the first three keep their previous contents
        let mut stream = Vec::new();
        stream.extend(&(0b000_0000000000011u16).to_le_bytes());
        stream.extend(&(0b011_0000000000001u16).to_le_bytes());
        let frame = decoder.decode_frame(&chunk("VPTR", &stream)).unwrap();
        let FramePixels::HiColor { pixels } = &frame.pixels else {
            panic!("expected a hicolor frame");
        };
        assert_eq!(pixels[0], 0x0300); // block 0, unchanged from frame 1
        assert_eq!(pixels[4], 0x7fff); // block 1, unchanged from frame 1
        assert_eq!(pixels[2 * 8 + 4], 0x0300); // block 3, rewritten
    }

    #[test]
    fn hicolor_run_and_indexed_write_commands() {
        // a 16x4 movie: 4x2 = 8 blocks, room for the five writes below
        let mut header = hicolor_header();
        header.width = 16;
        let mut decoder = FrameDecoder::new(&header).unwrap();

        let mut codebook = Vec::new();
        for value in [1u16, 2, 3] {
            for _ in 0..8 {
                codebook.extend(&value.to_le_bytes());
            }
        }

        // prefix 001: write entry 0 at (run+1)*2 = 2 blocks; then prefix
        // 010: write entry 1, then 2 more entries from stream bytes (2, 2)
        let mut stream = Vec::new();
        stream.extend(&(0b001_00000_00000000u16).to_le_bytes());
        stream.extend(&(0b010_00000_00000001u16).to_le_bytes());
        stream.push(2);
        stream.push(2);

        let mut vqfr = chunk("CBF0", &codebook);
        vqfr.extend(chunk("VPTR", &stream));
        let frame = decoder.decode_frame(&vqfr).unwrap();
        let FramePixels::HiColor { pixels } = &frame.pixels else {
            panic!("expected a hicolor frame");
        };
        // blocks 0-4 hold entries 0, 0, 1, 2, 2; blocks 5-7 stay black
        assert_eq!(pixels[0], 1); // block 0
        assert_eq!(pixels[4], 1); // block 1
        assert_eq!(pixels[8], 2); // block 2
        assert_eq!(pixels[12], 3); // block 3
        assert_eq!(pixels[2 * 16], 3); // block 4, second block row
        assert_eq!(pixels[2 * 16 + 4], 0); // block 5, never written
    }

    #[test]
    fn drops_stray_bytes_after_the_last_codebook_entry() {
        let mut decoder = FrameDecoder::new(&v2_header()).unwrap();
        // one full entry plus two stray trailing bytes, as retail HiColor
        // movies contain; the partial entry must not become addressable
        let codebook: Vec<u8> = (0..8).chain([9, 9]).collect();
        let table = [0u8, 0, 0, 0, 0, 0, 0, 0];

        let mut vqfr = chunk("CBF0", &codebook);
        vqfr.extend(chunk("VPT0", &table));
        let frame = decoder.decode_frame(&vqfr).unwrap();
        assert!(matches!(&frame.pixels, FramePixels::Indexed { pixels, .. }
            if pixels[0] == 0 && pixels[15] == 7));

        let table = [0u8, 1, 0, 0, 0, 0, 0, 0]; // block 1 wants entry 1
        assert_eq!(
            decoder.decode_frame(&chunk("VPT0", &table)),
            Err(Error::Video("block index outside the codebook"))
        );
    }

    #[test]
    fn rejects_block_indices_outside_the_codebook() {
        let mut decoder = FrameDecoder::new(&v2_header()).unwrap();
        let codebook: Vec<u8> = (0..8).collect(); // one entry
        let table = [0u8, 1, 0, 0, 0, 0, 0, 0]; // block 1 wants entry 1

        let mut vqfr = chunk("CBF0", &codebook);
        vqfr.extend(chunk("VPT0", &table));
        assert_eq!(
            decoder.decode_frame(&vqfr),
            Err(Error::Video("block index outside the codebook"))
        );
    }

    #[test]
    fn skips_hicolor_writes_with_indices_outside_the_codebook() {
        let mut decoder = FrameDecoder::new(&hicolor_header()).unwrap();

        let mut codebook = Vec::new();
        for pixel in [0x0300u16; 8] {
            codebook.extend(&pixel.to_le_bytes());
        }

        // write block 0, then a stray command indexing far past the
        // codebook (as retail movies contain), then block 0 again
        let mut stream = Vec::new();
        stream.extend(&(0b011_0000000000000u16).to_le_bytes());
        stream.extend(&(0b011_1111101010100u16).to_le_bytes());
        stream.extend(&(0b011_0000000000000u16).to_le_bytes());

        let mut vqfr = chunk("CBF0", &codebook);
        vqfr.extend(chunk("VPTR", &stream));
        let frame = decoder.decode_frame(&vqfr).unwrap();
        let FramePixels::HiColor { pixels } = &frame.pixels else {
            panic!("expected a hicolor frame");
        };
        // blocks 0 and 2 drawn, block 1 skipped but still advanced past
        assert_eq!(pixels[0], 0x0300);
        assert_eq!(pixels[4], 0); // untouched
        assert_eq!(pixels[2 * 8], 0x0300);
    }

    #[test]
    fn rejects_unknown_pointer_stream_commands() {
        let mut decoder = FrameDecoder::new(&hicolor_header()).unwrap();
        let stream = (0b111_0000000000000u16).to_le_bytes();
        assert_eq!(
            decoder.decode_frame(&chunk("VPTR", &stream)),
            Err(Error::Video("unknown pointer stream command"))
        );
    }

    #[test]
    fn renders_8bit_blocks_without_a_specialized_size() {
        // a 4x2 movie with 2x2 blocks: 2 blocks, drawn by the run-time
        // sized path rather than a monomorphized 4x2 or 4x4 one
        let mut header = v2_header();
        (header.width, header.height) = (4, 2);
        (header.block_width, header.block_height) = (2, 2);
        let mut decoder = FrameDecoder::new(&header).unwrap();

        let codebook: Vec<u8> = (0..8).collect(); // entries 0..4 and 4..8
        let table = [1u8, 9, /* hi half */ 0, 0x0f]; // entry 1, fill with 9

        let mut vqfr = chunk("CBF0", &codebook);
        vqfr.extend(chunk("VPT0", &table));
        let frame = decoder.decode_frame(&vqfr).unwrap();
        let FramePixels::Indexed { pixels, .. } = &frame.pixels else {
            panic!("expected an indexed frame");
        };
        assert_eq!(pixels, &vec![4, 5, 9, 9, 6, 7, 9, 9]);
    }

    #[test]
    fn hicolor_alpha_writes_keep_flagged_pixels() {
        // a 4x2 movie with 2x2 blocks, so this also covers the run-time
        // sized path of HiColor writes
        let mut header = hicolor_header();
        (header.width, header.height) = (4, 2);
        (header.block_width, header.block_height) = (2, 2);
        let mut decoder = FrameDecoder::new(&header).unwrap();

        // entry 0 has its first pixel flagged as transparent
        let mut codebook = Vec::new();
        for pixel in [0x8001u16, 2, 3, 4, 10, 11, 12, 13] {
            codebook.extend(&pixel.to_le_bytes());
        }

        // frame 1: entry 1 into both blocks
        let mut stream = Vec::new();
        stream.extend(&(0b011_0000000000001u16).to_le_bytes());
        stream.extend(&(0b011_0000000000001u16).to_le_bytes());
        let mut vqfr = chunk("CBF0", &codebook);
        vqfr.extend(chunk("VPTR", &stream));
        decoder.decode_frame(&vqfr).unwrap();

        // frame 2: entry 0 over block 0, skipping its transparent pixel
        let stream = (0b100_0000000000000u16).to_le_bytes();
        let frame = decoder.decode_frame(&chunk("VPTR", &stream)).unwrap();
        let FramePixels::HiColor { pixels } = &frame.pixels else {
            panic!("expected a hicolor frame");
        };
        assert_eq!(pixels, &vec![10, 2, 10, 11, 3, 4, 12, 13]);
    }
}
