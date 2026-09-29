//! Frame assembly: turning the codebook, palette, and vector-pointer chunks
//! nested inside VQFR (and VQFL) chunks into pixel frames.
//!
//! A VQA frame is a mosaic of `block_width` x `block_height` blocks. The
//! codebook is the lookup table of block pixel data; the pointer chunks say
//! which codebook entry every screen block uses. 8-bit movies redraw every
//! block each frame from a VPT? table; HiColor movies update the previous
//! frame differentially with a VPTR/VPRZ command stream.

use fearless_simd::prelude::*;
use fearless_simd::{Level, dispatch, mask16x16, u8x16, u8x32, u16x16, u32x4, u64x2};

use crate::chunk::{Chunk, Chunks};
use crate::error::{Error, ErrorKind, Limit, VideoError};
use crate::header::{VQAHeader, VQAVersion};
use crate::lcw::{self, LcwError};
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
    /// 15-bit `xrrrrrgg gggbbbbb` pixels (5 bits per channel). The top bit
    /// is not color. Blade Runner's overlay codebooks set it on transparent
    /// pixels, which the alpha-skip pointer-stream commands leave out,
    /// keeping the frame's previous pixel (0 before anything is drawn); the
    /// other commands copy a codebook pixel whole, bit included. The RGB
    /// conversions ignore it.
    HiColor {
        /// One packed value per pixel.
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
    /// 15-bit `xrrrrrgg gggbbbbb` pixels (5 bits per channel), the top bit
    /// not color; see [`FramePixels::HiColor`].
    HiColor {
        /// One packed value per pixel.
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

    /// Convert the frame to RGBA8888 bytes, row-major, alpha opaque; see
    /// [`FrameRef::to_rgba8888`].
    pub fn to_rgba8888(&self) -> Vec<u8> {
        self.view().to_rgba8888()
    }

    /// Like [`Frame::to_rgba8888`], but writes into `out`.
    ///
    /// # Panics
    ///
    /// If `out` isn't exactly four bytes per pixel long.
    pub fn write_rgba8888(&self, out: &mut [u8]) {
        self.view().write_rgba8888(out);
    }

    /// Convert the frame to packed `0x00RRGGBB` words, row-major; see
    /// [`FrameRef::to_xrgb8888`].
    pub fn to_xrgb8888(&self) -> Vec<u32> {
        self.view().to_xrgb8888()
    }

    /// Like [`Frame::to_xrgb8888`], but writes into `out`.
    ///
    /// # Panics
    ///
    /// If `out` isn't exactly one word per pixel long.
    pub fn write_xrgb8888(&self, out: &mut [u32]) {
        self.view().write_xrgb8888(out);
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

    /// Convert the frame to RGBA8888 bytes, row-major: R, G, B, then an
    /// opaque 0xff alpha, the layout GPU textures and most image libraries
    /// take. Indexed pixels with no palette entry come out black, and
    /// HiColor pixels' top bit is ignored, as for [`FrameRef::to_rgb888`].
    pub fn to_rgba8888(&self) -> Vec<u8> {
        let mut out = vec![0; self.pixel_count() * 4];
        self.write_rgba8888(&mut out);
        out
    }

    /// Like [`FrameRef::to_rgba8888`], but writes into `out`, so one buffer
    /// can be reused across frames.
    ///
    /// # Panics
    ///
    /// If `out` isn't exactly four bytes per pixel long.
    pub fn write_rgba8888(&self, out: &mut [u8]) {
        assert_eq!(
            out.len(),
            self.pixel_count() * 4,
            "RGBA8888 output must hold four bytes per pixel"
        );
        match self.pixels {
            FramePixelsRef::Indexed { pixels, palette } => {
                rgb::indexed_to_rgba8888(pixels, palette, out);
            }
            FramePixelsRef::HiColor { pixels } => rgb::hicolor_to_rgba8888(pixels, out),
        }
    }

    /// Convert the frame to one `0x00RRGGBB` word per pixel, row-major: the
    /// layout of software framebuffers such as minifb's and softbuffer's.
    /// Indexed pixels with no palette entry come out black, and HiColor
    /// pixels' top bit is ignored, as for [`FrameRef::to_rgb888`].
    pub fn to_xrgb8888(&self) -> Vec<u32> {
        let mut out = vec![0; self.pixel_count()];
        self.write_xrgb8888(&mut out);
        out
    }

    /// Like [`FrameRef::to_xrgb8888`], but writes into `out`, so one buffer
    /// can be reused across frames.
    ///
    /// # Panics
    ///
    /// If `out` isn't exactly one word per pixel long.
    pub fn write_xrgb8888(&self, out: &mut [u32]) {
        assert_eq!(
            out.len(),
            self.pixel_count(),
            "XRGB8888 output must hold one word per pixel"
        );
        match self.pixels {
            FramePixelsRef::Indexed { pixels, palette } => {
                rgb::indexed_to_xrgb8888(pixels, palette, out);
            }
            FramePixelsRef::HiColor { pixels } => rgb::hicolor_to_xrgb8888(pixels, out),
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
/// Feed it every VQFL chunk ([`FrameDecoder::process_vqfl`]) and VQFR (or
/// VQFK) chunk ([`FrameDecoder::decode_frame`]) in file order; it maintains
/// the codebook (including accumulation of partial codebooks across
/// `cbparts` frames), the palette, and the previous frame that HiColor
/// movies update differentially. Movies with `cbparts` 0 in their header
/// need [`FrameDecoder::swap_in_codebook_parts`] called where their CINF
/// chunk says; [`Frames`](crate::Frames) handles that and the older layout
/// without VQFR chunks. Cloning the decoder saves its state, e.g. to resume
/// decoding from a checkpoint after seeking.
///
/// # Malformed data
///
/// A block index past the end of the codebook fails an 8-bit frame
/// ([`VideoError::BlockIndexOutOfRange`]) but not a HiColor one, whose
/// block keeps its pixels. Retail HiColor movies hold the occasional stray
/// index, which the original players drew as garbage; no retail 8-bit movie
/// does, so there it means the table was misread.
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
    /// current codebook - 8-bit movies store palette indices, the
    /// codebook's `entries8` entries and then one solid-color entry per
    /// color, which fill blocks draw...
    codebook8: Vec<u8>,
    entries8: usize,
    /// ...HiColor movies 15-bit pixels
    codebook16: Vec<u16>,
    /// staged partial codebook data and how many parts are in
    parts: Vec<u8>,
    parts_count: usize,
    parts_compressed: bool,
    palette: Vec<[u8; 3]>,
    frame8: Vec<u8>,
    frame16: Vec<u16>,
    /// the codebook entry each block of a row draws, as 8-bit drawing
    /// works it out
    sources: Vec<u32>,
    level: Level,
}

impl FrameDecoder {
    /// Build a decoder sized from the header.
    ///
    /// # Errors
    ///
    /// - [`ErrorKind::InvalidHeader`] if the header's block width or height
    ///   is 0
    /// - [`ErrorKind::TooLarge`] ([`Limit::FrameSize`]) if a frame would
    ///   hold more than 2^24 pixels
    ///
    /// The error has no location.
    pub fn new(header: &VQAHeader) -> Result<FrameDecoder, Error> {
        let block_w = usize::from(header.block_width);
        let block_h = usize::from(header.block_height);
        if block_w == 0 || block_h == 0 {
            return Err(ErrorKind::InvalidHeader.into());
        }

        let width = usize::from(header.width);
        let height = usize::from(header.height);
        if width * height > MAX_FRAME_PIXELS {
            return Err(ErrorKind::TooLarge(Limit::FrameSize).into());
        }

        let hicolor = header.is_hicolor();
        let max_blocks = match header.maxblocks {
            0 => 0xff00,
            n => usize::from(n),
        };
        let entry_bytes = block_w * block_h * if hicolor { 2 } else { 1 };
        // saturating: 0xff00 HiColor entries of 255x255 pixels overflow a
        // 32-bit usize
        let max_codebook_bytes = max_blocks.saturating_mul(entry_bytes).min(MAX_FRAME_PIXELS);

        Ok(FrameDecoder {
            version: header.version,
            hicolor,
            width,
            height,
            block_w,
            block_h,
            blocks_x: width / block_w,
            blocks_y: height / block_h,
            // 4x2 movies flag a fill with 0x0f (hardcoded in Westwood's own
            // 4x2 drawer), 4x4 ones - Lands of Lore's - with 0xff whatever
            // their codebook size
            fill_sentinel: if block_h == 4 { 0xff } else { 0x0f },
            max_codebook_bytes,
            cbparts: usize::from(header.cbparts),
            codebook8: if hicolor {
                Vec::new()
            } else {
                with_fill_entries(Vec::new(), block_w * block_h)
            },
            entries8: 0,
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
            sources: if hicolor {
                Vec::new()
            } else {
                vec![0; width / block_w]
            },
            level: Level::new(),
        })
    }

    /// Process a VQFL chunk's payload: codebook (and palette) sub-chunks
    /// that apply to the following frames.
    ///
    /// # Errors
    ///
    /// Fails on the first bad sub-chunk of `data`:
    ///
    /// - [`ErrorKind::InvalidChunk`] if `data` doesn't split into whole
    ///   chunks
    /// - [`ErrorKind::Lcw`] if a compressed codebook or palette isn't valid
    ///   LCW data
    /// - [`ErrorKind::TooLarge`] ([`Limit::Codebook`]) if a codebook, or the
    ///   codebook parts staged so far, would hold more than the header's
    ///   `maxblocks` entries
    /// - [`ErrorKind::Video`] if a palette isn't whole colors or holds more
    ///   than 256, or codebook parts mix `CBP0` and `CBPZ`
    ///
    /// A compressed codebook or palette fails as the uncompressed one would,
    /// whether it expands too far or not. [`Error::chunk`] names the
    /// sub-chunk and [`Error::offset`] gives its position from the start of
    /// `data`. The sub-chunks before it stay applied.
    pub fn process_vqfl(&mut self, data: &[u8]) -> Result<(), Error> {
        self.apply_side_chunks(Chunks::payload(data))
    }

    /// Apply the codebook and palette chunks among `chunks`, a VQFL
    /// chunk's sub-chunks.
    pub(crate) fn apply_side_chunks(&mut self, chunks: Chunks<'_>) -> Result<(), Error> {
        for chunk in chunks {
            let chunk = chunk?;
            self.side_chunk(&chunk)
                .map_err(|kind| Error::in_chunk(kind, &chunk))?;
        }
        Ok(())
    }

    /// Decode one VQFR chunk's payload into the next frame.
    ///
    /// # Errors
    ///
    /// As for [`FrameDecoder::decode_frame_ref`].
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<Frame, Error> {
        Ok(self.decode_frame_ref(data)?.to_frame())
    }

    /// Like [`FrameDecoder::decode_frame`], but borrows the frame from the
    /// decoder instead of copying it out.
    ///
    /// # Errors
    ///
    /// Fails on the first bad sub-chunk of `data`:
    ///
    /// - [`ErrorKind::InvalidChunk`] if `data` doesn't split into whole
    ///   chunks
    /// - [`ErrorKind::Lcw`] if a compressed sub-chunk (`CBFZ`, `CPLZ`,
    ///   `VPTZ`, `VPTK`, `VPTD` or `VPRZ`) isn't valid LCW data, or a
    ///   compressed pointer stream (`VPRZ`) expands past 8 bytes per block
    /// - [`ErrorKind::TooLarge`] ([`Limit::Codebook`]) if a codebook, or the
    ///   codebook parts staged so far, would hold more than the header's
    ///   `maxblocks` entries
    /// - [`ErrorKind::Video`] if a palette, pointer table or pointer stream
    ///   doesn't fit the frame or the movie's pixel format, an 8-bit pointer
    ///   table points past the end of the codebook, or codebook parts mix
    ///   `CBP0` and `CBPZ` ([`VideoError`] says which)
    ///
    /// A compressed codebook, palette or pointer table fails as the
    /// uncompressed one would, whether it expands too far or not.
    ///
    /// [`Error::chunk`] names the sub-chunk and [`Error::offset`] gives its
    /// position from the start of `data`. Once the sub-chunks are applied,
    /// a codebook the frame's part completes is swapped in, failing as
    /// [`FrameDecoder::swap_in_codebook_parts`] does, with no location.
    ///
    /// The sub-chunks before a bad one stay applied, so the decoder is left
    /// part-way through the frame: clone it first to be able to go back.
    pub fn decode_frame_ref(&mut self, data: &[u8]) -> Result<FrameRef<'_>, Error> {
        self.decode_chunks(Chunks::payload(data))
    }

    /// Decode the next frame from `chunks`, a VQFR chunk's sub-chunks.
    pub(crate) fn decode_chunks(&mut self, chunks: Chunks<'_>) -> Result<FrameRef<'_>, Error> {
        for chunk in chunks {
            let chunk = chunk?;
            self.frame_chunk(&chunk)
                .map_err(|kind| Error::in_chunk(kind, &chunk))?;
        }
        Ok(self.end_frame()?)
    }

    /// Apply one of a frame's sub-chunks: draw a pointer table or stream,
    /// or take in a codebook, codebook part, or palette.
    pub(crate) fn frame_chunk(&mut self, chunk: &Chunk<'_>) -> Result<(), ErrorKind> {
        match &chunk.id {
            b"VPT0" => self.render_vpt(chunk.data),
            // VPTK marks a key frame and VPTD a delta one; Westwood's loader
            // reads both like VPTZ
            b"VPTZ" | b"VPTK" | b"VPTD" => {
                let size = ErrorKind::Video(VideoError::PointerTableSize);
                let table = decompress(chunk.data, self.pointer_table_len(), size)?;
                self.render_vpt(&table)
            }
            b"VPTR" => self.render_vptr(chunk.data),
            b"VPRZ" => {
                // a command stream has no fixed size; bound it generously
                let cap = self.blocks_x * self.blocks_y * 8 + 256;
                let stream = lcw::decompress(chunk.data, cap)?;
                self.render_vptr(&stream)
            }
            _ => self.side_chunk(chunk),
        }
    }

    /// Finish the frame the sub-chunks drew, and borrow it.
    pub(crate) fn end_frame(&mut self) -> Result<FrameRef<'_>, ErrorKind> {
        // a codebook completed by this frame's part takes effect only after
        // the frame is drawn
        if self.cbparts != 0 && self.parts_count >= self.cbparts {
            self.swap_parts()?;
        }
        Ok(self.frame())
    }

    /// Swap in the codebook assembled from the codebook parts (`CBP0` or
    /// `CBPZ` chunks) staged so far. The decoder does this by itself once
    /// the header's `cbparts` frames have each brought a part; movies whose
    /// header gives no count (`cbparts` 0, as in Lands of Lore) list the
    /// frames where each codebook takes over in a CINF chunk instead
    /// ([`VQA::codebook_starts`](crate::VQA::codebook_starts)), and need
    /// this called before decoding those frames. [`Frames`](crate::Frames)
    /// does that for them.
    ///
    /// # Errors
    ///
    /// - [`ErrorKind::Lcw`] if compressed (`CBPZ`) parts don't join into
    ///   valid LCW data
    /// - [`ErrorKind::TooLarge`] ([`Limit::Codebook`]) if the codebook, once
    ///   joined and decompressed, would hold more than the header's
    ///   `maxblocks` entries
    ///
    /// The error has no location. The staged parts are used up either way.
    pub fn swap_in_codebook_parts(&mut self) -> Result<(), Error> {
        Ok(self.swap_parts()?)
    }

    fn swap_parts(&mut self) -> Result<(), ErrorKind> {
        if self.parts_count == 0 {
            return Ok(());
        }
        let staged = std::mem::take(&mut self.parts);
        self.parts_count = 0;
        let bytes = if self.parts_compressed {
            let too_large = ErrorKind::TooLarge(Limit::Codebook);
            decompress(&staged, self.max_codebook_bytes, too_large)?
        } else {
            staged
        };
        self.set_codebook(bytes)
    }

    /// Handle the non-pointer sub-chunks: codebooks, codebook parts, and
    /// palettes. Anything unrecognized is skipped.
    fn side_chunk(&mut self, chunk: &Chunk<'_>) -> Result<(), ErrorKind> {
        match &chunk.id {
            b"CBF0" => self.set_codebook(chunk.data.to_vec()),
            b"CBFZ" => {
                let too_large = ErrorKind::TooLarge(Limit::Codebook);
                let data = decompress(chunk.data, self.max_codebook_bytes, too_large)?;
                self.set_codebook(data)
            }
            b"CBP0" | b"CBPZ" => self.stage_codebook_part(chunk),
            b"CPL0" => self.set_palette(chunk.data),
            b"CPLZ" => {
                let size = ErrorKind::Video(VideoError::PaletteSize);
                let data = decompress(chunk.data, 256 * 3, size)?;
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

    fn set_codebook(&mut self, mut bytes: Vec<u8>) -> Result<(), ErrorKind> {
        if bytes.len() > self.max_codebook_bytes {
            return Err(ErrorKind::TooLarge(Limit::Codebook));
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
            self.entries8 = bytes.len() / entry_bytes;
            self.codebook8 = with_fill_entries(bytes, entry_bytes);
        }
        Ok(())
    }

    fn stage_codebook_part(&mut self, chunk: &Chunk<'_>) -> Result<(), ErrorKind> {
        let compressed = chunk.id[3] == b'Z';
        if self.parts_count == 0 {
            self.parts.clear();
            self.parts_compressed = compressed;
        } else if compressed != self.parts_compressed {
            return Err(ErrorKind::Video(VideoError::MixedCodebookParts));
        }
        if self.parts.len() + chunk.data.len() > self.max_codebook_bytes {
            return Err(ErrorKind::TooLarge(Limit::Codebook));
        }
        self.parts.extend_from_slice(chunk.data);
        self.parts_count += 1;
        Ok(())
    }

    fn set_palette(&mut self, data: &[u8]) -> Result<(), ErrorKind> {
        if !data.len().is_multiple_of(3) || data.len() > 256 * 3 {
            return Err(ErrorKind::Video(VideoError::PaletteSize));
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
    fn render_vpt(&mut self, table: &[u8]) -> Result<(), ErrorKind> {
        if self.hicolor {
            return Err(ErrorKind::Video(VideoError::WrongPointerFormat));
        }
        if table.len() != self.blocks_x * self.blocks_y * 2 {
            return Err(ErrorKind::Video(VideoError::PointerTableSize));
        }
        if table.is_empty() {
            return Ok(());
        }
        let level = self.level;
        // 8-bit movies use 4x2 blocks; 4x4 covers the rest seen in the wild
        match (self.block_w, self.block_h) {
            (4, 2) => dispatch!(level, simd => self.render_vpt_with::<_, 4, 2>(simd, table)),
            (4, 4) => dispatch!(level, simd => self.render_vpt_with::<_, 4, 4>(simd, table)),
            _ => dispatch!(level, simd => self.render_vpt_with::<_, 0, 0>(simd, table)),
        }
    }

    /// Draw the frame a row of blocks at a time: first work out which
    /// codebook entry each block of the row draws, then copy them in. A
    /// block indexing past the codebook fails the frame, with the blocks
    /// before it drawn.
    #[inline(always)]
    fn render_vpt_with<S: Simd, const BW: usize, const BH: usize>(
        &mut self,
        simd: S,
        table: &[u8],
    ) -> Result<(), ErrorKind> {
        let (bw, bh) = self.block_size::<BW, BH>();
        let (blocks_x, blocks) = (self.blocks_x, self.blocks_x * self.blocks_y);
        let (entries, fill) = (self.entries8, self.fill_sentinel);
        // without a SIMD level, vector code would only be emulated
        let vector = !self.level.is_fallback();
        let sources = &mut self.sources[..blocks_x];
        let strips = self.frame8.chunks_exact_mut(self.width * bh);
        for (by, strip) in strips.take(self.blocks_y).enumerate() {
            let row = by * blocks_x;
            let drawable = match self.version {
                // v1: interleaved 16-bit entries
                VQAVersion::One => {
                    let words = &table[row * 2..][..blocks_x * 2];
                    match vector {
                        true => interleaved_sources(simd, words, entries, sources),
                        false => interleaved_sources_scalar(words, entries, sources),
                    }
                }
                // v2: a LoVal half, then a HiVal half
                _ => {
                    let lo = &table[row..][..blocks_x];
                    let hi = &table[blocks + row..][..blocks_x];
                    match vector {
                        true => split_sources(simd, lo, hi, fill, entries, sources),
                        false => split_sources_scalar(lo, hi, fill, entries, sources),
                    }
                }
            };
            let sources = &sources[..drawable];
            match (BW, BH, vector) {
                (0, _, _) => draw_row_any(strip, self.width, bw, &self.codebook8, sources),
                (4, 2, true) => {
                    draw_row_simd::<_, 2>(simd, strip, self.width, &self.codebook8, sources);
                }
                (4, 4, true) => {
                    draw_row_simd::<_, 4>(simd, strip, self.width, &self.codebook8, sources);
                }
                _ => draw_row::<BW, BH>(strip, self.width, &self.codebook8, sources),
            }
            if drawable < blocks_x {
                return Err(ErrorKind::Video(VideoError::BlockIndexOutOfRange));
            }
        }
        Ok(())
    }

    /// Apply a (decompressed) HiColor VPTR command stream to the previous
    /// frame. Commands walk the frame's blocks row-major.
    fn render_vptr(&mut self, stream: &[u8]) -> Result<(), ErrorKind> {
        if !self.hicolor {
            return Err(ErrorKind::Video(VideoError::WrongPointerFormat));
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
    ) -> Result<(), ErrorKind> {
        let mut pos = 0usize; // current block, row-major
        let mut sp = 0;
        while sp < stream.len() {
            let val = match stream.get(sp..sp + 2) {
                Some(b) => u16::from_le_bytes([b[0], b[1]]),
                None => return Err(ErrorKind::Video(VideoError::TruncatedPointerStream)),
            };
            sp += 2;

            let run_count = usize::from(val >> 8 & 0x1f) + 1;
            match val >> 13 {
                // skip blocks (leave them unchanged); saturating, since a
                // long run of skips could otherwise overflow a 32-bit usize
                0b000 => pos = pos.saturating_add(usize::from(val & 0x1fff)),
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
                            .ok_or(ErrorKind::Video(VideoError::TruncatedPointerStream))?;
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
                        .ok_or(ErrorKind::Video(VideoError::TruncatedPointerStream))?;
                    sp += 1;
                    let alpha = val >> 13 == 0b110;
                    for _ in 0..count {
                        self.write_block16::<BW, BH>(&mut pos, usize::from(val & 0x1fff), alpha)?;
                    }
                }
                _ => return Err(ErrorKind::Video(VideoError::UnknownPointerCommand)),
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
    ) -> Result<(), ErrorKind> {
        let (bw, bh) = self.block_size::<BW, BH>();
        if *pos >= self.blocks_x * self.blocks_y {
            return Err(ErrorKind::Video(VideoError::PointerStreamOverrun));
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

/// `codebook` followed by one solid-color entry of `entry` bytes per color,
/// in color order.
fn with_fill_entries(mut codebook: Vec<u8>, entry: usize) -> Vec<u8> {
    codebook.reserve(256 * entry);
    for color in 0..=255 {
        codebook.extend(std::iter::repeat_n(color, entry));
    }
    codebook
}

// Where the blocks of a row of an 8-bit pointer table draw from: each
// block's codebook entry, or for a fill block the solid-color entry of its
// color after the codebook's `entries` own. Each gives how many of the
// row's leading blocks can be drawn: all of them, or those before the first
// that indexes past the codebook. The vector versions take 16 blocks at a
// time, with no branch between fills and copies.

/// The sources of a row of a v2 pointer table, from its LoVal (`lo`) and
/// HiVal (`hi`) bytes. A HiVal of `fill` marks a fill with color LoVal.
#[inline(always)]
fn split_sources<S: Simd>(
    simd: S,
    lo: &[u8],
    hi: &[u8],
    fill: u8,
    entries: usize,
    out: &mut [u32],
) -> usize {
    let (lo_v, lo_rest) = lo.as_chunks::<16>();
    let (hi_v, hi_rest) = hi.as_chunks::<16>();
    let (out_v, out_rest) = out.as_chunks_mut::<16>();
    // the codebook's size fits: it holds at most 0xffff entries
    let entries_v = u16x16::splat(simd, entries as u16);
    for (i, ((lo, hi), out)) in lo_v.iter().zip(hi_v).zip(out_v).enumerate() {
        let (lo, hi) = (u8x16::from_slice(simd, lo), u8x16::from_slice(simd, hi));
        let index: u16x16<S> = simd
            .combine_u8x16(simd.zip_low_u8x16(lo, hi), simd.zip_high_u8x16(lo, hi))
            .bitcast();
        let is_fill = (index >> 8).simd_eq(u16x16::splat(simd, u16::from(fill)));
        let is_bad = !is_fill & index.simd_ge(entries_v);
        store_sources(simd, is_fill, index & 0xff, index, entries_v, out);
        if is_bad.any_true() {
            return i * 16 + is_bad.to_bitmask().trailing_zeros() as usize;
        }
    }
    let done = lo_v.len() * 16;
    done + split_sources_scalar(lo_rest, hi_rest, fill, entries, out_rest)
}

fn split_sources_scalar(lo: &[u8], hi: &[u8], fill: u8, entries: usize, out: &mut [u32]) -> usize {
    for (i, ((&lo, &hi), out)) in lo.iter().zip(hi).zip(out).enumerate() {
        let index = usize::from(hi) << 8 | usize::from(lo);
        *out = match hi == fill {
            true => entries + usize::from(lo),
            false if index < entries => index,
            false => return i,
        } as u32;
    }
    lo.len()
}

/// The sources of a row of a v1 pointer table, from its little-endian
/// words: a HiVal of 0xff marks a fill with color 255 - LoVal, and other
/// words hold the entry's index times 8.
#[inline(always)]
fn interleaved_sources<S: Simd>(simd: S, words: &[u8], entries: usize, out: &mut [u32]) -> usize {
    let (words_v, words_rest) = words.as_chunks::<32>();
    let (out_v, out_rest) = out.as_chunks_mut::<16>();
    let entries_v = u16x16::splat(simd, entries as u16);
    for (i, (words, out)) in words_v.iter().zip(out_v).enumerate() {
        let word: u16x16<S> = u8x32::from_slice(simd, words).bitcast();
        let is_fill = (word >> 8).simd_eq(u16x16::splat(simd, 0xff));
        let index = word >> 3;
        let is_bad = !is_fill & index.simd_ge(entries_v);
        store_sources(simd, is_fill, (word & 0xff) ^ 0xff, index, entries_v, out);
        if is_bad.any_true() {
            return i * 16 + is_bad.to_bitmask().trailing_zeros() as usize;
        }
    }
    let done = words_v.len() * 16;
    done + interleaved_sources_scalar(words_rest, entries, out_rest)
}

fn interleaved_sources_scalar(words: &[u8], entries: usize, out: &mut [u32]) -> usize {
    let words = words.as_chunks::<2>().0;
    for (i, (&[lo, hi], out)) in words.iter().zip(out).enumerate() {
        let index = (usize::from(hi) << 8 | usize::from(lo)) / 8;
        *out = match hi == 0xff {
            true => entries + usize::from(255 - lo),
            false if index < entries => index,
            false => return i,
        } as u32;
    }
    words.len()
}

/// Store 16 blocks' sources: `color` for the fills (`is_fill`), offset by
/// the codebook's `entries`, and `index` for the rest. Fills can land past
/// what 16 bits hold, so the sources widen to 32.
#[inline(always)]
fn store_sources<S: Simd>(
    simd: S,
    is_fill: mask16x16<S>,
    color: u16x16<S>,
    index: u16x16<S>,
    entries: u16x16<S>,
    out: &mut [u32; 16],
) {
    let (base_lo, base_hi) = simd.widen_u16x16(is_fill.select(color, index));
    let (offset_lo, offset_hi) = simd.widen_u16x16(is_fill.select(entries, u16x16::splat(simd, 0)));
    let (out_lo, out_hi) = out.split_at_mut(8);
    (base_lo + offset_lo).store_slice(out_lo);
    (base_hi + offset_hi).store_slice(out_hi);
}

/// Copy each block of a row of `BW` x `BH` blocks from its source entry
/// in `codebook` into `strip`, the row's `BH` lines of pixels.
#[inline(always)]
fn draw_row<const BW: usize, const BH: usize>(
    strip: &mut [u8],
    width: usize,
    codebook: &[u8],
    sources: &[u32],
) {
    let entries = codebook.as_chunks::<BW>().0.as_chunks::<BH>().0;
    let mut lines = block_lines::<BW, BH>(strip, width, sources.len());
    copy_blocks(&mut lines, entries, sources);
}

/// [`draw_row`] for blocks 4 pixels wide, four blocks at a time: their
/// entries, one line per lane, transposed so that each line of the four
/// is one 16-byte store.
#[inline(always)]
fn draw_row_simd<S: Simd, const BH: usize>(
    simd: S,
    strip: &mut [u8],
    width: usize,
    codebook: &[u8],
    sources: &[u32],
) {
    let entries = codebook.as_chunks::<4>().0.as_chunks::<BH>().0;
    let Some(last) = entries.len().checked_sub(1) else {
        return;
    };
    let mut lines = block_lines::<4, BH>(strip, width, sources.len());
    let (groups, rest) = sources.as_chunks::<4>();
    let done = groups.len() * 4;
    let mut heads = lines
        .each_mut()
        .map(|line| line[..done].as_chunks_mut::<4>().0);
    for (g, &[a, b, c, d]) in groups.iter().enumerate() {
        let a = entry_u32x4(simd, &entries[(a as usize).min(last)]);
        let b = entry_u32x4(simd, &entries[(b as usize).min(last)]);
        let c = entry_u32x4(simd, &entries[(c as usize).min(last)]);
        let d = entry_u32x4(simd, &entries[(d as usize).min(last)]);
        let (ab_lo, ab_hi) = (simd.zip_low_u32x4(a, b), simd.zip_high_u32x4(a, b));
        let (cd_lo, cd_hi) = (simd.zip_low_u32x4(c, d), simd.zip_high_u32x4(c, d));
        let (ab_lo, ab_hi): (u64x2<S>, u64x2<S>) = (ab_lo.bitcast(), ab_hi.bitcast());
        let (cd_lo, cd_hi): (u64x2<S>, u64x2<S>) = (cd_lo.bitcast(), cd_hi.bitcast());
        let transposed = [
            simd.zip_low_u64x2(ab_lo, cd_lo),
            simd.zip_high_u64x2(ab_lo, cd_lo),
            simd.zip_low_u64x2(ab_hi, cd_hi),
            simd.zip_high_u64x2(ab_hi, cd_hi),
        ];
        for (line, pixels) in heads.iter_mut().zip(transposed) {
            pixels
                .bitcast::<u8x16<S>>()
                .store_slice(line[g].as_flattened_mut());
        }
    }
    let mut tails = lines.each_mut().map(|line| &mut line[done..]);
    copy_blocks(&mut tails, entries, rest);
}

/// A block row's `BH` lines of pixels, each cut into `n` block-wide
/// pieces.
#[inline(always)]
fn block_lines<const BW: usize, const BH: usize>(
    strip: &mut [u8],
    width: usize,
    n: usize,
) -> [&mut [[u8; BW]]; BH] {
    let mut lines = strip.chunks_exact_mut(width);
    let lines: [_; BH] = std::array::from_fn(|_| lines.next().unwrap_or_default());
    lines.map(|line| &mut line.as_chunks_mut::<BW>().0[..n])
}

/// Copy the entry each of `sources` names into the blocks of `lines`.
#[inline(always)]
fn copy_blocks<const BW: usize, const BH: usize>(
    lines: &mut [&mut [[u8; BW]]; BH],
    entries: &[[[u8; BW]; BH]],
    sources: &[u32],
) {
    // the clamp never bites: a source is past the codebook's own entries
    // only if it's a fill's, and there are 256 of those after them
    let Some(last) = entries.len().checked_sub(1) else {
        return;
    };
    // every line exactly as long as the row, for the loop to see
    let n = sources.len();
    let mut lines = lines.each_mut().map(|line| &mut line[..n]);
    for (i, &source) in sources.iter().enumerate() {
        let entry = &entries[(source as usize).min(last)];
        for (line, pixels) in lines.iter_mut().zip(entry) {
            line[i] = *pixels;
        }
    }
}

/// A codebook entry of 4-pixel lines, one line per lane from lane 0.
#[inline(always)]
fn entry_u32x4<S: Simd, const BH: usize>(simd: S, entry: &[[u8; 4]; BH]) -> u32x4<S> {
    let mut bytes = [0; 16];
    bytes[..BH * 4].copy_from_slice(entry.as_flattened());
    u8x16::simd_from(simd, bytes).bitcast()
}

/// [`draw_row`] for blocks of any size, `bw` pixels wide.
fn draw_row_any(strip: &mut [u8], width: usize, bw: usize, codebook: &[u8], sources: &[u32]) {
    let bh = strip.len() / width;
    let entry = bw * bh;
    let last = codebook.len() / entry - 1;
    for (y, line) in strip.chunks_exact_mut(width).enumerate() {
        for (pixels, &source) in line.chunks_exact_mut(bw).zip(sources) {
            let at = (source as usize).min(last) * entry + y * bw;
            pixels.copy_from_slice(&codebook[at..at + bw]);
        }
    }
}

/// Decompress LCW data whose output can't be longer than `max` bytes: more
/// fails as `too_long`, as the same data uncompressed would, rather than as
/// an LCW error.
fn decompress(data: &[u8], max: usize, too_long: ErrorKind) -> Result<Vec<u8>, ErrorKind> {
    lcw::decompress(data, max).map_err(|e| match e {
        LcwError::TooLarge => too_long,
        e => ErrorKind::Lcw(e),
    })
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
            x_pos: 0,
            y_pos: 0,
            max_frame_size: 0,
            freq: 22050,
            channels: 1,
            bits: 16,
            alt_freq: 0,
            alt_channels: 0,
            alt_bits: 0,
            future_use: [0; 5],
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
            decoder
                .decode_frame(&chunk("VPT0", &table))
                .map_err(|e| e.kind()),
            Err(ErrorKind::Video(VideoError::BlockIndexOutOfRange))
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
            decoder.decode_frame(&vqfr).map_err(|e| e.kind()),
            Err(ErrorKind::Video(VideoError::BlockIndexOutOfRange))
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
    fn long_skip_runs_do_not_overflow_the_block_position() {
        // 600,000 maximal skips move 4.9 billion blocks - past u32::MAX,
        // which overflowed `usize` on 32-bit targets - then a write, which
        // lands past the frame
        let mut decoder = FrameDecoder::new(&hicolor_header()).unwrap();
        let mut stream = 0b000_1111111111111u16.to_le_bytes().repeat(600_000);
        stream.extend(0b011_0000000000000u16.to_le_bytes());
        assert_eq!(
            decoder
                .decode_frame(&chunk("VPTR", &stream))
                .map_err(|e| e.kind()),
            Err(ErrorKind::Video(VideoError::PointerStreamOverrun))
        );
    }

    #[test]
    fn rejects_unknown_pointer_stream_commands() {
        let mut decoder = FrameDecoder::new(&hicolor_header()).unwrap();
        let stream = (0b111_0000000000000u16).to_le_bytes();
        assert_eq!(
            decoder
                .decode_frame(&chunk("VPTR", &stream))
                .map_err(|e| e.kind()),
            Err(ErrorKind::Video(VideoError::UnknownPointerCommand))
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
    fn vector_block_sources_match_the_scalar_ones() {
        // rows of every length up to 70 blocks, whole vectors and leftovers
        // alike, with fills, copies and the odd index past the codebook, at
        // the detected SIMD level and the baseline one
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut random = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for level in [Level::new(), Level::baseline()] {
            for len in 0..=70 {
                for entries in [0, 1, 300, 0xff00, 0xffff] {
                    let lo: Vec<u8> = (0..len).map(|_| random() as u8).collect();
                    // mostly low HiVals, so that fills, copies and indexes
                    // past a small codebook all turn up
                    let hi: Vec<u8> = (0..len)
                        .map(|_| match random() % 8 {
                            0 => 0x0f,
                            1 => 0xff,
                            2 => random() as u8,
                            _ => (random() % 2) as u8,
                        })
                        .collect();
                    let words: Vec<u8> = lo.iter().zip(&hi).flat_map(|(&l, &h)| [l, h]).collect();

                    let (mut vector, mut scalar) = (vec![0; len], vec![0; len]);
                    let n = dispatch!(level, simd => split_sources(simd, &lo, &hi, 0x0f, entries, &mut vector));
                    let m = split_sources_scalar(&lo, &hi, 0x0f, entries, &mut scalar);
                    assert_eq!(
                        (n, &vector[..n]),
                        (m, &scalar[..m]),
                        "v2 at {level:?}, {len} blocks, {entries} entries"
                    );

                    let n = dispatch!(level, simd => interleaved_sources(simd, &words, entries, &mut vector));
                    let m = interleaved_sources_scalar(&words, entries, &mut scalar);
                    assert_eq!(
                        (n, &vector[..n]),
                        (m, &scalar[..m]),
                        "v1 at {level:?}, {len} blocks, {entries} entries"
                    );
                }
            }
        }
    }

    #[test]
    fn a_block_past_the_codebook_fails_with_the_blocks_before_it_drawn() {
        // one row of 40 blocks: two vectors' worth and 8 left over, with the
        // bad block in the second vector, then among the leftovers
        let mut header = v2_header();
        (header.width, header.height) = (160, 2);
        let codebook = [[1; 8], [2; 8]].concat();
        for bad in [20, 37] {
            let mut decoder = FrameDecoder::new(&header).unwrap();
            let mut table = vec![0; 80];
            for block in 0..40 {
                (table[block], table[40 + block]) = match block {
                    _ if block == bad => (5, 0),      // entry 5 of 2
                    _ if block % 3 == 0 => (7, 0x0f), // a fill with color 7
                    _ => (block as u8 % 2, 0),
                };
            }
            let mut vqfr = chunk("CBF0", &codebook);
            vqfr.extend(chunk("VPT0", &table));
            assert_eq!(
                decoder.decode_frame(&vqfr).map_err(|e| e.kind()),
                Err(ErrorKind::Video(VideoError::BlockIndexOutOfRange))
            );

            let frame = decoder.decode_frame_ref(&[]).unwrap();
            let FramePixelsRef::Indexed { pixels, .. } = frame.pixels else {
                panic!("expected an indexed frame");
            };
            for block in 0..40 {
                let color = match block {
                    _ if block >= bad => 0, // never drawn
                    _ if block % 3 == 0 => 7,
                    _ => block as u8 % 2 + 1,
                };
                for line in 0..2 {
                    let at = line * 160 + block * 4;
                    assert_eq!(pixels[at..at + 4], [color; 4], "block {block} of bad {bad}");
                }
            }
        }
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
