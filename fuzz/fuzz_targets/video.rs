#![no_main]

use libfuzzer_sys::fuzz_target;
use vqa::lcw::{self, LcwError};
use vqa::{
    ErrorKind, FrameDecoder, FramePixelsRef, FrameRef, Limit, VQAHeader, VQAVersion, VideoError,
};

/// The sub-chunks a record can become, by its kind byte.
const IDS: [&[u8; 4]; 8] = [
    b"CBF0", b"VPT0", b"VPTR", b"CPL0", b"CBFZ", b"VPTZ", b"VPRZ", b"CPLZ",
];

fuzz_target!(|data: &[u8]| {
    // Three bytes pick the movie: version and pixel format, block size,
    // frame size and codebook limit. The rest is a list of records - a kind
    // byte, a little-endian u16 length, and that many bytes - each of which
    // becomes one frame of a single sub-chunk. The decoder must agree with
    // the plain per-pixel renderer below on every frame: the error, if any,
    // and all the pixels, including those a failed chunk leaves drawn, and
    // the pixels' RGB conversions.
    let Some((&[format, size, dims], mut rest)) = data.split_first_chunk::<3>() else {
        return;
    };
    let header = header(format, size, dims);
    let mut decoder = FrameDecoder::new(&header).expect("the header is valid");
    let mut reference = Reference::new(&header);

    for _ in 0..32 {
        let Some((&kind, tail)) = rest.split_first() else {
            break;
        };
        let len = tail
            .first_chunk::<2>()
            .map_or(0, |&len| usize::from(u16::from_le_bytes(len)));
        let tail = tail.get(2..).unwrap_or_default();
        let (body, tail) = tail.split_at(len.min(tail.len()));
        rest = tail;

        let id = IDS[usize::from(kind) % IDS.len()];
        let mut payload = id.to_vec();
        payload.extend((body.len() as u32).to_be_bytes());
        payload.extend(body);
        if body.len() % 2 == 1 {
            payload.push(0);
        }

        let error = decoder.decode_frame_ref(&payload).err().map(|e| e.kind());
        assert_eq!(
            error,
            reference.apply(id, body).err(),
            "{:?}",
            std::str::from_utf8(id)
        );
        // a frame of no chunks shows what the last one left
        let frame = decoder
            .decode_frame_ref(&[])
            .expect("an empty frame decodes");
        reference.check(frame);
    }
});

/// The movie `format`, `size` and `dims` pick.
fn header(format: u8, size: u8, dims: u8) -> VQAHeader {
    let (block_width, block_height) = match size {
        0..=95 => (4, 2),
        96..=191 => (4, 4),
        _ => (1 + (size & 7), 1 + (size >> 3 & 7)),
    };
    let (blocks_x, blocks_y) = (u16::from(1 + (dims & 15)), u16::from(1 + (dims >> 4)));
    // some frames a pixel or two wider or taller than their blocks cover
    let (extra_x, extra_y) = (u16::from(format >> 4 & 3), u16::from(format >> 6));
    VQAHeader {
        version: match format & 3 {
            0 => VQAVersion::One,
            1 => VQAVersion::Two,
            _ => VQAVersion::Three,
        },
        flags: 0,
        num_frames: 32,
        width: blocks_x * u16::from(block_width) + extra_x,
        height: blocks_y * u16::from(block_height) + extra_y,
        block_width,
        block_height,
        frame_rate: 15,
        cbparts: 0,
        colors: if format & 4 != 0 { 0 } else { 256 },
        // 0 is the default of 0xff00 entries; a handful makes the codebook
        // limit reachable
        maxblocks: if format & 8 != 0 { u16::from(dims) } else { 0 },
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

/// The frame assembly as plainly as it can be written: every pixel drawn
/// one at a time, every index checked where it's used.
struct Reference {
    version: VQAVersion,
    hicolor: bool,
    width: usize,
    block_w: usize,
    block_h: usize,
    blocks_x: usize,
    blocks: usize,
    max_codebook_bytes: usize,
    /// the codebook, whole entries only: palette indices or little-endian
    /// 15-bit pixels
    codebook: Vec<u8>,
    palette: Vec<[u8; 3]>,
    pixels8: Vec<u8>,
    pixels16: Vec<u16>,
}

impl Reference {
    fn new(header: &VQAHeader) -> Reference {
        let (width, height) = (usize::from(header.width), usize::from(header.height));
        let (block_w, block_h) = (
            usize::from(header.block_width),
            usize::from(header.block_height),
        );
        let hicolor = header.colors == 0;
        let max_blocks = match header.maxblocks {
            0 => 0xff00,
            n => usize::from(n),
        };
        Reference {
            version: header.version,
            hicolor,
            width,
            block_w,
            block_h,
            blocks_x: width / block_w,
            blocks: (width / block_w) * (height / block_h),
            max_codebook_bytes: (max_blocks * block_w * block_h * if hicolor { 2 } else { 1 })
                .min(1 << 24),
            codebook: Vec::new(),
            palette: Vec::new(),
            pixels8: if hicolor {
                Vec::new()
            } else {
                vec![0; width * height]
            },
            pixels16: if hicolor {
                vec![0; width * height]
            } else {
                Vec::new()
            },
        }
    }

    fn apply(&mut self, id: &[u8; 4], body: &[u8]) -> Result<(), ErrorKind> {
        match id {
            b"CBF0" => self.set_codebook(body),
            b"CBFZ" => {
                let too_large = ErrorKind::TooLarge(Limit::Codebook);
                self.set_codebook(&decompress(body, self.max_codebook_bytes, too_large)?)
            }
            b"VPT0" => self.render_vpt(body),
            b"VPTZ" => {
                let size = ErrorKind::Video(VideoError::PointerTableSize);
                self.render_vpt(&decompress(body, self.blocks * 2, size)?)
            }
            b"VPTR" => self.render_vptr(body),
            b"VPRZ" => {
                let stream =
                    lcw::decompress(body, self.blocks * 8 + 256).map_err(ErrorKind::Lcw)?;
                self.render_vptr(&stream)
            }
            b"CPL0" => self.set_palette(body),
            b"CPLZ" => {
                let size = ErrorKind::Video(VideoError::PaletteSize);
                self.set_palette(&decompress(body, 256 * 3, size)?)
            }
            _ => unreachable!(),
        }
    }

    fn entry_bytes(&self) -> usize {
        self.block_w * self.block_h * if self.hicolor { 2 } else { 1 }
    }

    fn set_codebook(&mut self, bytes: &[u8]) -> Result<(), ErrorKind> {
        if bytes.len() > self.max_codebook_bytes {
            return Err(ErrorKind::TooLarge(Limit::Codebook));
        }
        let whole = bytes.len() - bytes.len() % self.entry_bytes();
        self.codebook = bytes[..whole].to_vec();
        Ok(())
    }

    fn set_palette(&mut self, data: &[u8]) -> Result<(), ErrorKind> {
        if data.len() % 3 != 0 || data.len() > 256 * 3 {
            return Err(ErrorKind::Video(VideoError::PaletteSize));
        }
        let scale = |v: u8| (v & 0x3f) << 2 | (v & 0x3f) >> 4;
        self.palette = data
            .chunks(3)
            .map(|c| [scale(c[0]), scale(c[1]), scale(c[2])])
            .collect();
        Ok(())
    }

    /// The frame index of pixel (x, y) of block `block`.
    fn at(&self, block: usize, x: usize, y: usize) -> usize {
        let (bx, by) = (block % self.blocks_x, block / self.blocks_x);
        (by * self.block_h + y) * self.width + bx * self.block_w + x
    }

    fn render_vpt(&mut self, table: &[u8]) -> Result<(), ErrorKind> {
        if self.hicolor {
            return Err(ErrorKind::Video(VideoError::WrongPointerFormat));
        }
        if table.len() != self.blocks * 2 {
            return Err(ErrorKind::Video(VideoError::PointerTableSize));
        }
        let fill_sentinel = if self.block_h == 4 { 0xff } else { 0x0f };
        for block in 0..self.blocks {
            let (fill, index) = match self.version {
                VQAVersion::One => {
                    let (lo, hi) = (table[block * 2], table[block * 2 + 1]);
                    let index = (usize::from(hi) << 8 | usize::from(lo)) / 8;
                    ((hi == 0xff).then_some(255 - lo), index)
                }
                _ => {
                    let (lo, hi) = (table[block], table[self.blocks + block]);
                    let index = usize::from(hi) << 8 | usize::from(lo);
                    ((hi == fill_sentinel).then_some(lo), index)
                }
            };
            let entry = self.block_w * self.block_h;
            if fill.is_none() && (index + 1) * entry > self.codebook.len() {
                return Err(ErrorKind::Video(VideoError::BlockIndexOutOfRange));
            }
            for y in 0..self.block_h {
                for x in 0..self.block_w {
                    let at = self.at(block, x, y);
                    self.pixels8[at] = match fill {
                        Some(color) => color,
                        None => self.codebook[index * entry + y * self.block_w + x],
                    };
                }
            }
        }
        Ok(())
    }

    fn render_vptr(&mut self, stream: &[u8]) -> Result<(), ErrorKind> {
        if !self.hicolor {
            return Err(ErrorKind::Video(VideoError::WrongPointerFormat));
        }
        let truncated = ErrorKind::Video(VideoError::TruncatedPointerStream);
        let mut pos = 0usize;
        let mut sp = 0;
        while sp < stream.len() {
            let val = match stream.get(sp..sp + 2) {
                Some(b) => u16::from_le_bytes([b[0], b[1]]),
                None => return Err(truncated),
            };
            sp += 2;
            let run = usize::from(val >> 8 & 0x1f) + 1;
            match val >> 13 {
                0b000 => pos = pos.saturating_add(usize::from(val & 0x1fff)),
                0b001 => {
                    for _ in 0..run * 2 {
                        self.write(&mut pos, usize::from(val & 0xff), false)?;
                    }
                }
                0b010 => {
                    self.write(&mut pos, usize::from(val & 0xff), false)?;
                    for _ in 0..run * 2 {
                        let index = *stream.get(sp).ok_or(truncated)?;
                        sp += 1;
                        self.write(&mut pos, usize::from(index), false)?;
                    }
                }
                0b011 => self.write(&mut pos, usize::from(val & 0x1fff), false)?,
                0b100 => self.write(&mut pos, usize::from(val & 0x1fff), true)?,
                0b101 | 0b110 => {
                    let count = *stream.get(sp).ok_or(truncated)?;
                    sp += 1;
                    for _ in 0..count {
                        self.write(&mut pos, usize::from(val & 0x1fff), val >> 13 == 0b110)?;
                    }
                }
                _ => return Err(ErrorKind::Video(VideoError::UnknownPointerCommand)),
            }
        }
        Ok(())
    }

    /// Write HiColor codebook entry `index` at block `pos`, and move on.
    fn write(&mut self, pos: &mut usize, index: usize, alpha_skip: bool) -> Result<(), ErrorKind> {
        if *pos >= self.blocks {
            return Err(ErrorKind::Video(VideoError::PointerStreamOverrun));
        }
        let entry = self.block_w * self.block_h;
        // a stray index past the codebook leaves the block as it was
        if (index + 1) * entry * 2 <= self.codebook.len() {
            for y in 0..self.block_h {
                for x in 0..self.block_w {
                    let byte = (index * entry + y * self.block_w + x) * 2;
                    let pixel = u16::from_le_bytes([self.codebook[byte], self.codebook[byte + 1]]);
                    if !(alpha_skip && pixel & 0x8000 != 0) {
                        let at = self.at(*pos, x, y);
                        self.pixels16[at] = pixel;
                    }
                }
            }
        }
        *pos += 1;
        Ok(())
    }

    /// Check `frame` against the reference's, pixels and RGB conversions.
    fn check(&self, frame: FrameRef<'_>) {
        let rgb: Vec<[u8; 3]> = match frame.pixels {
            FramePixelsRef::Indexed { pixels, palette } => {
                assert!(!self.hicolor);
                assert_eq!(pixels, self.pixels8, "8-bit pixels");
                assert_eq!(palette, self.palette, "palette");
                let color = |&p: &u8| {
                    self.palette
                        .get(usize::from(p))
                        .copied()
                        .unwrap_or_default()
                };
                pixels.iter().map(color).collect()
            }
            FramePixelsRef::HiColor { pixels } => {
                assert!(self.hicolor);
                assert_eq!(pixels, self.pixels16, "HiColor pixels");
                let scale = |v: u16| (v << 3 | v >> 2) as u8;
                let color = |&p: &u16| [scale(p >> 10 & 31), scale(p >> 5 & 31), scale(p & 31)];
                pixels.iter().map(color).collect()
            }
        };
        assert_eq!(frame.to_rgb888(), rgb.as_flattened(), "RGB888");
        let rgba: Vec<u8> = rgb.iter().flat_map(|&[r, g, b]| [r, g, b, 0xff]).collect();
        assert_eq!(frame.to_rgba8888(), rgba, "RGBA8888");
        let xrgb: Vec<u32> = rgb
            .iter()
            .map(|&[r, g, b]| u32::from_be_bytes([0, r, g, b]))
            .collect();
        assert_eq!(frame.to_xrgb8888(), xrgb, "XRGB8888");
    }
}

/// Decompress LCW data whose output can't be longer than `max` bytes: more
/// fails as `too_long`, as the same data uncompressed would.
fn decompress(data: &[u8], max: usize, too_long: ErrorKind) -> Result<Vec<u8>, ErrorKind> {
    lcw::decompress(data, max).map_err(|e| match e {
        LcwError::TooLarge => too_long,
        e => ErrorKind::Lcw(e),
    })
}
