//! Low-level nom parsers for the individual chunks of a VQA file.
//!
//! Every parser here consumes one chunk - tag, big-endian size, payload, and
//! the pad byte that follows an odd-sized payload - and borrows the payload
//! from the input (zero-copy). For whole-file access prefer [`crate::VQA`],
//! which composes these parsers and handles chunk types this module has no
//! dedicated parser for.

use std::convert::TryInto;

use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::{tag, take},
    combinator::{cond, map, map_opt, opt, value, verify},
    multi::count,
    number::complete::{be_u32, le_u16, le_u32, u8},
};

/// Take `size` bytes of chunk payload, also consuming the pad byte that
/// follows an odd-sized chunk - chunks always start at even offsets. The pad
/// byte may be absent when the chunk ends the input.
fn chunk_data<'a>(size: u32) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], &'a [u8]> {
    move |input| {
        let (input, data) = take(size).parse(input)?;
        let (input, _) = cond(size % 2 == 1, opt(take(1usize))).parse(input)?;

        Ok((input, data))
    }
}

/// The prelude shared by every chunk parser: the chunk's 4-character ID, its
/// big-endian size, and the (padded) payload.
fn plain_chunk<'a>(id: &'static str) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], (u32, &'a [u8])> {
    move |input| {
        let (input, _) = tag(id).parse(input)?;
        let (input, size) = be_u32(input)?;
        let (input, data) = chunk_data(size)(input)?;

        Ok((input, (size, data)))
    }
}

/// `(compressed, size, data)` of a chunk parsed by [`compressible_chunk`]
type CompressiblePayload<'a> = (bool, u32, &'a [u8]);

/// A chunk whose last ID character selects LCW compression: `Z` means the
/// payload is compressed, `0` that it is raw.
fn compressible_chunk<'a>(
    prefix: &'static str,
) -> impl FnMut(&'a [u8]) -> IResult<&'a [u8], CompressiblePayload<'a>> {
    move |input| {
        let (input, _) = tag(prefix).parse(input)?;
        let (input, compressed) =
            alt((value(true, tag("Z")), value(false, tag("0")))).parse(input)?;
        let (input, size) = be_u32(input)?;
        let (input, data) = chunk_data(size)(input)?;

        Ok((input, (compressed, size, data)))
    }
}

/// Any chunk, without knowing its type: the generic escape hatch for walking
/// over chunks this crate has no dedicated parser for (LINF, CINF, PINF, ...).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawChunk<'a> {
    /// The chunk's 4-character ID, e.g. `b"SND2"`
    pub id: [u8; 4],
    /// Payload size in bytes.
    pub size: u32,
    /// The raw payload, borrowed from the input.
    pub data: &'a [u8],
}

/// Parse any single chunk. IDs are validated to be uppercase ASCII letters
/// and digits, so a desynced or corrupt stream fails instead of producing
/// nonsense chunks.
pub fn raw_chunk(input: &[u8]) -> IResult<&[u8], RawChunk<'_>> {
    let (input, id) = verify(take(4usize), |id: &[u8]| {
        id.iter()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
    })
    .parse(input)?;
    let (input, size) = be_u32(input)?;
    let (input, data) = chunk_data(size)(input)?;

    Ok((
        input,
        RawChunk {
            id: id.try_into().expect("take(4) yields 4 bytes"),
            size,
            data,
        },
    ))
}

/// The outer `FORM` container that wraps the whole file (an EA IFF 85 form).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormChunk {
    /// Size of the form's contents in bytes. In v2/v3 movies this is the
    /// file size minus the 8-byte chunk header.
    pub size: u32,
}

/// Parse the `FORM` chunk that opens every VQA file. Consumes only the tag
/// and size; the form's contents (the `WVQA` signature followed by all other
/// chunks) stay in the input.
pub fn form_chunk(input: &[u8]) -> IResult<&[u8], FormChunk> {
    let (input, _) = tag("FORM").parse(input)?;
    let (input, size) = be_u32(input)?;

    Ok((input, FormChunk { size }))
}

/// The format version stored in the header.
///
/// The version alone does not imply the pixel format: most HiColor movies
/// are v3, but some v2 movies (Dune 2000) are HiColor too - check
/// [`VQAHeader::is_hicolor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VQAVersion {
    /// Version 1, used only in The Legend of Kyrandia III.
    One,
    /// Version 2, used in C&C, Red Alert, Lands of Lore II, and Dune 2000.
    Two,
    /// Version 3, used in the HiColor-era games (Tiberian Sun, Lands of
    /// Lore III, Blade Runner, Nox).
    Three,
}

/// Parse the header's little-endian version field; values other than 1-3
/// fail.
pub fn vqa_version(input: &[u8]) -> IResult<&[u8], VQAVersion> {
    map_opt(le_u16, |n| match n {
        1 => Some(VQAVersion::One),
        2 => Some(VQAVersion::Two),
        3 => Some(VQAVersion::Three),
        _ => None,
    })
    .parse(input)
}

/// The fixed 42-byte `VQHD` header describing the whole movie.
///
/// v1 movies leave several sound fields zeroed; the [`sample_rate`],
/// [`num_channels`], and [`bit_depth`] helpers apply the documented
/// fallbacks, so prefer them over reading `freq`, `channels`, and `bits`
/// directly.
///
/// [`sample_rate`]: VQAHeader::sample_rate
/// [`num_channels`]: VQAHeader::num_channels
/// [`bit_depth`]: VQAHeader::bit_depth
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VQAHeader {
    /// VQA version number
    pub version: VQAVersion,
    /// Flag bits. Bit 0 marks a soundtrack (see [`has_sound`]); the HiColor
    /// movies seen so far also set bits 2-4, whose meaning is unknown.
    ///
    /// [`has_sound`]: VQAHeader::has_sound
    pub flags: u16,
    /// Number of frames
    pub num_frames: u16,
    /// Movie width (pixels)
    pub width: u16,
    /// Movie height (pixels)
    pub height: u16,
    /// Width of each image block (pixels)
    pub block_width: u8,
    /// Height of each image block (pixels)
    pub block_height: u8,
    /// Frame rate of the VQA
    pub frame_rate: u8,
    /// How many images use the same lookup table
    pub cbparts: u8,
    /// Max number of colors used in VQA
    pub colors: u16,
    /// Max number of image blocks
    pub maxblocks: u16,
    /// Always 0?
    pub unk1: u32,
    /// Some kind of size?
    pub unk2: u16,
    /// Sound sampling frequency
    pub freq: u16,
    /// Number of sound channels
    pub channels: u8,
    /// Sound resolution
    pub bits: u8,
    /// Always 0?
    pub unk3: u32,
    /// 0 in old VQAs, 4 in HiColor VQAs?
    pub unk4: u16,
    /// 0 in old VQAs, CBFZ size in HiColor
    pub max_cbfz_size: u32,
    /// Always 0?
    pub unk5: u32,
}

impl VQAHeader {
    /// The sound sampling rate in Hz, applying the documented v1 fallback
    /// (a stored 0 means 22050 Hz).
    pub fn sample_rate(&self) -> u32 {
        match self.freq {
            0 => 22050,
            freq => u32::from(freq),
        }
    }

    /// The number of sound channels (a stored 0 means mono).
    pub fn num_channels(&self) -> u8 {
        match self.channels {
            0 => 1,
            channels => channels,
        }
    }

    /// The sound resolution in bits (a stored 0 means 8-bit).
    pub fn bit_depth(&self) -> u8 {
        match self.bits {
            0 => 8,
            bits => bits,
        }
    }

    /// Whether the movie carries a soundtrack (bit 0 of `flags`).
    pub fn has_sound(&self) -> bool {
        self.flags & 1 != 0
    }

    /// Whether the movie is HiColor (15-bit pixels) rather than 8-bit
    /// palettized. HiColor movies store 0 in the `colors` field.
    pub fn is_hicolor(&self) -> bool {
        self.colors == 0
    }
}

/// Parse the `VQHD` header chunk: the tag, the fixed 42-byte size, and every
/// header field.
pub fn vqa_header(input: &[u8]) -> IResult<&[u8], VQAHeader> {
    let (input, _) = tag("VQHD").parse(input)?;
    let (input, _) = verify(be_u32, |&size| size == 42).parse(input)?;
    let (input, version) = vqa_version(input)?;
    let (input, flags) = le_u16(input)?;
    let (input, num_frames) = le_u16(input)?;
    let (input, width) = le_u16(input)?;
    let (input, height) = le_u16(input)?;
    let (input, block_width) = u8(input)?;
    let (input, block_height) = u8(input)?;
    let (input, frame_rate) = u8(input)?;
    let (input, cbparts) = u8(input)?;
    let (input, colors) = le_u16(input)?;
    let (input, maxblocks) = le_u16(input)?;
    let (input, unk1) = le_u32(input)?;
    let (input, unk2) = le_u16(input)?;
    let (input, freq) = le_u16(input)?;
    let (input, channels) = u8(input)?;
    let (input, bits) = u8(input)?;
    let (input, unk3) = le_u32(input)?;
    let (input, unk4) = le_u16(input)?;
    let (input, max_cbfz_size) = le_u32(input)?;
    let (input, unk5) = le_u32(input)?;

    Ok((
        input,
        VQAHeader {
            version,
            flags,
            num_frames,
            width,
            height,
            block_width,
            block_height,
            frame_rate,
            cbparts,
            colors,
            maxblocks,
            unk1,
            unk2,
            freq,
            channels,
            bits,
            unk3,
            unk4,
            max_cbfz_size,
            unk5,
        },
    ))
}

/// Position of one frame's data, decoded from a FINF entry.
///
/// Stored FINF values are in 16-bit words with bit 30 flagging a new palette;
/// `offset` is the decoded absolute byte position of the frame's data
/// (its SND? chunk when the movie has sound, its VQFR chunk otherwise).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameInfo {
    /// Absolute byte offset of the frame's data from the start of the file.
    pub offset: u32,
    /// Whether the frame carries a new palette (stored bit 30).
    pub has_palette: bool,
}

const FINF_PALETTE_FLAG: u32 = 0x4000_0000;

/// Parse one raw FINF entry, applying the transforms described on
/// [`FrameInfo`].
pub fn frame_info(input: &[u8]) -> IResult<&[u8], FrameInfo> {
    map(le_u32, |raw| FrameInfo {
        offset: (raw & (FINF_PALETTE_FLAG - 1)) * 2,
        has_palette: raw & FINF_PALETTE_FLAG != 0,
    })
    .parse(input)
}

/// The frame index, sitting between the header and the first frame's data:
/// the position of every frame in the file, in frame order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FINFChunk {
    /// Payload size in bytes (4 bytes per frame).
    pub size: u32,
    /// One decoded entry per frame.
    pub frames: Vec<FrameInfo>,
}

/// Parse the `FINF` chunk into decoded per-frame positions.
pub fn finf_chunk(input: &[u8]) -> IResult<&[u8], FINFChunk> {
    let (input, _) = tag("FINF").parse(input)?;
    let (input, size) = be_u32(input)?;
    let (input, frames) = count(frame_info, size as usize / 4).parse(input)?;

    Ok((input, FINFChunk { size, frames }))
}

/// A sound chunk holding IMA ADPCM compressed 16-bit samples; decode with
/// [`crate::audio::decompress`]. Stereo data is laid out per VQA version:
/// v3 stores the left channel in the first half of the chunk and the right
/// in the second, v1/v2 alternate bytes between the channels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SND2Chunk<'a> {
    /// Payload size in bytes.
    pub size: u32,
    /// The compressed sample data, borrowed from the input.
    pub data: &'a [u8],
}

/// Parse an `SND2` (IMA ADPCM audio) chunk.
pub fn snd2_chunk(input: &[u8]) -> IResult<&[u8], SND2Chunk<'_>> {
    let (input, (size, data)) = plain_chunk("SND2")(input)?;

    Ok((input, SND2Chunk { size, data }))
}

/// One frame's video data; `data` holds the nested sub-chunks (CBF?, CBP?,
/// CPL?, VPT?, VPTR/VPRZ).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VQFRChunk<'a> {
    /// Payload size in bytes.
    pub size: u32,
    /// The nested sub-chunks, unparsed, borrowed from the input.
    pub data: &'a [u8],
}

/// Parse a `VQFR` (video frame) chunk; its sub-chunks stay unparsed in
/// `data`.
pub fn vqfr_chunk(input: &[u8]) -> IResult<&[u8], VQFRChunk<'_>> {
    let (input, (size, data)) = plain_chunk("VQFR")(input)?;

    Ok((input, VQFRChunk { size, data }))
}

/// A full codebook (block lookup table), replacing the current one. In
/// HiColor movies a compressed payload starting with a NUL byte uses the
/// relative LCW variant ([`crate::lcw::decompress`] handles both).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CBFChunk<'a> {
    /// Payload size in bytes.
    pub size: u32,
    /// Whether `data` is LCW-compressed (a `CBFZ` chunk).
    pub compressed: bool,
    /// The codebook data, borrowed from the input.
    pub data: &'a [u8],
}

/// Parse a `CBF0`/`CBFZ` (full codebook) chunk.
pub fn cbf_chunk(input: &[u8]) -> IResult<&[u8], CBFChunk<'_>> {
    let (input, (compressed, size, data)) = compressible_chunk("CBF")(input)?;

    Ok((
        input,
        CBFChunk {
            size,
            compressed,
            data,
        },
    ))
}

/// One part of the next codebook. `cbparts` (from the header) parts are
/// concatenated in frame order to form the new codebook; compressed parts
/// are decompressed only after concatenation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CBPChunk<'a> {
    /// Payload size in bytes.
    pub size: u32,
    /// Whether the *concatenated* parts are LCW-compressed (`CBPZ` chunks).
    pub compressed: bool,
    /// This part's bytes, borrowed from the input.
    pub data: &'a [u8],
}

/// Parse a `CBP0`/`CBPZ` (codebook part) chunk.
pub fn cbp_chunk(input: &[u8]) -> IResult<&[u8], CBPChunk<'_>> {
    let (input, (compressed, size, data)) = compressible_chunk("CBP")(input)?;

    Ok((
        input,
        CBPChunk {
            size,
            compressed,
            data,
        },
    ))
}

/// A color palette: consecutive R, G, B bytes per color. Only the low 6 bits
/// of each value are significant (VGA hardware).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CPLChunk<'a> {
    /// Payload size in bytes.
    pub size: u32,
    /// Whether `data` is LCW-compressed (a `CPLZ` chunk).
    pub compressed: bool,
    /// The palette data, borrowed from the input.
    pub data: &'a [u8],
}

/// Parse a `CPL0`/`CPLZ` (palette) chunk.
pub fn cpl_chunk(input: &[u8]) -> IResult<&[u8], CPLChunk<'_>> {
    let (input, (compressed, size, data)) = compressible_chunk("CPL")(input)?;

    Ok((
        input,
        CPLChunk {
            size,
            compressed,
            data,
        },
    ))
}

/// A vector pointer table for one 8-bit frame: per-block indexes into the
/// codebook (layout differs between v1 and v2, see `doc/vqa.txt`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VPTChunk<'a> {
    /// Payload size in bytes.
    pub size: u32,
    /// Whether `data` is LCW-compressed (a `VPTZ` chunk).
    pub compressed: bool,
    /// The pointer table, borrowed from the input.
    pub data: &'a [u8],
}

/// Parse a `VPT0`/`VPTZ` (8-bit vector pointer table) chunk.
pub fn vpt_chunk(input: &[u8]) -> IResult<&[u8], VPTChunk<'_>> {
    let (input, (compressed, size, data)) = compressible_chunk("VPT")(input)?;

    Ok((
        input,
        VPTChunk {
            size,
            compressed,
            data,
        },
    ))
}

/// A HiColor vector pointer stream (`VPTR` raw, `VPRZ` LCW-compressed): a
/// differential command stream updating the previous frame's blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VPTRChunk<'a> {
    /// Payload size in bytes.
    pub size: u32,
    /// Whether `data` is LCW-compressed (a `VPRZ` chunk).
    pub compressed: bool,
    /// The command stream, borrowed from the input.
    pub data: &'a [u8],
}

/// Parse a `VPTR`/`VPRZ` (HiColor vector pointer stream) chunk.
pub fn vptr_chunk(input: &[u8]) -> IResult<&[u8], VPTRChunk<'_>> {
    let (input, compressed) =
        alt((value(false, tag("VPTR")), value(true, tag("VPRZ")))).parse(input)?;
    let (input, size) = be_u32(input)?;
    let (input, data) = chunk_data(size)(input)?;

    Ok((
        input,
        VPTRChunk {
            size,
            compressed,
            data,
        },
    ))
}

/// A HiColor full-codebook refresh between frames; `data` holds the nested
/// sub-chunks (in practice a single CBFZ).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VQFLChunk<'a> {
    /// Payload size in bytes.
    pub size: u32,
    /// The nested sub-chunks, unparsed, borrowed from the input.
    pub data: &'a [u8],
}

/// Parse a `VQFL` (HiColor codebook refresh) chunk.
pub fn vqfl_chunk(input: &[u8]) -> IResult<&[u8], VQFLChunk<'_>> {
    let (input, (size, data)) = plain_chunk("VQFL")(input)?;

    Ok((input, VQFLChunk { size, data }))
}

/// An IMA ADPCM seek-state chunk accompanying SND2 data (decoder state for
/// jumping into the stream); not needed for sequential playback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SN2JChunk<'a> {
    /// Payload size in bytes.
    pub size: u32,
    /// The seek-state data, borrowed from the input.
    pub data: &'a [u8],
}

/// Parse an `SN2J` (ADPCM seek state) chunk.
pub fn sn2j_chunk(input: &[u8]) -> IResult<&[u8], SN2JChunk<'_>> {
    let (input, (size, data)) = plain_chunk("SN2J")(input)?;

    Ok((input, SN2JChunk { size, data }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successfully_parses_form_chunk() {
        let input = b"FORM\x00\x00\x00\x10trailing";
        let res = form_chunk(input);

        assert!(matches!(res, Ok((b"trailing", FormChunk { size: 16 }))));
    }

    #[test]
    fn successfully_parses_vqa_version() {
        assert!(matches!(
            vqa_version(b"\x01\x00trailing"),
            Ok((b"trailing", VQAVersion::One))
        ));

        assert!(matches!(
            vqa_version(b"\x02\x00trailing"),
            Ok((b"trailing", VQAVersion::Two))
        ));

        assert!(matches!(
            vqa_version(b"\x03\x00trailing"),
            Ok((b"trailing", VQAVersion::Three))
        ));

        assert!(vqa_version(b"\x04\x00trailing").is_err());
    }

    #[test]
    fn snd2_chunk_leaves_even_sized_payload_unpadded() {
        let input = b"SND2\x00\x00\x00\x04abcdnext";
        let (rest, chunk) = snd2_chunk(input).unwrap();

        assert_eq!(chunk.size, 4);
        assert_eq!(chunk.data, b"abcd");
        assert_eq!(rest, b"next");
    }

    #[test]
    fn snd2_chunk_consumes_pad_byte_after_odd_sized_payload() {
        let input = b"SND2\x00\x00\x00\x03abc\x00next";
        let (rest, chunk) = snd2_chunk(input).unwrap();

        assert_eq!(chunk.size, 3);
        assert_eq!(chunk.data, b"abc");
        assert_eq!(rest, b"next");
    }

    #[test]
    fn odd_sized_chunk_at_end_of_input_needs_no_pad_byte() {
        let input = b"SND2\x00\x00\x00\x03abc";
        let (rest, chunk) = snd2_chunk(input).unwrap();

        assert_eq!(chunk.data, b"abc");
        assert!(rest.is_empty());
    }

    #[test]
    fn cbf_chunk_parses_both_variants_and_pads() {
        let (rest, chunk) = cbf_chunk(b"CBFZ\x00\x00\x00\x03abc\x00next").unwrap();
        assert!(chunk.compressed);
        assert_eq!(chunk.data, b"abc");
        assert_eq!(rest, b"next");

        let (rest, chunk) = cbf_chunk(b"CBF0\x00\x00\x00\x04abcdnext").unwrap();
        assert!(!chunk.compressed);
        assert_eq!(chunk.data, b"abcd");
        assert_eq!(rest, b"next");
    }

    #[test]
    fn vqfr_chunk_consumes_pad_byte_after_odd_sized_payload() {
        let (rest, chunk) = vqfr_chunk(b"VQFR\x00\x00\x00\x05abcde\x00next").unwrap();
        assert_eq!(chunk.data, b"abcde");
        assert_eq!(rest, b"next");
    }

    #[test]
    fn raw_chunk_parses_any_id_and_pads() {
        let (rest, chunk) = raw_chunk(b"LINF\x00\x00\x00\x03abc\x00next").unwrap();
        assert_eq!(&chunk.id, b"LINF");
        assert_eq!(chunk.data, b"abc");
        assert_eq!(rest, b"next");
    }

    #[test]
    fn raw_chunk_rejects_non_chunk_ids() {
        assert!(raw_chunk(b"lin \x00\x00\x00\x00").is_err());
        assert!(raw_chunk(b"\x00\x01\x02\x03\x00\x00\x00\x00").is_err());
    }

    #[test]
    fn cbp_cpl_vpt_chunks_parse_both_variants() {
        let (_, chunk) = cbp_chunk(b"CBPZ\x00\x00\x00\x02ab").unwrap();
        assert!(chunk.compressed);
        assert_eq!(chunk.data, b"ab");

        let (_, chunk) = cpl_chunk(b"CPL0\x00\x00\x00\x03rgb\x00").unwrap();
        assert!(!chunk.compressed);
        assert_eq!(chunk.data, b"rgb");

        let (_, chunk) = vpt_chunk(b"VPTZ\x00\x00\x00\x02ab").unwrap();
        assert!(chunk.compressed);
        assert_eq!(chunk.data, b"ab");
    }

    #[test]
    fn vptr_chunk_parses_raw_and_compressed_ids() {
        let (_, chunk) = vptr_chunk(b"VPTR\x00\x00\x00\x02ab").unwrap();
        assert!(!chunk.compressed);
        assert_eq!(chunk.data, b"ab");

        let (_, chunk) = vptr_chunk(b"VPRZ\x00\x00\x00\x02ab").unwrap();
        assert!(chunk.compressed);

        // VPT0/VPTZ are a different chunk type
        assert!(vptr_chunk(b"VPTZ\x00\x00\x00\x02ab").is_err());
        assert!(vpt_chunk(b"VPTR\x00\x00\x00\x02ab").is_err());
    }

    #[test]
    fn vqfl_and_sn2j_chunks_parse() {
        let (_, chunk) = vqfl_chunk(b"VQFL\x00\x00\x00\x04abcd").unwrap();
        assert_eq!(chunk.data, b"abcd");

        let (_, chunk) = sn2j_chunk(b"SN2J\x00\x00\x00\x04abcd").unwrap();
        assert_eq!(chunk.data, b"abcd");
    }

    #[test]
    fn finf_chunk_decodes_frame_positions() {
        let mut input = b"FINF\x00\x00\x00\x08".to_vec();
        input.extend(100u32.to_le_bytes());
        input.extend((0x4000_0000u32 | 150).to_le_bytes());

        let (_, chunk) = finf_chunk(&input).unwrap();
        assert_eq!(
            chunk.frames,
            vec![
                FrameInfo {
                    offset: 200,
                    has_palette: false
                },
                FrameInfo {
                    offset: 300,
                    has_palette: true
                },
            ]
        );
    }
}
