//! Integration tests for the container layer (`VQA::parse`, `VQA::chunks`,
//! the `Frames` iterator, `VQA::frame_index`), the audio path
//! (`VQA::decode_audio`, `audio::decompress`, `CodecState`), the
//! `VQAHeader` helpers, and the `Error` type. Expected values are worked out
//! by hand on tiny inputs from doc/vqa.txt, doc/hc-vqa.txt and
//! doc/ima-adpcm.txt (IMA samples follow the latter's shift-and-add "usual
//! algorithm"), or are marked as locking current behavior.

mod common;

use std::collections::HashSet;
use std::error::Error as StdError;

use common::{chunk, header_8bit, header_hicolor, lcw_literals, movie, vqhd};
use vqa::audio::{CodecState, decompress};
use vqa::lcw::LcwError;
use vqa::{Error, Frame, FramePixels, VQA, VQAHeader, VQAVersion};

/// Wrap already-built chunks in `FORM` + `WVQA`, without adding a header,
/// so tests can supply a malformed VQHD.
fn form(chunks: &[Vec<u8>]) -> Vec<u8> {
    let mut body = b"WVQA".to_vec();
    for chunk in chunks {
        body.extend(chunk);
    }
    let mut file = b"FORM".to_vec();
    file.extend((body.len() as u32).to_be_bytes());
    file.extend(body);
    file
}

/// A header flagged as having sound, with the given stereo/mono setup.
fn sound_header(version: VQAVersion, channels: u8, bits: u8) -> VQAHeader {
    VQAHeader {
        version,
        flags: 1,
        channels,
        bits,
        ..header_8bit()
    }
}

fn decode_audio(file: &[u8]) -> Result<Vec<i16>, Error> {
    VQA::parse(file).expect("movie should parse").decode_audio()
}

// ---------------------------------------------------------------------------
// VQAHeader helpers

#[test]
fn parses_every_vqhd_field_in_documented_order() {
    // vqa.txt, VQHD chunk: the struct VQAHeader field order, 42 bytes, in
    // the "usual Intel" byte order (only chunk sizes are Motorola)
    #[rustfmt::skip]
    let payload = [
        0x02, 0x00,             // Version 2
        0x1d, 0x00,             // Flags
        0x02, 0x01,             // NumFrames 258
        0x40, 0x01,             // Width 320
        0xc8, 0x00,             // Height 200
        0x04, 0x02, 0x0f, 0x08, // BlockW, BlockH, FrameRate, CBParts
        0x00, 0x01,             // Colors 256
        0x00, 0x0f,             // MaxBlocks 0x0f00
        0x01, 0x02, 0x03, 0x04, // Unknown1
        0x05, 0x06,             // Unknown2
        0x22, 0x56,             // Freq 22050
        0x01, 0x10,             // Channels 1, Bits 16
        0x07, 0x08, 0x09, 0x0a, // Unknown3
        0x0b, 0x0c,             // Unknown4
        0x0d, 0x0e, 0x0f, 0x10, // MaxCBFZSize
        0x11, 0x12, 0x13, 0x14, // Unknown5
    ];
    let file = form(&[chunk(b"VQHD", &payload)]);
    let vqa = VQA::parse(&file).unwrap();

    assert_eq!(
        vqa.header,
        VQAHeader {
            version: VQAVersion::Two,
            flags: 0x001d,
            num_frames: 258,
            width: 320,
            height: 200,
            block_width: 4,
            block_height: 2,
            frame_rate: 15,
            cbparts: 8,
            colors: 256,
            maxblocks: 0x0f00,
            unk1: 0x0403_0201,
            unk2: 0x0605,
            freq: 22050,
            channels: 1,
            bits: 16,
            unk3: 0x0a09_0807,
            unk4: 0x0c0b,
            max_cbfz_size: 0x100f_0e0d,
            unk5: 0x1413_1211,
        }
    );
    // vqa.txt, FORM chunk: v2/v3 store the file size minus the 8-byte
    // chunk header
    assert_eq!(vqa.form_size as usize, file.len() - 8);
}

#[test]
fn zero_sound_fields_fall_back_to_the_v1_defaults() {
    // vqa.txt, VQHD chunk: v1 may store Freq 0 (use 22050 Hz), Channels 0
    // (use mono), and Bits 0 (assume 8 bits)
    let header = VQAHeader {
        version: VQAVersion::One,
        freq: 0,
        channels: 0,
        bits: 0,
        ..header_8bit()
    };
    let file = movie(&header, &[]);
    let header = VQA::parse(&file).unwrap().header;

    assert_eq!(header.sample_rate(), 22050);
    assert_eq!(header.num_channels(), 1);
    assert_eq!(header.bit_depth(), 8);
}

#[test]
fn nonzero_sound_fields_are_returned_as_stored() {
    // vqa.txt, VQHD chunk: Freq, Channels (1 mono, 2 stereo), Bits (8 or 16)
    for (freq, channels, bits) in [(11025, 2, 8), (44100, 1, 16), (22050, 2, 16)] {
        let header = VQAHeader {
            freq,
            channels,
            bits,
            ..header_8bit()
        };
        assert_eq!(header.sample_rate(), u32::from(freq));
        assert_eq!(header.num_channels(), channels);
        assert_eq!(header.bit_depth(), bits);
    }
}

#[test]
fn has_sound_reads_only_flags_bit_0() {
    // vqa.txt, VQHD chunk: bit 0 (LSB) of Flags marks a soundtrack; the
    // HiColor movies also set bits 2-4, which must not count
    for (flags, has_sound) in [
        (0x0000, false),
        (0x0001, true),
        (0x001c, false),
        (0x001d, true),
        (0xfffe, false),
        (0xffff, true),
    ] {
        let header = VQAHeader {
            flags,
            ..header_8bit()
        };
        assert_eq!(header.has_sound(), has_sound, "flags {flags:#06x}");
    }
}

#[test]
fn is_hicolor_means_colors_is_0() {
    // vqa.txt, VQHD chunk: HiColor movies store Colors 0, the old ones 256
    // or less; hc-vqa.txt: some v2 movies are HiColor too, so the version
    // does not decide it
    assert!(header_hicolor().is_hicolor());
    assert!(!header_8bit().is_hicolor());
    for (version, colors, hicolor) in [
        (VQAVersion::Two, 0, true),
        (VQAVersion::Three, 256, false),
        (VQAVersion::One, 1, false),
    ] {
        let header = VQAHeader {
            version,
            colors,
            ..header_8bit()
        };
        assert_eq!(header.is_hicolor(), hicolor, "{version:?}, colors {colors}");
    }
}

// ---------------------------------------------------------------------------
// VQA::parse failures

#[test]
fn parse_accepts_versions_1_to_3() {
    // vqa.txt, VQHD chunk: valid Version values are 1, 2 and 3
    for version in [VQAVersion::One, VQAVersion::Two, VQAVersion::Three] {
        let header = VQAHeader {
            version,
            ..header_8bit()
        };
        let file = movie(&header, &[]);
        assert_eq!(VQA::parse(&file).unwrap().header.version, version);
    }
}

#[test]
fn parse_rejects_files_that_are_not_form_files() {
    // vqa.txt, FORM chunk: the file is one FORM chunk
    let mut file = movie(&header_8bit(), &[]);
    file[..4].copy_from_slice(b"RIFF");
    assert_eq!(VQA::parse(&file), Err(Error::Parse));
    assert_eq!(VQA::parse(&[]), Err(Error::Parse));
    assert_eq!(VQA::parse(b"FOR"), Err(Error::Parse));
}

#[test]
fn parse_rejects_a_missing_wvqa_signature() {
    // vqa.txt, FORM chunk: "WVQA" follows the FORM chunk header
    let mut file = movie(&header_8bit(), &[]);
    assert_eq!(&file[8..12], b"WVQA");
    file[8..12].copy_from_slice(b"WAVE");
    assert_eq!(VQA::parse(&file), Err(Error::Parse));
}

#[test]
fn parse_rejects_a_vqhd_size_other_than_42() {
    // vqa.txt, VQHD chunk: its size is always 42 bytes. All 42 payload
    // bytes follow in every case, so only the size field is wrong: a
    // smaller size must fail on the check, not on running out of input
    for size in [0u32, 40, 44] {
        let mut vqhd_chunk = chunk(b"VQHD", &vqhd(&header_8bit()));
        vqhd_chunk[4..8].copy_from_slice(&size.to_be_bytes());
        let file = form(&[vqhd_chunk]);
        assert_eq!(VQA::parse(&file), Err(Error::Parse), "size {size}");
    }

    // and a 44-byte payload that really is 44 bytes long
    let mut long_payload = vqhd(&header_8bit());
    long_payload.extend([0, 0]);
    let long = form(&[chunk(b"VQHD", &long_payload)]);
    assert_eq!(VQA::parse(&long), Err(Error::Parse));
}

#[test]
fn parse_rejects_a_truncated_vqhd() {
    // vqa.txt, VQHD chunk: the header holds 42 bytes of fields; here the
    // size is right but the file ends after 20 of them
    let mut file = form(&[chunk(b"VQHD", &vqhd(&header_8bit()))]);
    file.truncate(file.len() - 22);
    assert_eq!(VQA::parse(&file), Err(Error::Parse));
}

#[test]
fn parse_rejects_versions_0_and_4() {
    // vqa.txt, VQHD chunk: Version is 1, 2 or 3
    for version in [0u16, 4] {
        let mut payload = vqhd(&header_8bit());
        payload[..2].copy_from_slice(&version.to_le_bytes());
        let file = form(&[chunk(b"VQHD", &payload)]);
        assert_eq!(VQA::parse(&file), Err(Error::Parse), "version {version}");
    }
}

// ---------------------------------------------------------------------------
// VQA::chunks

#[test]
fn chunks_walks_every_chunk_after_the_header_in_file_order() {
    // vqa.txt: every chunk is a 4-letter ID, a big-endian LONG size, then
    // the data; other chunks ("PINF, PINH, SN2J") can be skipped, and a
    // 0x00 byte keeps the chunk after an odd-sized one at an even offset
    // (FINF, NOTE #2). LINF and CINF are not in vqa.txt: the bundled
    // examples/wwlogo.vqa carries them. Offsets count from the start of the
    // file, where the body starts after FORM, WVQA and the 50-byte VQHD
    let file = movie(
        &header_8bit(),
        &[
            chunk(b"PINF", &[0xaa, 0xbb]),
            chunk(b"LINF", &[1, 2, 3, 4, 5, 6]),
            chunk(b"CINF", b"abc"),
            chunk(b"FINF", &[0; 4]),
            chunk(b"SND2", &[0x77]),
            chunk(b"VQFR", &[]),
        ],
    );
    let vqa = VQA::parse(&file).unwrap();

    let chunks: Vec<([u8; 4], usize, Vec<u8>)> = vqa
        .chunks()
        .map(|chunk| {
            let chunk = chunk.unwrap();
            (chunk.id, chunk.offset, chunk.data.to_vec())
        })
        .collect();
    assert_eq!(
        chunks,
        vec![
            (*b"PINF", 62, vec![0xaa, 0xbb]),
            (*b"LINF", 72, vec![1, 2, 3, 4, 5, 6]),
            (*b"CINF", 86, b"abc".to_vec()),
            (*b"FINF", 98, vec![0; 4]),
            (*b"SND2", 110, vec![0x77]),
            (*b"VQFR", 120, vec![]),
        ]
    );
    for (id, offset, _) in &chunks {
        assert_eq!(&file[*offset..offset + 4], id);
    }
}

#[test]
fn chunks_accepts_an_odd_sized_last_chunk_without_its_pad_byte() {
    // locks current behavior: the pad byte may be missing at end of file
    let mut file = movie(&header_8bit(), &[chunk(b"CINF", b"abc")]);
    assert_eq!(file.pop(), Some(0));
    let vqa = VQA::parse(&file).unwrap();

    let mut chunks = vqa.chunks();
    let cinf = chunks.next().unwrap().unwrap();
    assert_eq!((&cinf.id, cinf.data), (b"CINF", &b"abc"[..]));
    assert!(chunks.next().is_none());
}

#[test]
fn chunks_yields_one_parse_error_on_a_desync_then_stops() {
    // vqa.txt: chunk IDs are 4 uppercase letters (digits in SND0-2), so
    // anything else, or a size past the end of the file, is a desync;
    // yielding exactly one error and then None locks current behavior

    // a bad ID followed by a valid chunk, which the walk must not resync to
    let bad_id = |id: &[u8; 4]| [&id[..], &[0; 4], &chunk(b"SND2", &[0x77, 0x3f])].concat();
    let mut size_past_end = b"SND2".to_vec();
    size_past_end.extend(100u32.to_be_bytes());
    size_past_end.extend([1, 2, 3, 4]);

    let desyncs: [(&str, Vec<u8>); 5] = [
        ("lowercase ID", bad_id(b"linf")),
        ("space in ID", bad_id(b"SND ")),
        ("non-ASCII ID", bad_id(b"\xff\xfe\xfd\xfc")),
        ("size past the end", size_past_end),
        ("partial chunk header", b"SND2\0\0".to_vec()),
    ];
    for (what, bad) in desyncs {
        let mut file = movie(&header_8bit(), &[chunk(b"LINF", &[1, 2])]);
        file.extend(bad);
        let vqa = VQA::parse(&file).unwrap();

        let mut chunks = vqa.chunks();
        assert_eq!(&chunks.next().unwrap().unwrap().id, b"LINF", "{what}");
        assert_eq!(chunks.next(), Some(Err(Error::Parse)), "{what}");
        assert_eq!(chunks.next(), None, "{what}");
        assert_eq!(chunks.next(), None, "{what}");
    }
}

// ---------------------------------------------------------------------------
// VQA::frame_index

#[test]
fn frame_index_finds_finf_behind_other_chunks() {
    // vqa.txt, FINF chunk: Intel LONGs holding each frame's position / 2,
    // absolute from the start of the file, pointing at the frame's SND?
    // chunk; 0x40000000 flags a frame with a new palette
    //
    // layout: FORM header 8 + WVQA 4 + VQHD 50 = 62; LINF 8+6 -> 76;
    // CINF 8+3+pad -> 88; FINF 8+8 -> 104; SND2 8+2 -> 114; VQFR 8 -> 122
    let finf: Vec<u8> = [104u32 / 2, 0x4000_0000 | (122 / 2)]
        .iter()
        .flat_map(|entry| entry.to_le_bytes())
        .collect();
    let file = movie(
        &header_8bit(),
        &[
            chunk(b"LINF", &[0; 6]),
            chunk(b"CINF", &[0; 3]),
            chunk(b"FINF", &finf),
            chunk(b"SND2", &[0x77, 0x3f]),
            chunk(b"VQFR", &[]),
            chunk(b"SND2", &[0x77, 0x3f]),
            chunk(b"VQFR", &[]),
        ],
    );
    let vqa = VQA::parse(&file).unwrap();

    let index = vqa.frame_index.expect("the movie carries a FINF chunk");
    let entries: Vec<_> = index.iter().map(|e| (e.offset, e.has_palette)).collect();
    assert_eq!(entries, vec![(104, false), (122, true)]);
    for entry in &index {
        let at = entry.offset as usize;
        assert_eq!(&file[at..at + 4], b"SND2");
    }
}

#[test]
fn frame_index_is_none_without_finf() {
    // vqa.txt, FINF chunk: the index is a chunk of its own; no chunk, no
    // index
    let file = movie(
        &header_8bit(),
        &[chunk(b"LINF", &[0; 6]), chunk(b"VQFR", &[])],
    );
    assert_eq!(VQA::parse(&file).unwrap().frame_index, None);
    assert_eq!(
        VQA::parse(&movie(&header_8bit(), &[])).unwrap().frame_index,
        None
    );
}

#[test]
fn finf_offset_is_low_28_bits_times_2_and_bit_30_flags_a_palette() {
    // vqa.txt, FINF chunk: subtract the 0x40000000 palette flag, then
    // multiply by 2; Westwood's VQAFILE.H makes all top four bits flags,
    // VQAFRAME_OFFSET(a) = (a & VQAFINF_OFFSET) << 1 with VQAFINF_OFFSET
    // 0x0FFFFFFF (github.com/electronicarts/CnC_Red_Alert, WINVQ/VQA32)
    let entries = [0x0000_0010u32, 0x4000_0015, 0x0fff_ffff, 0x4fff_ffff];
    let finf: Vec<u8> = entries.iter().flat_map(|e| e.to_le_bytes()).collect();
    let file = movie(&header_8bit(), &[chunk(b"FINF", &finf)]);

    let index = VQA::parse(&file).unwrap().frame_index.unwrap();
    let decoded: Vec<(u32, bool)> = index
        .iter()
        .map(|entry| (entry.offset, entry.has_palette))
        .collect();
    assert_eq!(
        decoded,
        vec![
            (0x20, false),
            (0x2a, true),
            (0x1fff_fffe, false),
            (0x1fff_fffe, true),
        ]
    );
}

#[test]
fn finf_flag_bits_other_than_the_palette_leave_the_offset_alone() {
    // VQAFILE.H: bit 31 flags a key frame and bit 29 a sync point, and bit
    // 28 is also outside VQAFINF_OFFSET; none of them is the palette flag
    let entries = [0x8000_0003u32, 0x2000_0003, 0x1000_0003, 0xf000_0003];
    let finf: Vec<u8> = entries.iter().flat_map(|e| e.to_le_bytes()).collect();
    let file = movie(&header_8bit(), &[chunk(b"FINF", &finf)]);

    let index = VQA::parse(&file).unwrap().frame_index.unwrap();
    let decoded: Vec<(u32, bool)> = index
        .iter()
        .map(|entry| (entry.offset, entry.has_palette))
        .collect();
    assert_eq!(decoded, [(6, false), (6, false), (6, false), (6, true)]);
}

// ---------------------------------------------------------------------------
// Frames

/// A two-entry 4x2 codebook: entry 0 holds `base..base + 8`, entry 1
/// `base + 10..base + 18`.
fn codebook(base: u8) -> Vec<u8> {
    (base..base + 8).chain(base + 10..base + 18).collect()
}

/// A VQFR chunk holding the given sub-chunks.
fn vqfr(sub_chunks: &[Vec<u8>]) -> Vec<u8> {
    chunk(b"VQFR", &sub_chunks.concat())
}

/// Frame 1's VQFR without a palette: codebook(0) and FRAME1's pointers.
fn frame1_without_palette() -> Vec<u8> {
    vqfr(&[
        chunk(b"CBF0", &codebook(0)),
        chunk(b"VPT0", &[0, 1, 9, 1, /* hi */ 0, 0, 0x0f, 0]),
    ])
}

/// Three 8x4 v2 frames (2x2 blocks of 4x2): the first carries a codebook
/// and palette, a VQFL swaps in a new codebook before the second, and
/// sound and unknown chunks sit in between.
fn three_frame_movie() -> Vec<u8> {
    // vqa.txt, VERSION 2 INDEX TABLE LAYOUT: LoVal half, then HiVal half;
    // HiVal 0x0f fills the block with LoVal, else entry HiVal*256+LoVal
    let frame1 = vqfr(&[
        chunk(b"CBF0", &codebook(0)),
        chunk(b"CPL0", &[0x3f, 0, 0, 0, 0x3f, 0]),
        chunk(b"VPT0", &[0, 1, 9, 1, /* hi */ 0, 0, 0x0f, 0]),
    ]);
    let frame2 = vqfr(&[chunk(b"VPT0", &[1, 0, 0, 1, /* hi */ 0, 0, 0, 0])]);
    let frame3 = vqfr(&[chunk(b"VPT0", &[0, 2, 3, 1, /* hi */ 0, 0x0f, 0x0f, 0])]);
    // hc-vqa.txt: a VQFL chunk carries a CBFZ for the following frames.
    // It describes VQFL for HiColor movies only; applying one in this 8-bit
    // movie locks current behavior
    let vqfl = chunk(b"VQFL", &chunk(b"CBFZ", &lcw_literals(&codebook(20))));

    movie(
        &header_8bit(),
        &[
            chunk(b"SND2", &[0x77, 0x3f]),
            frame1,
            vqfl,
            chunk(b"SND2", &[0x77]),
            frame2,
            chunk(b"CINF", &[1, 2, 3]),
            chunk(b"SND2", &[0x77, 0x3f]),
            frame3,
        ],
    )
}

#[rustfmt::skip]
const FRAME1: [u8; 32] = [
    0, 1, 2, 3,  10, 11, 12, 13,
    4, 5, 6, 7,  14, 15, 16, 17,
    9, 9, 9, 9,  10, 11, 12, 13,
    9, 9, 9, 9,  14, 15, 16, 17,
];

#[rustfmt::skip]
const FRAME2: [u8; 32] = [
    30, 31, 32, 33,  20, 21, 22, 23,
    34, 35, 36, 37,  24, 25, 26, 27,
    20, 21, 22, 23,  30, 31, 32, 33,
    24, 25, 26, 27,  34, 35, 36, 37,
];

#[rustfmt::skip]
const FRAME3: [u8; 32] = [
    20, 21, 22, 23,  2, 2, 2, 2,
    24, 25, 26, 27,  2, 2, 2, 2,
     3,  3,  3,  3, 30, 31, 32, 33,
     3,  3,  3,  3, 34, 35, 36, 37,
];

/// Frame 1's CPL0 palette. vqa.txt, CPL? chunk: red, green, blue triples
/// of 6-bit VGA values. The (v << 2) | (v >> 4) widening to 8 bits locks
/// current behavior: 0x3f -> 0xfc | 0x03 = 0xff, and 0 -> 0.
const PALETTE: [[u8; 3]; 2] = [[0xff, 0, 0], [0, 0xff, 0]];

/// The palette indices and palette of a decoded 8-bit frame.
fn indexed(frame: Option<Result<Frame, Error>>) -> (Vec<u8>, Vec<[u8; 3]>) {
    let frame = frame
        .expect("expected a frame")
        .expect("frame should decode");
    assert_eq!((frame.width, frame.height), (8, 4));
    match frame.pixels {
        FramePixels::Indexed { pixels, palette } => (pixels, palette),
        FramePixels::HiColor { .. } => panic!("expected an 8-bit frame"),
    }
}

#[test]
fn frames_yields_one_frame_per_vqfr_in_file_order() {
    // vqa.txt, Typical VQA File: frame data is a VQFR chunk, other chunks
    // are skipped; hc-vqa.txt: VQFL codebooks apply to the frames after it
    // (in an 8-bit movie this locks current behavior, see
    // three_frame_movie). Only frame 1 carries a palette, and frames 2 and
    // 3 still use it: vqa.txt, Typical VQA File, gives CPL? only in the
    // first frame
    let file = three_frame_movie();
    let vqa = VQA::parse(&file).unwrap();
    let mut frames = vqa.frames().unwrap();

    assert_eq!(indexed(frames.next()), (FRAME1.to_vec(), PALETTE.to_vec()));
    assert_eq!(indexed(frames.next()), (FRAME2.to_vec(), PALETTE.to_vec()));
    assert_eq!(indexed(frames.next()), (FRAME3.to_vec(), PALETTE.to_vec()));
    assert!(frames.next().is_none());
}

#[test]
#[allow(clippy::iter_nth_zero)]
fn frames_nth_skips_frames_and_is_none_past_the_end() {
    // first principles: Iterator::nth consumes n items and returns the
    // next, so nth(0) is next()
    let file = three_frame_movie();
    let vqa = VQA::parse(&file).unwrap();

    let mut frames = vqa.frames().unwrap();
    assert_eq!(indexed(frames.nth(0)), (FRAME1.to_vec(), PALETTE.to_vec()));
    assert_eq!(indexed(frames.nth(0)), (FRAME2.to_vec(), PALETTE.to_vec()));

    assert_eq!(
        indexed(vqa.frames().unwrap().nth(2)),
        (FRAME3.to_vec(), PALETTE.to_vec())
    );

    let mut frames = vqa.frames().unwrap();
    assert_eq!(indexed(frames.nth(1)), (FRAME2.to_vec(), PALETTE.to_vec()));
    assert_eq!(indexed(frames.next()), (FRAME3.to_vec(), PALETTE.to_vec()));
    assert!(frames.nth(1).is_none());

    assert!(vqa.frames().unwrap().nth(3).is_none());
    assert!(vqa.frames().unwrap().nth(10).is_none());
}

/// A valid frame 1, a bad frame 2 (a 6-byte pointer table where 2x2
/// blocks need 8 bytes), and a valid frame 3.
fn bad_frame_2_movie() -> Vec<u8> {
    movie(
        &header_8bit(),
        &[
            frame1_without_palette(),
            vqfr(&[chunk(b"VPT0", &[0; 6])]),
            vqfr(&[chunk(b"VPT0", &[0; 8])]),
        ],
    )
}

#[test]
fn frames_stops_after_the_first_error() {
    // locks current behavior: a bad frame ends the iteration even though a
    // valid frame follows
    let file = bad_frame_2_movie();
    let vqa = VQA::parse(&file).unwrap();
    let mut frames = vqa.frames().unwrap();

    assert_eq!(indexed(frames.next()), (FRAME1.to_vec(), vec![]));
    assert!(matches!(frames.next(), Some(Err(Error::Video(_)))));
    assert!(frames.next().is_none());
    assert!(frames.next().is_none());
}

#[test]
fn frames_stops_after_a_bad_vqfl() {
    // locks current behavior: a VQFL whose codebook fails to decode ends
    // the iteration like a bad frame, even though a valid frame follows.
    // vqa.txt, Appendix A: 0x85 is command (1), "copy next Count bytes",
    // with Count 5, but only 1 byte follows
    let file = movie(
        &header_8bit(),
        &[
            frame1_without_palette(),
            chunk(b"VQFL", &chunk(b"CBFZ", &[0x85, 1])),
            vqfr(&[chunk(b"VPT0", &[0; 8])]),
        ],
    );
    let vqa = VQA::parse(&file).unwrap();
    let mut frames = vqa.frames().unwrap();

    assert_eq!(indexed(frames.next()), (FRAME1.to_vec(), vec![]));
    assert!(matches!(
        frames.next(),
        Some(Err(Error::Lcw(LcwError::Truncated)))
    ));
    assert!(frames.next().is_none());
    assert!(frames.next().is_none());
}

#[test]
fn frames_nth_over_a_bad_frame_is_none() {
    // locks current behavior, with first principles: Frames stops after
    // its first error, so an nth that skips over the error finds nothing
    // after it (the default nth would get the error, then None), and an
    // nth that lands on the error returns it
    let file = bad_frame_2_movie();
    let vqa = VQA::parse(&file).unwrap();

    let mut frames = vqa.frames().unwrap();
    assert!(frames.nth(2).is_none());
    assert!(frames.next().is_none());

    assert!(matches!(
        vqa.frames().unwrap().nth(1),
        Some(Err(Error::Video(_)))
    ));
}

#[test]
fn frames_of_a_movie_cut_off_mid_chunk_end_with_one_parse_error() {
    // vqa.txt: a chunk's size says how many bytes follow; a cut-off chunk
    // is a desync; the complete frames before it still decode
    let mut file = three_frame_movie();
    // the last chunk is frame 3's VQFR: 8 + a 16-byte VPT0 sub-chunk
    file.truncate(file.len() - 5);
    let vqa = VQA::parse(&file).unwrap();
    let mut frames = vqa.frames().unwrap();

    assert_eq!(indexed(frames.next()), (FRAME1.to_vec(), PALETTE.to_vec()));
    assert_eq!(indexed(frames.next()), (FRAME2.to_vec(), PALETTE.to_vec()));
    assert!(matches!(frames.next(), Some(Err(Error::Parse))));
    assert!(frames.next().is_none());
}

/// The frame `codebook(base)` draws under the pointer table
/// [`TWO_ENTRY_TABLE`]: entry 0 in the left blocks, entry 1 in the right.
fn two_entry_frame(base: u8) -> Vec<u8> {
    let (e0, e1) = (base, base + 10);
    let row = |r: u8| [e0 + 4 * r..e0 + 4 * r + 4, e1 + 4 * r..e1 + 4 * r + 4];
    let block_row: Vec<u8> = (0..2).flat_map(row).flatten().collect();
    [block_row.clone(), block_row].concat()
}

/// Blocks drawing entries 0, 1, 0, 1 (vqa.txt, VERSION 2 INDEX TABLE LAYOUT)
const TWO_ENTRY_TABLE: [u8; 8] = [0, 1, 0, 1, /* hi */ 0, 0, 0, 0];

/// The palette indices of every frame of `file`, which must all decode.
fn all_frames(file: &[u8]) -> Vec<Vec<u8>> {
    let vqa = VQA::parse(file).unwrap();
    let frames = vqa.frames().unwrap();
    frames.map(|frame| indexed(Some(frame)).0).collect()
}

/// A CINF chunk scheduling codebooks at the frames `starts`: a CINH count,
/// then CIND entries of a start frame and a compressed size (unused here).
fn cinf(starts: &[u16]) -> Vec<u8> {
    let mut cinh = (starts.len() as u16).to_le_bytes().to_vec();
    cinh.extend([0; 6]);
    let cind: Vec<u8> = starts
        .iter()
        .flat_map(|&start| [start.to_le_bytes().as_slice(), &[0; 4]].concat())
        .collect();
    chunk(
        b"CINF",
        &[chunk(b"CINH", &cinh), chunk(b"CIND", &cind)].concat(),
    )
}

#[test]
fn frames_swap_in_codebook_parts_where_the_cinf_schedule_starts_a_codebook() {
    // Lands of Lore's movies store cbparts 0 and send each codebook in
    // parts over a group of frames of varying length. Their CINF chunk's
    // CIND entries give each group's first frame (516EFA98.VQA: 0, 31, 53,
    // ...), where the codebook the previous group's parts built takes over.
    // No document describes CINF: this locks current behavior, found on
    // that movie
    let table = chunk(b"VPT0", &TWO_ENTRY_TABLE);
    let new = codebook(20);
    let frames = [
        vqfr(&[
            chunk(b"CBF0", &codebook(0)),
            chunk(b"CBP0", &new[..5]),
            table.clone(),
        ]),
        vqfr(&[chunk(b"CBP0", &new[5..]), table.clone()]),
        vqfr(std::slice::from_ref(&table)),
    ];

    // frames 0 and 1 are one group; frame 2 starts the next
    let scheduled = movie(&header_8bit(), &[&[cinf(&[0, 2])], &frames[..]].concat());
    assert_eq!(
        all_frames(&scheduled),
        [two_entry_frame(0), two_entry_frame(0), two_entry_frame(20)]
    );

    // without a schedule the parts never complete
    let unscheduled = movie(&header_8bit(), &frames);
    assert_eq!(all_frames(&unscheduled), vec![two_entry_frame(0); 3]);
}

#[test]
fn cinf_schedule_applies_before_a_frames_own_chunks_in_either_layout() {
    // Westwood's loader fixes a frame's codebook before reading any of its
    // chunks (LOADER.CPP: curframe->Codebook = loader->FullCB), so the part
    // a group's first frame brings goes to the next codebook in the older
    // layout too
    let (a, b) = (codebook(20), codebook(40));
    let table = chunk(b"VPT0", &TWO_ENTRY_TABLE);
    let frames = [
        vec![
            chunk(b"CBF0", &codebook(0)),
            chunk(b"CBP0", &a[..8]),
            table.clone(),
        ],
        vec![chunk(b"CBP0", &a[8..]), table.clone()],
        vec![chunk(b"CBP0", &b[..8]), table.clone()],
        vec![chunk(b"CBP0", &b[8..]), table.clone()],
        vec![table.clone()],
    ];
    let expected = [
        two_entry_frame(0),
        two_entry_frame(0),
        two_entry_frame(20),
        two_entry_frame(20),
        two_entry_frame(40),
    ];

    let wrapped: Vec<Vec<u8>> = frames.iter().map(|frame| vqfr(frame)).collect();
    let top_level: Vec<Vec<u8>> = frames.concat();
    for chunks in [wrapped, top_level] {
        let file = movie(&header_8bit(), &[&[cinf(&[0, 2, 4])], &chunks[..]].concat());
        assert_eq!(all_frames(&file), expected);
    }
}

#[test]
fn frames_ignore_the_cinf_schedule_when_the_header_counts_parts() {
    // locks current behavior: with cbparts in the header, parts complete by
    // count (vqa.txt, CBP? chunk). Were the schedule's frame 1 honored, its
    // swap would take in half a codebook, and frame 1 would fail
    let header = VQAHeader {
        cbparts: 2,
        ..header_8bit()
    };
    let new = codebook(20);
    let table = chunk(b"VPT0", &TWO_ENTRY_TABLE);
    let file = movie(
        &header,
        &[
            cinf(&[0, 1]),
            vqfr(&[
                chunk(b"CBF0", &codebook(0)),
                chunk(b"CBP0", &new[..8]),
                table.clone(),
            ]),
            vqfr(&[chunk(b"CBP0", &new[8..]), table.clone()]),
            vqfr(&[table]),
        ],
    );
    assert_eq!(
        all_frames(&file),
        [two_entry_frame(0), two_entry_frame(0), two_entry_frame(20)]
    );
}

#[test]
fn codebook_starts_lists_the_cind_start_frames() {
    // the CIND entries' start frames, in file order (516EFA98.VQA: 0, 31,
    // 53, ...); none without a CINF chunk or with one that doesn't parse
    let file = movie(&header_8bit(), &[cinf(&[0, 31, 53])]);
    let vqa = VQA::parse(&file).unwrap();
    assert_eq!(vqa.codebook_starts().collect::<Vec<_>>(), [0, 31, 53]);

    let file = movie(&header_8bit(), &[]);
    assert_eq!(VQA::parse(&file).unwrap().codebook_starts().count(), 0);
    let file = movie(&header_8bit(), &[chunk(b"CINF", b"abc")]);
    assert_eq!(VQA::parse(&file).unwrap().codebook_starts().count(), 0);
}

#[test]
fn frames_decode_vqfk_key_frames_and_vptk_and_vptd_tables() {
    // Westwood's VQA loader (WINVQ/VQA32/LOADER.CPP in EA's GPL Red Alert
    // source) reads a VQFK chunk like a VQFR, flagging a key frame, and
    // VPTK and VPTD pointer tables like VPTZ, LCW-compressed
    let file = movie(
        &header_8bit(),
        &[
            chunk(
                b"VQFK",
                &[
                    chunk(b"CBF0", &codebook(0)),
                    chunk(b"VPTK", &lcw_literals(&TWO_ENTRY_TABLE)),
                ]
                .concat(),
            ),
            vqfr(&[chunk(b"VPTD", &lcw_literals(&TWO_ENTRY_TABLE))]),
        ],
    );
    assert_eq!(all_frames(&file), [two_entry_frame(0), two_entry_frame(0)]);
}

#[test]
fn frames_decode_the_older_layout_without_vqfr_chunks() {
    // LOADER.CPP reads "either the older non-frame-grouped VQA file format,
    // or the new frame-chunk format. For the older format, it's assumed
    // that the last chunk in a frame is the pointer data." Kyrandia 3's
    // benchl.vqa is laid out so: CBFZ, CBPZ, CPL0 and VPTZ at the top level
    let other_table = [1, 0, 1, 0, /* hi */ 0, 0, 0, 0];
    let file = movie(
        &header_8bit(),
        &[
            chunk(b"CBF0", &codebook(0)),
            chunk(b"CPL0", &[0x3f, 0, 0]),
            chunk(b"VPT0", &TWO_ENTRY_TABLE),
            chunk(b"SND2", &[0x77]),
            chunk(b"VPTZ", &lcw_literals(&other_table)),
        ],
    );
    let vqa = VQA::parse(&file).unwrap();
    let frames: Vec<_> = vqa.frames().unwrap().map(|f| indexed(Some(f))).collect();

    // the second frame swaps the two entries' columns
    let swapped: Vec<u8> = two_entry_frame(0)
        .as_chunks::<4>()
        .0
        .chunks(2)
        .flat_map(|pair| [pair[1], pair[0]])
        .flatten()
        .collect();
    assert_eq!(
        frames,
        [
            (two_entry_frame(0), vec![[0xff, 0, 0]]),
            (swapped, vec![[0xff, 0, 0]])
        ]
    );
}

#[test]
fn frames_apply_top_level_parts_palettes_and_key_tables_in_the_older_layout() {
    // LOADER.CPP reads the older layout's top-level CBP0/CBPZ and CPL0/CPLZ
    // chunks as it does inside a VQFR, and ends a frame at VPTK or VPTD as
    // at VPT0 or VPTZ
    let header = VQAHeader {
        cbparts: 2,
        ..header_8bit()
    };
    let new = codebook(20);
    let file = movie(
        &header,
        &[
            chunk(b"CBF0", &codebook(0)),
            chunk(b"CPLZ", &lcw_literals(&[0x3f, 0, 0])),
            chunk(b"CBP0", &new[..8]),
            chunk(b"VPTK", &lcw_literals(&TWO_ENTRY_TABLE)),
            chunk(b"CBP0", &new[8..]),
            chunk(b"VPTD", &lcw_literals(&TWO_ENTRY_TABLE)),
            chunk(b"VPT0", &TWO_ENTRY_TABLE),
        ],
    );
    let vqa = VQA::parse(&file).unwrap();
    let frames: Vec<_> = vqa.frames().unwrap().map(|f| indexed(Some(f))).collect();
    let red = vec![[0xff, 0, 0]];
    assert_eq!(
        frames,
        [
            (two_entry_frame(0), red.clone()),
            (two_entry_frame(0), red.clone()),
            (two_entry_frame(20), red)
        ]
    );
}

// ---------------------------------------------------------------------------
// SND0 and SND1

#[test]
fn snd0_16bit_samples_are_signed_little_endian() {
    // vqa.txt, SND0 chunk: raw PCM; 16-bit samples are signed (Intel order)
    let data = [0x34, 0x12, 0xff, 0xff, 0x00, 0x80, 0xff, 0x7f];
    let file = movie(
        &sound_header(VQAVersion::Two, 1, 16),
        &[chunk(b"SND0", &data)],
    );
    assert_eq!(decode_audio(&file), Ok(vec![0x1234, -1, -32768, 32767]));
}

#[test]
fn snd0_16bit_chunk_with_an_odd_byte_count_drops_its_last_byte() {
    // locks current behavior: the half sample is dropped, not carried into
    // the next chunk
    let file = movie(
        &sound_header(VQAVersion::Two, 1, 16),
        &[
            chunk(b"SND0", &[0x34, 0x12, 0xff]),
            chunk(b"SND0", &[0x01, 0x00]),
        ],
    );
    assert_eq!(decode_audio(&file), Ok(vec![0x1234, 1]));
}

#[test]
fn snd0_8bit_samples_are_unsigned_and_widened_to_signed_16bit() {
    // vqa.txt, SND0 chunk: 8-bit samples are unsigned (0x80 is silence);
    // the (b - 128) << 8 widening locks current behavior
    let data = [0x00, 0x80, 0xff, 0x7f, 0x81];
    let expected = vec![-32768, 0, 32512, -256, 256];

    let file = movie(
        &sound_header(VQAVersion::Two, 1, 8),
        &[chunk(b"SND0", &data)],
    );
    assert_eq!(decode_audio(&file), Ok(expected.clone()));

    // vqa.txt, VQHD chunk: a v1 Bits of 0 means 8-bit
    let file = movie(
        &sound_header(VQAVersion::One, 0, 0),
        &[chunk(b"SND0", &data)],
    );
    assert_eq!(decode_audio(&file), Ok(expected));
}

#[test]
fn snd1_westwood_adpcm_decodes_each_chunk_from_silence() {
    // vqa.txt, SND1 chunk and Appendix C: a header of OutSize and Size
    // words, then commands; CurSample starts at 0x80 in every chunk, and a
    // chunk whose sizes are equal is stored raw. The unsigned 8-bit samples
    // widen as SND0's do, (b - 128) << 8
    let file = movie(
        &sound_header(VQAVersion::One, 1, 8),
        &[
            // 0x41: 4-bit deltas for 2 bytes; nibbles 0, f, 7, 8 are -9, +8,
            // -1, 0 (WSTable4bit): 0x77, 0x7f, 0x7e, 0x7e
            chunk(b"SND1", &[4, 0, 3, 0, 0x41, 0xf0, 0x87]),
            // 0x00: 2-bit deltas for 1 byte, all +1 (WSTable2bit[3]), from
            // 0x80 again: 0x81, 0x82, 0x83, 0x84
            chunk(b"SND1", &[4, 0, 2, 0, 0x00, 0xff]),
            // raw
            chunk(b"SND1", &[2, 0, 2, 0, 0x00, 0xff]),
        ],
    );
    let widen = |b: i16| (b - 128) << 8;
    let expected: Vec<i16> = [0x77, 0x7f, 0x7e, 0x7e, 0x81, 0x82, 0x83, 0x84, 0x00, 0xff]
        .into_iter()
        .map(widen)
        .collect();
    assert_eq!(decode_audio(&file), Ok(expected));
}

// ---------------------------------------------------------------------------
// VQA::audio_chunks

#[test]
fn audio_chunks_yield_each_sound_chunk_in_file_order() {
    // one item per SND? chunk, the same samples decode_audio returns, with
    // the IMA predictor carried across (vqa.txt, Appendix B)
    let file = movie(
        &sound_header(VQAVersion::Two, 1, 16),
        &[
            chunk(b"SND2", &MONO[..2]),
            chunk(b"VQFR", &[]),
            chunk(b"SND2", &MONO[2..]),
            chunk(b"SND0", &[0x34, 0x12]),
        ],
    );
    let vqa = VQA::parse(&file).unwrap();
    let chunks: Vec<Vec<i16>> = vqa.audio_chunks().map(Result::unwrap).collect();
    assert_eq!(
        chunks,
        [
            MONO_SAMPLES[..4].to_vec(),
            MONO_SAMPLES[4..].to_vec(),
            vec![0x1234]
        ]
    );
}

#[test]
fn audio_chunks_keep_the_sound_before_a_cut_off_chunk() {
    // a chunk whose size runs past the end of the file is a desync (vqa.txt:
    // the size says how many bytes follow): the chunks before it still
    // decode, then one error ends the iteration. decode_audio fails outright
    let mut file = movie(
        &sound_header(VQAVersion::Two, 1, 16),
        &[chunk(b"SND2", &MONO[..2]), chunk(b"SND2", &MONO[2..])],
    );
    // the last chunk is 1 data byte and its pad: cut both
    file.truncate(file.len() - 2);
    let vqa = VQA::parse(&file).unwrap();

    let mut chunks = vqa.audio_chunks();
    assert_eq!(chunks.next(), Some(Ok(MONO_SAMPLES[..4].to_vec())));
    assert_eq!(chunks.next(), Some(Err(Error::Parse)));
    assert_eq!(chunks.next(), None);
    assert_eq!(vqa.decode_audio(), Err(Error::Parse));
}

#[test]
fn audio_chunks_next_into_appends_and_clones_resume_in_step() {
    // first principles: next_into appends where next allocates; a clone
    // carries the IMA predictors, so both copies decode the rest alike
    let file = movie(
        &sound_header(VQAVersion::Two, 1, 16),
        &[chunk(b"SND2", &MONO[..1]), chunk(b"SND2", &MONO[1..])],
    );
    let vqa = VQA::parse(&file).unwrap();

    let mut chunks = vqa.audio_chunks();
    let mut samples = vec![7];
    assert_eq!(chunks.next_into(&mut samples), Some(Ok(())));
    assert_eq!(samples, [7, MONO_SAMPLES[0], MONO_SAMPLES[1]]);

    let mut resumed = chunks.clone();
    assert_eq!(chunks.next_into(&mut samples), Some(Ok(())));
    assert_eq!(samples[1..], MONO_SAMPLES);
    assert_eq!(resumed.next(), Some(Ok(MONO_SAMPLES[2..].to_vec())));
    assert_eq!(chunks.next_into(&mut samples), None);
}

// ---------------------------------------------------------------------------
// IMA ADPCM (SND2)
//
// ima-adpcm.txt, Optimization, "The usual algorithm" (shift-and-add), with
// the tables from Decoding Tables: from predictor 0 and step index 0, each
// nibble adds or (sign bit 8) subtracts diff = step>>3 [+ step if bit 2]
// [+ step>>1 if bit 1] [+ step>>2 if bit 0]; then index +=
// {-1,-1,-1,-1,2,4,6,8}[n & 7]; ima-adpcm.txt, Decoding IMA: the predictor
// saturates to i16 and the index to 0..=88. vqa.txt Appendix B: the first
// code of a byte is in the lower bits. (The documents' other delta
// formulas round differently: vqa.txt Appendix B's (Step * Code) div 4 +
// Step div 8 gives 12, not 11, for nibble 7 at step 7.)
//
// [0x77, 0x3f, 0x09] = nibbles 7, 7, f, 3, 9, 0:
//   7 at index 0 (step 7):   0 + 7 + 3 + 1      = 11  ->  11, index 8
//   7 at index 8 (step 16):  2 + 16 + 8 + 4     = 30  ->  41, index 16
//   f at index 16 (step 34): -(4 + 34 + 17 + 8) = -63 -> -22, index 24
//   3 at index 24 (step 73): 9 + 36 + 18        = 63  ->  41, index 23
//   9 at index 23 (step 66): -(8 + 16)          = -24 ->  17, index 22
//   0 at index 22 (step 60): 7                        ->  24, index 21
const MONO: [u8; 3] = [0x77, 0x3f, 0x09];
const MONO_SAMPLES: [i16; 6] = [11, 41, -22, 41, 17, 24];

// [0x4c, 0x05] = nibbles c, 4, 5, 0:
//   c at index 0 (step 7):  -(0 + 7)     = -7 -> -7, index 2
//   4 at index 2 (step 9):  1 + 9        = 10 ->  3, index 4
//   5 at index 4 (step 11): 1 + 11 + 2   = 14 -> 17, index 8
//   0 at index 8 (step 16): 2                 -> 19, index 7
const OTHER: [u8; 2] = [0x4c, 0x05];
const OTHER_SAMPLES: [i16; 4] = [-7, 3, 17, 19];

#[test]
fn ima_adpcm_decodes_hand_computed_samples() {
    // ima-adpcm.txt, Optimization, "The usual algorithm"; see the worked
    // table above
    assert_eq!(decompress(&mut CodecState::new(), &MONO), MONO_SAMPLES);
    assert_eq!(decompress(&mut CodecState::new(), &OTHER), OTHER_SAMPLES);
    assert_eq!(decompress(&mut CodecState::default(), &MONO), MONO_SAMPLES);
    assert_eq!(decompress(&mut CodecState::new(), &[]), Vec::<i16>::new());
}

#[test]
fn ima_adpcm_decodes_the_low_nibble_first() {
    // vqa.txt, Appendix B: two codes per byte, the first in the lower bits
    // 0x07: 7 at index 0 -> 11, index 8; 0 at step 16 -> 11 + 2 = 13
    assert_eq!(decompress(&mut CodecState::new(), &[0x07]), [11, 13]);
    // 0x70: 0 at index 0 -> 0, index stays 0; 7 at step 7 -> 11
    assert_eq!(decompress(&mut CodecState::new(), &[0x70]), [0, 11]);
}

#[test]
fn ima_adpcm_saturates_the_predictor_and_the_step_index() {
    // ima-adpcm.txt, Decoding IMA: saturate the step index to 0..=88 and
    // the predictor to -32768..=32767. Twelve 7s climb the index by 8 each
    // (11 and 41 as in MONO above, then):
    //   index 16 (34):     4 + 34 + 17 + 8           = 63    ->   104
    //   index 24 (73):     9 + 73 + 36 + 18          = 136   ->   240
    //   index 32 (157):    19 + 157 + 78 + 39        = 293   ->   533
    //   index 40 (337):    42 + 337 + 168 + 84       = 631   ->  1164
    //   index 48 (724):    90 + 724 + 362 + 181      = 1357  ->  2521
    //   index 56 (1552):   194 + 1552 + 776 + 388    = 2910  ->  5431
    //   index 64 (3327):   415 + 3327 + 1663 + 831   = 6236  -> 11667
    //   index 72 (7132):   891 + 7132 + 3566 + 1783  = 13372 -> 25039
    //   index 80 (15289):  1911 + 15289 + 7644 + 3822 = 28666 -> 32767
    //   index 88 (32767):  4095 + 32767 + 16383 + 8191 = 61436 -> 32767,
    //                      index 96 saturates to 88
    // then f, f at 88: 32767 - 61436 = -28669, then -90105 -> -32768;
    // 0 at 88: -32768 + 4095 = -28673 (from the saturated value), index 87;
    // 8 at 87 (29794): -3724 -> -32397
    let data = [0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0xff, 0x80];
    #[rustfmt::skip]
    let expected = [
        11, 41, 104, 240, 533, 1164, 2521, 5431,
        11667, 25039, 32767, 32767, -28669, -32768, -28673, -32397,
    ];
    assert_eq!(decompress(&mut CodecState::new(), &data), expected);

    // at index 0, nibble 0 moves the index to -1, which saturates to 0:
    // the step stays 7, so nibbles 0, 0, 1, 0 add 0, 0, (7>>3) + (7>>2) = 1,
    // and 0
    assert_eq!(
        decompress(&mut CodecState::new(), &[0x00, 0x01]),
        [0, 0, 1, 1]
    );
}

#[test]
fn ima_adpcm_state_carries_across_calls() {
    // vqa.txt, Appendix B: the sample and index are kept across chunks
    let mut state = CodecState::new();
    let mut samples = decompress(&mut state, &MONO[..2]);
    samples.extend(decompress(&mut state, &MONO[2..]));
    assert_eq!(samples, MONO_SAMPLES);
    assert_eq!(state, {
        let mut whole = CodecState::new();
        decompress(&mut whole, &MONO);
        whole
    });
}

#[test]
fn decode_audio_keeps_one_mono_predictor_across_snd2_chunks() {
    // vqa.txt, SND2 chunk and Appendix B: one state for the whole stream,
    // with other chunks in between; a reset state would decode the last
    // chunk as -1, -1
    let file = movie(
        &sound_header(VQAVersion::Two, 1, 16),
        &[
            chunk(b"SND2", &MONO[..2]),
            chunk(b"VQFR", &[]),
            chunk(b"SND2", &MONO[2..]),
        ],
    );
    assert_eq!(decode_audio(&file), Ok(MONO_SAMPLES.to_vec()));
}

/// A v3 (Tiberian Sun era, HiColor) movie with sound.
fn v3_sound_header(channels: u8) -> VQAHeader {
    VQAHeader {
        flags: 1,
        channels,
        ..header_hicolor()
    }
}

#[test]
fn decode_audio_decodes_v3_mono_as_one_stream() {
    // vqa.txt, SND2 chunk: Tiberian Sun (v3) stereo "is encoded the same
    // way as mono except the SND2 chunk is split into two halfs", so a mono
    // v3 chunk is one plain stream, not two halves; vqa.txt, Appendix B:
    // with one state across chunks
    let file = movie(&v3_sound_header(1), &[chunk(b"SND2", &MONO)]);
    assert_eq!(decode_audio(&file), Ok(MONO_SAMPLES.to_vec()));

    let file = movie(
        &v3_sound_header(1),
        &[chunk(b"SND2", &MONO[..1]), chunk(b"SND2", &MONO[1..])],
    );
    assert_eq!(decode_audio(&file), Ok(MONO_SAMPLES.to_vec()));
}

// MONO[..2] on the left (11, 41, -22, 41) and OTHER on the right (-7, 3,
// 17, 19), interleaved L, R, L, R. A swapped layout would start -7, 11,
// and a mis-split one would decode other bytes for each channel.
const STEREO_SAMPLES: [i16; 8] = [11, -7, 41, 3, -22, 17, 41, 19];

#[test]
fn decode_audio_splits_v3_stereo_chunks_into_halves() {
    // vqa.txt, SND2 chunk: Tiberian Sun (v3) stores the left channel in
    // the first half of the chunk and the right in the second
    let file = movie(
        &v3_sound_header(2),
        &[chunk(b"SND2", &[&MONO[..2], &OTHER[..]].concat())],
    );
    assert_eq!(decode_audio(&file), Ok(STEREO_SAMPLES.to_vec()));

    // vqa.txt, SND2 chunk and Appendix B: each channel keeps its own state
    // across chunks
    let file = movie(
        &v3_sound_header(2),
        &[
            chunk(b"SND2", &[MONO[0], OTHER[0]]),
            chunk(b"SND2", &[MONO[1], OTHER[1]]),
        ],
    );
    assert_eq!(decode_audio(&file), Ok(STEREO_SAMPLES.to_vec()));
}

/// MONO[..2] as the left channel and OTHER as the right in the
/// byte-alternating layout (L R L R): in one chunk, and in two.
fn alternating_stereo_movies(version: VQAVersion) -> [Vec<u8>; 2] {
    let header = sound_header(version, 2, 16);
    [
        movie(
            &header,
            &[chunk(b"SND2", &[MONO[0], OTHER[0], MONO[1], OTHER[1]])],
        ),
        movie(
            &header,
            &[
                chunk(b"SND2", &[MONO[0], OTHER[0]]),
                chunk(b"SND2", &[MONO[1], OTHER[1]]),
            ],
        ),
    ]
}

#[test]
fn decode_audio_alternates_v2_stereo_bytes() {
    // vqa.txt, SND2 chunk: old movies (C&C, RA: version 2, NOTE #1) pack
    // bytes as LL RR LL RR: two left samples in one byte, then two right
    // samples in the next; each channel keeps its own Index and Cur_Sample
    // (Appendix B), across chunks too
    for (i, file) in alternating_stereo_movies(VQAVersion::Two)
        .iter()
        .enumerate()
    {
        assert_eq!(decode_audio(file), Ok(STEREO_SAMPLES.to_vec()), "movie {i}");
    }
}

#[test]
fn decode_audio_alternates_v1_stereo_bytes_too() {
    // locks current behavior: vqa.txt documents the LL RR layout only for
    // C&C and RA (v2); the stereo SND2 layout of v1 (Kyrandia III, which
    // uses 8-bit sound) is undocumented, and the crate treats it like v2
    for (i, file) in alternating_stereo_movies(VQAVersion::One)
        .iter()
        .enumerate()
    {
        assert_eq!(decode_audio(file), Ok(STEREO_SAMPLES.to_vec()), "movie {i}");
    }
}

#[test]
fn decode_audio_keeps_the_right_predictor_in_step_over_an_odd_v3_chunk() {
    // locks current behavior: vqa.txt says a stereo SND2 chunk is exactly
    // twice a mono one, so odd chunks are undocumented. For an odd v3
    // chunk the crate splits at len / 2, which gives the right half the
    // extra byte; it decodes that byte to keep the right predictor in
    // step and drops its samples, which have no left partner.
    //
    // [0x77, 0x4c, 0x05]: left [0x77] -> 11, 41; right [0x4c] -> -7, 3,
    // then 0x05 -> 17, 19 (dropped), leaving right at 19, index 7.
    // [0x3f, 0x09]: left 0x3f from 41, index 16 -> -22, 41 (as in MONO);
    // right 0x09 from 19, index 7 (step 14):
    //   9: -(1 + 3) = -4 -> 15, index 6
    //   0 at index 6 (step 13): 1 -> 16, index 5
    // (skipping 0x05 would decode 0x09 from 3, index 4 -> 0, 1 instead)
    let file = movie(
        &v3_sound_header(2),
        &[
            chunk(b"SND2", &[0x77, 0x4c, 0x05]),
            chunk(b"SND2", &[0x3f, 0x09]),
        ],
    );
    assert_eq!(
        decode_audio(&file),
        Ok(vec![11, -7, 41, 3, -22, 15, 41, 16])
    );
}

#[test]
fn decode_audio_keeps_the_left_predictor_in_step_over_an_odd_v1_v2_chunk() {
    // locks current behavior: as above, but an odd v1/v2 chunk ends on an
    // unpaired left byte, which the crate decodes and drops.
    //
    // [0x77, 0x4c, 0x3f]: left 0x77 -> 11, 41; right 0x4c -> -7, 3; then
    // left 0x3f -> -22, 41 (dropped), leaving left at 41, index 23 as in
    // MONO. [0x09, 0x05]: left 0x09 -> 17, 24 and right 0x05 -> 17, 19,
    // the rest of MONO and OTHER
    // (skipping 0x3f would decode 0x09 from 41, index 16 -> 29, 32 instead)
    for version in [VQAVersion::One, VQAVersion::Two] {
        let file = movie(
            &sound_header(version, 2, 16),
            &[
                chunk(b"SND2", &[0x77, 0x4c, 0x3f]),
                chunk(b"SND2", &[0x09, 0x05]),
            ],
        );
        assert_eq!(
            decode_audio(&file),
            Ok(vec![11, -7, 41, 3, 17, 17, 24, 19]),
            "{version:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Error

#[test]
fn error_messages_are_nonempty_and_distinct_per_variant() {
    // locks current behavior
    let errors = [
        Error::Parse,
        Error::Lcw(LcwError::Truncated),
        Error::TooLarge("x"),
        Error::Video("x"),
        Error::UnsupportedSound("x"),
    ];
    let messages: HashSet<String> = errors.iter().map(ToString::to_string).collect();
    assert_eq!(messages.len(), errors.len());
    assert!(messages.iter().all(|message| !message.is_empty()));
}

#[test]
fn lcw_errors_convert_into_error_and_are_its_source() {
    // locks current behavior
    assert_eq!(
        Error::from(LcwError::BadOffset),
        Error::Lcw(LcwError::BadOffset)
    );

    fn fails() -> Result<(), Error> {
        let lcw: Result<(), LcwError> = Err(LcwError::Truncated);
        lcw?;
        Ok(())
    }
    let error = fails().unwrap_err();
    assert_eq!(error, Error::Lcw(LcwError::Truncated));

    let source = error.source().expect("an LCW error has a source");
    assert_eq!(
        source.downcast_ref::<LcwError>(),
        Some(&LcwError::Truncated)
    );
}

#[test]
fn other_errors_have_no_source() {
    // locks current behavior
    for error in [
        Error::Parse,
        Error::TooLarge("x"),
        Error::Video("x"),
        Error::UnsupportedSound("x"),
    ] {
        assert!(error.source().is_none(), "{error:?}");
    }
}
