//! HiColor (15-bit) video decoding through `FrameDecoder` and the
//! `VQA::frames` iterator, checked against doc/hc-vqa.txt and the HC_VQA
//! copy in doc/vqa.txt: LCW-compressed VPRZ streams, every write command on
//! the 4x4-block render path, the Blade Runner alpha-skip commands, VQFL
//! codebook refreshes, relative-LCW codebooks, stray codebook bytes,
//! command-stream errors, and the conversion of 15-bit pixels to RGB888.

// binary literals below group digits by field (prefix_run_index, or
// alpha_red_green_blue), not by four
#![allow(clippy::unusual_byte_groupings)]

mod common;

use common::{chunk, header_hicolor, lcw_literals, movie};
use vqa::{Error, Frame, FrameDecoder, FramePixels, VQA, VQAHeader};

/// Little-endian bytes of 16-bit words: codebook pixels or VPTR commands.
fn le(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// The pixels of a HiColor frame.
fn pixels(frame: &Frame) -> &[u16] {
    match &frame.pixels {
        FramePixels::HiColor { pixels } => pixels,
        FramePixels::Indexed { .. } => panic!("expected a HiColor frame"),
    }
}

/// [`header_hicolor`] resized to a `width` x `height` frame of `block_width`
/// x `block_height` blocks.
fn sized(width: u16, height: u16, block_width: u8, block_height: u8) -> VQAHeader {
    VQAHeader {
        width,
        height,
        block_width,
        block_height,
        ..header_hicolor()
    }
}

/// Decode a VQFR payload made of the sub-chunks `chunks` into the decoder's
/// next frame, and return its pixels.
fn decode(decoder: &mut FrameDecoder, chunks: &[Vec<u8>]) -> Vec<u16> {
    let frame = decoder.decode_frame(&chunks.concat()).unwrap();
    pixels(&frame).to_vec()
}

/// The pixels of a frame of solid 4x2 blocks, `blocks_x` to a row, given
/// each block's value in row-major block order.
fn solid_4x2_blocks(values: &[u16], blocks_x: usize) -> Vec<u16> {
    let mut pixels = Vec::new();
    for row in values.chunks(blocks_x) {
        for _ in 0..2 {
            for &value in row {
                pixels.extend([value; 4]);
            }
        }
    }
    pixels
}

/// Three 4x2 entries, 48 bytes: 1..=8, 11..=18, 21..=28.
fn three_entries() -> Vec<u8> {
    #[rustfmt::skip]
    let entries = le(&[
         1,  2,  3,  4,  5,  6,  7,  8,
        11, 12, 13, 14, 15, 16, 17, 18,
        21, 22, 23, 24, 25, 26, 27, 28,
    ]);
    entries
}

// hc-vqa.txt, The VPTR chunks: "VPRZ chunks ... are just VPTR chunks
// compressed with the standard Format80 algorithm" (vqa.txt, Appendix A)
#[test]
fn vprz_decodes_like_the_same_stream_sent_raw_as_vptr() {
    // a 16x4 movie with 4x2 blocks: 4x2 = 8 blocks
    let header = sized(16, 4, 4, 2);

    // entry 1 three times, entry 2, skip 2 blocks, entry 0 twice (101)
    let stream = [
        le(&[
            0b011_0000000000001,
            0b011_0000000000001,
            0b011_0000000000001,
            0b011_0000000000010,
            0b000_0000000000010,
            0b101_0000000000000,
        ]),
        vec![2],
    ]
    .concat();
    assert_eq!(
        stream,
        [1, 0x60, 1, 0x60, 1, 0x60, 2, 0x60, 2, 0, 0, 0xa0, 2]
    );

    // Format80: literal "01 60"; command (2) copying 4 bytes from 2 behind,
    // overlapping its own output ("01 60 01 60"); a literal of the last 7
    // bytes; the 0x80 end marker
    let compressed = [
        &[0x80 | 2, 1, 0x60][..],
        &[0b0_001_0000, 2],
        &[0x80 | 7, 2, 0x60, 2, 0, 0, 0xa0, 2],
        &[0x80],
    ]
    .concat();

    #[rustfmt::skip]
    let expected = [
        11, 12, 13, 14,  11, 12, 13, 14,  11, 12, 13, 14,  21, 22, 23, 24,
        15, 16, 17, 18,  15, 16, 17, 18,  15, 16, 17, 18,  25, 26, 27, 28,
         0,  0,  0,  0,   0,  0,  0,  0,   1,  2,  3,  4,   1,  2,  3,  4,
         0,  0,  0,  0,   0,  0,  0,  0,   5,  6,  7,  8,   5,  6,  7,  8,
    ];

    let mut decoder = FrameDecoder::new(&header).unwrap();
    let raw = decode(
        &mut decoder,
        &[chunk(b"CBF0", &three_entries()), chunk(b"VPTR", &stream)],
    );
    let mut decoder = FrameDecoder::new(&header).unwrap();
    let packed = decode(
        &mut decoder,
        &[
            chunk(b"CBF0", &three_entries()),
            chunk(b"VPRZ", &compressed),
        ],
    );
    assert_eq!(raw, expected);
    assert_eq!(packed, raw);
}

// hc-vqa.txt: VPRZ is "the standard Format80 algorithm", whose long copies
// (3) and (5) read from a Pos "absolute from the start of the destination
// buffer" (vqa.txt, Appendix A); only a NUL-led CBFZ uses relative ones
#[test]
fn vprz_long_copies_read_absolute_positions() {
    // a 16x4 movie with 4x2 blocks: 4x2 = 8 blocks
    let header = sized(16, 4, 4, 2);

    // entry 1, entry 2, skip 2 blocks, entry 1, entry 2, entry 0 twice (101)
    let stream = [
        le(&[
            0b011_0000000000001,
            0b011_0000000000010,
            0b000_0000000000010,
            0b011_0000000000001,
            0b011_0000000000010,
            0b101_0000000000000,
        ]),
        vec![2],
    ]
    .concat();
    assert_eq!(
        stream,
        [1, 0x60, 2, 0x60, 2, 0, 1, 0x60, 2, 0x60, 0, 0xa0, 2]
    );

    #[rustfmt::skip]
    let expected = [
        11, 12, 13, 14,  21, 22, 23, 24,  0, 0, 0, 0,  0, 0, 0, 0,
        15, 16, 17, 18,  25, 26, 27, 28,  0, 0, 0, 0,  0, 0, 0, 0,
        11, 12, 13, 14,  21, 22, 23, 24,  1, 2, 3, 4,  1, 2, 3, 4,
        15, 16, 17, 18,  25, 26, 27, 28,  5, 6, 7, 8,  5, 6, 7, 8,
    ];
    let mut decoder = FrameDecoder::new(&header).unwrap();
    let raw = decode(
        &mut decoder,
        &[chunk(b"CBF0", &three_entries()), chunk(b"VPTR", &stream)],
    );
    assert_eq!(raw, expected);

    // Format80: a literal of the first 6 bytes; a long copy of 4 bytes from
    // position 0 ("01 60 02 60"); a literal of the last 3 bytes; the 0x80
    // end marker. Read relative, position 0 would be the write position
    // itself, with nothing behind it to copy
    let head = &[0x80 | 6, 1, 0x60, 2, 0x60, 2, 0][..];
    let tail = &[0x80 | 3, 0, 0xa0, 2, 0x80][..];
    let copies = [
        // (3): Count-3 = 1, then the word Pos = 0
        &[0b11_000001, 0, 0][..],
        // (5): the words Count = 4 and Pos = 0
        &[0xff, 4, 0, 0, 0],
    ];
    for copy in copies {
        let compressed = [head, copy, tail].concat();
        let mut decoder = FrameDecoder::new(&header).unwrap();
        let packed = decode(
            &mut decoder,
            &[
                chunk(b"CBF0", &three_entries()),
                chunk(b"VPRZ", &compressed),
            ],
        );
        assert_eq!(packed, raw, "{compressed:02x?}");
    }
}

/// A 12x8 HiColor movie with 4x4 blocks: 3x2 = 6 blocks, drawn by the
/// render path specialized for 4x4. The block grid isn't square, so mixing
/// up its column and row counts, or the frame's width and height, shows.
fn header_4x4() -> VQAHeader {
    sized(12, 8, 4, 4)
}

/// Four 4x4 entries: entry `e` holds pixels 16e+1 ..= 16e+16, row-major.
fn codebook_4x4() -> Vec<u8> {
    le(&(1..=64).collect::<Vec<_>>())
}

// hc-vqa.txt: "011 - Write block (Val & 0x1fff)"; "000 - Skip Count blocks";
// blocks go left to right, top to down
#[test]
fn prefix_011_writes_one_4x4_block() {
    let mut decoder = FrameDecoder::new(&header_4x4()).unwrap();
    // entry 2, skip a block, entry 1, entry 0 (starting the second block
    // row), skip a block, entry 3
    let stream = le(&[
        0b011_0000000000010,
        0b000_0000000000001,
        0b011_0000000000001,
        0b011_0000000000000,
        0b000_0000000000001,
        0b011_0000000000011,
    ]);
    let frame = decode(
        &mut decoder,
        &[chunk(b"CBF0", &codebook_4x4()), chunk(b"VPTR", &stream)],
    );
    #[rustfmt::skip]
    assert_eq!(frame, [
        33, 34, 35, 36,   0,  0,  0,  0,  17, 18, 19, 20,
        37, 38, 39, 40,   0,  0,  0,  0,  21, 22, 23, 24,
        41, 42, 43, 44,   0,  0,  0,  0,  25, 26, 27, 28,
        45, 46, 47, 48,   0,  0,  0,  0,  29, 30, 31, 32,
         1,  2,  3,  4,   0,  0,  0,  0,  49, 50, 51, 52,
         5,  6,  7,  8,   0,  0,  0,  0,  53, 54, 55, 56,
         9, 10, 11, 12,   0,  0,  0,  0,  57, 58, 59, 60,
        13, 14, 15, 16,   0,  0,  0,  0,  61, 62, 63, 64,
    ]);
}

// hc-vqa.txt: "101 - Write block (Val & 0x1fff) Count times. Count is the
// next byte"; the next command word follows that byte
#[test]
fn prefix_101_writes_a_4x4_block_count_times() {
    let mut decoder = FrameDecoder::new(&header_4x4()).unwrap();
    // entry 3 four times, wrapping to the second block row, then entry 1
    // from an odd stream offset
    let stream = [
        le(&[0b101_0000000000011]),
        vec![4],
        le(&[0b011_0000000000001]),
    ]
    .concat();
    let frame = decode(
        &mut decoder,
        &[chunk(b"CBF0", &codebook_4x4()), chunk(b"VPTR", &stream)],
    );
    #[rustfmt::skip]
    assert_eq!(frame, [
        49, 50, 51, 52,  49, 50, 51, 52,  49, 50, 51, 52,
        53, 54, 55, 56,  53, 54, 55, 56,  53, 54, 55, 56,
        57, 58, 59, 60,  57, 58, 59, 60,  57, 58, 59, 60,
        61, 62, 63, 64,  61, 62, 63, 64,  61, 62, 63, 64,
        49, 50, 51, 52,  17, 18, 19, 20,   0,  0,  0,  0,
        53, 54, 55, 56,  21, 22, 23, 24,   0,  0,  0,  0,
        57, 58, 59, 60,  25, 26, 27, 28,   0,  0,  0,  0,
        61, 62, 63, 64,  29, 30, 31, 32,   0,  0,  0,  0,
    ]);
}

// hc-vqa.txt: "001 - Write block number (Val & 0xff) Count times. Count is
// (((Val/256) & 0x1f)+1)*2"
#[test]
fn prefix_001_writes_a_4x4_block_twice_per_run_step() {
    let mut decoder = FrameDecoder::new(&header_4x4()).unwrap();
    // entry 0, then entry 3 (1+1)*2 = 4 times, wrapping to the second block
    // row
    let stream = le(&[0b011_0000000000000, 0b001_00001_00000011]);
    let frame = decode(
        &mut decoder,
        &[chunk(b"CBF0", &codebook_4x4()), chunk(b"VPTR", &stream)],
    );
    #[rustfmt::skip]
    assert_eq!(frame, [
         1,  2,  3,  4,  49, 50, 51, 52,  49, 50, 51, 52,
         5,  6,  7,  8,  53, 54, 55, 56,  53, 54, 55, 56,
         9, 10, 11, 12,  57, 58, 59, 60,  57, 58, 59, 60,
        13, 14, 15, 16,  61, 62, 63, 64,  61, 62, 63, 64,
        49, 50, 51, 52,  49, 50, 51, 52,   0,  0,  0,  0,
        53, 54, 55, 56,  53, 54, 55, 56,   0,  0,  0,  0,
        57, 58, 59, 60,  57, 58, 59, 60,   0,  0,  0,  0,
        61, 62, 63, 64,  61, 62, 63, 64,   0,  0,  0,  0,
    ]);
}

// hc-vqa.txt: "010 - Write block number (Val & 0xff) and then write Count
// blocks getting their indexes by reading next Count bytes", Count as for 001
#[test]
fn prefix_010_writes_a_4x4_block_then_blocks_from_index_bytes() {
    let mut decoder = FrameDecoder::new(&header_4x4()).unwrap();
    // entry 2, then (1+1)*2 = 4 index bytes (entries 0, 3, 1, 3), then
    // entry 0
    let stream = [
        le(&[0b010_00001_00000010]),
        vec![0, 3, 1, 3],
        le(&[0b011_0000000000000]),
    ]
    .concat();
    let frame = decode(
        &mut decoder,
        &[chunk(b"CBF0", &codebook_4x4()), chunk(b"VPTR", &stream)],
    );
    #[rustfmt::skip]
    assert_eq!(frame, [
        33, 34, 35, 36,   1,  2,  3,  4,  49, 50, 51, 52,
        37, 38, 39, 40,   5,  6,  7,  8,  53, 54, 55, 56,
        41, 42, 43, 44,   9, 10, 11, 12,  57, 58, 59, 60,
        45, 46, 47, 48,  13, 14, 15, 16,  61, 62, 63, 64,
        17, 18, 19, 20,  49, 50, 51, 52,   1,  2,  3,  4,
        21, 22, 23, 24,  53, 54, 55, 56,   5,  6,  7,  8,
        25, 26, 27, 28,  57, 58, 59, 60,   9, 10, 11, 12,
        29, 30, 31, 32,  61, 62, 63, 64,  13, 14, 15, 16,
    ]);
}

/// Two 4x4 entries: entry 0 holds 1..=16; entry 1 holds 17..=32 with its
/// diagonal flagged transparent by the top (alpha) bit.
fn alpha_codebook_4x4() -> Vec<u8> {
    #[rustfmt::skip]
    let codebook = le(&[
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16,

        0x8000 | 17, 18, 19, 20,
        21, 0x8000 | 22, 23, 24,
        25, 26, 0x8000 | 27, 28,
        29, 30, 31, 0x8000 | 32,
    ]);
    codebook
}

/// A 4x4-block decoder whose first frame drew entry 0 of
/// [`alpha_codebook_4x4`] into all six blocks.
fn alpha_decoder_4x4() -> FrameDecoder {
    let mut decoder = FrameDecoder::new(&header_4x4()).unwrap();
    let stream = [le(&[0b101_0000000000000]), vec![6]].concat();
    decode(
        &mut decoder,
        &[
            chunk(b"CBF0", &alpha_codebook_4x4()),
            chunk(b"VPTR", &stream),
        ],
    );
    decoder
}

// vqa.txt, HC_VQA.txt copy: "100 - Same as 011 but skip pixels with alpha
// bit set" (the a of arrrrrgg gggbbbbb); flagged pixels keep the previous
// frame's value
#[test]
fn prefix_100_writes_a_4x4_block_keeping_alpha_pixels() {
    let mut decoder = alpha_decoder_4x4();
    // skip four blocks, then entry 1 over the fifth (second block row)
    let stream = le(&[0b000_0000000000100, 0b100_0000000000001]);
    let frame = decode(&mut decoder, &[chunk(b"VPTR", &stream)]);
    #[rustfmt::skip]
    assert_eq!(frame, [
         1,  2,  3,  4,   1,  2,  3,  4,   1,  2,  3,  4,
         5,  6,  7,  8,   5,  6,  7,  8,   5,  6,  7,  8,
         9, 10, 11, 12,   9, 10, 11, 12,   9, 10, 11, 12,
        13, 14, 15, 16,  13, 14, 15, 16,  13, 14, 15, 16,
         1,  2,  3,  4,   1, 18, 19, 20,   1,  2,  3,  4,
         5,  6,  7,  8,  21,  6, 23, 24,   5,  6,  7,  8,
         9, 10, 11, 12,  25, 26, 11, 28,   9, 10, 11, 12,
        13, 14, 15, 16,  29, 30, 31, 16,  13, 14, 15, 16,
    ]);
}

// vqa.txt, HC_VQA.txt copy: "110 - Same as 101 but skip pixels with alpha
// bit set"
#[test]
fn prefix_110_writes_a_4x4_block_count_times_keeping_alpha_pixels() {
    let mut decoder = alpha_decoder_4x4();
    // skip a block, then entry 1 over the next four, wrapping to the second
    // block row
    let stream = [le(&[0b000_0000000000001, 0b110_0000000000001]), vec![4]].concat();
    let frame = decode(&mut decoder, &[chunk(b"VPTR", &stream)]);
    #[rustfmt::skip]
    assert_eq!(frame, [
         1,  2,  3,  4,   1, 18, 19, 20,   1, 18, 19, 20,
         5,  6,  7,  8,  21,  6, 23, 24,  21,  6, 23, 24,
         9, 10, 11, 12,  25, 26, 11, 28,  25, 26, 11, 28,
        13, 14, 15, 16,  29, 30, 31, 16,  29, 30, 31, 16,
         1, 18, 19, 20,   1, 18, 19, 20,   1,  2,  3,  4,
        21,  6, 23, 24,  21,  6, 23, 24,   5,  6,  7,  8,
        25, 26, 11, 28,  25, 26, 11, 28,   9, 10, 11, 12,
        29, 30, 31, 16,  29, 30, 31, 16,  13, 14, 15, 16,
    ]);
}

// vqa.txt, HC_VQA.txt copy: "110 - Same as 101 but skip pixels with alpha
// bit set": Count writes, each keeping the previous pixel under a flagged
// one whatever its color bits (0xffff included)
#[test]
fn prefix_110_writes_a_block_count_times_keeping_alpha_pixels() {
    // the 8x4 movie with 4x2 blocks: 2x2 = 4 blocks
    let mut decoder = FrameDecoder::new(&header_hicolor()).unwrap();
    #[rustfmt::skip]
    let codebook = le(&[
        1, 2, 3, 4, 5, 6, 7, 8,
        0x8000 | 11, 12, 0xffff, 14, 15, 16, 17, 0x8000 | 18,
    ]);

    // frame 1: entry 0 into every block
    let stream = [le(&[0b101_0000000000000]), vec![4]].concat();
    decode(
        &mut decoder,
        &[chunk(b"CBF0", &codebook), chunk(b"VPTR", &stream)],
    );

    // frame 2: entry 1 three times, crossing into the second block row
    let stream = [le(&[0b110_0000000000001]), vec![3]].concat();
    let frame = decode(&mut decoder, &[chunk(b"VPTR", &stream)]);
    #[rustfmt::skip]
    assert_eq!(frame, [
         1, 12,  3, 14,   1, 12,  3, 14,
        15, 16, 17,  8,  15, 16, 17,  8,
         1, 12,  3, 14,   1,  2,  3,  4,
        15, 16, 17,  8,   5,  6,  7,  8,
    ]);
}

// vqa.txt, HC_VQA.txt copy: "110 - Same as 101 but skip pixels with alpha
// bit set", on a block size with no specialized render path
#[test]
fn prefix_110_keeps_alpha_pixels_with_unspecialized_block_sizes() {
    // a 4x2 movie with 2x2 blocks: 2 blocks
    let mut decoder = FrameDecoder::new(&sized(4, 2, 2, 2)).unwrap();
    let codebook = le(&[1, 2, 3, 4, 0x8000 | 11, 12, 13, 0x8000 | 14]);

    // frame 1: entry 0 into both blocks
    let stream = [le(&[0b101_0000000000000]), vec![2]].concat();
    let frame = decode(
        &mut decoder,
        &[chunk(b"CBF0", &codebook), chunk(b"VPTR", &stream)],
    );
    assert_eq!(frame, [1, 2, 1, 2, 3, 4, 3, 4]);

    // frame 2: entry 1 into both, around its flagged corners
    let stream = [le(&[0b110_0000000000001]), vec![2]].concat();
    let frame = decode(&mut decoder, &[chunk(b"VPTR", &stream)]);
    assert_eq!(frame, [1, 12, 1, 12, 13, 4, 13, 4]);
}

// locks current behavior, as the FramePixels::HiColor docs describe it: the
// plain writes (001, 010, 011, 101) copy an entry whole, its top bit
// included, so a frame can hold values above 0x7fff. doc/ says only "Write
// block" (hc-vqa.txt) and that the bit flags a transparent pixel (vqa.txt,
// HC_VQA copy)
#[test]
fn plain_writes_carry_the_alpha_bit_into_the_frame() {
    // a 16x4 movie with 4x2 blocks: 4x2 = 8 blocks
    let mut decoder = FrameDecoder::new(&sized(16, 4, 4, 2)).unwrap();
    let (a, w, b) = (0x8000 | 11, 0xffff, 0x8000 | 18);
    let codebook = le(&[a, 12, w, 14, 15, 16, 17, b]);

    // entry 0 once by 011, once by 101, twice by 001 (run 0), then by 010
    // (run 0) and its 2 index bytes
    let stream = [
        le(&[0b011_0000000000000, 0b101_0000000000000]),
        vec![1],
        le(&[0b001_00000_00000000, 0b010_00000_00000000]),
        vec![0, 0],
    ]
    .concat();
    let frame = decode(
        &mut decoder,
        &[chunk(b"CBF0", &codebook), chunk(b"VPTR", &stream)],
    );
    #[rustfmt::skip]
    assert_eq!(frame, [
         a, 12,  w, 14,   a, 12,  w, 14,   a, 12,  w, 14,   a, 12,  w, 14,
        15, 16, 17,  b,  15, 16, 17,  b,  15, 16, 17,  b,  15, 16, 17,  b,
         a, 12,  w, 14,   a, 12,  w, 14,   a, 12,  w, 14,   0,  0,  0,  0,
        15, 16, 17,  b,  15, 16, 17,  b,  15, 16, 17,  b,   0,  0,  0,  0,
    ]);
}

// hc-vqa.txt: 101 writes a block "Count times. Count is the next byte";
// vqa.txt's HC_VQA copy: "110 - Same as 101". Neither says what a zero byte
// does; writing nothing is the literal reading of "Count times" (locks
// current behavior)
#[test]
fn repeat_count_zero_writes_nothing() {
    let mut decoder = FrameDecoder::new(&header_hicolor()).unwrap();
    let codebook = le(&[1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]);
    // entry 0 zero times twice, then entry 1, which lands in the first block
    let stream = [
        le(&[0b101_0000000000000]),
        vec![0],
        le(&[0b110_0000000000000]),
        vec![0],
        le(&[0b011_0000000000001]),
    ]
    .concat();
    let frame = decode(
        &mut decoder,
        &[chunk(b"CBF0", &codebook), chunk(b"VPTR", &stream)],
    );
    #[rustfmt::skip]
    assert_eq!(frame, [
        11, 12, 13, 14,  0, 0, 0, 0,
        15, 16, 17, 18,  0, 0, 0, 0,
         0,  0,  0,  0,  0, 0, 0, 0,
         0,  0,  0,  0,  0, 0, 0, 0,
    ]);
}

// hc-vqa.txt: 001 and 010 take block numbers from Val & 0xff ("can only
// index the first 256 blocks") with bits 8-12 the run; 011 and 101 take
// Val & 0x1fff, and vqa.txt's HC_VQA copy makes 100 and 110 "Same as" 011
// and 101
#[test]
fn run_commands_index_8_bits_and_single_writes_index_13_bits() {
    // a 16x4 movie with 4x2 blocks: 4x2 = 8 blocks
    let mut decoder = FrameDecoder::new(&sized(16, 4, 4, 2)).unwrap();
    // 258 solid entries: entry e is all e+1
    let codebook = le(&(0..258).flat_map(|e| [e + 1; 8]).collect::<Vec<_>>());

    // frame 1: entry 0x101 (011), entry 0x01 four times (001, run 1 - read
    // as 13 bits, 0x101), entry 0x101 once (101)
    let stream = [
        le(&[
            0b011_0000100000001,
            0b001_00001_00000001,
            0b101_0000100000001,
        ]),
        vec![1],
    ]
    .concat();
    let frame = decode(
        &mut decoder,
        &[chunk(b"CBF0", &codebook), chunk(b"VPTR", &stream)],
    );
    #[rustfmt::skip]
    assert_eq!(frame, [
        258, 258, 258, 258,    2,   2,   2,   2,  2, 2, 2, 2,  2, 2, 2, 2,
        258, 258, 258, 258,    2,   2,   2,   2,  2, 2, 2, 2,  2, 2, 2, 2,
          2,   2,   2,   2,  258, 258, 258, 258,  0, 0, 0, 0,  0, 0, 0, 0,
          2,   2,   2,   2,  258, 258, 258, 258,  0, 0, 0, 0,  0, 0, 0, 0,
    ]);

    // frame 2: entry 0x02 (010, run 1 - read as 13 bits, 0x102, past the
    // codebook), then 4 index bytes: entries 3, 255, 0, 1
    let stream = [le(&[0b010_00001_00000010]), vec![3, 255, 0, 1]].concat();
    let frame = decode(&mut decoder, &[chunk(b"VPTR", &stream)]);
    #[rustfmt::skip]
    assert_eq!(frame, [
        3, 3, 3, 3,    4,   4,   4,   4,  256, 256, 256, 256,  1, 1, 1, 1,
        3, 3, 3, 3,    4,   4,   4,   4,  256, 256, 256, 256,  1, 1, 1, 1,
        2, 2, 2, 2,  258, 258, 258, 258,    0,   0,   0,   0,  0, 0, 0, 0,
        2, 2, 2, 2,  258, 258, 258, 258,    0,   0,   0,   0,  0, 0, 0, 0,
    ]);

    // frame 3: entry 0x101 by 100, then once by 110; no entry has the alpha
    // bit to skip, so both draw it whole (read as 8 bits, 0x01 would draw 2)
    let stream = [le(&[0b100_0000100000001, 0b110_0000100000001]), vec![1]].concat();
    let frame = decode(&mut decoder, &[chunk(b"VPTR", &stream)]);
    #[rustfmt::skip]
    assert_eq!(frame, [
        258, 258, 258, 258,  258, 258, 258, 258,  256, 256, 256, 256,  1, 1, 1, 1,
        258, 258, 258, 258,  258, 258, 258, 258,  256, 256, 256, 256,  1, 1, 1, 1,
          2,   2,   2,   2,  258, 258, 258, 258,    0,   0,   0,   0,  0, 0, 0, 0,
          2,   2,   2,   2,  258, 258, 258, 258,    0,   0,   0,   0,  0, 0, 0, 0,
    ]);
}

// hc-vqa.txt: 001 and 010 take Count from all five bits 8-12, "(((Val/256)
// & 0x1f)+1)*2"; run 16 sets bit 12 alone, for (16+1)*2 = 34 blocks
#[test]
fn run_counts_take_five_bits() {
    // a 40x8 movie with 4x2 blocks: 10x4 = 40 blocks
    let mut decoder = FrameDecoder::new(&sized(40, 8, 4, 2)).unwrap();
    // 3 solid entries: entry e is all e+1
    let codebook = le(&(0..3).flat_map(|e| [e + 1; 8]).collect::<Vec<_>>());

    // frame 1: entry 1 34 times (001, run 16), then entry 0 (011)
    let stream = le(&[0b001_10000_00000001, 0b011_0000000000000]);
    let frame = decode(
        &mut decoder,
        &[chunk(b"CBF0", &codebook), chunk(b"VPTR", &stream)],
    );
    let blocks = [vec![2; 34], vec![1], vec![0; 5]].concat();
    assert_eq!(frame, solid_4x2_blocks(&blocks, 10));

    // frame 2: entry 2, then 34 index bytes alternating entries 0 and 1
    // (010, run 16), then entry 2 (011)
    let stream = [
        le(&[0b010_10000_00000010]),
        [0, 1].repeat(17),
        le(&[0b011_0000000000010]),
    ]
    .concat();
    let frame = decode(&mut decoder, &[chunk(b"VPTR", &stream)]);
    let blocks = [vec![3], [1, 2].repeat(17), vec![3], vec![0; 4]].concat();
    assert_eq!(frame, solid_4x2_blocks(&blocks, 10));
}

/// The VQFL scenario on the 8x4 movie with 4x2 blocks: the VQFR payloads
/// of frames 1-3, and the VQFL between frames 1 and 2 whose CBFZ replaces
/// codebook A (1..=8, 11..=18) with B (21..=28, 31..=38).
fn refresh_chunks() -> ([Vec<u8>; 3], Vec<u8>) {
    let codebook_a = le(&[1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]);
    let codebook_b = le(&[
        21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
    ]);
    // frame 1: codebook A, entries 0, 1, 0, 1
    let frame_1 = [
        chunk(b"CBF0", &codebook_a),
        chunk(
            b"VPTR",
            &le(&[
                0b011_0000000000000,
                0b011_0000000000001,
                0b011_0000000000000,
                0b011_0000000000001,
            ]),
        ),
    ]
    .concat();
    // frame 2: skip a block, entry 0 into the second
    let frame_2 = chunk(b"VPTR", &le(&[0b000_0000000000001, 0b011_0000000000000]));
    // frame 3: skip three blocks, entry 1 into the fourth
    let frame_3 = chunk(b"VPTR", &le(&[0b000_0000000000011, 0b011_0000000000001]));
    let vqfl = chunk(b"CBFZ", &lcw_literals(&codebook_b));
    ([frame_1, frame_2, frame_3], vqfl)
}

/// The frames [`refresh_chunks`] decodes to: frame 1 from codebook A; frame
/// 2 rewrites only its second block, from B; frame 3 only its fourth,
/// from B still.
#[rustfmt::skip]
const REFRESH_FRAMES: [[u16; 32]; 3] = [
    [
        1, 2, 3, 4,  11, 12, 13, 14,
        5, 6, 7, 8,  15, 16, 17, 18,
        1, 2, 3, 4,  11, 12, 13, 14,
        5, 6, 7, 8,  15, 16, 17, 18,
    ],
    [
        1, 2, 3, 4,  21, 22, 23, 24,
        5, 6, 7, 8,  25, 26, 27, 28,
        1, 2, 3, 4,  11, 12, 13, 14,
        5, 6, 7, 8,  15, 16, 17, 18,
    ],
    [
        1, 2, 3, 4,  21, 22, 23, 24,
        5, 6, 7, 8,  25, 26, 27, 28,
        1, 2, 3, 4,  31, 32, 33, 34,
        5, 6, 7, 8,  35, 36, 37, 38,
    ],
];

// hc-vqa.txt: later CBFZ chunks come in VQFL chunks ahead of the VQFR they
// apply to, "Subsequent frames use the last lookup table loaded", and VPTR
// "only records changes from the previous frame"
#[test]
fn vqfl_codebook_refresh_applies_to_later_frames_differentially() {
    let ([frame_1, frame_2, frame_3], vqfl) = refresh_chunks();
    let mut decoder = FrameDecoder::new(&header_hicolor()).unwrap();

    let frame = decoder.decode_frame(&frame_1).unwrap();
    assert_eq!(pixels(&frame), REFRESH_FRAMES[0]);
    decoder.process_vqfl(&vqfl).unwrap();
    let frame = decoder.decode_frame(&frame_2).unwrap();
    assert_eq!(pixels(&frame), REFRESH_FRAMES[1]);
    let frame = decoder.decode_frame(&frame_3).unwrap();
    assert_eq!(pixels(&frame), REFRESH_FRAMES[2]);
}

// hc-vqa.txt, as above, through the whole-movie iterator, which applies
// VQFL chunks as it walks past them
#[test]
fn frames_iterator_applies_vqfl_refreshes_between_frames() {
    let ([frame_1, frame_2, frame_3], vqfl) = refresh_chunks();
    // the header counts the movie's three frames
    let header = VQAHeader {
        num_frames: 3,
        ..header_hicolor()
    };
    let file = movie(
        &header,
        &[
            chunk(b"VQFR", &frame_1),
            chunk(b"VQFL", &vqfl),
            chunk(b"VQFR", &frame_2),
            chunk(b"VQFR", &frame_3),
        ],
    );
    let vqa = VQA::parse(&file).unwrap();
    let frames: Vec<Vec<u16>> = vqa
        .frames()
        .unwrap()
        .map(|frame| pixels(&frame.unwrap()).to_vec())
        .collect();
    assert_eq!(frames, REFRESH_FRAMES);
}

// hc-vqa.txt, The CBFZ chunks and The modified Format80 scheme: a CBFZ
// starting with NUL is Format80 from the next byte, with commands (3) and
// (5) copying from DP-Word(...), relative to the write position
#[test]
fn cbfz_with_a_leading_nul_uses_relative_long_copies() {
    let mut cbfz = vec![0x00, 0x80 | 48];
    cbfz.extend(three_entries());
    // (3): 13+3 = 16 bytes from 48-32 = 16, so entry 3 repeats entry 1;
    // absolute, it would copy from 32 and repeat entry 2
    cbfz.extend([0b11_001101, 32, 0]);
    // (5): 16 bytes from 64-64 = 0, so entry 4 repeats entry 0; absolute,
    // position 64 isn't written yet
    cbfz.extend([0xff, 16, 0, 64, 0]);
    cbfz.push(0x80);

    let mut decoder = FrameDecoder::new(&header_hicolor()).unwrap();
    // entries 3, 4, 1, 0
    let stream = le(&[
        0b011_0000000000011,
        0b011_0000000000100,
        0b011_0000000000001,
        0b011_0000000000000,
    ]);
    let frame = decode(
        &mut decoder,
        &[chunk(b"CBFZ", &cbfz), chunk(b"VPTR", &stream)],
    );
    #[rustfmt::skip]
    assert_eq!(frame, [
        11, 12, 13, 14,  1, 2, 3, 4,
        15, 16, 17, 18,  5, 6, 7, 8,
        11, 12, 13, 14,  1, 2, 3, 4,
        15, 16, 17, 18,  5, 6, 7, 8,
    ]);
}

// hc-vqa.txt, The CBFZ chunks: without a leading NUL, a CBFZ is standard
// Format80, whose command (3) copies from a position absolute from the
// start of the output (vqa.txt, Appendix A)
#[test]
fn cbfz_without_a_leading_nul_uses_absolute_long_copies() {
    let mut cbfz = vec![0x80 | 48];
    cbfz.extend(three_entries());
    // (3): 16 bytes from position 32, so entry 3 repeats entry 2
    cbfz.extend([0b11_001101, 32, 0]);
    cbfz.push(0x80);

    let mut decoder = FrameDecoder::new(&header_hicolor()).unwrap();
    // entries 3, 2, 1, 0
    let stream = le(&[
        0b011_0000000000011,
        0b011_0000000000010,
        0b011_0000000000001,
        0b011_0000000000000,
    ]);
    let frame = decode(
        &mut decoder,
        &[chunk(b"CBFZ", &cbfz), chunk(b"VPTR", &stream)],
    );
    #[rustfmt::skip]
    assert_eq!(frame, [
        21, 22, 23, 24,  21, 22, 23, 24,
        25, 26, 27, 28,  25, 26, 27, 28,
        11, 12, 13, 14,   1,  2,  3,  4,
        15, 16, 17, 18,   5,  6,  7,  8,
    ]);
}

// locks current behavior: Westwood's compressor leaves stray bytes after
// the last entry of some retail HiColor codebooks. A trailing partial entry
// is accepted but never drawn: a write of it keeps the block's previous
// pixels and still advances the position (whether set_codebook truncates
// the partial entry is unobservable)
#[test]
fn trailing_partial_codebook_entry_is_accepted_but_never_drawn() {
    let old_codebook = le(&[0x7c00; 16]);
    let entry_0 = le(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let entry_1 = le(&[11, 12, 13, 14, 15, 16, 17, 18]);
    for id in [b"CBF0", b"CBFZ"] {
        for stray in [1, 2, 15] {
            let mut decoder = FrameDecoder::new(&header_hicolor()).unwrap();

            // frame 1: entry 0 of a two-entry codebook everywhere
            let stream = [le(&[0b101_0000000000000]), vec![4]].concat();
            decode(
                &mut decoder,
                &[chunk(b"CBF0", &old_codebook), chunk(b"VPTR", &stream)],
            );

            // frame 2: a new codebook of entry 0 plus the first `stray`
            // bytes of entry 1; draw entries 0, 1, 0
            let bytes = [&entry_0[..], &entry_1[..stray]].concat();
            let payload = if id == b"CBFZ" {
                lcw_literals(&bytes)
            } else {
                bytes
            };
            let stream = le(&[
                0b011_0000000000000,
                0b011_0000000000001,
                0b011_0000000000000,
            ]);
            let frame = decode(
                &mut decoder,
                &[chunk(id, &payload), chunk(b"VPTR", &stream)],
            );
            #[rustfmt::skip]
            assert_eq!(frame, [
                1, 2, 3, 4,  0x7c00, 0x7c00, 0x7c00, 0x7c00,
                5, 6, 7, 8,  0x7c00, 0x7c00, 0x7c00, 0x7c00,
                1, 2, 3, 4,  0x7c00, 0x7c00, 0x7c00, 0x7c00,
                5, 6, 7, 8,  0x7c00, 0x7c00, 0x7c00, 0x7c00,
            ], "{} with {stray} stray bytes", String::from_utf8_lossy(id));
        }
    }
}

/// 5-bit channel values expanded to 8 bits by bit replication, v << 3 |
/// v >> 2: the top three bits repeat below, so 0 -> 0 and 31 -> 255.
#[rustfmt::skip]
const EXPAND5: [u8; 32] = [
      0,   8,  16,  24,  33,  41,  49,  57,
     66,  74,  82,  90,  99, 107, 115, 123,
    132, 140, 148, 156, 165, 173, 181, 189,
    198, 206, 214, 222, 231, 239, 247, 255,
];

// hc-vqa.txt, The CBFZ chunks: pixels are 0rrrrrgg gggbbbbb, 0-31 per
// channel (vqa.txt's copy makes the top bit alpha, not color); first
// principles: bit replication stretches 0-31 over the full 0-255
#[test]
fn hicolor_channels_expand_to_8_bits_by_bit_replication() {
    let mut pixels = Vec::new();
    let mut expected: Vec<[u8; 3]> = Vec::new();
    // each row ramps one channel through 0-31, then adds that channel at
    // full scale with the top bit set, a pixel mixing 21, 10 and 30 in
    // some order, and one more top-bit case
    for v in 0..32 {
        pixels.push(v << 10);
        expected.push([EXPAND5[usize::from(v)], 0, 0]);
    }
    pixels.extend([0b1_11111_00000_00000, 0b0_10101_01010_11110, 0xffff]);
    expected.extend([[255, 0, 0], [173, 82, 247], [255, 255, 255]]);
    for v in 0..32 {
        pixels.push(v << 5);
        expected.push([0, EXPAND5[usize::from(v)], 0]);
    }
    pixels.extend([0b1_00000_11111_00000, 0b0_01010_11110_10101, 0x8000]);
    expected.extend([[0, 255, 0], [82, 247, 173], [0, 0, 0]]);
    for v in 0..32 {
        pixels.push(v);
        expected.push([0, 0, EXPAND5[usize::from(v)]]);
    }
    pixels.extend([0b1_00000_00000_11111, 0b0_11110_10101_01010, 0x7fff]);
    expected.extend([[0, 0, 255], [247, 173, 82], [255, 255, 255]]);
    let expected = expected.as_flattened();

    // 35x3 = 105 pixels: when the CPU has a SIMD level, six 16-pixel vector
    // steps, then a 9-pixel scalar tail; at the fallback level all of it is
    // scalar (the rgb.rs unit tests cover the baseline level)
    let frame = Frame {
        width: 35,
        height: 3,
        pixels: FramePixels::HiColor { pixels },
    };
    assert_eq!(frame.to_rgb888(), expected);
    assert_eq!(frame.view().to_rgb888(), expected);

    // the write_ variants overwrite every byte of the buffer they're given
    let mut out = vec![0xaa; 105 * 3];
    frame.write_rgb888(&mut out);
    assert_eq!(out, expected);
    let mut out = vec![0xaa; 105 * 3];
    frame.view().write_rgb888(&mut out);
    assert_eq!(out, expected);
}

/// A 2x1 HiColor frame.
fn two_pixel_frame() -> Frame {
    Frame {
        width: 2,
        height: 1,
        pixels: FramePixels::HiColor {
            pixels: vec![0x7fff, 0],
        },
    }
}

// Frame::write_rgb888 docs, # Panics: out must be exactly three bytes per
// pixel long; the message locks current behavior
#[test]
#[should_panic(expected = "RGB888 output must hold three bytes per pixel")]
fn write_rgb888_panics_on_a_short_buffer() {
    two_pixel_frame().write_rgb888(&mut [0; 5]);
}

// FrameRef::write_rgb888 docs, # Panics, as above
#[test]
#[should_panic(expected = "RGB888 output must hold three bytes per pixel")]
fn write_rgb888_panics_on_a_long_buffer() {
    two_pixel_frame().view().write_rgb888(&mut [0; 7]);
}

// locks current behavior: command words are 16 bits, so a single byte left
// after the last command is an error
#[test]
fn dangling_odd_byte_after_the_last_command_is_an_error() {
    for stream in [vec![0x60], [le(&[0b011_0000000000000]), vec![0]].concat()] {
        let mut decoder = FrameDecoder::new(&header_hicolor()).unwrap();
        assert_eq!(
            decoder.decode_frame(&chunk(b"VPTR", &stream)),
            Err(Error::Video("dangling byte in pointer stream")),
            "{stream:02x?}"
        );
    }
}

// locks current behavior: a stream ending where prefix 010 needs index
// bytes, or 101 and 110 their count byte, is an error
#[test]
fn stream_ending_before_an_index_or_count_byte_is_truncated() {
    let streams = [
        // 010, run 0: none of its 2 index bytes
        le(&[0b010_00000_00000000]),
        // 010, run 0: 1 of 2
        [le(&[0b010_00000_00000000]), vec![0]].concat(),
        // 010, run 1: 3 of 4
        [le(&[0b010_00001_00000000]), vec![0, 0, 0]].concat(),
        // 101 and 110 without their count byte
        le(&[0b101_0000000000000]),
        le(&[0b110_0000000000000]),
    ];
    for stream in streams {
        let mut decoder = FrameDecoder::new(&header_hicolor()).unwrap();
        assert_eq!(
            decoder.decode_frame(&chunk(b"VPTR", &stream)),
            Err(Error::Video("truncated pointer stream")),
            "{stream:02x?}"
        );
    }
}

// locks current behavior: a write beyond the last block is an error; the
// last block itself is writable, and a skip past it writes nothing. The
// counts come from hc-vqa.txt: 000 skips Val & 0x1fff blocks, 001 writes
// (((Val/256) & 0x1f)+1)*2 times
#[test]
fn writes_past_the_last_block_are_errors() {
    // the 8x4 movie with 4x2 blocks: 4 blocks
    let codebook = chunk(b"CBF0", &le(&[1; 8]));
    let past = [
        // 101 and 110, 5 times
        [le(&[0b101_0000000000000]), vec![5]].concat(),
        [le(&[0b110_0000000000000]), vec![5]].concat(),
        // 001, run 2: (2+1)*2 = 6 times
        le(&[0b001_00010_00000000]),
        // 001, run 16 (bit 12 of the run): (16+1)*2 = 34 times
        le(&[0b001_10000_00000000]),
        // 010, run 1: 1 + 4 blocks
        [le(&[0b010_00001_00000000]), vec![0; 4]].concat(),
        // skip all 4 blocks, then 011 or 100
        le(&[0b000_0000000000100, 0b011_0000000000000]),
        le(&[0b000_0000000000100, 0b100_0000000000000]),
        // skip 0x103 = 259 blocks (bit 8 of the count), then 011
        le(&[0b000_0000100000011, 0b011_0000000000000]),
    ];
    for stream in past {
        let mut decoder = FrameDecoder::new(&header_hicolor()).unwrap();
        assert_eq!(
            decoder.decode_frame(&[codebook.clone(), chunk(b"VPTR", &stream)].concat()),
            Err(Error::Video("pointer stream writes past the frame")),
            "{stream:02x?}"
        );
    }

    #[rustfmt::skip]
    let fits = [
        // 101, 4 times: every block
        ([le(&[0b101_0000000000000]), vec![4]].concat(), [
            1, 1, 1, 1,  1, 1, 1, 1,
            1, 1, 1, 1,  1, 1, 1, 1,
            1, 1, 1, 1,  1, 1, 1, 1,
            1, 1, 1, 1,  1, 1, 1, 1,
        ]),
        // skip 3, then 011 into the last block
        (le(&[0b000_0000000000011, 0b011_0000000000000]), [
            0, 0, 0, 0,  0, 0, 0, 0,
            0, 0, 0, 0,  0, 0, 0, 0,
            0, 0, 0, 0,  1, 1, 1, 1,
            0, 0, 0, 0,  1, 1, 1, 1,
        ]),
        // skip 5
        (le(&[0b000_0000000000101]), [0; 32]),
    ];
    for (stream, expected) in fits {
        let mut decoder = FrameDecoder::new(&header_hicolor()).unwrap();
        let frame = decode(&mut decoder, &[codebook.clone(), chunk(b"VPTR", &stream)]);
        assert_eq!(frame, expected, "{stream:02x?}");
    }
}

// locks current behavior: the largest header values make a decoder, on
// 32-bit targets too - 0xff00 entries (maxblocks 0) of 255x255 pixels would
// overflow a 32-bit usize unless the codebook cap saturates
#[test]
fn new_takes_the_largest_blocks_and_codebook() {
    let header = VQAHeader {
        maxblocks: 0,
        ..sized(255, 255, 255, 255)
    };
    assert!(FrameDecoder::new(&header).is_ok());
}

// locks current behavior: HiColor frames come from VPTR/VPRZ command
// streams (hc-vqa.txt), so an 8-bit VPT0 or VPTZ pointer table is an error
#[test]
fn vpt0_and_vptz_pointer_tables_are_errors_in_a_hicolor_movie() {
    // a well-formed table for 4 blocks: LoVal and HiVal halves
    let table = [0u8; 8];
    for vqfr in [
        chunk(b"VPT0", &table),
        chunk(b"VPTZ", &lcw_literals(&table)),
    ] {
        let mut decoder = FrameDecoder::new(&header_hicolor()).unwrap();
        assert_eq!(
            decoder.decode_frame(&vqfr),
            Err(Error::Video("VPT? pointer table in a HiColor movie"))
        );
    }
}
