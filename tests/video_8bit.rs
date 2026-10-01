//! Integration tests for 8-bit palettized video through `FrameDecoder`: v1
//! and v2 pointer tables and their fill markers, block grids, full
//! codebooks and codebooks sent in parts (over more than one cycle),
//! palettes, the LCW-compressed chunk variants, 4x4 blocks, conversion of
//! indexed frames to RGB888, and the errors on malformed 8-bit video data.
//! Expected frames are worked out by hand from doc/vqa.txt on movies of a
//! few blocks.

// the VPTR command word below groups its bits by field (prefix_index)
#![allow(clippy::unusual_byte_groupings)]

mod common;

use common::{chunk, header_8bit, kind, lcw_literals};
use vqa::lcw::{self, LcwError};
use vqa::{
    Error, ErrorKind, Frame, FrameDecoder, FramePixels, FramePixelsRef, Limit, VQAHeader,
    VQAVersion, VideoError,
};

/// The palette indices and palette of an 8-bit frame.
fn indexed(frame: &Frame) -> (&[u8], &[[u8; 3]]) {
    match &frame.pixels {
        FramePixels::Indexed { pixels, palette } => (pixels, palette),
        FramePixels::HiColor { .. } => panic!("expected an indexed frame"),
    }
}

/// The palette indices of an 8-bit frame.
fn pixels(frame: &Frame) -> &[u8] {
    indexed(frame).0
}

/// Decode one frame made of `chunks` with a fresh decoder for `header`.
fn decode_one(header: &VQAHeader, chunks: &[Vec<u8>]) -> Result<Frame, Error> {
    FrameDecoder::new(header)
        .unwrap()
        .decode_frame(&chunks.concat())
}

/// A codebook of `count` 4x2 entries, all 0xee (a value no expected frame
/// holds) except the listed ones.
fn codebook_4x2(count: usize, entries: &[(usize, [u8; 8])]) -> Vec<u8> {
    let mut book = vec![0xee; count * 8];
    for (index, entry) in entries {
        book[index * 8..][..8].copy_from_slice(entry);
    }
    book
}

/// A v2 table for the 8x4 movie drawing entries 0, 1, 2, 3 into its blocks
/// (top left, top right, bottom left, bottom right).
const IDENTITY_TABLE: [u8; 8] = [0, 1, 2, 3, /* HiVal */ 0, 0, 0, 0];

/// `IDENTITY_TABLE` over the codebook of bytes 0..32 (entry k = 8k..8k+8).
#[rustfmt::skip]
const OLD_FRAME: [u8; 32] = [
     0,  1,  2,  3,    8,  9, 10, 11,
     4,  5,  6,  7,   12, 13, 14, 15,
    16, 17, 18, 19,   24, 25, 26, 27,
    20, 21, 22, 23,   28, 29, 30, 31,
];

/// `IDENTITY_TABLE` over the codebook of bytes 100..132.
#[rustfmt::skip]
const NEW_FRAME: [u8; 32] = [
    100, 101, 102, 103,   108, 109, 110, 111,
    104, 105, 106, 107,   112, 113, 114, 115,
    116, 117, 118, 119,   124, 125, 126, 127,
    120, 121, 122, 123,   128, 129, 130, 131,
];

/// `IDENTITY_TABLE` over the codebook of bytes 200..232.
#[rustfmt::skip]
const NEXT_FRAME: [u8; 32] = [
    200, 201, 202, 203,   208, 209, 210, 211,
    204, 205, 206, 207,   212, 213, 214, 215,
    216, 217, 218, 219,   224, 225, 226, 227,
    220, 221, 222, 223,   228, 229, 230, 231,
];

#[test]
fn v1_tables_pair_loval_and_hival_per_block() {
    // vqa.txt, VERSION 1 INDEX TABLE LAYOUT: one (LoVal, HiVal) byte pair
    // per block; HiVal 0xff fills the block with color 255-LoVal, anything
    // else (0x0f included) draws entry (HiVal*256+LoVal)/8
    let header = VQAHeader {
        version: VQAVersion::One,
        ..header_8bit()
    };
    let book = codebook_4x2(
        481,
        &[
            (0, [0, 1, 2, 3, 4, 5, 6, 7]),
            (1, [10, 11, 12, 13, 14, 15, 16, 17]),
            (480, [100, 101, 102, 103, 104, 105, 106, 107]),
        ],
    );
    let table = [
        5, 0xff, // fill with 255-5 = 250
        8, 0x00, // entry 8/8 = 1
        0, 0x00, // entry 0
        0, 0x0f, // entry 0xf00/8 = 480
    ];

    let frame = decode_one(&header, &[chunk(b"CBF0", &book), chunk(b"VPT0", &table)]).unwrap();
    #[rustfmt::skip]
    assert_eq!(pixels(&frame), [
        250, 250, 250, 250,    10,  11,  12,  13,
        250, 250, 250, 250,    14,  15,  16,  17,
          0,   1,   2,   3,   100, 101, 102, 103,
          4,   5,   6,   7,   104, 105, 106, 107,
    ]);
}

#[test]
fn v2_hival_0x0f_fills_the_block_with_loval() {
    // vqa.txt, VERSION 2 INDEX TABLE LAYOUT: LoVal in the table's first
    // half, HiVal in the second; HiVal 0x0f fills the block with color
    // LoVal, anything else draws entry HiVal*256+LoVal. 0x0f is the marker
    // of 4x2 blocks, hardcoded in Westwood's 4x2 drawer (`cmp bh,00Fh` in
    // UnVQ_4x2, WINVQ/VQA32/UNVQBUFF.ASM of EA's GPL Red Alert source)
    let book = codebook_4x2(
        0x0103,
        &[
            (0, [0, 1, 2, 3, 4, 5, 6, 7]),
            (1, [10, 11, 12, 13, 14, 15, 16, 17]),
            (0x0102, [20, 21, 22, 23, 24, 25, 26, 27]),
        ],
    );
    let table = [
        66, 0x02, 0x01, 0x00, /* HiVal */ 0x0f, 0x01, 0x00, 0x00,
    ];

    let frame = decode_one(
        &header_8bit(),
        &[chunk(b"CBF0", &book), chunk(b"VPT0", &table)],
    )
    .unwrap();
    #[rustfmt::skip]
    assert_eq!(pixels(&frame), [
        66, 66, 66, 66,   20, 21, 22, 23,
        66, 66, 66, 66,   24, 25, 26, 27,
        10, 11, 12, 13,    0,  1,  2,  3,
        14, 15, 16, 17,    4,  5,  6,  7,
    ]);
}

#[test]
fn v2_4x2_blocks_fill_on_hival_0x0f_whatever_maxblocks() {
    // Westwood's 4x2 drawer compares HiVal with 0x0f alone (UNVQBUFF.ASM,
    // see above), whatever the header's maxblocks (CBentries), which sizes
    // only the codebook buffer (LOADER.CPP); so even at maxblocks 0xff00
    // HiVal 0xff is an index, here past the codebook
    let header = VQAHeader {
        maxblocks: 0xff00,
        ..header_8bit()
    };
    let table = [66, 66, 66, 66, /* HiVal */ 0x0f, 0x0f, 0x0f, 0x0f];
    let frame = decode_one(&header, &[chunk(b"VPT0", &table)]).unwrap();
    assert_eq!(pixels(&frame), [66; 32]);

    let table = [66, 66, 66, 66, /* HiVal */ 0xff, 0xff, 0xff, 0xff];
    assert_eq!(
        kind(decode_one(&header, &[chunk(b"VPT0", &table)])),
        Err(ErrorKind::Video(VideoError::BlockIndexOutOfRange))
    );
}

#[test]
fn maxblocks_0_caps_the_codebook_at_0xff00_entries() {
    // locks current behavior: FrameDecoder::new reads maxblocks 0 as 0xff00
    // entries for the codebook cap, so entries past 0x0f00 are allowed and
    // reachable with HiVal 0x10 and up; 0x0f still marks a 4x2 fill
    let header = VQAHeader {
        maxblocks: 0,
        ..header_8bit()
    };
    let book = codebook_4x2(0x1001, &[(0x1000, [1, 2, 3, 4, 5, 6, 7, 8])]);
    let table = [0x00, 66, 66, 66, /* HiVal */ 0x10, 0x0f, 0x0f, 0x0f];
    let chunks = [chunk(b"CBF0", &book), chunk(b"VPT0", &table)];

    let frame = decode_one(&header, &chunks).unwrap();
    #[rustfmt::skip]
    assert_eq!(pixels(&frame), [
         1,  2,  3,  4,   66, 66, 66, 66,
         5,  6,  7,  8,   66, 66, 66, 66,
        66, 66, 66, 66,   66, 66, 66, 66,
        66, 66, 66, 66,   66, 66, 66, 66,
    ]);

    // the same 0x1001-entry codebook is past the cap at maxblocks 0x0f00
    assert_eq!(
        kind(decode_one(&header_8bit(), &chunks)),
        Err(ErrorKind::TooLarge(Limit::Codebook))
    );

    // the cap at maxblocks 0 is 0xff00 entries of 4x2 bytes
    let at_cap = chunk(b"CBF0", &vec![0; 0xff00 * 8]);
    assert!(decode_one(&header, &[at_cap]).is_ok());
    let past_cap = chunk(b"CBF0", &vec![0; 0xff01 * 8]);
    assert_eq!(
        kind(decode_one(&header, &[past_cap])),
        Err(ErrorKind::TooLarge(Limit::Codebook))
    );
}

/// The 8x8 movie with 4x4 blocks, 2x2 = 4 blocks per frame.
fn header_4x4(version: VQAVersion, maxblocks: u16) -> VQAHeader {
    VQAHeader {
        version,
        height: 8,
        block_height: 4,
        maxblocks,
        ..header_8bit()
    }
}

/// A codebook of two 4x4 entries: 0..16 and 20..36.
fn codebook_4x4() -> Vec<u8> {
    (0..16).chain(20..36).collect()
}

/// Entry 1, a fill with color 99, entry 0, and entry 1 over `codebook_4x4`.
#[rustfmt::skip]
const FRAME_4X4: [u8; 64] = [
    20, 21, 22, 23,   99, 99, 99, 99,
    24, 25, 26, 27,   99, 99, 99, 99,
    28, 29, 30, 31,   99, 99, 99, 99,
    32, 33, 34, 35,   99, 99, 99, 99,
     0,  1,  2,  3,   20, 21, 22, 23,
     4,  5,  6,  7,   24, 25, 26, 27,
     8,  9, 10, 11,   28, 29, 30, 31,
    12, 13, 14, 15,   32, 33, 34, 35,
];

#[test]
fn renders_v2_4x4_blocks() {
    // vqa.txt, VERSION 2 INDEX TABLE LAYOUT on an 8x8 frame of 4x4 blocks
    // (the renderer's 4x4 path), whose fill marker is HiVal 0xff: the NOTE
    // guesses so for BlockH=4, and Lands of Lore's 4x4 movies use it
    let header = header_4x4(VQAVersion::Two, 0xff00);
    let table = [1, 99, 0, 1, /* HiVal */ 0, 0xff, 0, 0];

    let chunks = [chunk(b"CBF0", &codebook_4x4()), chunk(b"VPT0", &table)];
    let frame = decode_one(&header, &chunks).unwrap();
    assert_eq!((frame.width, frame.height), (8, 8));
    assert_eq!(pixels(&frame), FRAME_4X4);
}

#[test]
fn v2_4x4_blocks_fill_on_hival_0xff_whatever_maxblocks() {
    // as above; Lands of Lore's 516EFA98.VQA has 4x4 blocks and maxblocks
    // 0x07d0, and fills with 0xff: 0x0f is an index, entry 0x0f63 here,
    // past the two-entry codebook
    let header = header_4x4(VQAVersion::Two, 0x07d0);
    let book = chunk(b"CBF0", &codebook_4x4());

    let table = chunk(b"VPT0", &[1, 99, 0, 1, /* HiVal */ 0, 0xff, 0, 0]);
    let frame = decode_one(&header, &[book.clone(), table]).unwrap();
    assert_eq!(pixels(&frame), FRAME_4X4);

    let table = chunk(b"VPT0", &[1, 99, 0, 1, /* HiVal */ 0, 0x0f, 0, 0]);
    assert_eq!(
        kind(decode_one(&header, &[book, table])),
        Err(ErrorKind::Video(VideoError::BlockIndexOutOfRange))
    );
}

#[test]
fn renders_v1_4x4_blocks() {
    // vqa.txt, VERSION 1 INDEX TABLE LAYOUT on an 8x8 frame of 4x4 blocks:
    // HiVal 0xff fills with 255-LoVal, else entry (HiVal*256+LoVal)/8
    let header = header_4x4(VQAVersion::One, 0x0f00);
    let table = [
        5, 0xff, // fill with 250
        8, 0x00, // entry 1
        0, 0x00, // entry 0
        8, 0x00, // entry 1
    ];

    let chunks = [chunk(b"CBF0", &codebook_4x4()), chunk(b"VPT0", &table)];
    let frame = decode_one(&header, &chunks).unwrap();
    #[rustfmt::skip]
    assert_eq!(pixels(&frame), [
        250, 250, 250, 250,   20, 21, 22, 23,
        250, 250, 250, 250,   24, 25, 26, 27,
        250, 250, 250, 250,   28, 29, 30, 31,
        250, 250, 250, 250,   32, 33, 34, 35,
          0,   1,   2,   3,   20, 21, 22, 23,
          4,   5,   6,   7,   24, 25, 26, 27,
          8,   9,  10,  11,   28, 29, 30, 31,
         12,  13,  14,  15,   32, 33, 34, 35,
    ]);
}

#[test]
fn block_grids_wider_than_tall_are_walked_row_major_in_both_layouts() {
    // vqa.txt, VPT? chunk and both INDEX TABLE LAYOUTs: Width/BlockW blocks
    // across and Height/BlockH down, block (bx, by) at table position
    // by*(Width/BlockW)+bx (doubled for v1's byte pairs; v2's HiVal half
    // starts after (Width/BlockW)*(Height/BlockH) bytes). A 16x4 frame of
    // 4x2 blocks is a 4x2 grid, so swapping the two counts shows
    let header = VQAHeader {
        width: 16,
        ..header_8bit()
    };
    let book: Vec<u8> = (0..64).collect(); // entry k = 8k..8k+8
    // block k draws entry k
    let v2_table = [
        0, 1, 2, 3, 4, 5, 6, 7, /* HiVal */ 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let v1_table = [0, 0, 8, 0, 16, 0, 24, 0, 32, 0, 40, 0, 48, 0, 56, 0];
    #[rustfmt::skip]
    let expected = [
         0,  1,  2,  3,    8,  9, 10, 11,   16, 17, 18, 19,   24, 25, 26, 27,
         4,  5,  6,  7,   12, 13, 14, 15,   20, 21, 22, 23,   28, 29, 30, 31,
        32, 33, 34, 35,   40, 41, 42, 43,   48, 49, 50, 51,   56, 57, 58, 59,
        36, 37, 38, 39,   44, 45, 46, 47,   52, 53, 54, 55,   60, 61, 62, 63,
    ];

    let chunks = [chunk(b"CBF0", &book), chunk(b"VPT0", &v2_table)];
    let frame = decode_one(&header, &chunks).unwrap();
    assert_eq!((frame.width, frame.height), (16, 4));
    assert_eq!(pixels(&frame), expected);

    let v1 = VQAHeader {
        version: VQAVersion::One,
        ..header
    };
    let chunks = [chunk(b"CBF0", &book), chunk(b"VPT0", &v1_table)];
    assert_eq!(pixels(&decode_one(&v1, &chunks).unwrap()), expected);
}

#[test]
fn cbp0_parts_are_appended_and_swapped_in_after_the_last_parts_frame() {
    // vqa.txt, CBP? chunk: the next codebook comes as cbparts parts, one
    // per frame, appended in frame order; once it is complete and the
    // current frame is displayed, it replaces the old codebook
    let header = VQAHeader {
        cbparts: 3,
        ..header_8bit()
    };
    let mut decoder = FrameDecoder::new(&header).unwrap();
    let old: Vec<u8> = (0..32).collect();
    let new: Vec<u8> = (100..132).collect();
    let table = chunk(b"VPT0", &IDENTITY_TABLE);

    // uneven parts that cut through entries, so only byte-wise appending
    // in frame order rebuilds the codebook
    let parts = [&new[..5], &new[5..21], &new[21..]];

    let vqfr = [
        chunk(b"CBF0", &old),
        chunk(b"CBP0", parts[0]),
        table.clone(),
    ]
    .concat();
    assert_eq!(pixels(&decoder.decode_frame(&vqfr).unwrap()), OLD_FRAME);
    let vqfr = [chunk(b"CBP0", parts[1]), table.clone()].concat();
    assert_eq!(pixels(&decoder.decode_frame(&vqfr).unwrap()), OLD_FRAME);
    // the frame carrying the last part still draws with the old codebook
    let vqfr = [chunk(b"CBP0", parts[2]), table.clone()].concat();
    assert_eq!(pixels(&decoder.decode_frame(&vqfr).unwrap()), OLD_FRAME);

    assert_eq!(pixels(&decoder.decode_frame(&table).unwrap()), NEW_FRAME);
    assert_eq!(pixels(&decoder.decode_frame(&table).unwrap()), NEW_FRAME);
}

#[test]
fn cbpz_parts_are_joined_before_decompressing() {
    // vqa.txt, CBP? chunk and its NOTE (which says CBFZ; it means CBPZ, the
    // chunks that come in parts): compressed parts are appended first, then
    // LCW-decompressed as one stream, not one by one. Appendix A: the 0xc5
    // command copies from a position absolute from the start of the output,
    // so in part 2 it reads back into bytes part 1 produced
    let header = VQAHeader {
        cbparts: 2,
        ..header_8bit()
    };
    let mut decoder = FrameDecoder::new(&header).unwrap();
    #[rustfmt::skip]
    let stream = [
        0x88, 100, 101, 102, 103, 104, 105, 106, 107, // entry 0: 8 literals
        0xfe, 0x08, 0x00, 50,                         // entry 1: fill 8 x 50
        0xc5, 0x02, 0x00,                             // entry 2: 5+3 bytes from position 2
        0x88, 120, 121, 122, 123, 124, 125, 126, 127, // entry 3: 8 literals
        0x80,
    ];
    // split between the 0xc5 command and its position word, so neither half
    // decompresses alone: the first ends mid-command, and the second starts
    // with 0x02 0x00, a copy of 3 bytes from 0x200 back in an empty output
    let (first, second) = stream.split_at(14);
    assert_eq!(first[13..], [0xc5]);
    assert_eq!(lcw::decompress(first, 32), Err(LcwError::Truncated));
    assert_eq!(lcw::decompress(second, 32), Err(LcwError::BadOffset));

    let old: Vec<u8> = (0..32).collect();
    let table = chunk(b"VPT0", &IDENTITY_TABLE);
    let vqfr = [chunk(b"CBF0", &old), chunk(b"CBPZ", first), table.clone()].concat();
    assert_eq!(pixels(&decoder.decode_frame(&vqfr).unwrap()), OLD_FRAME);
    let vqfr = [chunk(b"CBPZ", second), table.clone()].concat();
    assert_eq!(pixels(&decoder.decode_frame(&vqfr).unwrap()), OLD_FRAME);

    // entry 2 is bytes 2..10 of the whole codebook: 102..=107, 50, 50
    let frame = decoder.decode_frame(&table).unwrap();
    #[rustfmt::skip]
    assert_eq!(pixels(&frame), [
        100, 101, 102, 103,    50,  50,  50,  50,
        104, 105, 106, 107,    50,  50,  50,  50,
        102, 103, 104, 105,   120, 121, 122, 123,
        106, 107,  50,  50,   124, 125, 126, 127,
    ]);
}

#[test]
fn codebook_parts_cycles_repeat_each_with_its_own_chunk_type() {
    // vqa.txt, CBP? chunk: every cbparts frames bring the parts of the next
    // codebook, so after one swap the next cycle starts from scratch. That a
    // cycle of CBP0 parts may follow one of CBPZ parts locks current
    // behavior (the crate checks the type within a cycle only)
    let header = VQAHeader {
        cbparts: 2,
        ..header_8bit()
    };
    let mut decoder = FrameDecoder::new(&header).unwrap();
    let old: Vec<u8> = (0..32).collect();
    let new: Vec<u8> = (100..132).collect();
    let next: Vec<u8> = (200..232).collect();
    let table = chunk(b"VPT0", &IDENTITY_TABLE);

    // cycle 1, CBPZ: a 16-byte literal run without the end marker, then the
    // rest (lcw_literals ends with 0x80, which would cut the joined stream
    // short if it ended part 1)
    let part1 = [&[0x90][..], &new[..16]].concat();
    let vqfr = [chunk(b"CBF0", &old), chunk(b"CBPZ", &part1), table.clone()].concat();
    assert_eq!(pixels(&decoder.decode_frame(&vqfr).unwrap()), OLD_FRAME);
    let vqfr = [chunk(b"CBPZ", &lcw_literals(&new[16..])), table.clone()].concat();
    assert_eq!(pixels(&decoder.decode_frame(&vqfr).unwrap()), OLD_FRAME);

    // cycle 2, CBP0, drawn with cycle 1's codebook until it completes
    let vqfr = [chunk(b"CBP0", &next[..16]), table.clone()].concat();
    assert_eq!(pixels(&decoder.decode_frame(&vqfr).unwrap()), NEW_FRAME);
    let vqfr = [chunk(b"CBP0", &next[16..]), table.clone()].concat();
    assert_eq!(pixels(&decoder.decode_frame(&vqfr).unwrap()), NEW_FRAME);

    assert_eq!(pixels(&decoder.decode_frame(&table).unwrap()), NEXT_FRAME);
}

#[test]
fn codebook_parts_swap_in_before_the_last_parts_frame_returns() {
    // vqa.txt, CBP? chunk: "Once you get the complete table and display the
    // current frame, replace the old table with the new one" - the swap
    // follows the frame carrying the last part, so a codebook loaded after
    // that frame replaces it in turn. The later codebook comes in a VQFL
    // chunk, which hc-vqa.txt describes for HiColor movies and this crate
    // also accepts in 8-bit ones (locks current behavior)
    let header = VQAHeader {
        cbparts: 2,
        ..header_8bit()
    };
    let mut decoder = FrameDecoder::new(&header).unwrap();
    let old: Vec<u8> = (0..32).collect();
    let new: Vec<u8> = (100..132).collect();
    let next: Vec<u8> = (200..232).collect();
    let table = chunk(b"VPT0", &IDENTITY_TABLE);

    let vqfr = [
        chunk(b"CBF0", &old),
        chunk(b"CBP0", &new[..16]),
        table.clone(),
    ]
    .concat();
    assert_eq!(pixels(&decoder.decode_frame(&vqfr).unwrap()), OLD_FRAME);
    let vqfr = [chunk(b"CBP0", &new[16..]), table.clone()].concat();
    assert_eq!(pixels(&decoder.decode_frame(&vqfr).unwrap()), OLD_FRAME);
    decoder.process_vqfl(&chunk(b"CBF0", &next)).unwrap();
    assert_eq!(pixels(&decoder.decode_frame(&table).unwrap()), NEXT_FRAME);
}

#[test]
fn swap_in_codebook_parts_completes_the_staged_parts_on_demand() {
    // with cbparts 0 the header gives no part count (Lands of Lore's
    // movies schedule their codebooks in a CINF chunk instead): parts
    // stay staged until the caller swaps them in (locks current behavior)
    let mut decoder = FrameDecoder::new(&header_8bit()).unwrap();
    let old: Vec<u8> = (0..32).collect();
    let new: Vec<u8> = (100..132).collect();
    let table = chunk(b"VPT0", &IDENTITY_TABLE);

    // nothing staged: no change
    decoder.swap_in_codebook_parts().unwrap();
    let vqfr = [
        chunk(b"CBF0", &old),
        chunk(b"CBP0", &new[..7]),
        table.clone(),
    ]
    .concat();
    assert_eq!(pixels(&decoder.decode_frame(&vqfr).unwrap()), OLD_FRAME);
    let vqfr = [chunk(b"CBP0", &new[7..]), table.clone()].concat();
    assert_eq!(pixels(&decoder.decode_frame(&vqfr).unwrap()), OLD_FRAME);
    assert_eq!(pixels(&decoder.decode_frame(&table).unwrap()), OLD_FRAME);

    decoder.swap_in_codebook_parts().unwrap();
    assert_eq!(pixels(&decoder.decode_frame(&table).unwrap()), NEW_FRAME);
}

#[test]
fn vptk_and_vptd_tables_decode_like_vptz() {
    // Westwood's VQA loader (WINVQ/VQA32/LOADER.CPP in EA's GPL Red Alert
    // source) loads VPTK (a key frame's table) and VPTD like VPTZ
    let book: Vec<u8> = (0..32).collect();
    for id in [b"VPTK", b"VPTD"] {
        let chunks = [
            chunk(b"CBF0", &book),
            chunk(id, &lcw_literals(&IDENTITY_TABLE)),
        ];
        assert_eq!(
            pixels(&decode_one(&header_8bit(), &chunks).unwrap()),
            OLD_FRAME
        );
    }
}

#[test]
fn a_bad_joined_cbpz_stream_fails_the_frame_carrying_the_last_part() {
    // locks current behavior: the joined parts are decompressed as the frame
    // with the last part finishes (vqa.txt, CBP? chunk), so that frame
    // reports the error. Here part 2 cuts the fill command 0xfe short of its
    // count's high byte and its color
    let header = VQAHeader {
        cbparts: 2,
        ..header_8bit()
    };
    let mut decoder = FrameDecoder::new(&header).unwrap();
    let old: Vec<u8> = (0..32).collect();
    let table = chunk(b"VPT0", &IDENTITY_TABLE);

    let part1 = [0x88, 1, 2, 3, 4, 5, 6, 7, 8]; // 8 literals
    let vqfr = [chunk(b"CBF0", &old), chunk(b"CBPZ", &part1), table.clone()].concat();
    assert_eq!(pixels(&decoder.decode_frame(&vqfr).unwrap()), OLD_FRAME);
    let vqfr = [chunk(b"CBPZ", &[0xfe, 0x08]), table].concat();
    assert_eq!(
        kind(decoder.decode_frame(&vqfr)),
        Err(ErrorKind::Lcw(LcwError::Truncated))
    );
}

#[test]
fn palette_values_keep_only_their_low_six_bits() {
    // vqa.txt, CPL? chunk: R, G, B bytes with bits 6 and 7 masked out (VGA
    // uses bits 0-5). vqa.txt gives no 8-bit scale; the endpoints are first
    // principles: 6-bit 0 is black and 63 VGA full intensity, which is 0xff
    // in 8 bits (reaching it rests on the crate's scaling, locked below; a
    // plain v << 2 would stop at 0xfc)
    #[rustfmt::skip]
    let cpl = [
        0x00, 0x3f, 0x40, // 0, 63, 0
        0x80, 0xc0, 0xff, // 0, 0, 63
        0x7f, 0xbf, 0x3f, // 63, 63, 63
    ];
    let frame = decode_one(&header_8bit(), &[chunk(b"CPL0", &cpl)]).unwrap();
    assert_eq!(
        indexed(&frame).1,
        [[0, 0xff, 0], [0, 0, 0xff], [0xff, 0xff, 0xff]]
    );

    // whatever the scale, a mid value reads the same under any bits 6 and 7
    #[rustfmt::skip]
    let cpl = [
        0x15, 0x55, 0x95, // 0x15 under bits 6-7 of 00, 01, 10...
        0xd5, 0x2a, 0x6a, // ...and 11; 0x2a under 00, 01...
        0xaa, 0xea, 0x2a, // ...10, 11, and 00 again
    ];
    let frame = decode_one(&header_8bit(), &[chunk(b"CPL0", &cpl)]).unwrap();
    let values = indexed(&frame).1.concat();
    let (x15, x2a) = (values[0], values[4]);
    assert_eq!(values, [x15, x15, x15, x15, x2a, x2a, x2a, x2a, x2a]);
    assert_ne!(x15, x2a);
}

#[test]
fn cplz_palettes_are_lcw_compressed() {
    // vqa.txt, VQFR chunk ('Z' means Format80) and CPL? chunk: masked
    // 6-bit values; first principles, as above: 0 black, 63 full intensity
    #[rustfmt::skip]
    let cplz = [
        0xfe, 0x06, 0x00, 0x3f, // fill 6 x 0x3f
        0x83, 0x00, 0x40, 0xff, // 3 literals
        0x80,
    ];
    let frame = decode_one(&header_8bit(), &[chunk(b"CPLZ", &cplz)]).unwrap();
    assert_eq!(indexed(&frame).1, [[0xff; 3], [0xff; 3], [0, 0, 0xff]]);
}

#[test]
fn palette_scales_six_bits_to_eight_by_repeating_the_top_bits() {
    // locks current behavior: v -> (v << 2) | (v >> 4) after masking
    #[rustfmt::skip]
    let cpl = [
        0x01, 0x10, 0x20, // 4, 65, 130
        0x2a, 0x30, 0x3e, // 170, 195, 251
        0x60, 0xa1, 0xd5, // masked 0x20, 0x21, 0x15: 130, 134, 85
    ];
    let frame = decode_one(&header_8bit(), &[chunk(b"CPL0", &cpl)]).unwrap();
    assert_eq!(
        indexed(&frame).1,
        [[0x04, 0x41, 0x82], [0xaa, 0xc3, 0xfb], [0x82, 0x86, 0x55]]
    );
}

#[test]
fn cbfz_codebook_and_vptz_table_in_one_frame() {
    // vqa.txt, CBF? and VPT? chunks with Appendix A (Format80) streams
    // using literal, fill, absolute copy, and end commands
    #[rustfmt::skip]
    let cbfz = [
        0x88, 0, 1, 2, 3, 4, 5, 6, 7,         // entry 0: 8 literals
        0xfe, 0x08, 0x00, 9,                  // entry 1: fill 8 x 9
        0xc5, 0x04, 0x00,                     // entry 2: 5+3 bytes from position 4
        0x88, 20, 21, 22, 23, 24, 25, 26, 27, // entry 3: 8 literals
        0x80,
    ];
    #[rustfmt::skip]
    let vptz = [
        0x84, 3, 2, 1, 85,      // LoVal: entries 3, 2, 1, fill color 85
        0xfe, 0x03, 0x00, 0x00, // HiVal: 0, 0, 0...
        0x81, 0x0f,             // ...and the fill marker
        0x80,
    ];

    let frame = decode_one(
        &header_8bit(),
        &[chunk(b"CBFZ", &cbfz), chunk(b"VPTZ", &vptz)],
    )
    .unwrap();
    #[rustfmt::skip]
    assert_eq!(pixels(&frame), [
        20, 21, 22, 23,    4,  5,  6,  7,
        24, 25, 26, 27,    9,  9,  9,  9,
         9,  9,  9,  9,   85, 85, 85, 85,
         9,  9,  9,  9,   85, 85, 85, 85,
    ]);
}

#[test]
fn vqfl_codebook_and_palette_apply_to_the_frames_that_follow() {
    // locks current behavior: hc-vqa.txt describes VQFL chunks carrying the
    // codebooks for the VQFR chunks after them in HiColor movies; 8-bit
    // movies get the same treatment, palettes included
    let mut decoder = FrameDecoder::new(&header_8bit()).unwrap();
    let old: Vec<u8> = (0..32).collect();
    let new: Vec<u8> = (100..132).collect();
    let table = chunk(b"VPT0", &IDENTITY_TABLE);

    let vqfr = [
        chunk(b"CBF0", &old),
        chunk(b"CPL0", &[0x3f, 0, 0]),
        table.clone(),
    ]
    .concat();
    let frame = decoder.decode_frame(&vqfr).unwrap();
    assert_eq!(indexed(&frame), (&OLD_FRAME[..], &[[0xff, 0, 0]][..]));

    let vqfl = [
        chunk(b"CBFZ", &lcw_literals(&new)),
        chunk(b"CPLZ", &lcw_literals(&[0, 0x3f, 0])),
    ]
    .concat();
    decoder.process_vqfl(&vqfl).unwrap();
    let frame = decoder.decode_frame(&table).unwrap();
    assert_eq!(indexed(&frame), (&NEW_FRAME[..], &[[0, 0xff, 0]][..]));
}

#[test]
fn fills_and_entries_draw_alike_as_codebooks_grow_and_shrink() {
    // the decoder keeps a solid-color entry per color next to the
    // codebook; each new codebook, larger or smaller, must leave both
    // where the table finds them
    let mut decoder = FrameDecoder::new(&header_8bit()).unwrap();
    // blocks: fill with color 9, entry 1, entry 0, fill with color 255
    let table = chunk(b"VPT0", &[9, 1, 0, 255, /* HiVal */ 0x0f, 0, 0, 0x0f]);
    for (n, count) in (1..).zip([2, 300, 2, 0x0f00, 2]) {
        let m = n + 100;
        let book = codebook_4x2(count, &[(0, [n; 8]), (1, [m; 8])]);
        let frame = decoder
            .decode_frame(&[chunk(b"CBF0", &book), table.clone()].concat())
            .unwrap();
        #[rustfmt::skip]
        assert_eq!(pixels(&frame), [
            9, 9, 9, 9,   m,   m,   m,   m,
            9, 9, 9, 9,   m,   m,   m,   m,
            n, n, n, n,   255, 255, 255, 255,
            n, n, n, n,   255, 255, 255, 255,
        ], "codebook {n}, of {count} entries");
    }

    // an empty codebook still leaves the fills
    let fills = chunk(b"VPT0", &[1, 2, 3, 4, /* HiVal */ 0x0f, 0x0f, 0x0f, 0x0f]);
    let frame = decoder
        .decode_frame(&[chunk(b"CBF0", &[]), fills].concat())
        .unwrap();
    #[rustfmt::skip]
    assert_eq!(pixels(&frame), [
        1, 1, 1, 1,   2, 2, 2, 2,
        1, 1, 1, 1,   2, 2, 2, 2,
        3, 3, 3, 3,   4, 4, 4, 4,
        3, 3, 3, 3,   4, 4, 4, 4,
    ]);
}

#[test]
fn codebook_swaps_cost_only_the_codebook_whatever_the_block_size() {
    // a swap used to rebuild the 256 solid-color entries after the new
    // codebook: 16.6 MB with 255x255 blocks, so these 16 KB of empty
    // codebooks took seconds in a release build, and this test over two
    // minutes in a debug one
    let header = VQAHeader {
        width: 255,
        height: 255,
        block_width: 255,
        block_height: 255,
        ..header_8bit()
    };
    let mut decoder = FrameDecoder::new(&header).unwrap();
    decoder
        .process_vqfl(&chunk(b"CBF0", &[]).repeat(2000))
        .unwrap();

    // the one block, filled with color 7
    let frame = decoder.decode_frame(&chunk(b"VPT0", &[7, 0x0f])).unwrap();
    assert!(pixels(&frame).iter().all(|&p| p == 7));
}

/// A 4x2 movie of a single 4x2 block.
fn rgb_header() -> VQAHeader {
    VQAHeader {
        width: 4,
        height: 2,
        ..header_8bit()
    }
}

/// A frame for `rgb_header` whose block holds `indices` over a red, green,
/// blue palette.
fn rgb_vqfr(indices: [u8; 8]) -> Vec<u8> {
    [
        chunk(b"CBF0", &indices),
        chunk(b"CPL0", &[0x3f, 0, 0, 0, 0x3f, 0, 0, 0, 0x3f]),
        chunk(b"VPT0", &[0, /* HiVal */ 0]),
    ]
    .concat()
}

/// Indices 0 1 2 0 / 2 1 0 1 over red, green, blue.
#[rustfmt::skip]
const RGB: [u8; 24] = [
    0xff, 0, 0,   0, 0xff, 0,   0, 0, 0xff,   0xff, 0, 0,
    0, 0, 0xff,   0, 0xff, 0,   0xff, 0, 0,   0, 0xff, 0,
];

#[test]
fn indexed_frames_convert_to_palette_colors_row_major() {
    // first principles: an indexed pixel's color is palette[index], packed
    // R, G, B, pixel by pixel in row-major order
    let mut decoder = FrameDecoder::new(&rgb_header()).unwrap();
    let frame = decoder
        .decode_frame(&rgb_vqfr([0, 1, 2, 0, 2, 1, 0, 1]))
        .unwrap();
    assert_eq!(frame.to_rgb888(), RGB);

    let mut out = [0xaa; 24];
    frame.write_rgb888(&mut out);
    assert_eq!(out, RGB);

    // the 4-byte formats hold the same colors: R, G, B, opaque alpha
    // bytes, and 0x00RRGGBB words
    let (rgba, xrgb) = four_byte(&RGB);
    assert_eq!(frame.to_rgba8888(), rgba);
    assert_eq!(frame.to_xrgb8888(), xrgb);
    let mut out = [0xaa; 32];
    frame.write_rgba8888(&mut out);
    assert_eq!(out[..], rgba);
    let mut out = [0xaaaa_aaaa; 8];
    frame.write_xrgb8888(&mut out);
    assert_eq!(out[..], xrgb);
}

/// RGB888 bytes as RGBA8888 bytes (alpha opaque) and XRGB8888 words.
fn four_byte(rgb: &[u8]) -> (Vec<u8>, Vec<u32>) {
    let rgb = rgb.as_chunks::<3>().0;
    let rgba = rgb.iter().flat_map(|&[r, g, b]| [r, g, b, 0xff]);
    let xrgb = rgb
        .iter()
        .map(|&[r, g, b]| u32::from_be_bytes([0, r, g, b]));
    (rgba.collect(), xrgb.collect())
}

#[test]
fn borrowed_frames_convert_to_palette_colors_row_major() {
    // first principles, as above, on the frame decode_frame_ref lends out
    let mut decoder = FrameDecoder::new(&rgb_header()).unwrap();
    let frame = decoder
        .decode_frame_ref(&rgb_vqfr([0, 1, 2, 0, 2, 1, 0, 1]))
        .unwrap();
    let FramePixelsRef::Indexed { pixels, palette } = frame.pixels else {
        panic!("expected an indexed frame");
    };
    assert_eq!(pixels, [0, 1, 2, 0, 2, 1, 0, 1]);
    assert_eq!(palette, [[0xff, 0, 0], [0, 0xff, 0], [0, 0, 0xff]]);
    assert_eq!(frame.to_rgb888(), RGB);

    let mut out = [0xaa; 24];
    frame.write_rgb888(&mut out);
    assert_eq!(out, RGB);

    let (rgba, xrgb) = four_byte(&RGB);
    assert_eq!(frame.to_rgba8888(), rgba);
    let mut out = [0xaaaa_aaaa; 8];
    frame.write_xrgb8888(&mut out);
    assert_eq!(out[..], xrgb);
}

#[test]
fn indices_past_the_palette_come_out_black() {
    // locks current behavior (documented on Frame::to_rgb888)
    let vqfr = [
        chunk(b"CBF0", &[0, 1, 2, 3, 255, 1, 0, 2]),
        chunk(b"CPL0", &[0x3f, 0x3f, 0x3f, 0x3f, 0, 0]), // white, red
        chunk(b"VPT0", &[0, 0]),
    ]
    .concat();
    #[rustfmt::skip]
    let rgb = [
        0xff, 0xff, 0xff,   0xff, 0, 0,   0, 0, 0,        0, 0, 0,
        0, 0, 0,            0xff, 0, 0,   0xff, 0xff, 0xff,   0, 0, 0,
    ];
    let mut decoder = FrameDecoder::new(&rgb_header()).unwrap();
    let frame = decoder.decode_frame(&vqfr).unwrap();
    assert_eq!(frame.to_rgb888(), rgb);
    let mut out = [0xaa; 24];
    frame.write_rgb888(&mut out);
    assert_eq!(out, rgb);
    // opaque black in the 4-byte formats
    let (rgba, xrgb) = four_byte(&rgb);
    assert_eq!(frame.to_rgba8888(), rgba);
    assert_eq!(frame.to_xrgb8888(), xrgb);

    // with no palette at all, every pixel is black
    let vqfr = [
        chunk(b"CBF0", &[0, 1, 2, 3, 4, 5, 6, 7]),
        chunk(b"VPT0", &[0, 0]),
    ];
    let frame = decode_one(&rgb_header(), &vqfr).unwrap();
    let mut out = [0xaa; 24];
    frame.write_rgb888(&mut out);
    assert_eq!(out, [0; 24]);
}

#[test]
#[should_panic(expected = "three bytes per pixel")]
fn write_rgb888_panics_on_a_short_buffer() {
    // locks current behavior (documented on Frame::write_rgb888)
    let frame = decode_one(&rgb_header(), &[rgb_vqfr([0; 8])]).unwrap();
    frame.write_rgb888(&mut [0; 23]);
}

#[test]
#[should_panic(expected = "three bytes per pixel")]
fn borrowed_write_rgb888_panics_on_a_long_buffer() {
    // locks current behavior (documented on FrameRef::write_rgb888)
    let mut decoder = FrameDecoder::new(&rgb_header()).unwrap();
    let frame = decoder.decode_frame_ref(&rgb_vqfr([0; 8])).unwrap();
    frame.write_rgb888(&mut [0; 25]);
}

#[test]
#[should_panic(expected = "four bytes per pixel")]
fn write_rgba8888_panics_on_a_short_buffer() {
    // locks current behavior (documented on Frame::write_rgba8888)
    let frame = decode_one(&rgb_header(), &[rgb_vqfr([0; 8])]).unwrap();
    frame.write_rgba8888(&mut [0; 31]);
}

#[test]
#[should_panic(expected = "one word per pixel")]
fn borrowed_write_xrgb8888_panics_on_a_long_buffer() {
    // locks current behavior (documented on FrameRef::write_xrgb8888)
    let mut decoder = FrameDecoder::new(&rgb_header()).unwrap();
    let frame = decoder.decode_frame_ref(&rgb_vqfr([0; 8])).unwrap();
    frame.write_xrgb8888(&mut [0; 9]);
}

#[test]
fn new_rejects_a_zero_block_size() {
    // locks current behavior
    for (block_width, block_height) in [(0, 2), (4, 0), (0, 0)] {
        let header = VQAHeader {
            block_width,
            block_height,
            ..header_8bit()
        };
        assert_eq!(
            FrameDecoder::new(&header).err().map(|e| e.kind()),
            Some(ErrorKind::InvalidHeader)
        );
    }
}

#[test]
fn new_caps_frames_at_1_shl_24_pixels() {
    // locks current behavior
    let header = |width, height| VQAHeader {
        width,
        height,
        ..header_8bit()
    };
    assert!(FrameDecoder::new(&header(4096, 4096)).is_ok());
    for (width, height) in [(4097, 4096), (4096, 4097), (0xffff, 0xffff)] {
        assert_eq!(
            FrameDecoder::new(&header(width, height))
                .err()
                .map(|e| e.kind()),
            Some(ErrorKind::TooLarge(Limit::FrameSize))
        );
    }
}

#[test]
fn rejects_pointer_tables_of_the_wrong_size() {
    // locks current behavior: the 8x4 movie's table is (8/4)*(4/2)*2 = 8
    // bytes (vqa.txt, VPT? chunk)
    let book: Vec<u8> = (0..32).collect();
    let mismatch = Err(ErrorKind::Video(VideoError::PointerTableSize));
    for table in [&[0u8; 6][..], &[0; 7], &[0; 9], &[0; 10]] {
        let chunks = [chunk(b"CBF0", &book), chunk(b"VPT0", table)];
        assert_eq!(kind(decode_one(&header_8bit(), &chunks)), mismatch);
    }

    // a VPTZ, VPTK or VPTD decompressing short or long fails the same way,
    // although one decompressing long stops at the LCW output cap
    for id in [b"VPTZ", b"VPTK", b"VPTD"] {
        for len in [6, 10] {
            let chunks = [
                chunk(b"CBF0", &book),
                chunk(id, &lcw_literals(&[0; 32][..len])),
            ];
            assert_eq!(kind(decode_one(&header_8bit(), &chunks)), mismatch);
        }
    }
}

#[test]
fn rejects_block_indices_past_the_codebook() {
    // locks current behavior
    let outside = Err(ErrorKind::Video(VideoError::BlockIndexOutOfRange));
    let book: Vec<u8> = (0..16).collect(); // entries 0 and 1

    // v2: entry 2, and entry 0x0100 (HiVal 1)
    let v2 = header_8bit();
    let table = [0, 1, 2, 0, /* HiVal */ 0, 0, 0, 0];
    assert_eq!(
        kind(decode_one(
            &v2,
            &[chunk(b"CBF0", &book), chunk(b"VPT0", &table)]
        )),
        outside
    );
    let table = [0, 1, 0, 0, /* HiVal */ 0, 0, 0, 1];
    assert_eq!(
        kind(decode_one(
            &v2,
            &[chunk(b"CBF0", &book), chunk(b"VPT0", &table)]
        )),
        outside
    );
    // no codebook at all
    assert_eq!(kind(decode_one(&v2, &[chunk(b"VPT0", &[0; 8])])), outside);

    // v1: block 0 is entry 0x0100/8 = 32; read as a v2 table the same bytes
    // would be entries 0, 1, 0, 0, all inside the codebook
    let v1 = VQAHeader {
        version: VQAVersion::One,
        ..header_8bit()
    };
    let table = [0, 1, 0, 0, 0, 0, 0, 0];
    assert_eq!(
        kind(decode_one(
            &v1,
            &[chunk(b"CBF0", &book), chunk(b"VPT0", &table)]
        )),
        outside
    );
}

#[test]
fn rejects_hicolor_pointer_streams_in_8bit_movies() {
    // locks current behavior
    let book: Vec<u8> = (0..8).collect();
    let stream = 0b011_0000000000000u16.to_le_bytes(); // write block 0
    let expected = Err(ErrorKind::Video(VideoError::WrongPointerFormat));
    assert_eq!(
        kind(decode_one(
            &header_8bit(),
            &[chunk(b"CBF0", &book), chunk(b"VPTR", &stream)]
        )),
        expected
    );
    assert_eq!(
        kind(decode_one(
            &header_8bit(),
            &[
                chunk(b"CBF0", &book),
                chunk(b"VPRZ", &lcw_literals(&stream))
            ]
        )),
        expected
    );
}

#[test]
fn rejects_mixed_cbp0_and_cbpz_parts() {
    // locks current behavior
    let header = VQAHeader {
        cbparts: 2,
        ..header_8bit()
    };
    let mixed = Err(ErrorKind::Video(VideoError::MixedCodebookParts));
    let book: Vec<u8> = (0..32).collect();
    let table = chunk(b"VPT0", &IDENTITY_TABLE);

    // CBP0 in one frame, CBPZ in the next
    let mut decoder = FrameDecoder::new(&header).unwrap();
    let vqfr = [
        chunk(b"CBF0", &book),
        chunk(b"CBP0", &book[..16]),
        table.clone(),
    ]
    .concat();
    decoder.decode_frame(&vqfr).unwrap();
    let vqfr = [chunk(b"CBPZ", &lcw_literals(&book[16..])), table.clone()].concat();
    assert_eq!(kind(decoder.decode_frame(&vqfr)), mixed);

    // CBPZ then CBP0
    let mut decoder = FrameDecoder::new(&header).unwrap();
    let vqfr = [
        chunk(b"CBF0", &book),
        chunk(b"CBPZ", &lcw_literals(&book[..16])),
        table.clone(),
    ]
    .concat();
    decoder.decode_frame(&vqfr).unwrap();
    let vqfr = [chunk(b"CBP0", &book[16..]), table].concat();
    assert_eq!(kind(decoder.decode_frame(&vqfr)), mixed);
}

#[test]
fn rejects_palettes_of_bad_sizes() {
    // locks current behavior: whole R, G, B triples, at most 256 colors
    let size = Err(ErrorKind::Video(VideoError::PaletteSize));
    let v2 = header_8bit();
    assert_eq!(kind(decode_one(&v2, &[chunk(b"CPL0", &[0; 4])])), size);
    assert_eq!(
        kind(decode_one(&v2, &[chunk(b"CPL0", &[0; 257 * 3])])),
        size
    );
    assert_eq!(
        kind(decode_one(&v2, &[chunk(b"CPLZ", &lcw_literals(&[0; 5]))])),
        size
    );

    // exactly 256 colors is fine
    let frame = decode_one(&v2, &[chunk(b"CPL0", &[0x3f; 256 * 3])]).unwrap();
    assert_eq!(indexed(&frame).1, [[0xff; 3]; 256]);

    // a CPLZ expanding past 256 colors fails the same way, although it
    // stops at the LCW output cap
    let cplz = [0xfe, 0x03, 0x03, 0x3f, 0x80]; // fill 0x303 = 257*3 bytes
    assert_eq!(kind(decode_one(&v2, &[chunk(b"CPLZ", &cplz)])), size);
}

#[test]
fn rejects_codebooks_larger_than_maxblocks_entries() {
    // locks current behavior: maxblocks 2 of 4x2 blocks caps a codebook at
    // 16 bytes
    let header = VQAHeader {
        maxblocks: 2,
        ..header_8bit()
    };
    let book: Vec<u8> = (0..24).collect(); // 3 entries

    // 16 bytes is fine: block 3 draws entry 1
    let table = [0, 0, 0, 1, /* HiVal */ 0, 0, 0, 0];
    let frame = decode_one(
        &header,
        &[chunk(b"CBF0", &book[..16]), chunk(b"VPT0", &table)],
    )
    .unwrap();
    assert_eq!(pixels(&frame)[2 * 8 + 4..][..4], [8, 9, 10, 11]);

    assert_eq!(
        kind(decode_one(&header, &[chunk(b"CBF0", &book)])),
        Err(ErrorKind::TooLarge(Limit::Codebook))
    );
    // compressed, it fails the same way, although it stops at the LCW
    // output cap
    assert_eq!(
        kind(decode_one(&header, &[chunk(b"CBFZ", &lcw_literals(&book))])),
        Err(ErrorKind::TooLarge(Limit::Codebook))
    );

    // parts adding up past the cap fail as the overflowing one arrives
    let header = VQAHeader {
        cbparts: 2,
        ..header
    };
    let mut decoder = FrameDecoder::new(&header).unwrap();
    decoder.decode_frame(&chunk(b"CBP0", &book[..16])).unwrap();
    assert_eq!(
        kind(decoder.decode_frame(&chunk(b"CBP0", &book[16..]))),
        Err(ErrorKind::TooLarge(Limit::Codebook))
    );

    // compressed parts that fit the cap but expand past it fail when the
    // codebook they complete is swapped in: here a 5-byte LCW fill of 24
    // bytes, in two parts
    let fill = [0xfe, 24, 0, 7, 0x80];
    let mut decoder = FrameDecoder::new(&header).unwrap();
    decoder.decode_frame(&chunk(b"CBPZ", &fill[..2])).unwrap();
    assert_eq!(
        kind(decoder.decode_frame(&chunk(b"CBPZ", &fill[2..]))),
        Err(ErrorKind::TooLarge(Limit::Codebook))
    );
    let mut decoder = FrameDecoder::new(&header).unwrap();
    decoder.process_vqfl(&chunk(b"CBPZ", &fill)).unwrap();
    assert_eq!(
        kind(decoder.swap_in_codebook_parts()),
        Err(ErrorKind::TooLarge(Limit::Codebook))
    );
}
