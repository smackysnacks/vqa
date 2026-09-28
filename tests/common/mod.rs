//! Builders shared by the integration tests: chunks, LCW streams, headers,
//! and whole movies, all small enough to check by hand.

// each test crate uses only some of these
#![allow(dead_code)]

use vqa::{Error, ErrorKind, VQAHeader, VQAVersion};

/// Wrap `data` in a chunk: the ID, the big-endian size, the payload, and a
/// pad byte after an odd-sized payload.
pub fn chunk(id: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.extend((data.len() as u32).to_be_bytes());
    out.extend(data);
    if data.len() % 2 == 1 {
        out.push(0);
    }
    out
}

/// Encode `data` as an absolute-mode LCW stream of literal runs (`0x80 | n`
/// then `n` bytes, up to 63 at a time), ending with the `0x80` end marker.
pub fn lcw_literals(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for run in data.chunks(63) {
        out.push(0x80 | run.len() as u8);
        out.extend(run);
    }
    out.push(0x80);
    out
}

/// An 8x4 8-bit v2 movie with 4x2 blocks (2x2 = 4 blocks per frame), a
/// 256-color palette, and mono 16-bit sound.
pub fn header_8bit() -> VQAHeader {
    VQAHeader {
        version: VQAVersion::Two,
        flags: 0,
        num_frames: 1,
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

/// The same 8x4 movie with 4x2 blocks, as v3 HiColor (`colors` = 0) with
/// stereo sound.
pub fn header_hicolor() -> VQAHeader {
    VQAHeader {
        version: VQAVersion::Three,
        colors: 0,
        channels: 2,
        ..header_8bit()
    }
}

/// Serialize `header` as the 42-byte VQHD payload.
pub fn vqhd(header: &VQAHeader) -> Vec<u8> {
    let version: u16 = match header.version {
        VQAVersion::One => 1,
        VQAVersion::Two => 2,
        VQAVersion::Three => 3,
    };
    let mut out = Vec::new();
    out.extend(version.to_le_bytes());
    out.extend(header.flags.to_le_bytes());
    out.extend(header.num_frames.to_le_bytes());
    out.extend(header.width.to_le_bytes());
    out.extend(header.height.to_le_bytes());
    out.extend([
        header.block_width,
        header.block_height,
        header.frame_rate,
        header.cbparts,
    ]);
    out.extend(header.colors.to_le_bytes());
    out.extend(header.maxblocks.to_le_bytes());
    out.extend(header.unk1.to_le_bytes());
    out.extend(header.unk2.to_le_bytes());
    out.extend(header.freq.to_le_bytes());
    out.extend([header.channels, header.bits]);
    out.extend(header.unk3.to_le_bytes());
    out.extend(header.unk4.to_le_bytes());
    out.extend(header.max_cbfz_size.to_le_bytes());
    out.extend(header.unk5.to_le_bytes());
    assert_eq!(out.len(), 42);
    out
}

/// A whole movie file: the FORM chunk, the WVQA signature, the VQHD chunk
/// for `header`, then `chunks` (already wrapped, e.g. by [`chunk`]) in order.
pub fn movie(header: &VQAHeader, chunks: &[Vec<u8>]) -> Vec<u8> {
    let mut body = b"WVQA".to_vec();
    body.extend(chunk(b"VQHD", &vqhd(header)));
    for chunk in chunks {
        body.extend(chunk);
    }
    let mut file = b"FORM".to_vec();
    file.extend((body.len() as u32).to_be_bytes());
    file.extend(body);
    file
}

/// A result with its error's kind in place of the error, so tests compare
/// what went wrong rather than where.
pub fn kind<T>(result: Result<T, Error>) -> Result<T, ErrorKind> {
    result.map_err(|e| e.kind())
}

/// FNV-1a 64 offset basis.
pub const FNV_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// Fold `bytes` into an FNV-1a 64 hash.
pub fn fnv1a(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}
