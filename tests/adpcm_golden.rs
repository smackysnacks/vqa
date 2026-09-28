//! Golden test: decode all SND2 audio in the bundled wwlogo.vqa and verify
//! the output against a known checksum, locking in the ADPCM decoder's
//! behavior across refactors.

use vqa::VQA;
use vqa::audio::{CodecState, decompress};

fn fnv1a(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}

#[test]
fn decodes_wwlogo_audio_to_known_checksum() {
    let buffer = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/wwlogo.vqa"))
        .expect("failed to read wwlogo.vqa");
    let vqa = VQA::parse(&buffer).expect("failed to parse the container");

    let chunks: Vec<_> = vqa
        .chunks()
        .map(|chunk| chunk.expect("failed to parse a chunk"))
        .filter(|chunk| &chunk.id == b"SND2")
        .collect();
    assert_eq!(chunks.len(), 130);

    let mut left_state = CodecState::new();
    let mut right_state = CodecState::new();
    let mut num_samples = 0;
    let mut hash = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
    for chunk in &chunks {
        let half = chunk.data.len() / 2;
        let left = decompress(&mut left_state, &chunk.data[..half]);
        let right = decompress(&mut right_state, &chunk.data[half..]);

        num_samples += left.len() + right.len();
        for sample in left.iter().chain(right.iter()) {
            hash = fnv1a(hash, &sample.to_le_bytes());
        }
    }

    assert_eq!(num_samples, 382_200);
    assert_eq!(hash, 0x44e9_d409_c096_7436);
}
