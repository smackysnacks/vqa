//! Golden tests over movies from FFmpeg's sample archive, listed in
//! tests/samples.txt: every video frame and every audio sample of each,
//! hashed. The movies aren't in the repository; download them with
//! `just samples`, then run `just test-samples`.
//!
//! The hashes were pinned after checking the output against other
//! decoders:
//!
//! - Every 8-bit movie's palette indices and palettes match FFmpeg 7.0.2's
//!   frame for frame, except where FFmpeg can't decode the movie: Lands of
//!   Lore's 516EFA98.VQA, whose codebooks come in groups scheduled by its
//!   CINF chunk, and Kyrandia 3's benchl.vqa, which has no VQFR chunks
//!   (Westwood's loader reads that layout too).
//! - Every HiColor movie's pixels match FFmpeg's, except in two known
//!   FFmpeg bugs. It ignores the alpha skip of Blade Runner's
//!   76B70801.VQA, where the crate matches ScummVM's Blade Runner engine.
//!   And from frame 76 of Dune 2000's t_titl_e.vqa, an off-by-one in its
//!   relative LCW decoder drops a frame, where the crate matches OpenRA.
//! - Raw PCM (SND0) and Westwood ADPCM (SND1) audio match FFmpeg's sample
//!   for sample. IMA ADPCM (SND2) follows Westwood's own decoder (the
//!   shift-and-add deltas of SOSCODEC.ASM in EA's GPL Red Alert source),
//!   which FFmpeg rounds differently.
//!
//! The truncated samples (FFmpeg's FATE cuts, and cc-demo1.vqa, one byte
//! short) end in an error after their last whole chunk.

mod common;

use std::fmt::Write;
use std::path::Path;

use common::{FNV_BASIS, fnv1a};
use vqa::{FramePixelsRef, VQA};

/// What decoding one sample movie gives.
#[derive(Debug, PartialEq, Eq)]
struct Decoded {
    /// frames decoded before the end or the first error
    frames: usize,
    /// FNV-1a over them: palette indices then RGB palette bytes for 8-bit
    /// frames, little-endian pixels for HiColor ones
    video: u64,
    /// the error that ended the frames, if any
    video_error: Option<String>,
    /// samples decoded by `audio_chunks` before the end or the first error
    samples: usize,
    /// FNV-1a over their little-endian bytes
    audio: u64,
    /// the error that ended the audio, if any
    audio_error: Option<String>,
}

fn decode(data: &[u8]) -> Decoded {
    let vqa = VQA::parse(data).expect("sample should parse");

    let mut frames = vqa.frames().expect("sample's header should be valid");
    let (mut count, mut video, mut video_error) = (0, FNV_BASIS, None);
    while let Some(frame) = frames.next_ref() {
        match frame {
            Ok(frame) => match frame.pixels {
                FramePixelsRef::Indexed { pixels, palette } => {
                    video = fnv1a(video, pixels);
                    video = fnv1a(video, palette.as_flattened());
                }
                FramePixelsRef::HiColor { pixels } => {
                    for pixel in pixels {
                        video = fnv1a(video, &pixel.to_le_bytes());
                    }
                }
            },
            Err(e) => video_error = Some(format!("{e:?}")),
        }
        count += 1;
    }
    // the frame that failed isn't a decoded one
    let frames = count - usize::from(video_error.is_some());

    let mut samples = Vec::new();
    let mut chunks = vqa.audio_chunks();
    let mut audio_error = None;
    while let Some(result) = chunks.next_into(&mut samples) {
        if let Err(e) = result {
            audio_error = Some(format!("{e:?}"));
        }
    }
    let audio = samples
        .iter()
        .fold(FNV_BASIS, |hash, sample| fnv1a(hash, &sample.to_le_bytes()));

    Decoded {
        frames,
        video,
        video_error,
        samples: samples.len(),
        audio,
        audio_error,
    }
}

/// `(file, frames, video hash, video error, samples, audio hash, audio
/// error)`
type Golden = (
    &'static str,
    usize,
    u64,
    Option<&'static str>,
    usize,
    u64,
    Option<&'static str>,
);

/// The decoded output of every movie in tests/samples.txt.
#[rustfmt::skip]
const GOLDEN: &[Golden] = &[
    ("fate/ws_snd.vqa", 10, 0xbaebfbb984fa5121, Some("Parse"), 40639, 0x24ac094ac484a97b, Some("Parse")),
    ("fate/small-cut-v3.vqa", 11, 0x2900d8658e24f3db, Some("Parse"), 17640, 0xe9c1600e5447816f, Some("Parse")),
    ("fate/cc-demo1-partial.vqa", 38, 0x0f0a5d84bf5d2375, Some("Parse"), 68354, 0x9fa3569e19bf7b37, Some("Parse")),
    ("td/cc-demo1.vqa", 417, 0x4bb73ad2d4627d52, Some("Parse"), 625484, 0x93a6601fd612ed00, Some("Parse")),
    ("td/NOD1PRE.VQA", 32, 0x52fd085d87e77402, None, 58064, 0x794ccddc9d9d9467, None),
    ("td/DINO.VQA", 116, 0x83cb6ad907b824ae, None, 181544, 0x3e2f978aa85abeab, None),
    ("td/SPYCRASH.VQA", 255, 0xc37832321f208dbf, None, 385874, 0xaa074c21c0a1d29e, None),
    ("ra/allymorf.vqa", 85, 0x2a4accd5920a454e, None, 135974, 0x61c5ab5c9e50def3, None),
    ("ra/nukestok.vqa", 160, 0xca92b18ef6c8e0aa, None, 246224, 0x7ddcdf4ca52cd345, None),
    ("ra/redintro.vqa", 188, 0x980cc702c970436f, None, 286718, 0xff4731c34ca486c2, None),
    ("k3/bench0.vqa", 6, 0x3d1d0e3e7f8a2cc1, None, 0, 0xcbf29ce484222325, None),
    ("k3/bench1.vqa", 6, 0xd31080999d35420a, None, 0, 0xcbf29ce484222325, None),
    ("k3/bench2.vqa", 6, 0x280d7d9e8f815998, None, 0, 0xcbf29ce484222325, None),
    ("k3/benchl.vqa", 96, 0xec5ce75d2af36fd1, None, 0, 0xcbf29ce484222325, None),
    ("k3/boat2.vqa", 176, 0x7f392ef8d18b6cc6, None, 404464, 0x43ee19970a4959a2, None),
    ("lol/4D6EFA9C.VQA", 75, 0xf6b38db63ce00b72, None, 0, 0xcbf29ce484222325, None),
    ("lol/596EFA96.VQA", 75, 0xb08838b9406cb4ce, None, 0, 0xcbf29ce484222325, None),
    ("lol/5D6EFA8C.VQA", 75, 0xed4b6c0b4184cb61, None, 0, 0xcbf29ce484222325, None),
    ("lol/516EFA98.VQA", 450, 0x115ef740ace57f59, None, 0, 0xcbf29ce484222325, None),
    ("d2k/t_titl_e.vqa", 96, 0xa4c98866a6aa8e19, None, 141120, 0xd2574cbf238fd976, None),
    ("ts/small.vqa", 96, 0x36abe7a6c8a3fa97, None, 141120, 0xb882bb57a3b00e3a, None),
    ("ts/gdilogo.vqa", 119, 0xd4a5e2c5e843c04e, None, 349860, 0x5bb2139720d268cc, None),
    ("ts/orcas.vqa", 150, 0xb6a82fd4f86f97b3, None, 441000, 0xba0a065a68f3a52c, None),
    ("br/76B70801.VQA", 60, 0xb258803424e0fb0e, None, 0, 0xcbf29ce484222325, None),
    ("br/A5B1D8C8.VQA", 61, 0x6d7657b156cba83e, None, 0, 0xcbf29ce484222325, None),
    ("br/A5B1DCD2.VQA", 61, 0x774fd0028d3c2b67, None, 0, 0xcbf29ce484222325, None),
    ("br/A1B3FCCE.VQA", 61, 0x3dfa82728e3d08de, None, 0, 0xcbf29ce484222325, None),
    ("custom/VOYAGER.VQA", 61, 0x3fddc2efe502ef81, None, 0, 0xcbf29ce484222325, None),
    ("custom/SATURN.VQA", 447, 0xdaeb38de46e58a7a, None, 0, 0xcbf29ce484222325, None),
];

/// The files tests/samples.txt lists, relative to samples/.
fn manifest() -> Vec<String> {
    let manifest = include_str!("samples.txt");
    manifest
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| line.split_whitespace().nth(1))
        .map(String::from)
        .collect()
}

#[test]
#[ignore = "needs the sample movies: run `just samples`, then `just test-samples`"]
fn sample_movies_decode_to_known_checksums() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("samples");
    let mut actual = String::new();
    let mut mismatches = Vec::new();
    for file in manifest() {
        let data = std::fs::read(dir.join(&file))
            .unwrap_or_else(|e| panic!("{file}: {e} (run `just samples` first)"));
        let decoded = decode(&data);

        let golden = GOLDEN.iter().find(|golden| golden.0 == file).map(
            |&(_, frames, video, video_error, samples, audio, audio_error)| Decoded {
                frames,
                video,
                video_error: video_error.map(String::from),
                samples,
                audio,
                audio_error: audio_error.map(String::from),
            },
        );
        if golden.as_ref() != Some(&decoded) {
            mismatches.push(format!("{file}: expected {golden:?}, got {decoded:?}"));
        }

        let Decoded {
            frames,
            video,
            video_error,
            samples,
            audio,
            audio_error,
        } = decoded;
        writeln!(
            actual,
            "    ({file:?}, {frames}, {video:#018x}, {video_error:?}, {samples}, {audio:#018x}, {audio_error:?}),"
        )
        .unwrap();
    }
    assert!(
        mismatches.is_empty(),
        "{}\n\nthe table for every sample as decoded now:\n{actual}",
        mismatches.join("\n")
    );
    assert_eq!(GOLDEN.len(), manifest().len(), "GOLDEN has stale entries");
}
