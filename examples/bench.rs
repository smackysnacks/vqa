//! Benchmark the decoder and check that its output is unchanged. Build with
//! `--release`.
//!
//! Usage: bench hash <vqa file>...    hash each movie's decoded output
//!        bench time <vqa file>...    time each decoding stage
//!        bench synth8                time the 8-bit path on generated frames
//!
//! `hash` covers every frame's RGB888 bytes and the whole soundtrack, so a
//! performance change that alters any output shows up as a new hash.
//! Timings are the best of several runs.

use std::hint::black_box;
use std::time::{Duration, Instant};

use vqa::{FrameDecoder, VQA, VQAHeader, VQAVersion, lcw, raw_chunk};

const RUNS: usize = 7;

const FNV_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let files = args.get(2..).unwrap_or_default();
    match args.get(1).map(String::as_str) {
        Some("hash") if !files.is_empty() => files.iter().for_each(|file| hash(file)),
        Some("time") if !files.is_empty() => files.iter().for_each(|file| time(file)),
        Some("synth8") => synth8(),
        _ => {
            println!("usage: {} hash <vqa file>...", args[0]);
            println!("       {} time <vqa file>...", args[0]);
            println!("       {} synth8", args[0]);
        }
    }
}

fn fnv1a(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}

/// The shortest of `RUNS` runs of `run`.
fn best(mut run: impl FnMut()) -> Duration {
    (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed()
        })
        .min()
        .expect("RUNS is nonzero")
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

/// Print a hash of every frame's RGB888 bytes followed by every audio
/// sample's little-endian bytes (FNV-1a 64).
fn hash(path: &str) {
    let buffer = std::fs::read(path).expect("failed to read file");
    let vqa = VQA::parse(&buffer).expect("failed to parse VQA");

    let mut hash = FNV_BASIS;
    for frame in vqa.frames().expect("bad video header") {
        hash = fnv1a(hash, &frame.expect("failed to decode frame").to_rgb888());
    }
    match vqa.decode_audio() {
        Ok(samples) => {
            for sample in samples {
                hash = fnv1a(hash, &sample.to_le_bytes());
            }
        }
        Err(e) => println!("{path}: audio not hashed: {e}"),
    }

    println!("{hash:016x}  {path}");
}

/// Time frame decoding, RGB conversion, LCW decompression, and audio
/// decoding separately.
fn time(path: &str) {
    let buffer = std::fs::read(path).expect("failed to read file");
    let vqa = VQA::parse(&buffer).expect("failed to parse VQA");
    let header = &vqa.header;
    println!(
        "{path}: {}x{}, {}x{} blocks, {} frames, version {:?}{}",
        header.width,
        header.height,
        header.block_width,
        header.block_height,
        header.num_frames,
        header.version,
        if header.is_hicolor() { ", HiColor" } else { "" }
    );

    let mut num_frames = 0;
    let decode = best(|| {
        num_frames = 0;
        for frame in vqa.frames().expect("bad video header") {
            black_box(frame.expect("failed to decode frame"));
            num_frames += 1;
        }
    });
    println!(
        "  frame decode   {:8.1} ms  ({:.3} ms/frame)",
        ms(decode),
        ms(decode) / num_frames.max(1) as f64
    );

    // the same, borrowing each frame instead of copying it out
    let decode = best(|| {
        let mut frames = vqa.frames().expect("bad video header");
        while let Some(frame) = frames.next_ref() {
            black_box(frame.expect("failed to decode frame"));
        }
    });
    println!(
        "  ... next_ref   {:8.1} ms  ({:.3} ms/frame)",
        ms(decode),
        ms(decode) / num_frames.max(1) as f64
    );

    // one frame from the middle of the movie, converted repeatedly while it
    // sits in cache - as it does right after decoding
    if let Some(frame) = vqa
        .frames()
        .expect("bad video header")
        .take(usize::from(header.num_frames / 2) + 1)
        .last()
    {
        let frame = frame.expect("failed to decode frame");
        let rgb = best(|| {
            for _ in 0..100 {
                black_box(frame.to_rgb888());
            }
        });
        println!(
            "  to_rgb888      {:8.1} us/frame",
            rgb.as_secs_f64() * 1e6 / 100.0
        );

        let mut out = vec![0; frame.width * frame.height * 3];
        let rgb = best(|| {
            for _ in 0..100 {
                frame.write_rgb888(&mut out);
                black_box(&out);
            }
        });
        println!(
            "  write_rgb888   {:8.1} us/frame  (reused buffer)",
            rgb.as_secs_f64() * 1e6 / 100.0
        );
    }

    // every compressed sub-chunk of the video stream
    let mut compressed = Vec::new();
    for chunk in vqa.chunks() {
        let chunk = chunk.expect("failed to parse chunk");
        if &chunk.id == b"VQFR" || &chunk.id == b"VQFL" {
            let mut data = chunk.data;
            while !data.is_empty() {
                let (rest, sub) = raw_chunk(data).expect("failed to parse sub-chunk");
                data = rest;
                if sub.id[3] == b'Z' {
                    compressed.push(sub.data);
                }
            }
        }
    }
    let mut out_bytes = 0;
    let lcw_time = best(|| {
        out_bytes = 0;
        for data in &compressed {
            out_bytes += black_box(lcw::decompress(data, 1 << 24).expect("bad LCW data")).len();
        }
    });
    println!(
        "  lcw            {:8.1} ms  ({} chunks, {} KiB out)",
        ms(lcw_time),
        compressed.len(),
        out_bytes / 1024
    );

    if header.has_sound() {
        match vqa.decode_audio() {
            Ok(_) => {
                let audio = best(|| {
                    black_box(vqa.decode_audio().expect("failed to decode audio"));
                });
                println!("  decode_audio   {:8.1} ms", ms(audio));
            }
            Err(e) => println!("  decode_audio   skipped: {e}"),
        }
    }
}

/// Time the 8-bit (VPT0) path on generated frames, since no 8-bit sample
/// movie is at hand: a random full codebook and palette, then random v2
/// pointer tables with about one fill block in 16.
fn synth8() {
    for (width, height) in [(320, 200), (640, 400)] {
        let frames = 1000;
        let header = VQAHeader {
            version: VQAVersion::Two,
            flags: 0,
            num_frames: frames as u16,
            width,
            height,
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
        };

        // xorshift64, fixed seed so every run decodes the same frames
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut random = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let codebook: Vec<u8> = (0..0x0f00 * 8).map(|_| random() as u8).collect();
        let palette: Vec<u8> = (0..256 * 3).map(|_| (random() & 0x3f) as u8).collect();
        let mut setup = chunk("CBF0", &codebook);
        setup.extend(chunk("CPL0", &palette));

        let blocks = usize::from(width / 4) * usize::from(height / 2);
        let tables: Vec<Vec<u8>> = (0..16)
            .map(|_| {
                let mut table = vec![0; blocks * 2];
                for i in 0..blocks {
                    let r = random();
                    table[i] = r as u8;
                    // HiVal 0x0f flags a fill block
                    table[blocks + i] = if r >> 60 == 0 {
                        0x0f
                    } else {
                        (r >> 8 & 0xff) as u8 % 0x0f
                    };
                }
                chunk("VPT0", &table)
            })
            .collect();

        let decode = best(|| {
            let mut decoder = FrameDecoder::new(&header).expect("bad synthetic header");
            decoder
                .process_vqfl(&setup)
                .expect("bad synthetic codebook");
            for table in tables.iter().cycle().take(frames) {
                black_box(decoder.decode_frame(table).expect("bad synthetic frame"));
            }
        });
        println!(
            "synthetic 8-bit v2 {width}x{height}, 4x2 blocks: {:.4} ms/frame",
            ms(decode) / frames as f64
        );

        // palette lookup, cache-hot as in `time`
        let mut decoder = FrameDecoder::new(&header).expect("bad synthetic header");
        decoder
            .process_vqfl(&setup)
            .expect("bad synthetic codebook");
        let frame = decoder
            .decode_frame(&tables[0])
            .expect("bad synthetic frame");
        let rgb = best(|| {
            for _ in 0..100 {
                black_box(frame.to_rgb888());
            }
        });
        println!(
            "  to_rgb888 {:8.1} us/frame",
            rgb.as_secs_f64() * 1e6 / 100.0
        );
    }
}

/// Wrap `data` in a chunk header with the given ID, padded to even length.
fn chunk(id: &str, data: &[u8]) -> Vec<u8> {
    let mut out = id.as_bytes().to_vec();
    out.extend((data.len() as u32).to_be_bytes());
    out.extend(data);
    if data.len() % 2 == 1 {
        out.push(0);
    }
    out
}
