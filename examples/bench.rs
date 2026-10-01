//! Benchmark the decoder and check that its output is unchanged. Build with
//! `--release`; `just bench` runs it natively, at the generic x86-64 level,
//! or on wasm32.
//!
//! Usage: bench hash <vqa file>...    hash each movie's decoded output
//!        bench time <vqa file>...    time each decoding stage
//!        bench synth8                time the 8-bit path on generated frames
//!
//! `hash` covers every frame's RGB888 bytes and the whole soundtrack, so a
//! performance change that alters any output shows up as a new hash.
//! Both commands take a damaged or cut-off movie as far as it decodes.
//!
//! `time` prints one row per movie, each time the best of several runs:
//!
//! - `decode`: [`Frames`](vqa::Frames), copying every frame out
//! - `next_ref`: the same without the copy
//! - `draw`: `next_ref` on a copy of the movie whose compressed chunks
//!   (VPTZ, VPRZ, CBFZ, CPLZ, ...) are swapped for uncompressed ones, so
//!   all that is left is drawing, and `ns/blk` divides it by the blocks
//!   drawn: every block of an 8-bit frame, the blocks a HiColor frame's
//!   command stream writes
//! - `lcw`: decompressing those chunks
//! - `rgb888`, `rgba`, `xrgb`: converting one frame into a reused buffer,
//!   cache-hot as it is right after decoding
//! - `audio`: decoding the whole soundtrack

use std::hint::black_box;
use std::time::{Duration, Instant};

use vqa::{Chunk, FrameDecoder, FrameRef, VQA, VQAHeader, VQAVersion, lcw};

const RUNS: usize = 15;

// FNV-1a, as in tests/common, which the published crate leaves out
const FNV_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let files = args.get(2..).unwrap_or_default();
    match args.get(1).map(String::as_str) {
        Some("hash") if !files.is_empty() => files.iter().for_each(|file| hash(file)),
        Some("time") if !files.is_empty() => {
            print_time_header();
            files.iter().for_each(|file| time(file));
        }
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

fn us(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e6
}

/// Print a hash of every frame's RGB888 bytes followed by every audio
/// sample's little-endian bytes (FNV-1a 64), up to the first error in
/// each.
fn hash(path: &str) {
    let buffer = std::fs::read(path).expect("failed to read file");
    let vqa = VQA::parse(&buffer).expect("failed to parse VQA");

    let mut hash = FNV_BASIS;
    let mut notes = Vec::new();
    let mut frames = vqa.frames().expect("bad video header");
    let (mut count, mut rgb) = (0, Vec::new());
    while let Some(frame) = frames.next_ref() {
        match frame {
            Ok(frame) => {
                rgb.resize(frame.width * frame.height * 3, 0);
                frame.write_rgb888(&mut rgb);
                hash = fnv1a(hash, &rgb);
                count += 1;
            }
            Err(e) => notes.push(format!("video: {e}")),
        }
    }

    let (mut samples, mut chunks) = (Vec::new(), vqa.audio_chunks());
    while let Some(result) = chunks.next_into(&mut samples) {
        if let Err(e) = result {
            notes.push(format!("audio: {e}"));
        }
    }
    for sample in &samples {
        hash = fnv1a(hash, &sample.to_le_bytes());
    }

    let notes = notes
        .iter()
        .map(|note| format!("; {note}"))
        .collect::<String>();
    println!(
        "{hash:016x}  {path}  ({count} frames, {} samples{notes})",
        samples.len()
    );
}

fn print_time_header() {
    println!(
        "{:<26} {:<18} {:>6} {:>8} {:>8} {:>8} {:>8} {:>7} {:>8} {:>8} {:>8} {:>7}",
        "", "", "", "decode", "next_ref", "draw", "lcw", "", "rgb888", "rgba", "xrgb", "audio"
    );
    println!(
        "{:<26} {:<18} {:>6} {:>35} {:>7} {:>26} {:>7}",
        "movie",
        "kind",
        "frames",
        "---------- us/frame ----------",
        "ns/blk",
        "------ us/frame ------",
        "ms"
    );
}

/// Time every decoding stage of one movie and print them as a row.
fn time(path: &str) {
    let buffer = std::fs::read(path).expect("failed to read file");
    let vqa = VQA::parse(&buffer).expect("failed to parse VQA");
    let header = &vqa.header;
    let kind = format!(
        "{} {}x{} {}x{}",
        match (header.version, header.is_hicolor()) {
            (_, true) => "hc",
            (VQAVersion::One, _) => "v1",
            _ => "8b",
        },
        header.width,
        header.height,
        header.block_width,
        header.block_height,
    );

    // frames up to the first error
    let count_frames = |vqa: &VQA<'_>| {
        let mut frames = vqa.frames().expect("bad video header");
        let mut count = 0;
        while let Some(Ok(_)) = frames.next_ref() {
            count += 1;
        }
        count
    };
    let num_frames = count_frames(&vqa);
    let per_frame = |time: Duration| us(time) / num_frames.max(1) as f64;

    let decode = best(|| {
        for frame in vqa.frames().expect("bad video header") {
            if black_box(frame).is_err() {
                break;
            }
        }
    });
    let next_ref = best(|| next_ref_all(&vqa));

    let (expanded, blocks) = expand(&buffer, &vqa);
    let expanded = VQA::parse(&expanded).expect("failed to parse the expanded copy");
    assert_eq!(
        count_frames(&expanded),
        num_frames,
        "{path}: expanded copy differs"
    );
    let draw = best(|| next_ref_all(&expanded));

    // every compressed chunk the decoder sees, but for codebook parts
    // (CBPZ), which are LCW data only once joined
    let compressed: Vec<&[u8]> = frame_chunks(&vqa)
        .filter(|chunk| chunk.id[3] == b'Z' || matches!(&chunk.id, b"VPTK" | b"VPTD"))
        .filter(|chunk| &chunk.id != b"CBPZ")
        .map(|chunk| chunk.data)
        .collect();
    let lcw_time = best(|| {
        for data in &compressed {
            let _ = black_box(lcw::decompress(data, 1 << 24));
        }
    });

    // one frame from the middle of the movie
    let mut frames = vqa.frames().expect("bad video header");
    let middle = frames.nth(num_frames / 2).and_then(Result::ok);
    let (rgb888, rgba, xrgb) = match &middle {
        Some(frame) => convert_times(frame.view()),
        None => (f64::NAN, f64::NAN, f64::NAN),
    };

    let audio = best(|| {
        let (mut samples, mut chunks) = (Vec::new(), vqa.audio_chunks());
        while let Some(Ok(())) = chunks.next_into(&mut samples) {}
        black_box(samples);
    });

    let name = path.rsplitn(3, '/').take(2).collect::<Vec<_>>();
    let name = name.into_iter().rev().collect::<Vec<_>>().join("/");
    println!(
        "{name:<26} {kind:<18} {num_frames:>6} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>7.2} {rgb888:>8.1} {rgba:>8.1} {xrgb:>8.1} {:>7.2}",
        per_frame(decode),
        per_frame(next_ref),
        per_frame(draw),
        per_frame(lcw_time),
        draw.as_secs_f64() * 1e9 / blocks.max(1) as f64,
        audio.as_secs_f64() * 1e3,
    );
}

/// Decode every frame up to the first error, borrowing each.
fn next_ref_all(vqa: &VQA<'_>) {
    let mut frames = vqa.frames().expect("bad video header");
    while let Some(frame) = frames.next_ref() {
        if black_box(frame).is_err() {
            break;
        }
    }
}

/// The chunks that make up frames: the sub-chunks of every VQFR, VQFK and
/// VQFL chunk, and the top-level ones of the older layout without them.
fn frame_chunks<'a>(vqa: &VQA<'a>) -> impl Iterator<Item = Chunk<'a>> {
    vqa.chunks().map_while(Result::ok).flat_map(|chunk| {
        let nested = matches!(&chunk.id, b"VQFR" | b"VQFK" | b"VQFL");
        let subs = nested.then(|| chunk.sub_chunks().map_while(Result::ok));
        let top = (!nested).then_some(chunk);
        subs.into_iter().flatten().chain(top)
    })
}

/// Convert `frame` to each output format into a reused buffer, giving the
/// best time per conversion in microseconds.
fn convert_times(frame: FrameRef<'_>) -> (f64, f64, f64) {
    let pixels = frame.width * frame.height;
    let mut rgb888 = vec![0; pixels * 3];
    let mut rgba = vec![0; pixels * 4];
    let mut xrgb = vec![0; pixels];
    (
        per_call(|| frame.write_rgb888(black_box(&mut rgb888))),
        per_call(|| frame.write_rgba8888(black_box(&mut rgba))),
        per_call(|| frame.write_xrgb8888(black_box(&mut xrgb))),
    )
}

/// The best time for one call of `run`, in microseconds, calling it 100
/// times a run.
fn per_call(mut run: impl FnMut()) -> f64 {
    us(best(|| (0..100).for_each(|_| run()))) / 100.0
}

/// A copy of the movie with every compressed frame chunk swapped for its
/// uncompressed form, and the number of blocks its frames draw. A chunk
/// that doesn't decompress stays as it is, to fail the same way.
fn expand(buffer: &[u8], vqa: &VQA<'_>) -> (Vec<u8>, usize) {
    // the FORM chunk header, the WVQA signature and the VQHD chunk
    let mut out = buffer[..12 + 8 + 42].to_vec();
    let header = &vqa.header;
    let blocks_per_frame = usize::from(header.width / u16::from(header.block_width.max(1)))
        * usize::from(header.height / u16::from(header.block_height.max(1)));
    let mut blocks = 0;

    let mut expand_one = |out: &mut Vec<u8>, chunk: &Chunk<'_>| {
        // the uncompressed ID, and how far the data may expand. Past the
        // decoder's own bound, a command stream fails as compressed data,
        // but would decode uncompressed; the other chunks fail either way
        let uncompressed = match &chunk.id {
            b"VPTZ" | b"VPTK" | b"VPTD" => Some((b"VPT0", 1 << 24)),
            b"VPRZ" => Some((b"VPTR", blocks_per_frame * 8 + 256)),
            b"CBFZ" => Some((b"CBF0", 1 << 24)),
            b"CPLZ" => Some((b"CPL0", 1 << 24)),
            _ => None,
        };
        let expanded =
            uncompressed.and_then(|(id, max)| Some((id, lcw::decompress(chunk.data, max).ok()?)));
        let (id, data) = match &expanded {
            Some((id, data)) => (*id, data.as_slice()),
            None => (&chunk.id, chunk.data),
        };
        match id {
            b"VPT0" => blocks += blocks_per_frame,
            b"VPTR" => blocks += vptr_writes(data),
            _ => {}
        }
        push_chunk(out, id, data);
    };

    for chunk in vqa.chunks().map_while(Result::ok) {
        if matches!(&chunk.id, b"VQFR" | b"VQFK" | b"VQFL") {
            let mut payload = Vec::new();
            for sub in chunk.sub_chunks().map_while(Result::ok) {
                expand_one(&mut payload, &sub);
            }
            push_chunk(&mut out, &chunk.id, &payload);
        } else {
            expand_one(&mut out, &chunk);
        }
    }
    (out, blocks)
}

/// Append a chunk: the ID, the big-endian size, the payload, and a pad
/// byte after an odd-sized payload.
fn push_chunk(out: &mut Vec<u8>, id: &[u8; 4], data: &[u8]) {
    out.extend(id);
    out.extend((data.len() as u32).to_be_bytes());
    out.extend(data);
    if data.len() % 2 == 1 {
        out.push(0);
    }
}

/// The number of blocks a HiColor command stream writes.
fn vptr_writes(stream: &[u8]) -> usize {
    let (mut writes, mut sp) = (0, 0);
    while let Some(bytes) = stream.get(sp..sp + 2) {
        let val = u16::from_le_bytes([bytes[0], bytes[1]]);
        sp += 2;
        let run = usize::from(val >> 8 & 0x1f) + 1;
        match val >> 13 {
            0b000 => {}
            0b001 => writes += run * 2,
            0b010 => {
                writes += 1 + run * 2;
                sp += run * 2;
            }
            0b011 | 0b100 => writes += 1,
            0b101 | 0b110 => {
                writes += usize::from(stream.get(sp).copied().unwrap_or(0));
                sp += 1;
            }
            _ => break,
        }
    }
    writes
}

/// Time the 8-bit (VPT0) path on generated frames: a random full codebook
/// and palette, then random v2 pointer tables with about one fill block in
/// 16.
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
        let mut setup = Vec::new();
        push_chunk(&mut setup, b"CBF0", &codebook);
        push_chunk(&mut setup, b"CPL0", &palette);

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
                let mut chunk = Vec::new();
                push_chunk(&mut chunk, b"VPT0", &table);
                chunk
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
            "synthetic 8-bit v2 {width}x{height}, 4x2 blocks: {:.1} us/frame",
            us(decode) / frames as f64
        );
    }
}
