# vqa

[![CI](https://github.com/smackysnacks/vqa/actions/workflows/rust.yml/badge.svg)](https://github.com/smackysnacks/vqa/actions/workflows/rust.yml)
[![crates.io](https://img.shields.io/crates/v/vqa.svg)](https://crates.io/crates/vqa)
[![Crates.io Total Downloads](https://img.shields.io/crates/d/vqa)](https://crates.io/crates/vqa)
[![docs.rs](https://img.shields.io/docsrs/vqa)](https://docs.rs/vqa)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A parser and decoder for Westwood Studios' VQA (Vector Quantized Animation)
format — the full-motion-video format of Westwood's 90s games, including
Command & Conquer, Red Alert, Lands of Lore, Dune 2000, Blade Runner,
Tiberian Sun, and Nox.

![Playing a VQA movie with the player example](assets/demo.jpg)

## Format support

| Area      | Coverage                                                                                        |
|-----------|-------------------------------------------------------------------------------------------------|
| Container | All three versions (v1–v3), both 8-bit and HiColor movies                                        |
| Video     | 8-bit palettized (`VPT?` pointer tables) and 15-bit HiColor (`VPTR`/`VPRZ` command streams, including the Blade Runner alpha-skip commands) |
| Audio     | IMA ADPCM (`SND2`), Westwood ADPCM (`SND1`, early 8-bit-audio movies), and raw PCM (`SND0`) |

Malformed input fails with an error rather than panicking, and allocation
sizes taken from the file are capped, so the crate is safe to run on
untrusted data. Its fuzz targets (see `fuzz/`) run nightly in CI.

## Quick start

```rust
use vqa::VQA;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = std::fs::read("movie.vqa")?;
    let vqa = VQA::parse(&data)?;

    let header = &vqa.header;
    println!(
        "{}x{}, {} frames at {} fps",
        header.width, header.height, header.num_frames, header.frame_rate
    );

    // The video, frame by frame
    for frame in vqa.frames()? {
        let rgb = frame?.to_rgb888(); // packed RGB bytes, row-major
    }

    // The soundtrack, as interleaved signed 16-bit PCM
    if header.has_sound() {
        let samples = vqa.decode_audio()?;
    }
    Ok(())
}
```

For consumers that want to walk the container themselves, `Chunks` walks
any run of chunks zero-copy, and `Chunk::sub_chunks` the chunks nested in
one. `FrameDecoder`, `lcw` (LCW/"Format80" decompression), and `audio` (IMA
and Westwood ADPCM) are the decoding layers underneath. See the
[API docs](https://docs.rs/vqa) for the full tour.

## Performance

A 640x400 frame decodes in 25–50 µs on a current desktop CPU, over a
thousand times faster than the movies play. Drawing 8-bit frames and
converting HiColor frames to RGB use SIMD through
[fearless_simd](https://crates.io/crates/fearless_simd): AVX-512, AVX2,
SSE4.2 or SSE2 on x86, chosen at run time, so a build for generic x86-64
runs as fast as one for the host CPU, and NEON on 64-bit ARM.

WebAssembly has no run-time detection, so enable `simd128` when building
for it; every major browser, Node and wasmtime support it. Without it, the
crate falls back to scalar code, which on wasm32 draws 8-bit frames 2.6
times as slowly and converts HiColor frames to RGB three times as slowly:

```sh
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown
```

## Examples

Runnable examples exercise the high-level API, using the bundled
`examples/wwlogo.vqa` sample movie:

```sh
# Play a movie in a window (video + audio; Space pauses, Esc/Q quits)
cargo run --release --example player -- examples/wwlogo.vqa 2
# or: just play examples/wwlogo.vqa

# Dump every video frame as PPM
cargo run --release --example dump_frames -- examples/wwlogo.vqa out/

# Time each decoding stage (or `hash` the decoded output)
cargo run --release --example bench -- time examples/wwlogo.vqa
```

The examples' audio/video output uses [cpal](https://crates.io/crates/cpal)
and [minifb](https://crates.io/crates/minifb) (dev-dependencies only; on
Linux, cpal needs the ALSA headers, e.g. `libasound2-dev`).

## Testing

`cargo test` runs the unit and integration tests. Movies from FFmpeg's
sample archive, at least one of every version, pixel format and sound codec
(27 of them from Westwood's games), are checked separately against pinned
hashes:

```sh
just samples        # download them into samples/ (31 MB), checking md5s
just test-samples
```

## Format documentation

The `doc/` directory of the repository carries the format references this
crate is written against: `vqa.txt` for v1/v2, `hc-vqa.txt` for the HiColor
scheme, and `ima-adpcm.txt` for the audio codec.

## License

MIT - see [LICENSE](LICENSE).
