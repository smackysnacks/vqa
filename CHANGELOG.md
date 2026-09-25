# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-09-24

Frame decoding is 2.5–3.4× faster and audio decoding 3.9× faster, and the
decoded output is unchanged. Decoded frames can now be borrowed instead of
copied.

### Added

- `FrameRef` and `FramePixelsRef`, borrowed versions of `Frame` and
  `FramePixels`. `Frame::view` and `FrameRef::to_frame` convert between them.
- `Frames::next_ref` and `FrameDecoder::decode_frame_ref`, which borrow each
  decoded frame from the decoder instead of copying it out.
- `Frame::write_rgb888` and `FrameRef::write_rgb888`, which convert into a
  caller's buffer so one buffer can be reused across frames.
- `audio::decompress_into`, which appends decoded samples to an existing
  `Vec`.
- `Clone` for `FrameDecoder` and `Frames`, so a decoding position can be saved
  and resumed later.
- A `bench` example that times each decoding stage and hashes the decoded
  output.

### Changed

- **Breaking:** `VQAHeader::flags` is now the raw `u16` from the header.
  Before, only bit 0 was kept, and bits 2–4 (set by every HiColor movie seen
  so far, meaning unknown) were dropped. `VQAHeader::has_sound` works as
  before.
- Faster decoding:
  - Frame decoding is 2.5–3.4× faster, from specialized 4×2 and 4×4 block
    copies.
  - LCW decompression is 2.2× faster.
  - `decode_audio` is 3.9× faster: ADPCM decoding is table-driven and decodes
    both stereo channels in one pass.
  - RGB888 conversion uses SIMD (AVX2, SSE4.2, SSE2 or NEON, chosen at run
    time) through the new `fearless_simd` dependency. It is now equally fast in
    builds for generic x86-64 and builds for the host CPU.
- `Frames::nth`, and so `Iterator::skip`, no longer copies out the frames it
  skips.
- The minimum supported Rust version is now 1.89, and is declared in
  `rust-version`. It was 1.87 before, but not declared.
- The `player` example seeks backwards from checkpoints saved every 5 seconds
  instead of re-decoding from the first frame, and prints the full header.
  `dump_frames` borrows frames and reuses one RGB buffer.

### Removed

- **Breaking:** `VQAFlags`. Use `VQAHeader::has_sound`, or test
  `VQAHeader::flags` directly.
- The `bitflags` dependency.
- The `play` example, which played only the soundtrack. `player` plays it
  too.

## [0.5.1] - 2026-07-24

### Changed

- Added the homepage and documentation links to the crate metadata, and
  pointed the repository link at the renamed GitHub repository.

## [0.5.0] - 2026-07-24

First release on crates.io.

### Added

- `VQA`, which parses a movie once and then decodes its video frame by frame
  (`VQA::frames`) and its soundtrack as interleaved 16-bit PCM
  (`VQA::decode_audio`).
- Zero-copy nom parsers for every chunk type (`parser`), for walking the
  container directly.
- Container versions 1–3.
- Video decoding for 8-bit palettized movies and 15-bit HiColor movies,
  including the Blade Runner alpha-skip commands, with conversion to RGB888.
- Audio decoding for IMA ADPCM (`SND2`) and raw PCM (`SND0`).
- LCW ("Format80") decompression (`lcw`).
- Errors instead of panics on malformed input, and caps on allocation sizes
  read from the file. Fuzz targets cover the parser, LCW and ADPCM.
- `player`, `play` and `dump_frames` examples.

[unreleased]: https://github.com/smackysnacks/vqa/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/smackysnacks/vqa/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/smackysnacks/vqa/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/smackysnacks/vqa/releases/tag/v0.5.0
