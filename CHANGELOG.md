# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Decoding is now checked against 29 movies from FFmpeg's sample archive, 27
of them from Westwood's games. Comparing their output with FFmpeg's and with
Westwood's own source found several of the fixes below. Westwood ADPCM audio
and three container variants now decode, and frames convert to 4-byte pixel
formats.

This release also breaks the API, once, so later releases don't have to:
- Errors say where they happened.
- One chunk type replaces the nom parsers, and nom is no longer a
  dependency.
- The header's unknown fields have Westwood's own names.
- Each public item has one path.

[Migrating from 0.6](#migrating-from-06) lists what to change. Apart from
the fixes below, decoding output is unchanged.

### Added

- Westwood ADPCM (`SND1`) audio, the soundtrack codec of Kyrandia 3 and
  other early movies, in `decode_audio`, `audio_chunks`, and the new
  `audio::westwood` module.
- `VQA::audio_chunks` and `AudioChunks`, which decode the soundtrack one
  sound chunk at a time. Unlike `decode_audio`, which fails outright, they
  yield every chunk before a malformed one, so a damaged or cut-off movie
  still gives up its sound.
- `to_rgba8888`, `write_rgba8888`, `to_xrgb8888` and `write_xrgb8888` on
  `Frame` and `FrameRef`. RGBA8888 bytes suit GPU textures and image
  libraries, and XRGB8888 (`0x00RRGGBB`) words suit software framebuffers.
  HiColor frames convert with a SIMD kernel, as for RGB888.
- `Frames` decodes three more kinds of movie:
  - Lands of Lore's movies whose codebooks come in groups of frames of
    varying length (`cbparts` 0 in the header), as their CINF chunk
    schedules them. `FrameDecoder::swap_in_codebook_parts` does the same
    for callers driving the decoder themselves.
  - The older layout that has each frame's chunks at the top level instead
    of in a VQFR chunk (Kyrandia 3's `benchl.vqa`).
  - VQFK key frames, and VPTK and VPTD pointer tables.
- `VQA::codebook_starts`, the frames where the CINF chunk starts each
  codebook.
- Where an error happened: `Error::chunk`, `Error::offset` and
  `Error::frame` give the chunk, its byte offset, and the frame number,
  when known, and the message includes them, e.g. `unexpected end of data
  (frame 417, VQFR chunk at offset 0x363b92)`.
- `Chunk::sub_chunks` walks the chunks nested in a VQFR, VQFK, VQFL or CINF
  chunk, and `Chunks::new` any run of chunks. Each `Chunk` has the `offset`
  of its header.
- `VQAHeader::parse` parses a VQHD payload, `FrameInfo::from_raw` a FINF
  entry, and `VQAVersion` converts to and from its number (`u16::from`,
  `VQAVersion::try_from`).
- `VQAHeader::max_cbfz_size`, the size of the largest compressed codebook,
  which HiColor, Lands of Lore and some Red Alert movies store in the
  header's reserved words.

### Changed

- **Breaking:** `Error` is a struct, and `Error::kind` says what went wrong.
  The `ErrorKind` enum splits the old `Parse` into `NotVqa`,
  `InvalidHeader`, `Truncated` (the file is cut short) and `InvalidChunk`
  (the chunks don't line up). `TooLarge` and `Video` carry typed causes,
  `Limit` and `VideoError`, in place of strings.
  - `Error` is no longer `Copy`.
  - `source()` no longer returns the LCW error, whose reason is part of the
    message, as with `std::io::Error`.
  - The message text has changed.
- **Breaking:** `RawChunk` is now `Chunk`. It drops `size`, which always
  equals `data.len()`, gains `offset`, and is `#[non_exhaustive]`.
  `VQA::chunks` counts offsets from the start of the file.
- **Breaking:** `VQAHeader`'s unknown fields take the names in Westwood's
  own `VQAHeader`:
  - `unk1` is `x_pos` and `y_pos`, where to draw the frames. Blade Runner
    places its overlays by it.
  - `unk2` is `max_frame_size`.
  - `unk3` is `alt_freq`, `alt_channels` and `alt_bits`, for an alternate
    soundtrack.
  - `unk4`, `max_cbfz_size` and `unk5` are the reserved `future_use` words;
    `max_cbfz_size` is now a method.
- **Breaking:** the crate root re-exports each public item by name, and the
  `movie`, `video`, `error` and `audio::codec` modules are private. `audio`
  and `lcw` are the only public modules.
- **Breaking:** `VQAVersion`'s discriminants are its version numbers, so
  `VQAVersion::One as u16` is 1, not 0.
- **Breaking:** `FrameInfo`, `ErrorKind` and `lcw::LcwError` are
  `#[non_exhaustive]`.
- `decode_audio` fails with `ErrorKind::TooLarge` past 2^26 samples (over 25
  minutes of stereo sound at 22050 Hz). Westwood ADPCM expands up to
  64-fold, so a small crafted movie could otherwise make it allocate
  gigabytes. `audio_chunks` holds one chunk at a time and has no limit.
- Every fallible function documents its errors in an `# Errors` section.
- Faster decoding, with output unchanged:
  - 8-bit frames draw 2.5–4.5 times as fast. Where each block draws from
    is worked out with SIMD, 16 blocks at a time, and blocks are copied four
    at a time, one store per line of pixels. A 640x400 frame decodes in
    42 µs instead of 82.
  - HiColor frames draw 20–30% faster.
  - 8-bit frames convert to RGB888 twice as fast, and HiColor frames a
    third faster, from a SIMD kernel that takes 32 pixels at a time.
  - Mono IMA ADPCM soundtracks decode 40% faster, and stereo ones 10%.
- The README and the crate docs say to build for WebAssembly with
  `simd128`, without which the SIMD paths fall back to scalar code.
- The `bench` example times drawing and LCW decompression apart, and takes
  damaged or cut-off movies as far as they decode. `just bench` runs it
  natively, built for generic x86-64, or on wasm32 under Node.

### Removed

- **Breaking:** the `parser` module: its 15 nom parsers and the chunk
  structs they returned (`SND2Chunk`, `VQFRChunk`, `CBFChunk` and so on).
  Walk chunks with `VQA::chunks`, `Chunks::new` and `Chunk::sub_chunks`,
  and match on `chunk.id`.
- **Breaking:** `Error::UnsupportedSound`. Nothing returns it, now that
  every sound chunk type decodes.
- The `nom` dependency.

### Fixed

- 8-bit movies with 4x4 blocks and at most 0x0f00 codebook entries (some
  of Lands of Lore's) failed on their first frame. The marker for a
  solid-color block now depends on the block size (0x0f for 4x2 blocks,
  0xff for 4x4) rather than on the header's `maxblocks`. So 4x2 movies
  whose header allows more than 0x0f00 entries now mark solid-color
  blocks with 0x0f, as Westwood's own 4x2 drawer does, rather than 0xff.
  None of the sample movies is one of them.
- `FrameInfo::offset` masked FINF entries with `0x3FFFFFFF`, so an entry
  with its sync flag (bit 29) or bit 28 set got a wrong offset. The top
  four bits are flags, as in Westwood's own VQA library.
- On 32-bit targets, `FrameDecoder::new` (and so `VQA::frames`) panicked in
  debug builds on a HiColor header with very large blocks and codebook.
- The `player` example panicked on a frame that failed to decode, and
  played without sound if any sound chunk was malformed. It now plays a
  damaged movie as far as it goes.
- On 32-bit targets such as wasm32, a HiColor pointer stream with long runs
  of skip commands overflowed the block position. Debug builds panicked, and
  release builds wrapped around silently. The position now saturates, so a
  write past the frame fails with an error, as on 64-bit targets.
- The `dump_frames` example no longer panics on a step of `0` or a malformed
  step.
- The `FramePixels::HiColor` docs said the top bit of every pixel is clear.
  Plain block writes copy a codebook pixel whole, that bit included, while
  the alpha-skip writes leave out the pixels that have it (Blade Runner's
  transparent ones). RGB conversion ignores it.
- `bench time` panicked on movies with compressed codebook parts, Red
  Alert's among them.

### Migrating from 0.6

| 0.6 | 0.7 |
|---|---|
| `use vqa::*`, `vqa::parser::X`, `vqa::movie::X`, `vqa::video::X`, `vqa::error::Error` | `vqa::X` |
| `vqa::audio::codec::X` | `vqa::audio::X` |
| `match e { Error::X => .. }` | `match e.kind() { ErrorKind::X => .., _ => .. }` |
| `Error::Parse` | `ErrorKind::NotVqa`, `InvalidHeader`, `Truncated` or `InvalidChunk` |
| `Error::Video("block size is zero")` | `ErrorKind::InvalidHeader` |
| `Error::Video("...")` | `ErrorKind::Video(VideoError::...)` |
| `Error::TooLarge("frame dimensions")`, `("codebook")`, `("codebook parts")`, `("soundtrack")` | `ErrorKind::TooLarge(Limit::FrameSize)`, `(Limit::Codebook)`, `(Limit::Codebook)`, `(Limit::Soundtrack)` |
| `Error::Lcw(e)`, `e.source()` | `ErrorKind::Lcw(e)` from `e.kind()` |
| `raw_chunk(input)` | `Chunks::new(input)`, an iterator |
| `RawChunk { id, size, data }` | `Chunk { id, offset, data, .. }`; `size` is `data.len()` |
| a `raw_chunk` loop over a VQFR, VQFL or CINF payload | `for sub in chunk.sub_chunks()` |
| `snd2_chunk`, `vqfr_chunk`, `cbf_chunk` and the other typed parsers | match on `chunk.id`; the payload is `chunk.data` |
| `form_chunk`, `vqa_header` | `VQA::parse(file)?.form_size` and `.header`, or `VQAHeader::parse(vqhd.data)` |
| `vqa_version`, `frame_info`, `finf_chunk` | `VQAVersion::try_from`, `FrameInfo::from_raw`, `vqa.frame_index` |
| `header.unk1`, `unk2`, `unk3` | `x_pos` and `y_pos`; `max_frame_size`; `alt_freq`, `alt_channels` and `alt_bits` |
| `header.unk4`, `max_cbfz_size`, `unk5` | `future_use[0]`, `max_cbfz_size()`, `future_use[3..5]` |

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
