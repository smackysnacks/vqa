//! A parser and decoder for Westwood Studios' VQA (Vector Quantized
//! Animation) format, the full-motion-video format of Westwood's 90s games
//! (Command & Conquer, Red Alert, Lands of Lore, Dune 2000, Blade Runner,
//! Tiberian Sun, Nox).
//!
//! # Quick start
//!
//! [`VQA`] parses the container once, then hands out decoded video frames
//! and audio samples:
//!
//! ```no_run
//! use vqa::VQA;
//!
//! let data = std::fs::read("movie.vqa")?;
//! let vqa = VQA::parse(&data)?;
//!
//! let header = &vqa.header;
//! println!(
//!     "{}x{}, {} frames at {} fps",
//!     header.width, header.height, header.num_frames, header.frame_rate
//! );
//!
//! // The video, frame by frame
//! for frame in vqa.frames()? {
//!     let rgb = frame?.to_rgb888(); // packed RGB bytes, row-major
//! }
//!
//! // The soundtrack, as interleaved signed 16-bit PCM
//! if header.has_sound() {
//!     let samples = vqa.decode_audio()?;
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Each frame the iterator yields is a copy of the decoder's own. A loop
//! that is done with each frame before asking for the next can borrow it
//! instead with [`Frames::next_ref`], and convert it into one reused
//! buffer:
//!
//! ```no_run
//! # let data = std::fs::read("movie.vqa")?;
//! # let vqa = vqa::VQA::parse(&data)?;
//! let mut frames = vqa.frames()?;
//! let mut rgb = Vec::new();
//! while let Some(frame) = frames.next_ref() {
//!     let frame = frame?;
//!     rgb.resize(frame.width * frame.height * 3, 0);
//!     frame.write_rgb888(&mut rgb);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! [`VQA::decode_audio`] fails on the first malformed chunk. To play a
//! damaged or cut-off movie as far as it goes, decode the soundtrack chunk
//! by chunk with [`VQA::audio_chunks`], which yields the sound before the
//! damage, and frames with [`Frames`], which stops at the first bad one.
//!
//! Runnable examples exercise the same API: `player` plays a movie (video
//! in a window, soundtrack on the default audio device), `dump_frames`
//! writes every video frame out as PPM, and `bench` times the decoder and
//! hashes its output.
//!
//! # Layers
//!
//! - [`VQA`] with [`Frames`] and [`AudioChunks`]: the high-level API above.
//! - [`Chunks`]: the zero-copy chunk walk underneath, for consumers that
//!   want to walk the container themselves. [`VQA::chunks`] walks a movie's
//!   chunks, [`Chunk::sub_chunks`] the chunks nested in one, and
//!   [`VQAHeader::parse`] and [`FrameInfo::from_raw`] decode the header and
//!   the frame index.
//! - [`FrameDecoder`]: the stateful codebook/palette/frame assembler driving
//!   [`Frames`].
//! - [`audio`]: the IMA ADPCM and Westwood ADPCM decoders behind `SND2` and
//!   `SND1` sound chunks.
//! - [`lcw`]: LCW ("Format80") decompression, used by every `*Z` chunk.
//!
//! # Format support
//!
//! All three container versions (v1-v3) parse. Video decoding covers both
//! the 8-bit palettized scheme (`VPT?` pointer tables) and the HiColor
//! 15-bit scheme (`VPTR`/`VPRZ` command streams, including the Blade Runner
//! alpha-skip commands). Audio decoding covers IMA ADPCM (`SND2`), Westwood
//! ADPCM (`SND1`, in early 8-bit-audio movies), and raw PCM (`SND0`).
//!
//! Malformed input fails with an [`Error`] rather than panicking, and
//! allocation sizes taken from the file are capped, so the crate is safe to
//! run on untrusted data (its fuzz targets run nightly in CI). An error says
//! what went wrong ([`Error::kind`]) and, where known, where: the frame, the
//! chunk, and the chunk's byte offset in the file.
//!
//! The `doc/` directory of the repository carries the format references this
//! crate is written against: `vqa.txt` for v1/v2 and `hc-vqa.txt` for the
//! HiColor scheme.
//!
//! # Performance
//!
//! Drawing 8-bit frames and converting HiColor frames to RGB use SIMD,
//! chosen at run time on x86 (AVX-512, AVX2, SSE4.2 or SSE2), and NEON on
//! 64-bit ARM. WebAssembly has no run-time detection: build with
//! `-C target-feature=+simd128`, or those paths fall back to scalar code,
//! which on wasm32 draws 8-bit frames 2.6 times as slowly and converts
//! HiColor frames three times as slowly.

#![warn(rust_2018_idioms)]
#![warn(missing_docs)]
#![warn(clippy::missing_errors_doc)]

pub use chunk::{Chunk, Chunks};
pub use error::{Error, ErrorKind, Limit, VideoError};
pub use header::{FrameInfo, VQAHeader, VQAVersion};
pub use movie::{AudioChunks, Frames, VQA};
pub use video::{Frame, FrameDecoder, FramePixels, FramePixelsRef, FrameRef};

pub mod audio;
mod chunk;
mod error;
mod header;
pub mod lcw;
mod movie;
mod rgb;
mod video;
