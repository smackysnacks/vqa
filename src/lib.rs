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
//! Runnable examples exercise the same API: `player` plays a movie (video
//! in a window, soundtrack on the default audio device), `play` plays just
//! the soundtrack, `dump_frames` writes every video frame out as PPM, and
//! `bench` times the decoder and hashes its output.
//!
//! # Layers
//!
//! - [`movie`]: the high-level API above ([`VQA`], [`Chunks`], [`Frames`]).
//! - [`parser`]: zero-copy nom parsers for the individual chunks, for
//!   consumers that want to walk the container themselves.
//! - [`video`]: [`FrameDecoder`], the stateful codebook/palette/frame
//!   assembler driving [`Frames`].
//! - [`audio`]: the IMA ADPCM decoder behind `SND2` sound chunks.
//! - [`lcw`]: LCW ("Format80") decompression, used by every `*Z` chunk.
//!
//! # Format support
//!
//! All three container versions (v1-v3) parse. Video decoding covers both
//! the 8-bit palettized scheme (`VPT?` pointer tables) and the HiColor
//! 15-bit scheme (`VPTR`/`VPRZ` command streams, including the Blade Runner
//! alpha-skip commands). Audio decoding covers IMA ADPCM (`SND2`) and raw
//! PCM (`SND0`); Westwood ADPCM (`SND1`, found in early 8-bit-audio movies)
//! is not supported yet.
//!
//! Malformed input fails with an [`Error`] rather than panicking, and
//! allocation sizes taken from the file are capped, so the crate is safe to
//! run on untrusted data (it is continuously fuzzed).
//!
//! The `doc/` directory of the repository carries the format references this
//! crate is written against: `vqa.txt` for v1/v2 and `hc-vqa.txt` for the
//! HiColor scheme.

#![warn(rust_2018_idioms)]
#![warn(missing_docs)]

pub use error::Error;
pub use movie::{Chunks, Frames, VQA};
pub use parser::*;
pub use video::{Frame, FrameDecoder, FramePixels, FramePixelsRef, FrameRef};

pub mod audio;
pub mod error;
pub mod lcw;
pub mod movie;
pub mod parser;
mod rgb;
pub mod video;
