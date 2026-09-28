//! Audio decoding: the IMA ADPCM codec behind `SND2` sound chunks
//! ([`decompress`] and [`CodecState`]) and Westwood's own ADPCM behind
//! `SND1` ones ([`westwood`]).
//!
//! [`VQA::decode_audio`](crate::VQA::decode_audio) and
//! [`VQA::audio_chunks`](crate::VQA::audio_chunks) drive these decoders and
//! handle the per-version stereo layouts; use [`decompress`] and
//! [`westwood::decompress`] directly when working with raw chunk data.

pub use self::codec::{CodecState, decompress, decompress_into};

mod codec;
pub mod westwood;
