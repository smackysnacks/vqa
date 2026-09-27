//! Error type shared by the high-level decoding API.

use std::fmt;

use crate::lcw::LcwError;

/// Errors produced while decoding a VQA movie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The container or a chunk failed to parse.
    Parse,
    /// LCW-compressed chunk data was malformed.
    Lcw(LcwError),
    /// A size in the file exceeds a sanity limit (the string names it).
    TooLarge(&'static str),
    /// Video data was malformed (the string says how).
    Video(&'static str),
    /// The soundtrack uses a codec this crate does not support. Every sound
    /// chunk type VQA movies use (`SND0`-`SND2`) now decodes, so nothing
    /// returns this.
    UnsupportedSound(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parse => write!(f, "malformed chunk structure"),
            Error::Lcw(e) => write!(f, "malformed LCW data: {}", e),
            Error::TooLarge(what) => write!(f, "{} exceeds sanity limits", what),
            Error::Video(what) => write!(f, "malformed video data: {}", what),
            Error::UnsupportedSound(what) => write!(f, "unsupported sound format: {}", what),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Lcw(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LcwError> for Error {
    fn from(e: LcwError) -> Self {
        Error::Lcw(e)
    }
}
