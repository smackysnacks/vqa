//! The error type every fallible part of the crate returns.

use std::fmt;

use crate::chunk::Chunk;
use crate::lcw::LcwError;

/// An error decoding a movie: what went wrong ([`Error::kind`]) and, when
/// known, where ([`Error::chunk`], [`Error::offset`] and [`Error::frame`]).
///
/// Match on the kind. The message `Display` prints, which names the place
/// too, is for people and may change in any release.
#[derive(Clone, PartialEq, Eq)]
pub struct Error {
    kind: ErrorKind,
    /// the chunk's ID, or all zeros (never a valid ID) when unknown
    chunk: [u8; 4],
    /// the frame number, or `u32::MAX` when unknown
    frame: u32,
    /// the byte offset, or `usize::MAX` when unknown
    offset: usize,
}

const NO_CHUNK: [u8; 4] = [0; 4];
const NO_FRAME: u32 = u32::MAX;
const NO_OFFSET: usize = usize::MAX;

/// What kind of [`Error`] happened.
///
/// New kinds may be added in minor releases, as may new causes in
/// [`VideoError`], [`Limit`] and [`LcwError`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// The input doesn't start with a `FORM` chunk holding a `WVQA` movie.
    NotVqa,
    /// The `VQHD` header is missing, isn't 42 bytes long, holds a version
    /// other than 1, 2 or 3, or describes a movie that can't be decoded
    /// (blocks of zero width or height).
    InvalidHeader,
    /// The input ends inside a chunk: the file is cut short.
    Truncated,
    /// The chunks don't line up: an ID that isn't four uppercase ASCII
    /// letters or digits (the walk has lost its place), or a sub-chunk
    /// running past the end of its container's payload.
    InvalidChunk,
    /// LCW-compressed chunk data is malformed.
    Lcw(LcwError),
    /// A size in the file exceeds a sanity limit.
    TooLarge(Limit),
    /// Video data is malformed.
    Video(VideoError),
}

/// The sanity limit an [`ErrorKind::TooLarge`] error exceeds.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Limit {
    /// The frame holds more than 2^24 pixels.
    FrameSize,
    /// A codebook, or the codebook parts staged for one, would hold more
    /// than the header's `maxblocks` entries (or more than 2^24 bytes).
    Codebook,
    /// The soundtrack holds more than 2^26 samples; see
    /// [`VQA::decode_audio`](crate::VQA::decode_audio).
    Soundtrack,
}

/// How the video data in an [`ErrorKind::Video`] error is malformed.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoError {
    /// Codebook parts mix uncompressed (`CBP0`) and compressed (`CBPZ`)
    /// ones.
    MixedCodebookParts,
    /// A palette isn't a whole number of colors, or holds more than 256.
    PaletteSize,
    /// A pointer chunk for the other pixel format: a `VPT?` table in a
    /// HiColor movie, or a `VPTR`/`VPRZ` stream in an 8-bit one.
    WrongPointerFormat,
    /// An 8-bit pointer table doesn't hold one entry per block.
    PointerTableSize,
    /// An 8-bit pointer table points past the end of the codebook. HiColor
    /// pointer streams never fail this way: retail movies contain the
    /// occasional stray block index, and those blocks keep their pixels.
    BlockIndexOutOfRange,
    /// A HiColor pointer stream ends in the middle of a command.
    TruncatedPointerStream,
    /// A HiColor pointer stream holds a command that doesn't exist.
    UnknownPointerCommand,
    /// A HiColor pointer stream writes more blocks than the frame has.
    PointerStreamOverrun,
}

impl Error {
    /// What went wrong.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The ID of the innermost chunk involved, e.g. the `VPTZ` inside a
    /// `VQFR`, when there is one and its ID could be read.
    pub fn chunk(&self) -> Option<[u8; 4]> {
        (self.chunk != NO_CHUNK).then_some(self.chunk)
    }

    /// The byte offset of the chunk's 8-byte header, or of the bytes that
    /// failed to read as one, when known. It counts from the start of the
    /// input the call that returned the error was given: the file, for
    /// [`VQA`](crate::VQA) and the iterators it hands out.
    pub fn offset(&self) -> Option<usize> {
        (self.offset != NO_OFFSET).then_some(self.offset)
    }

    /// The number of the frame [`Frames`](crate::Frames) was decoding,
    /// counted from 0 as [`Iterator::nth`] and
    /// [`VQA::codebook_starts`](crate::VQA::codebook_starts) count them.
    pub fn frame(&self) -> Option<usize> {
        (self.frame != NO_FRAME).then_some(self.frame as usize)
    }

    /// An error in the chunk `chunk` (if its ID is known) at `offset`.
    #[cold]
    #[inline(never)]
    pub(crate) fn at(kind: ErrorKind, chunk: Option<[u8; 4]>, offset: usize) -> Error {
        Error {
            kind,
            chunk: chunk.unwrap_or(NO_CHUNK),
            frame: NO_FRAME,
            offset,
        }
    }

    /// An error in `chunk`.
    #[cold]
    #[inline(never)]
    pub(crate) fn in_chunk(kind: ErrorKind, chunk: &Chunk<'_>) -> Error {
        Error::at(kind, Some(chunk.id), chunk.offset)
    }

    /// Place an error with no known location in `chunk`, the chunk that
    /// was being decoded.
    #[cold]
    #[inline(never)]
    pub(crate) fn or_in(mut self, chunk: &Chunk<'_>) -> Error {
        if self.offset == NO_OFFSET {
            self.chunk = chunk.id;
            self.offset = chunk.offset;
        }
        self
    }

    /// Note the frame being decoded.
    #[cold]
    #[inline(never)]
    pub(crate) fn in_frame(mut self, frame: usize) -> Error {
        self.frame = u32::try_from(frame).map_or(NO_FRAME - 1, |frame| frame.min(NO_FRAME - 1));
        self
    }
}

impl From<ErrorKind> for Error {
    /// An error with no location.
    fn from(kind: ErrorKind) -> Error {
        Error {
            kind,
            chunk: NO_CHUNK,
            frame: NO_FRAME,
            offset: NO_OFFSET,
        }
    }
}

impl From<LcwError> for Error {
    fn from(e: LcwError) -> Error {
        Error::from(ErrorKind::Lcw(e))
    }
}

impl From<LcwError> for ErrorKind {
    fn from(e: LcwError) -> ErrorKind {
        ErrorKind::Lcw(e)
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Error")
            .field("kind", &self.kind)
            .field(
                "chunk",
                &self
                    .chunk()
                    .map(|id| String::from_utf8_lossy(&id).into_owned()),
            )
            .field("offset", &self.offset())
            .field("frame", &self.frame())
            .finish()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)?;
        let (frame, chunk, offset) = (self.frame(), self.chunk(), self.offset());
        if frame.is_none() && offset.is_none() {
            return Ok(());
        }
        f.write_str(" (")?;
        if let Some(frame) = frame {
            write!(f, "frame {frame}")?;
            if offset.is_some() {
                f.write_str(", ")?;
            }
        }
        match (chunk, offset) {
            (Some(id), Some(offset)) => {
                write!(f, "{} chunk at offset {offset:#x}", id.escape_ascii())?
            }
            (None, Some(offset)) => write!(f, "at offset {offset:#x}")?,
            _ => {}
        }
        f.write_str(")")
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::NotVqa => f.write_str("not a VQA file"),
            ErrorKind::InvalidHeader => f.write_str("invalid VQHD header"),
            ErrorKind::Truncated => f.write_str("unexpected end of data"),
            ErrorKind::InvalidChunk => f.write_str("malformed chunk structure"),
            ErrorKind::Lcw(e) => write!(f, "malformed LCW data: {e}"),
            ErrorKind::TooLarge(limit) => write!(f, "{limit} exceeds sanity limits"),
            ErrorKind::Video(e) => write!(f, "malformed video data: {e}"),
        }
    }
}

impl fmt::Display for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Limit::FrameSize => "frame size",
            Limit::Codebook => "codebook",
            Limit::Soundtrack => "soundtrack",
        })
    }
}

impl fmt::Display for VideoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            VideoError::MixedCodebookParts => "mixed CBP0/CBPZ parts",
            VideoError::PaletteSize => "palette size",
            VideoError::WrongPointerFormat => "pointer chunk for the other pixel format",
            VideoError::PointerTableSize => "pointer table size mismatch",
            VideoError::BlockIndexOutOfRange => "block index outside the codebook",
            VideoError::TruncatedPointerStream => "truncated pointer stream",
            VideoError::UnknownPointerCommand => "unknown pointer stream command",
            VideoError::PointerStreamOverrun => "pointer stream writes past the frame",
        })
    }
}

/// The LCW reason is part of the message, and [`Error::kind`] holds it, so
/// there is no [`source`](std::error::Error::source) (as with
/// [`std::io::Error`]).
impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kind and cause, to check they all read differently.
    fn every_kind() -> Vec<ErrorKind> {
        let mut kinds = vec![
            ErrorKind::NotVqa,
            ErrorKind::InvalidHeader,
            ErrorKind::Truncated,
            ErrorKind::InvalidChunk,
        ];
        for e in [LcwError::Truncated, LcwError::BadOffset, LcwError::TooLarge] {
            kinds.push(ErrorKind::Lcw(e));
        }
        for limit in [Limit::FrameSize, Limit::Codebook, Limit::Soundtrack] {
            kinds.push(ErrorKind::TooLarge(limit));
        }
        for e in [
            VideoError::MixedCodebookParts,
            VideoError::PaletteSize,
            VideoError::WrongPointerFormat,
            VideoError::PointerTableSize,
            VideoError::BlockIndexOutOfRange,
            VideoError::TruncatedPointerStream,
            VideoError::UnknownPointerCommand,
            VideoError::PointerStreamOverrun,
        ] {
            kinds.push(ErrorKind::Video(e));
        }
        // a new cause fails to build here until it is listed above
        for kind in &kinds {
            match kind {
                ErrorKind::NotVqa
                | ErrorKind::InvalidHeader
                | ErrorKind::Truncated
                | ErrorKind::InvalidChunk
                | ErrorKind::Lcw(LcwError::Truncated | LcwError::BadOffset | LcwError::TooLarge)
                | ErrorKind::TooLarge(Limit::FrameSize | Limit::Codebook | Limit::Soundtrack)
                | ErrorKind::Video(
                    VideoError::MixedCodebookParts
                    | VideoError::PaletteSize
                    | VideoError::WrongPointerFormat
                    | VideoError::PointerTableSize
                    | VideoError::BlockIndexOutOfRange
                    | VideoError::TruncatedPointerStream
                    | VideoError::UnknownPointerCommand
                    | VideoError::PointerStreamOverrun,
                ) => {}
            }
        }
        kinds
    }

    #[test]
    fn every_kind_has_its_own_message() {
        let kinds = every_kind();
        let messages: std::collections::HashSet<_> =
            kinds.iter().map(ToString::to_string).collect();
        assert_eq!(messages.len(), kinds.len());
        for message in &messages {
            assert!(!message.ends_with('.'), "{message}");
        }
    }

    #[test]
    fn display_names_the_place_it_knows() {
        let kind = ErrorKind::Video(VideoError::PointerTableSize);
        let e = Error::at(kind, Some(*b"VPT0"), 0x28);
        assert_eq!(
            e.to_string(),
            "malformed video data: pointer table size mismatch (VPT0 chunk at offset 0x28)"
        );
        assert_eq!(
            e.clone().in_frame(1).to_string(),
            "malformed video data: pointer table size mismatch \
             (frame 1, VPT0 chunk at offset 0x28)"
        );
        let e = Error::at(ErrorKind::InvalidChunk, None, 0x5e);
        assert_eq!(e.to_string(), "malformed chunk structure (at offset 0x5e)");
        let e = Error::from(ErrorKind::TooLarge(Limit::Codebook)).in_frame(7);
        assert_eq!(e.to_string(), "codebook exceeds sanity limits (frame 7)");
        assert_eq!(Error::from(ErrorKind::NotVqa).to_string(), "not a VQA file");
    }

    #[test]
    fn accessors_report_only_what_is_known() {
        let e = Error::from(ErrorKind::Truncated);
        assert_eq!((e.chunk(), e.offset(), e.frame()), (None, None, None));
        let e = Error::at(ErrorKind::Truncated, Some(*b"SND2"), 0).in_frame(0);
        assert_eq!(
            (e.kind(), e.chunk(), e.offset(), e.frame()),
            (ErrorKind::Truncated, Some(*b"SND2"), Some(0), Some(0))
        );
        assert_eq!(
            format!("{e:?}"),
            r#"Error { kind: Truncated, chunk: Some("SND2"), offset: Some(0), frame: Some(0) }"#
        );
    }

    #[test]
    fn or_in_places_only_errors_without_a_location() {
        let vqfr = crate::Chunks::new(b"VQFR\0\0\0\0").next().unwrap().unwrap();
        let e = Error::from(ErrorKind::TooLarge(Limit::Codebook))
            .in_frame(3)
            .or_in(&vqfr);
        assert_eq!(
            (e.chunk(), e.offset(), e.frame()),
            (Some(*b"VQFR"), Some(0), Some(3))
        );
        let inner = Error::at(ErrorKind::InvalidChunk, None, 20);
        assert_eq!(inner.clone().or_in(&vqfr), inner);
    }

    #[test]
    fn stays_small() {
        assert!(size_of::<Error>() <= 24);
        assert_eq!(size_of::<ErrorKind>(), 2);
        assert_eq!(size_of::<Result<(), ErrorKind>>(), 2);
    }
}
