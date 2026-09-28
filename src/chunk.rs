//! Walking a movie's chunks. Every chunk is a 4-character ID, a big-endian
//! 32-bit payload size, the payload, and a pad byte after an odd-sized
//! payload, so that chunks start at even offsets.

use std::fmt;

use crate::error::{Error, ErrorKind};

/// The size of a chunk's header: the ID and the payload size.
const HEADER_LEN: usize = 8;

/// One chunk: its ID, where it starts, and its payload, borrowed from the
/// input.
///
/// Container chunks (VQFR, VQFK, VQFL and CINF) nest more chunks in their
/// payload; [`Chunk::sub_chunks`] walks them.
#[non_exhaustive]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Chunk<'a> {
    /// The chunk's 4-character ID, e.g. `*b"VQFR"`: uppercase ASCII letters
    /// and digits.
    pub id: [u8; 4],
    /// The byte offset of the chunk's 8-byte header, counted from the start
    /// of the input the walk began at: the file, for the chunks of
    /// [`VQA::chunks`](crate::VQA::chunks) and their sub-chunks.
    pub offset: usize,
    /// The payload, without the pad byte.
    pub data: &'a [u8],
}

impl<'a> Chunk<'a> {
    /// Walk the chunks nested in this one's payload, with offsets counted
    /// the same way as this chunk's.
    ///
    /// The payload is taken to be whole, so a sub-chunk running past its
    /// end is corrupt data rather than a cut-off file.
    pub fn sub_chunks(&self) -> Chunks<'a> {
        Chunks {
            input: self.data,
            offset: self.offset + HEADER_LEN,
            nested: true,
        }
    }
}

impl fmt::Debug for Chunk<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // the length rather than the payload, which runs to kilobytes
        f.debug_struct("Chunk")
            .field("id", &String::from_utf8_lossy(&self.id))
            .field("offset", &self.offset)
            .field("len", &self.data.len())
            .finish()
    }
}

/// Iterator over consecutive chunks, such as the body of a movie
/// ([`VQA::chunks`](crate::VQA::chunks)) or the payload of a container chunk
/// ([`Chunk::sub_chunks`]).
///
/// The pad byte after an odd-sized payload may be missing at the very end.
///
/// # Errors
///
/// It yields an error, and then stops, if the input doesn't split into
/// whole chunks:
///
/// - [`ErrorKind::InvalidChunk`] for a chunk ID that isn't four uppercase
///   ASCII letters or digits (the walk has lost its place), or a sub-chunk
///   running past the end of its container's payload
/// - [`ErrorKind::Truncated`] for a chunk running past the end of the
///   input, which is cut short
///
/// [`Error::offset`] is where the bad chunk starts, and [`Error::chunk`]
/// its ID, if all four bytes of a valid one are there.
#[derive(Debug, Clone)]
pub struct Chunks<'a> {
    input: &'a [u8],
    /// the offset of `input` from where the walk's offsets count
    offset: usize,
    /// whether `input` is a whole payload, where running past its end is
    /// corruption rather than a cut-off file
    nested: bool,
}

impl<'a> Chunks<'a> {
    /// Walk the chunks in `data`, counting offsets from its start.
    pub fn new(data: &'a [u8]) -> Chunks<'a> {
        Chunks::at(data, 0)
    }

    /// Walk the chunks in `data`, which starts `offset` bytes into the
    /// input the offsets count from.
    pub(crate) fn at(data: &'a [u8], offset: usize) -> Chunks<'a> {
        Chunks {
            input: data,
            offset,
            nested: false,
        }
    }

    /// Walk the chunks in `data`, a whole payload: [`Chunk::sub_chunks`]
    /// for a payload handed over without its chunk.
    pub(crate) fn payload(data: &'a [u8]) -> Chunks<'a> {
        Chunks {
            input: data,
            offset: 0,
            nested: true,
        }
    }

    fn next_chunk(&mut self) -> Result<Chunk<'a>, Error> {
        let input = self.input;
        if !is_chunk_id(&input[..input.len().min(4)]) {
            return Err(Error::at(ErrorKind::InvalidChunk, None, self.offset));
        }
        // past the end of a whole payload the data is corrupt; past the end
        // of the input the file is cut short
        let overrun = if self.nested {
            ErrorKind::InvalidChunk
        } else {
            ErrorKind::Truncated
        };
        let Some((header, rest)) = input.split_first_chunk::<HEADER_LEN>() else {
            let id = input.first_chunk::<4>().copied();
            return Err(Error::at(overrun, id, self.offset));
        };
        let (id, size) = header.split_first_chunk::<4>().expect("eight bytes");
        let size = u32::from_be_bytes(size.try_into().expect("four bytes"));
        let Some(data) = usize::try_from(size).ok().and_then(|size| rest.get(..size)) else {
            return Err(Error::at(overrun, Some(*id), self.offset));
        };
        let mut rest = &rest[data.len()..];
        if data.len() % 2 == 1 && !rest.is_empty() {
            rest = &rest[1..];
        }

        let chunk = Chunk {
            id: *id,
            offset: self.offset,
            data,
        };
        self.offset += input.len() - rest.len();
        self.input = rest;
        Ok(chunk)
    }
}

/// Whether `id` is made of what chunk IDs are made of: uppercase ASCII
/// letters and digits.
pub(crate) fn is_chunk_id(id: &[u8]) -> bool {
    id.iter()
        .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
}

impl<'a> Iterator for Chunks<'a> {
    type Item = Result<Chunk<'a>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.input.is_empty() {
            return None;
        }
        let chunk = self.next_chunk();
        if chunk.is_err() {
            self.input = &[];
        }
        Some(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A walked chunk's ID, offset and payload
    type Walked<'a> = ([u8; 4], usize, &'a [u8]);

    /// Every chunk in `input`, or the error.
    fn walk(input: &[u8]) -> Vec<Result<Walked<'_>, Error>> {
        Chunks::new(input)
            .map(|chunk| chunk.map(|c| (c.id, c.offset, c.data)))
            .collect()
    }

    #[test]
    fn walks_chunks_and_consumes_pad_bytes() {
        let input = b"SND2\x00\x00\x00\x04abcdSND1\x00\x00\x00\x03abc\x00VQFR\x00\x00\x00\x00";
        assert_eq!(
            walk(input),
            vec![
                Ok((*b"SND2", 0, &b"abcd"[..])),
                Ok((*b"SND1", 12, &b"abc"[..])),
                Ok((*b"VQFR", 24, &b""[..])),
            ]
        );
    }

    #[test]
    fn odd_sized_chunk_at_end_of_input_needs_no_pad_byte() {
        let chunks = walk(b"SND2\x00\x00\x00\x03abc");
        assert_eq!(chunks, vec![Ok((*b"SND2", 0, &b"abc"[..]))]);
    }

    /// An error's kind, chunk and offset.
    fn located(e: &Error) -> (ErrorKind, Option<[u8; 4]>, Option<usize>) {
        (e.kind(), e.chunk(), e.offset())
    }

    /// The error ending the walk of `input`.
    fn failure(chunks: Chunks<'_>) -> (ErrorKind, Option<[u8; 4]>, Option<usize>) {
        let errors: Vec<_> = chunks.filter_map(Result::err).collect();
        assert_eq!(errors.len(), 1);
        located(&errors[0])
    }

    #[test]
    fn rejects_non_chunk_ids_and_stops() {
        let invalid = (ErrorKind::InvalidChunk, None, Some(0));
        assert_eq!(failure(Chunks::new(b"lin \x00\x00\x00\x00")), invalid);
        assert_eq!(
            failure(Chunks::new(b"\x00\x01\x02\x03\x00\x00\x00\x00")),
            invalid
        );
        // a bad ID fails even when the input ends inside it
        assert_eq!(failure(Chunks::new(b"SNd")), invalid);
        assert_eq!(failure(Chunks::new(b"s")), invalid);
        // and it isn't taken for the end of a cut-off file
        let chunks: Vec<_> = Chunks::new(b"SND2\x00\x00\x00\x00snd2").collect();
        assert_eq!(chunks.len(), 2);
        assert_eq!(
            located(chunks[1].as_ref().unwrap_err()),
            (ErrorKind::InvalidChunk, None, Some(8))
        );
    }

    #[test]
    fn a_chunk_running_past_the_input_is_a_cut_off_file() {
        let cut = |id| (ErrorKind::Truncated, id, Some(0));
        assert_eq!(
            failure(Chunks::new(b"SND2\x00\x00\x00\x05abcd")),
            cut(Some(*b"SND2"))
        );
        assert_eq!(
            failure(Chunks::new(b"SND2\x00\x00\x00")),
            cut(Some(*b"SND2"))
        );
        assert_eq!(
            failure(Chunks::new(b"SND2\xff\xff\xff\xff")),
            cut(Some(*b"SND2"))
        );
        assert_eq!(failure(Chunks::new(b"SND")), cut(None));
        // the chunks before the bad one come out first
        let chunks = walk(b"SND2\x00\x00\x00\x00SND");
        assert_eq!(chunks[0], Ok((*b"SND2", 0, &b""[..])));
        assert_eq!(
            located(chunks[1].as_ref().unwrap_err()),
            (ErrorKind::Truncated, None, Some(8))
        );
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn a_sub_chunk_running_past_its_container_is_corrupt() {
        // a VQFR whose one sub-chunk claims 16 bytes but has 2
        let input = b"VQFR\x00\x00\x00\x0aVPT0\x00\x00\x00\x10ab";
        let vqfr = Chunks::new(input).next().unwrap().unwrap();
        assert_eq!(
            failure(vqfr.sub_chunks()),
            (ErrorKind::InvalidChunk, Some(*b"VPT0"), Some(8))
        );
        assert_eq!(
            failure(Chunks::payload(vqfr.data)),
            (ErrorKind::InvalidChunk, Some(*b"VPT0"), Some(0))
        );
        // the same bytes on their own read as a cut-off file
        assert_eq!(
            failure(Chunks::new(vqfr.data)),
            (ErrorKind::Truncated, Some(*b"VPT0"), Some(0))
        );
    }

    #[test]
    fn sub_chunks_count_offsets_from_the_same_origin() {
        let mut input = b"SND2\x00\x00\x00\x02abVQFR\x00\x00\x00\x12".to_vec();
        input.extend(b"CBF0\x00\x00\x00\x01x\x00VPT0\x00\x00\x00\x00");
        let chunks: Vec<_> = Chunks::at(&input, 100).map(Result::unwrap).collect();
        assert_eq!(chunks[1].offset, 110);
        let subs: Vec<_> = chunks[1].sub_chunks().map(Result::unwrap).collect();
        assert_eq!(
            (subs[0].id, subs[0].offset, subs[0].data),
            (*b"CBF0", 118, &b"x"[..])
        );
        assert_eq!((subs[1].id, subs[1].offset), (*b"VPT0", 128));
        // every offset points at the chunk's header in the input
        for chunk in chunks.iter().chain(&subs) {
            assert_eq!(&input[chunk.offset - 100..][..4], &chunk.id);
            let payload = chunk.offset - 100 + HEADER_LEN;
            assert_eq!(&input[payload..][..chunk.data.len()], chunk.data);
        }
    }

    #[test]
    fn debug_shows_the_length_rather_than_the_payload() {
        let chunk = Chunks::new(b"SND2\x00\x00\x00\x04abcd")
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(
            format!("{chunk:?}"),
            r#"Chunk { id: "SND2", offset: 0, len: 4 }"#
        );
    }
}
