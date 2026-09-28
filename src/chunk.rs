//! Walking a movie's chunks. Every chunk is a 4-character ID, a big-endian
//! 32-bit payload size, the payload, and a pad byte after an odd-sized
//! payload, so that chunks start at even offsets.

use std::fmt;

use crate::error::Error;

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
/// It yields an error and then stops if the input doesn't split into whole
/// chunks: a chunk ID that isn't four uppercase ASCII letters or digits (the
/// walk has lost its place), or a chunk running past the end of the input.
/// The pad byte after an odd-sized payload may be missing at the very end.
#[derive(Debug, Clone)]
pub struct Chunks<'a> {
    input: &'a [u8],
    /// the offset of `input` from where the walk's offsets count
    offset: usize,
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
        }
    }

    /// Walk the chunks in `data`, a whole payload: [`Chunk::sub_chunks`]
    /// for a payload handed over without its chunk.
    pub(crate) fn payload(data: &'a [u8]) -> Chunks<'a> {
        Chunks::at(data, 0)
    }

    /// The input not walked yet, and its offset.
    pub(crate) fn remaining(&self) -> (&'a [u8], usize) {
        (self.input, self.offset)
    }

    fn next_chunk(&mut self) -> Result<Chunk<'a>, Error> {
        let input = self.input;
        let id_bytes = &input[..input.len().min(4)];
        if !id_bytes
            .iter()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
        {
            return Err(Error::Parse);
        }
        let Some((header, rest)) = input.split_first_chunk::<HEADER_LEN>() else {
            return Err(Error::Parse);
        };
        let (id, size) = header.split_at(4);
        let size = u32::from_be_bytes(size.try_into().expect("four bytes"));
        let Some(data) = usize::try_from(size).ok().and_then(|size| rest.get(..size)) else {
            return Err(Error::Parse);
        };
        let mut rest = &rest[data.len()..];
        if data.len() % 2 == 1 && !rest.is_empty() {
            rest = &rest[1..];
        }

        let chunk = Chunk {
            id: id.try_into().expect("four bytes"),
            offset: self.offset,
            data,
        };
        self.offset += input.len() - rest.len();
        self.input = rest;
        Ok(chunk)
    }
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

    #[test]
    fn rejects_non_chunk_ids_and_stops() {
        assert_eq!(walk(b"lin \x00\x00\x00\x00"), vec![Err(Error::Parse)]);
        assert_eq!(
            walk(b"\x00\x01\x02\x03\x00\x00\x00\x00"),
            vec![Err(Error::Parse)]
        );
        // a bad ID fails even when the input ends inside it
        assert_eq!(walk(b"SN"), vec![Err(Error::Parse)]);
        assert_eq!(walk(b"s"), vec![Err(Error::Parse)]);
    }

    #[test]
    fn rejects_chunks_running_past_the_input() {
        assert_eq!(walk(b"SND2\x00\x00\x00\x05abcd"), vec![Err(Error::Parse)]);
        assert_eq!(walk(b"SND2\x00\x00\x00"), vec![Err(Error::Parse)]);
        assert_eq!(walk(b"SND2\xff\xff\xff\xff"), vec![Err(Error::Parse)]);
        // the chunks before the bad one come out first
        let chunks = walk(b"SND2\x00\x00\x00\x00SND");
        assert_eq!(chunks, vec![Ok((*b"SND2", 0, &b""[..])), Err(Error::Parse)]);
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
