//! LCW ("Format80") decompression, the scheme behind every `*Z` chunk in a
//! VQA file (CBFZ, CBPZ, CPLZ, VPTZ, VPRZ).
//!
//! The original variant addresses already-written output with offsets
//! absolute from the start of the buffer, capping the output at 64 KiB. The
//! HiColor-era files add a "relative" variant whose long copy commands
//! address backwards from the write position instead, signalled by a NUL
//! byte in front of the stream.

use std::fmt;

/// How the long copy commands (`0xC0..=0xFD` and `0xFF`) address the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Offsets count from the start of the output buffer (the original scheme).
    Absolute,
    /// Offsets count backwards from the write position (the HiColor scheme).
    Relative,
}

/// Errors produced by [`decompress`] on malformed streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LcwError {
    /// The stream ended in the middle of a command.
    Truncated,
    /// A copy command referenced data outside what has been written so far.
    BadOffset,
    /// The output would exceed the caller's size limit.
    TooLarge,
}

impl fmt::Display for LcwError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LcwError::Truncated => write!(f, "stream ended in the middle of a command"),
            LcwError::BadOffset => write!(f, "copy command references unwritten data"),
            LcwError::TooLarge => write!(f, "output exceeds the size limit"),
        }
    }
}

impl std::error::Error for LcwError {}

/// Decompress an LCW stream, auto-detecting the variant: a leading NUL byte
/// selects [`Mode::Relative`] (the HiColor convention - no valid absolute
/// stream starts with 0x00), anything else [`Mode::Absolute`].
///
/// `max_out` caps the output size so malformed data cannot demand unbounded
/// allocations; pass the expected decompressed size.
pub fn decompress(src: &[u8], max_out: usize) -> Result<Vec<u8>, LcwError> {
    match src.split_first() {
        Some((0, rest)) => decompress_with(rest, Mode::Relative, max_out),
        _ => decompress_with(src, Mode::Absolute, max_out),
    }
}

/// Decompress an LCW stream with an explicit offset [`Mode`].
pub fn decompress_with(src: &[u8], mode: Mode, max_out: usize) -> Result<Vec<u8>, LcwError> {
    let mut out = Output::new(src.len(), max_out);
    let mut sp = 0;

    // streams normally end with a 0x80 command; tolerate running off the end
    while let Some(&cmd) = src.get(sp) {
        sp += 1;

        if cmd == 0x80 {
            // "copy zero literal bytes" doubles as the end marker
            break;
        } else if cmd & 0x80 == 0 {
            // 0b0ccc_pppp P: copy count+3 bytes from pppp:P behind the
            // write position (relative in both variants)
            let count = usize::from(cmd >> 4) + 3;
            let offset = usize::from(cmd & 0x0f) << 8
                | usize::from(*src.get(sp).ok_or(LcwError::Truncated)?);
            sp += 1;
            out.copy_back(offset, count)?;
        } else if cmd & 0x40 == 0 {
            // 0b10cc_cccc: copy count literal bytes from the source
            let count = usize::from(cmd & 0x3f);
            out.literal(src, sp, count)?;
            sp += count;
        } else if cmd == 0xfe {
            // 0xFE C C V: write byte V count times
            let count = usize::from(read_u16(src, sp)?);
            let color = *src.get(sp + 2).ok_or(LcwError::Truncated)?;
            sp += 3;
            out.fill(color, count)?;
        } else {
            // 0b11cc_cccc P P: copy count+3 bytes from position P
            // 0xFF C C P P: copy count bytes from position P
            let (count, pos) = if cmd == 0xff {
                let count = usize::from(read_u16(src, sp)?);
                let pos = usize::from(read_u16(src, sp + 2)?);
                sp += 4;
                (count, pos)
            } else {
                let pos = usize::from(read_u16(src, sp)?);
                sp += 2;
                (usize::from(cmd & 0x3f) + 3, pos)
            };
            if count > 0 {
                let offset = match mode {
                    Mode::Absolute => out.len.checked_sub(pos).ok_or(LcwError::BadOffset)?,
                    Mode::Relative => pos,
                };
                out.copy_back(offset, count)?;
            }
        }
    }

    Ok(out.finish())
}

fn read_u16(src: &[u8], sp: usize) -> Result<u16, LcwError> {
    let bytes = src.get(sp..sp + 2).ok_or(LcwError::Truncated)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// Scratch space kept past the end of the output. Most copies and literals
/// are a handful of bytes, so rather than loop over a variable length they
/// move a fixed `WIDE` bytes - one unaligned 16-byte load and store - and
/// let the excess spill into the scratch space, where later writes cover it.
const WIDE: usize = 16;

/// The output being decompressed: `buf[..len]` is the data so far, the rest
/// zero-initialized scratch space for wide copies to spill into.
struct Output {
    buf: Vec<u8>,
    len: usize,
    max_out: usize,
}

impl Output {
    fn new(src_len: usize, max_out: usize) -> Output {
        // LCW output is typically 1-2x its input; start at 2x and grow
        let size = src_len.saturating_mul(2).min(max_out);
        Output {
            buf: vec![0; size + WIDE],
            len: 0,
            max_out,
        }
    }

    /// Check that `count` more bytes fit under the cap, and make room for
    /// them plus the scratch space.
    #[inline(always)]
    fn reserve(&mut self, count: usize) -> Result<(), LcwError> {
        if self.len + count > self.max_out {
            return Err(LcwError::TooLarge);
        }
        if self.len + count + WIDE > self.buf.len() {
            self.grow(self.len + count);
        }
        Ok(())
    }

    /// Grow to hold `needed` bytes plus the scratch space, at least doubling
    /// so appends stay amortized O(1).
    #[cold]
    fn grow(&mut self, needed: usize) {
        let size = needed
            .max(self.buf.len().saturating_mul(2))
            .min(self.max_out);
        self.buf.resize(size + WIDE, 0);
    }

    /// Append the `count` literal bytes at `src[sp..]`.
    #[inline(always)]
    fn literal(&mut self, src: &[u8], sp: usize, count: usize) -> Result<(), LcwError> {
        if count <= WIDE && sp + WIDE <= src.len() {
            // short, and clear of the end of the stream: move WIDE bytes
            self.reserve(count)?;
            self.buf[self.len..self.len + WIDE].copy_from_slice(&src[sp..sp + WIDE]);
        } else {
            let literal = src.get(sp..sp + count).ok_or(LcwError::Truncated)?;
            self.reserve(count)?;
            self.buf[self.len..self.len + count].copy_from_slice(literal);
        }
        self.len += count;
        Ok(())
    }

    /// Append `count` copies of `byte`.
    #[inline(always)]
    fn fill(&mut self, byte: u8, count: usize) -> Result<(), LcwError> {
        self.reserve(count)?;
        self.buf[self.len..self.len + count].fill(byte);
        self.len += count;
        Ok(())
    }

    /// Append `count` bytes read from `offset` bytes behind the write
    /// position. Copies may overlap the write position, RLE-style.
    #[inline(always)]
    fn copy_back(&mut self, offset: usize, count: usize) -> Result<(), LcwError> {
        if offset == 0 || offset > self.len {
            return Err(LcwError::BadOffset);
        }
        self.reserve(count)?;
        let (from, to) = (self.len - offset, self.len);
        if offset >= WIDE && count <= WIDE {
            // short, and the source ends before the write position: move
            // WIDE bytes
            self.buf.copy_within(from..from + WIDE, to);
        } else if offset >= count {
            self.buf.copy_within(from..from + count, to);
        } else {
            // the source runs into the bytes being written; copy one at a
            // time so the repeating pattern propagates
            for i in 0..count {
                self.buf[to + i] = self.buf[from + i];
            }
        }
        self.len += count;
        Ok(())
    }

    fn finish(mut self) -> Vec<u8> {
        self.buf.truncate(self.len);
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_literal_bytes() {
        assert_eq!(decompress(b"\x82ab\x80", 64).unwrap(), b"ab");
    }

    #[test]
    fn tolerates_missing_end_marker() {
        assert_eq!(decompress(b"\x82ab", 64).unwrap(), b"ab");
    }

    #[test]
    fn short_copy_references_recent_output() {
        // literal "abc", then copy 3 bytes from 3 behind the write position
        assert_eq!(decompress(b"\x83abc\x00\x03\x80", 64).unwrap(), b"abcabc");
    }

    #[test]
    fn short_copy_with_offset_one_repeats_last_byte() {
        // count bits 0b101 -> 5+3 = 8 copies of 'a'
        assert_eq!(decompress(b"\x81a\x50\x01\x80", 64).unwrap(), b"aaaaaaaaa");
    }

    #[test]
    fn fill_writes_color_count_times() {
        assert_eq!(decompress(b"\xfe\x05\x00A\x80", 64).unwrap(), b"AAAAA");
    }

    #[test]
    fn long_copy_absolute_addresses_from_start() {
        // literal "abcd", then copy 3 bytes from absolute position 1
        assert_eq!(
            decompress(b"\x84abcd\xc0\x01\x00\x80", 64).unwrap(),
            b"abcdbcd"
        );
    }

    #[test]
    fn long_copy_relative_addresses_from_write_position() {
        // literal "abcd", then copy 3 bytes from 4 behind the write position;
        // the leading NUL selects the relative variant
        assert_eq!(
            decompress(b"\x00\x84abcd\xc0\x04\x00\x80", 64).unwrap(),
            b"abcdabc"
        );
    }

    #[test]
    fn very_long_copy_takes_count_and_position_words() {
        let out = decompress(b"\x82ab\xff\x06\x00\x00\x00\x80", 64).unwrap();
        assert_eq!(out, b"abababab");
    }

    #[test]
    fn errors_on_backreference_before_start() {
        assert_eq!(
            decompress(b"\x81a\x00\x05\x80", 64),
            Err(LcwError::BadOffset)
        );
        // absolute position beyond what has been written
        assert_eq!(
            decompress(b"\x81a\xc0\x02\x00\x80", 64),
            Err(LcwError::BadOffset)
        );
    }

    #[test]
    fn errors_on_truncated_command() {
        assert_eq!(decompress(b"\x85ab", 64), Err(LcwError::Truncated));
        assert_eq!(decompress(b"\xfe\x05", 64), Err(LcwError::Truncated));
        assert_eq!(decompress(b"\xff\x06\x00", 64), Err(LcwError::Truncated));
    }

    #[test]
    fn errors_when_output_exceeds_cap() {
        assert_eq!(
            decompress(b"\xfe\xff\xff\x41\x80", 64),
            Err(LcwError::TooLarge)
        );
    }

    #[test]
    fn far_short_copy_writes_exactly_count_bytes() {
        // 20 literal bytes, then 5 bytes from 20 behind - far enough for a
        // wide copy - then a literal that must land right after them
        let mut src = vec![0x80 | 20];
        src.extend(b"abcdefghijklmnopqrst");
        src.extend(b"\x20\x14\x82XY\x80");
        assert_eq!(
            decompress(&src, 64).unwrap(),
            b"abcdefghijklmnopqrstabcdeXY"
        );
    }

    #[test]
    fn short_literals_away_from_the_stream_end_copy_only_their_bytes() {
        // each literal has 16+ stream bytes after it, so all three take
        // the wide path
        let mut src = b"\x83abc\x90".to_vec();
        src.extend(b"0123456789ABCDEF");
        src.extend(b"\x81Z\x80");
        src.extend([0; 16]);
        assert_eq!(decompress(&src, 64).unwrap(), b"abc0123456789ABCDEFZ");
    }

    #[test]
    fn grows_past_the_initial_buffer_and_copies_from_it() {
        // a 5-byte stream expanding to exactly the cap
        assert_eq!(
            decompress(b"\xfe\xe8\x03A\x80", 1000).unwrap(),
            vec![b'A'; 1000]
        );

        // literal, 256-byte fill, then 4 bytes from absolute position 0
        let out = decompress(b"\x82ab\xfe\x00\x01A\xff\x04\x00\x00\x00\x80", 1024).unwrap();
        let mut expected = b"ab".to_vec();
        expected.extend([b'A'; 256]);
        expected.extend(b"abAA");
        assert_eq!(out, expected);
    }

    #[test]
    fn long_overlapping_copy_repeats_the_pattern() {
        // 40 bytes from 3 behind: longer than a wide move and overlapping
        // its own output
        let out = decompress(b"\x83abc\xff\x28\x00\x00\x00\x80", 64).unwrap();
        let mut expected = b"abc".repeat(15);
        expected.truncate(43);
        assert_eq!(out, expected);
    }

    #[test]
    fn errors_when_a_short_literal_exceeds_cap() {
        let mut src = b"\x83abc".to_vec();
        src.extend([0x80; 16]);
        assert_eq!(decompress(&src, 2), Err(LcwError::TooLarge));
    }
}
