//! The movie header (the VQHD chunk) and the frame index entries (the FINF
//! chunk).

use crate::error::{Error, ErrorKind};

/// The format version stored in the header.
///
/// The version alone does not imply the pixel format: most HiColor movies
/// are v3, but some v2 movies (Dune 2000, Blade Runner) are HiColor too -
/// check [`VQAHeader::is_hicolor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VQAVersion {
    /// Version 1, used only in The Legend of Kyrandia III.
    One = 1,
    /// Version 2, used in C&C, Red Alert, Lands of Lore II, Dune 2000, and
    /// Blade Runner.
    Two = 2,
    /// Version 3, used in the later HiColor games (Tiberian Sun, Lands of
    /// Lore III, Nox).
    Three = 3,
}

impl From<VQAVersion> for u16 {
    /// The version number as stored in the header.
    fn from(version: VQAVersion) -> u16 {
        version as u16
    }
}

impl TryFrom<u16> for VQAVersion {
    type Error = Error;

    /// The version stored in the header as `number`.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidHeader`], with no location, unless `number` is 1,
    /// 2 or 3.
    fn try_from(number: u16) -> Result<VQAVersion, Error> {
        match number {
            1 => Ok(VQAVersion::One),
            2 => Ok(VQAVersion::Two),
            3 => Ok(VQAVersion::Three),
            _ => Err(ErrorKind::InvalidHeader.into()),
        }
    }
}

/// The fixed 42-byte `VQHD` header describing the whole movie.
///
/// The fields follow Westwood's own `VQAHeader` (`VQAFILE.H` in the VQA
/// library in EA's GPL release of the Red Alert source), named in
/// parentheses below, and hold the values as stored. v1 movies leave
/// several sound fields zeroed; the [`sample_rate`], [`num_channels`], and
/// [`bit_depth`] helpers apply the documented fallbacks, so prefer them over
/// reading `freq`, `channels`, and `bits` directly.
///
/// [`sample_rate`]: VQAHeader::sample_rate
/// [`num_channels`]: VQAHeader::num_channels
/// [`bit_depth`]: VQAHeader::bit_depth
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VQAHeader {
    /// The format version (`Version`).
    pub version: VQAVersion,
    /// Flag bits (`Flags`). Bit 0 marks a soundtrack (see [`has_sound`]),
    /// and bit 1 an alternate one, which no movie seen so far has. The
    /// HiColor movies seen so far also set some of bits 2-4 (Blade Runner's
    /// bits 2 and 4, the others all three), whose meaning is unknown.
    ///
    /// [`has_sound`]: VQAHeader::has_sound
    pub flags: u16,
    /// The number of frames (`Frames`).
    pub num_frames: u16,
    /// The movie's width in pixels (`ImageWidth`).
    pub width: u16,
    /// The movie's height in pixels (`ImageHeight`).
    pub height: u16,
    /// The width of each image block in pixels (`BlockWidth`).
    pub block_width: u8,
    /// The height of each image block in pixels (`BlockHeight`).
    pub block_height: u8,
    /// Frames per second (`FPS`).
    pub frame_rate: u8,
    /// Frames per codebook (`Groupsize`): how many frames each bring a part
    /// of the next codebook. 0 in movies whose codebooks come whole, and in
    /// those whose CINF chunk schedules them
    /// ([`VQA::codebook_starts`](crate::VQA::codebook_starts)).
    pub cbparts: u8,
    /// The number of colors solid-color blocks use (`Num1Colors`); 0 in
    /// HiColor movies (see [`is_hicolor`]).
    ///
    /// [`is_hicolor`]: VQAHeader::is_hicolor
    pub colors: u16,
    /// The number of codebook entries (`CBentries`), the most a codebook
    /// holds; the decoder takes 0 to mean 0xff00.
    pub maxblocks: u16,
    /// Where to draw the frames (`Xpos`): the left edge, or 0xffff
    /// (Westwood's -1) to center them. 0 in every movie seen so far except
    /// Blade Runner's overlays, which the game places by it.
    pub x_pos: u16,
    /// Where to draw the frames (`Ypos`): the top edge, or 0xffff to center
    /// them; see [`x_pos`](VQAHeader::x_pos).
    pub y_pos: u16,
    /// The size of the largest frame, in Westwood's naming (`MaxFramesize`).
    /// What encoders store varies: v2 HiColor movies (Dune 2000, Blade
    /// Runner) store their largest VPRZ chunk's size, while 8-bit and v3
    /// movies store values smaller than their frame chunks, whose meaning is
    /// unknown.
    pub max_frame_size: u16,
    /// The sound sampling rate in Hz (`SampleRate`); see
    /// [`sample_rate`](VQAHeader::sample_rate).
    pub freq: u16,
    /// The number of sound channels (`Channels`); see
    /// [`num_channels`](VQAHeader::num_channels).
    pub channels: u8,
    /// The sound resolution in bits (`BitsPerSample`); see
    /// [`bit_depth`](VQAHeader::bit_depth).
    pub bits: u8,
    /// The alternate soundtrack's sampling rate in Hz (`AltSampleRate`).
    /// Westwood's player could switch to an alternate soundtrack, carried
    /// in SNA? chunks, which this crate doesn't decode; the alternate
    /// fields are 0 in every movie seen so far.
    pub alt_freq: u16,
    /// The alternate soundtrack's channels (`AltChannels`).
    pub alt_channels: u8,
    /// The alternate soundtrack's sound resolution in bits
    /// (`AltBitsPerSample`).
    pub alt_bits: u8,
    /// Five words Westwood reserved (`FutureUse`). Later encoders use some
    /// of them: HiColor movies store 4 in the first, and HiColor, Lands of
    /// Lore and some Red Alert movies store their largest compressed
    /// codebook's size in the next two
    /// ([`max_cbfz_size`](VQAHeader::max_cbfz_size)).
    pub future_use: [u16; 5],
}

/// The size of the `VQHD` payload.
const HEADER_LEN: usize = 42;

impl VQAHeader {
    /// Parse the payload of a `VQHD` chunk, all of whose fields are
    /// little-endian.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidHeader`], with no location, unless `vqhd` is
    /// exactly 42 bytes long and holds version 1, 2 or 3.
    pub fn parse(vqhd: &[u8]) -> Result<VQAHeader, Error> {
        let vqhd: &[u8; HEADER_LEN] = vqhd
            .try_into()
            .map_err(|_| Error::from(ErrorKind::InvalidHeader))?;
        let u16_at = |at: usize| u16::from_le_bytes([vqhd[at], vqhd[at + 1]]);

        Ok(VQAHeader {
            version: VQAVersion::try_from(u16_at(0))?,
            flags: u16_at(2),
            num_frames: u16_at(4),
            width: u16_at(6),
            height: u16_at(8),
            block_width: vqhd[10],
            block_height: vqhd[11],
            frame_rate: vqhd[12],
            cbparts: vqhd[13],
            colors: u16_at(14),
            maxblocks: u16_at(16),
            x_pos: u16_at(18),
            y_pos: u16_at(20),
            max_frame_size: u16_at(22),
            freq: u16_at(24),
            channels: vqhd[26],
            bits: vqhd[27],
            alt_freq: u16_at(28),
            alt_channels: vqhd[30],
            alt_bits: vqhd[31],
            future_use: [32, 34, 36, 38, 40].map(u16_at),
        })
    }

    /// The sound sampling rate in Hz, applying the documented v1 fallback
    /// (a stored 0 means 22050 Hz).
    pub fn sample_rate(&self) -> u32 {
        match self.freq {
            0 => 22050,
            freq => u32::from(freq),
        }
    }

    /// The number of sound channels (a stored 0 means mono).
    pub fn num_channels(&self) -> u8 {
        match self.channels {
            0 => 1,
            channels => channels,
        }
    }

    /// The sound resolution in bits (a stored 0 means 8-bit).
    pub fn bit_depth(&self) -> u8 {
        match self.bits {
            0 => 8,
            bits => bits,
        }
    }

    /// The size of the movie's largest compressed codebook, a CBFZ chunk or
    /// CBPZ parts joined, where the encoder stored it: in the second and
    /// third [`future_use`](VQAHeader::future_use) words, low word first.
    /// HiColor, Lands of Lore and some Red Alert movies store it; the other
    /// movies seen so far store 0.
    pub fn max_cbfz_size(&self) -> u32 {
        u32::from(self.future_use[1]) | u32::from(self.future_use[2]) << 16
    }

    /// Whether the movie carries a soundtrack (bit 0 of `flags`).
    pub fn has_sound(&self) -> bool {
        self.flags & 1 != 0
    }

    /// Whether the movie is HiColor (15-bit pixels) rather than 8-bit
    /// palettized. HiColor movies store 0 in the `colors` field.
    pub fn is_hicolor(&self) -> bool {
        self.colors == 0
    }
}

/// Position of one frame's data, decoded from a FINF entry.
///
/// A stored FINF entry holds flags in its top four bits (31 a key frame, 30
/// a new palette, 29 a sync point) and the frame's position in 16-bit words
/// in the other 28; `offset` is that position decoded to an absolute byte
/// offset of the frame's data (its SND? chunk when the movie has sound, its
/// VQFR chunk otherwise).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameInfo {
    /// Absolute byte offset of the frame's data from the start of the file.
    pub offset: u32,
    /// Whether the frame carries a new palette (stored bit 30).
    pub has_palette: bool,
}

const FINF_PALETTE_FLAG: u32 = 0x4000_0000;
/// The bits below the flags (Westwood's `VQAFINF_OFFSET`)
const FINF_OFFSET: u32 = 0x0fff_ffff;

impl FrameInfo {
    /// Decode one FINF entry, a little-endian `u32` in the chunk's payload:
    ///
    /// ```
    /// # let finf = [0u8; 8];
    /// use vqa::FrameInfo;
    ///
    /// let (entries, _) = finf.as_chunks::<4>();
    /// let index: Vec<FrameInfo> = entries
    ///     .iter()
    ///     .map(|&entry| FrameInfo::from_raw(u32::from_le_bytes(entry)))
    ///     .collect();
    /// ```
    ///
    /// [`VQA::frame_index`](crate::VQA::frame_index) holds the whole index,
    /// decoded.
    pub const fn from_raw(entry: u32) -> FrameInfo {
        FrameInfo {
            offset: (entry & FINF_OFFSET) * 2,
            has_palette: entry & FINF_PALETTE_FLAG != 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_convert_to_and_from_their_numbers() {
        for (number, version) in [
            (1, VQAVersion::One),
            (2, VQAVersion::Two),
            (3, VQAVersion::Three),
        ] {
            assert_eq!(VQAVersion::try_from(number), Ok(version));
            assert_eq!(u16::from(version), number);
        }
        assert!(VQAVersion::try_from(0).is_err());
        assert!(VQAVersion::try_from(4).is_err());
    }

    #[test]
    fn header_payload_must_be_42_bytes() {
        let mut vqhd = [0u8; 42];
        vqhd[0] = 2;
        assert!(VQAHeader::parse(&vqhd).is_ok());
        assert!(VQAHeader::parse(&vqhd[..41]).is_err());
        assert!(VQAHeader::parse(&[&vqhd[..], &[0]].concat()).is_err());
    }

    #[test]
    fn frame_info_decodes_offset_and_palette_flag() {
        let info = FrameInfo::from_raw(0x4000_0000 | 150);
        assert_eq!((info.offset, info.has_palette), (300, true));
        // bits 28-31 are flags, not offset
        let info = FrameInfo::from_raw(0xb000_0000 | 100);
        assert_eq!((info.offset, info.has_palette), (200, false));
    }
}
