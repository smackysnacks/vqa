//! The movie header (the VQHD chunk) and the frame index entries (the FINF
//! chunk).

use crate::error::Error;

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
    /// Fails unless `number` is 1, 2 or 3.
    fn try_from(number: u16) -> Result<VQAVersion, Error> {
        match number {
            1 => Ok(VQAVersion::One),
            2 => Ok(VQAVersion::Two),
            3 => Ok(VQAVersion::Three),
            _ => Err(Error::Parse),
        }
    }
}

/// The fixed 42-byte `VQHD` header describing the whole movie.
///
/// v1 movies leave several sound fields zeroed; the [`sample_rate`],
/// [`num_channels`], and [`bit_depth`] helpers apply the documented
/// fallbacks, so prefer them over reading `freq`, `channels`, and `bits`
/// directly.
///
/// [`sample_rate`]: VQAHeader::sample_rate
/// [`num_channels`]: VQAHeader::num_channels
/// [`bit_depth`]: VQAHeader::bit_depth
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VQAHeader {
    /// VQA version number
    pub version: VQAVersion,
    /// Flag bits. Bit 0 marks a soundtrack (see [`has_sound`]); the HiColor
    /// movies seen so far also set bits 2-4, whose meaning is unknown.
    ///
    /// [`has_sound`]: VQAHeader::has_sound
    pub flags: u16,
    /// Number of frames
    pub num_frames: u16,
    /// Movie width (pixels)
    pub width: u16,
    /// Movie height (pixels)
    pub height: u16,
    /// Width of each image block (pixels)
    pub block_width: u8,
    /// Height of each image block (pixels)
    pub block_height: u8,
    /// Frame rate of the VQA
    pub frame_rate: u8,
    /// How many images use the same lookup table
    pub cbparts: u8,
    /// Max number of colors used in VQA
    pub colors: u16,
    /// Max number of image blocks
    pub maxblocks: u16,
    /// Always 0?
    pub unk1: u32,
    /// Some kind of size?
    pub unk2: u16,
    /// Sound sampling frequency
    pub freq: u16,
    /// Number of sound channels
    pub channels: u8,
    /// Sound resolution
    pub bits: u8,
    /// Always 0?
    pub unk3: u32,
    /// 0 in old VQAs, 4 in HiColor VQAs?
    pub unk4: u16,
    /// 0 in old VQAs, CBFZ size in HiColor
    pub max_cbfz_size: u32,
    /// Always 0?
    pub unk5: u32,
}

/// The size of the `VQHD` payload.
const HEADER_LEN: usize = 42;

impl VQAHeader {
    /// Parse the payload of a `VQHD` chunk, all of whose fields are
    /// little-endian.
    ///
    /// # Errors
    ///
    /// Fails unless `vqhd` is exactly 42 bytes long and holds version 1, 2
    /// or 3.
    pub fn parse(vqhd: &[u8]) -> Result<VQAHeader, Error> {
        let vqhd: &[u8; HEADER_LEN] = vqhd.try_into().map_err(|_| Error::Parse)?;
        let u16_at = |at: usize| u16::from_le_bytes([vqhd[at], vqhd[at + 1]]);
        let u32_at =
            |at: usize| u32::from_le_bytes(vqhd[at..at + 4].try_into().expect("four bytes"));

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
            unk1: u32_at(18),
            unk2: u16_at(22),
            freq: u16_at(24),
            channels: vqhd[26],
            bits: vqhd[27],
            unk3: u32_at(28),
            unk4: u16_at(32),
            max_cbfz_size: u32_at(34),
            unk5: u32_at(38),
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
