//! The IMA ADPCM decoder: 4 bits per sample, decoded to signed 16-bit PCM.

const STEP_TABLE: [u32; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
    494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272,
    2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
];

const INDEX_ADJUSTMENT: [i32; 16] = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8];

/// `DIFF[index][nibble]`: the signed change to the predicted sample when
/// `nibble` is decoded at step index `index`. Built with the reference
/// decoder's shift-and-add - `(2 * magnitude + 1) * step / 8` rounds
/// differently - so a table lookup replaces three data-dependent branches.
static DIFF: [[i32; 16]; 89] = {
    let mut table = [[0; 16]; 89];
    let mut index = 0;
    while index < 89 {
        let step = STEP_TABLE[index];
        let mut nibble = 0;
        while nibble < 16 {
            let mut diff = step >> 3;
            if nibble & 4 != 0 {
                diff += step;
            }
            if nibble & 2 != 0 {
                diff += step >> 1;
            }
            if nibble & 1 != 0 {
                diff += step >> 2;
            }
            table[index][nibble] = if nibble & 8 != 0 {
                -(diff as i32)
            } else {
                diff as i32
            };
            nibble += 1;
        }
        index += 1;
    }
    table
};

/// `NEXT_INDEX[index][nibble]`: the step index after decoding `nibble` at
/// step index `index`, already clamped to the step table.
static NEXT_INDEX: [[u8; 16]; 89] = {
    let mut table = [[0; 16]; 89];
    let mut index = 0;
    while index < 89 {
        let mut nibble = 0;
        while nibble < 16 {
            let next = index as i32 + INDEX_ADJUSTMENT[nibble];
            table[index][nibble] = if next < 0 {
                0
            } else if next > 88 {
                88
            } else {
                next as u8
            };
            nibble += 1;
        }
        index += 1;
    }
    table
};

/// Predictor state for one audio channel, carried across chunk boundaries.
///
/// Feed every chunk of a channel through [`decompress`] with the same state;
/// stereo streams need one state per channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecState {
    sample: i32,
    /// always a valid step table index, 0..=88
    index: usize,
}

impl CodecState {
    /// A fresh predictor state, for the start of a stream.
    pub fn new() -> CodecState {
        CodecState {
            sample: 0,
            index: 0,
        }
    }

    /// Decode one 4-bit sample, advancing the predictor.
    #[inline(always)]
    pub(crate) fn decode(&mut self, nibble: u8) -> i16 {
        let nibble = usize::from(nibble & 0xf);
        self.sample = (self.sample + DIFF[self.index][nibble]).clamp(-32768, 32767);
        self.index = usize::from(NEXT_INDEX[self.index][nibble]);
        self.sample as i16
    }

    /// Decode one byte: two samples, low nibble first.
    #[inline(always)]
    pub(crate) fn decode_byte(&mut self, byte: u8) -> [i16; 2] {
        [self.decode(byte), self.decode(byte >> 4)]
    }
}

impl Default for CodecState {
    fn default() -> Self {
        Self::new()
    }
}

/// Decompress IMA ADPCM data of a _single_ audio channel into signed 16-bit
/// PCM: two samples per input byte, low nibble first.
///
/// `state` carries the predictor across calls, so consecutive chunks of a
/// stream decode with the same state - and each channel needs its own.
pub fn decompress(state: &mut CodecState, input: &[u8]) -> Vec<i16> {
    let mut buffer = Vec::with_capacity(input.len() * 2);
    decompress_into(state, input, &mut buffer);
    buffer
}

/// Like [`decompress`], but appends the samples to `out`, so one buffer can
/// collect a whole stream.
pub fn decompress_into(state: &mut CodecState, input: &[u8], out: &mut Vec<i16>) {
    out.reserve(input.len() * 2);
    for &byte in input {
        out.extend_from_slice(&state.decode_byte(byte));
    }
}
