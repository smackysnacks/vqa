//! Westwood's own ADPCM codec ("WS ADPCM") behind `SND1` sound chunks: mono
//! unsigned 8-bit samples, compressed to 2 or 4 bits each, or stored raw.
//!
//! Every chunk decodes on its own: the predicted sample restarts at 0x80
//! (silence) in each one, so unlike IMA ADPCM there is no state to carry.
//!
//! Written against Westwood's own decoder, `AudioUnzap` in `AUDUNZAP.ASM`
//! of the VQA library in EA's GPL release of the Red Alert source
//! (github.com/electronicarts/CnC_Red_Alert, `WINVQ/VQM32`).

/// The 4-bit deltas, indexed by nibble.
const DELTA4: [i16; 16] = [-9, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8];

/// Decompress one `SND1` chunk - its 4-byte header (little-endian output
/// size, then compressed size) and the data after it - into unsigned
/// 8-bit samples.
///
/// A chunk whose two sizes are equal is stored raw. Decoding is lenient,
/// like the original player's: a command running past the output size is
/// cut off at it, and one the rest of the chunk can't complete ends the
/// chunk, so a malformed chunk yields fewer samples rather than an
/// error.
pub fn decompress(chunk: &[u8]) -> Vec<u8> {
    let mut samples = Vec::new();
    decompress_into(chunk, &mut samples);
    samples
}

/// Like [`decompress`], but appends the samples to `out`.
pub fn decompress_into(chunk: &[u8], out: &mut Vec<u8>) {
    let Some((header, mut src)) = chunk.split_first_chunk::<4>() else {
        return;
    };
    let out_size = usize::from(u16::from_le_bytes([header[0], header[1]]));
    let in_size = usize::from(u16::from_le_bytes([header[2], header[3]]));
    if in_size == out_size {
        out.extend_from_slice(&src[..out_size.min(src.len())]);
        return;
    }

    out.reserve(out_size);
    let end = out.len() + out_size;
    let mut sample: u8 = 0x80;
    while out.len() < end {
        let Some((&command, rest)) = src.split_first() else {
            break;
        };
        src = rest;
        // 0bCCnn_nnnn: a two-bit code and a six-bit count
        let count = usize::from(command & 0x3f);
        match command >> 6 {
            // count+1 bytes of four 2-bit deltas each, -2..=1, low bits first
            0 => {
                let Some(bytes) = src.get(..count + 1) else {
                    break;
                };
                src = &src[count + 1..];
                for &byte in bytes {
                    for shift in [0, 2, 4, 6] {
                        sample = saturating(sample, i16::from(byte >> shift & 3) - 2);
                        out.push(sample);
                    }
                }
            }
            // count+1 bytes of two 4-bit deltas each, low nibble first
            1 => {
                let Some(bytes) = src.get(..count + 1) else {
                    break;
                };
                src = &src[count + 1..];
                for &byte in bytes {
                    for nibble in [byte & 0xf, byte >> 4] {
                        sample = saturating(sample, DELTA4[usize::from(nibble)]);
                        out.push(sample);
                    }
                }
            }
            // with bit 5 set, the low five bits are one signed delta, which
            // unlike the others wraps around rather than saturating
            2 if count & 0x20 != 0 => {
                sample = sample.wrapping_add_signed((command << 3) as i8 >> 3);
                out.push(sample);
            }
            // else count+1 raw samples, the last one becoming the prediction
            2 => {
                let Some(raw) = src.get(..count + 1) else {
                    break;
                };
                src = &src[count + 1..];
                out.extend_from_slice(raw);
                sample = raw[count];
            }
            // count+1 repeats of the current sample
            _ => out.resize(out.len() + count + 1, sample),
        }
    }
    // Westwood's player decodes a command's samples whole and plays the
    // chunk's first OutSize of them
    out.truncate(end);
}

/// `sample + delta`, saturating at 0 and 255.
fn saturating(sample: u8, delta: i16) -> u8 {
    (i16::from(sample) + delta).clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A chunk with the given output size, compressed size, and data.
    fn chunk(out_size: u16, in_size: u16, data: &[u8]) -> Vec<u8> {
        let mut chunk = out_size.to_le_bytes().to_vec();
        chunk.extend(in_size.to_le_bytes());
        chunk.extend(data);
        chunk
    }

    #[test]
    fn equal_sizes_mean_raw_samples() {
        assert_eq!(decompress(&chunk(3, 3, &[1, 2, 250])), [1, 2, 250]);
    }

    #[test]
    fn two_bit_deltas_come_low_bits_first() {
        // one byte, 0b11_10_01_00: deltas -2, -1, 0, +1 from 0x80
        let samples = decompress(&chunk(4, 2, &[0x00, 0b11_10_01_00]));
        assert_eq!(samples, [0x7e, 0x7d, 0x7d, 0x7e]);
    }

    #[test]
    fn four_bit_deltas_come_low_nibble_first() {
        // two bytes: nibbles 0 (-9), f (+8), 7 (-1), 8 (0)
        let samples = decompress(&chunk(4, 3, &[0x41, 0xf0, 0x87]));
        assert_eq!(samples, [0x77, 0x7f, 0x7e, 0x7e]);
    }

    #[test]
    fn big_deltas_are_five_bit_signed() {
        // 0b10_1_01111 is +15, 0b10_1_10000 is -16
        let samples = decompress(&chunk(2, 2 + 1, &[0xaf, 0xb0]));
        assert_eq!(samples, [0x8f, 0x7f]);
    }

    #[test]
    fn big_deltas_wrap_around() {
        // a raw 250 then +15, and a raw 5 then -16, wrapping in 8 bits
        let samples = decompress(&chunk(4, 5, &[0x80, 250, 0xaf, 0x80, 5, 0xb0]));
        assert_eq!(samples, [250, 9, 5, 245]);
    }

    #[test]
    fn raw_runs_set_the_prediction() {
        // 3 raw samples, then a +1 big delta from the last of them
        let samples = decompress(&chunk(4, 5, &[0x82, 10, 20, 30, 0xa1]));
        assert_eq!(samples, [10, 20, 30, 31]);
    }

    #[test]
    fn repeats_copy_the_current_sample() {
        // big delta +3, then 3 repeats of it
        assert_eq!(decompress(&chunk(4, 2, &[0xa3, 0xc2])), [0x83; 4]);
    }

    #[test]
    fn deltas_saturate_at_0_and_255() {
        // a raw 250, then 4-bit deltas +8 twice; a raw 5, then -9 twice
        let samples = decompress(&chunk(6, 8, &[0x80, 250, 0x40, 0xff, 0x80, 5, 0x40, 0x00]));
        assert_eq!(samples, [250, 255, 255, 5, 0, 0]);
    }

    #[test]
    fn decompress_into_appends_and_cuts_off_relative_to_its_start() {
        let mut samples = vec![9, 9];
        decompress_into(&chunk(3, 2, &[0x00, 0xff]), &mut samples);
        assert_eq!(samples, [9, 9, 0x81, 0x82, 0x83]);
        decompress_into(&chunk(2, 2, &[7, 8]), &mut samples);
        assert_eq!(samples, [9, 9, 0x81, 0x82, 0x83, 7, 8]);
    }

    #[test]
    fn every_command_missing_its_data_ends_the_chunk() {
        // a 2-bit group, a 4-bit group, and a raw run each short of data;
        // the samples before them stay
        assert_eq!(decompress(&chunk(5, 3, &[0xa1, 0x01, 0xff])), [0x81]);
        assert_eq!(decompress(&chunk(5, 3, &[0xa1, 0x41, 0x12])), [0x81]);
        assert_eq!(decompress(&chunk(5, 3, &[0xa1, 0x82, 1, 2])), [0x81]);
    }

    #[test]
    fn cuts_off_at_the_output_size_and_stops_at_the_end_of_the_data() {
        // a 2-bit group of 4 samples cut to the 3 the header asks for
        assert_eq!(decompress(&chunk(3, 2, &[0x00, 0xff])), [0x81, 0x82, 0x83]);
        // a run of 6 after one big delta, cut to 3 in all
        assert_eq!(decompress(&chunk(3, 2, &[0xa1, 0xc5])), [0x81; 3]);
        // a 4-bit group missing its data byte ends the chunk
        assert_eq!(decompress(&chunk(2, 1, &[0x40])), []);
        // no header at all
        assert_eq!(decompress(&[0, 0, 0]), []);
    }
}
