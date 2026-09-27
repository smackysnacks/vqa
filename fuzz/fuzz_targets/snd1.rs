#![no_main]

use libfuzzer_sys::fuzz_target;
use vqa::audio::westwood;

fuzz_target!(|data: &[u8]| {
    // Any chunk decodes without panicking, agrees exactly with Westwood's
    // decoder as transliterated below, and appends after existing samples.
    let expected = reference(data);
    assert_eq!(westwood::decompress(data), expected);

    let mut samples = vec![1, 2, 3];
    westwood::decompress_into(data, &mut samples);
    assert_eq!(samples[..3], [1, 2, 3]);
    assert_eq!(samples[3..], expected);
});

const DELTA2: [i8; 4] = [-2, -1, 0, 1];
const DELTA4: [i8; 16] = [-9, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8];

/// `add dl,delta` followed by AudioUnzap's carry test: a positive delta
/// that carries out saturates at 0xFF, a negative one that doesn't at 0.
fn add_saturating(sample: u8, delta: i8) -> u8 {
    let (sum, carry) = sample.overflowing_add(delta as u8);
    match (delta < 0, carry) {
        (false, true) => 0xff,
        (true, false) => 0,
        _ => sum,
    }
}

/// Westwood's AudioUnzap (WINVQ/VQM32/AUDUNZAP.ASM in EA's GPL Red Alert
/// source), transliterated, with its player's cut of the output to
/// UnCompSize (LOADER.CPP). The original reads on past the end of the
/// chunk; here a command whose data isn't all there ends the chunk.
fn reference(chunk: &[u8]) -> Vec<u8> {
    if chunk.len() < 4 {
        return Vec::new();
    }
    let uncomp_size = usize::from(u16::from_le_bytes([chunk[0], chunk[1]]));
    let comp_size = usize::from(u16::from_le_bytes([chunk[2], chunk[3]]));
    let src = &chunk[4..];
    if uncomp_size == comp_size {
        return src[..uncomp_size.min(src.len())].to_vec();
    }

    let mut out = Vec::new();
    let mut previous: u8 = 0x80; // dl
    let mut count = uncomp_size as isize; // ecx
    let mut i = 0;
    while count > 0 {
        let Some(&code) = src.get(i) else {
            break;
        };
        i += 1;
        let (command, mut al) = (code >> 6, code & 0x3f);

        if command == 2 {
            if al & 0b0010_0000 != 0 {
                // shl al,3; sar al,3; add dl,al
                previous = previous.wrapping_add(((al << 3) as i8 >> 3) as u8);
                out.push(previous);
                count -= 1;
            } else {
                // al+1 raw samples; the last becomes 'previous'
                let n = usize::from(al) + 1;
                let Some(raw) = src.get(i..i + n) else {
                    break;
                };
                out.extend_from_slice(raw);
                i += n;
                count -= n as isize;
                previous = raw[n - 1];
            }
            continue;
        }

        al += 1; // the other codes use AL+1
        let n = usize::from(al);
        match command {
            1 => {
                let Some(bytes) = src.get(i..i + n) else {
                    break;
                };
                i += n;
                for &byte in bytes {
                    previous = add_saturating(previous, DELTA4[usize::from(byte & 0x0f)]);
                    let second = add_saturating(previous, DELTA4[usize::from(byte >> 4)]);
                    out.extend([previous, second]);
                    count -= 2;
                    previous = second;
                }
            }
            0 => {
                let Some(bytes) = src.get(i..i + n) else {
                    break;
                };
                i += n;
                for &byte in bytes {
                    for shift in [0, 2, 4, 6] {
                        previous = add_saturating(previous, DELTA2[usize::from(byte >> shift & 3)]);
                        out.push(previous);
                    }
                    count -= 4;
                }
            }
            _ => {
                out.extend(std::iter::repeat_n(previous, n));
                count -= n as isize;
            }
        }
    }
    out.truncate(uncomp_size);
    out
}
