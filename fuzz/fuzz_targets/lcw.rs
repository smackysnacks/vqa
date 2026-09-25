#![no_main]

use libfuzzer_sys::fuzz_target;
use vqa::lcw::{self, LcwError, Mode};

fuzz_target!(|data: &[u8]| {
    // Decompression must fail cleanly on arbitrary input in every mode,
    // never allocate beyond the caller's cap, and agree exactly - output
    // and error alike - with the straightforward decoder below.
    for max_out in [1 << 16, 64] {
        for mode in [Mode::Absolute, Mode::Relative] {
            assert_eq!(
                lcw::decompress_with(data, mode, max_out),
                reference(data, mode, max_out)
            );
        }
        let _ = lcw::decompress(data, max_out);
    }
});

/// The original byte-at-a-time decoder, kept as the oracle for the
/// optimized one: every copy goes through `Vec::push`.
fn reference(src: &[u8], mode: Mode, max_out: usize) -> Result<Vec<u8>, LcwError> {
    let mut out = Vec::new();
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
            copy_back(&mut out, offset, count, max_out)?;
        } else if cmd & 0x40 == 0 {
            // 0b10cc_cccc: copy count literal bytes from the source
            let count = usize::from(cmd & 0x3f);
            let literal = src.get(sp..sp + count).ok_or(LcwError::Truncated)?;
            sp += count;
            if out.len() + count > max_out {
                return Err(LcwError::TooLarge);
            }
            out.extend_from_slice(literal);
        } else if cmd == 0xfe {
            // 0xFE C C V: write byte V count times
            let count = usize::from(read_u16(src, sp)?);
            let color = *src.get(sp + 2).ok_or(LcwError::Truncated)?;
            sp += 3;
            if out.len() + count > max_out {
                return Err(LcwError::TooLarge);
            }
            out.resize(out.len() + count, color);
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
                    Mode::Absolute => out.len().checked_sub(pos).ok_or(LcwError::BadOffset)?,
                    Mode::Relative => pos,
                };
                copy_back(&mut out, offset, count, max_out)?;
            }
        }
    }

    Ok(out)
}

fn read_u16(src: &[u8], sp: usize) -> Result<u16, LcwError> {
    let bytes = src.get(sp..sp + 2).ok_or(LcwError::Truncated)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// Append `count` bytes read from `offset` bytes behind the write position,
/// one at a time - copies may overlap the write position, RLE-style.
fn copy_back(
    out: &mut Vec<u8>,
    offset: usize,
    count: usize,
    max_out: usize,
) -> Result<(), LcwError> {
    if offset == 0 || offset > out.len() {
        return Err(LcwError::BadOffset);
    }
    if out.len() + count > max_out {
        return Err(LcwError::TooLarge);
    }
    for pos in (out.len() - offset..).take(count) {
        let byte = out[pos];
        out.push(byte);
    }
    Ok(())
}
