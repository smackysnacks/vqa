//! Pixel conversion from decoded frames to packed RGB888.
//!
//! HiColor conversion is pure lane-wise bit shuffling, so it runs as a
//! `fearless_simd` kernel, dispatched at run time to the best instruction
//! set the CPU has. Every SIMD operation must sit in an `#[inline(always)]`
//! function reached from `dispatch!`: closures and other out-of-line code
//! don't inherit the enabled target features, which turns each operation
//! into a function call.

use fearless_simd::prelude::*;
use fearless_simd::{Level, dispatch, u8x16, u16x16};

/// `RGB_SHUFFLE[block][channel]` interleaves three 16-lane planes (R, G, B)
/// into 48 bytes of RGB888, 16 bytes per `block`: each output byte of the
/// channel it holds takes that plane's lane for its pixel, and 0x80 - out
/// of range, so it shuffles in a zero - marks the bytes of the other two
/// channels.
const RGB_SHUFFLE: [[[u8; 16]; 3]; 3] = {
    let mut table = [[[0x80; 16]; 3]; 3];
    let mut byte = 0;
    while byte < 48 {
        table[byte / 16][byte % 3][byte % 16] = (byte / 3) as u8;
        byte += 1;
    }
    table
};

/// Expand a 5-bit channel to 8 bits, repeating its top bits in the bottom
/// so 0 and 31 map to 0 and 255.
#[inline(always)]
fn scale5(v: u16) -> u8 {
    (v << 3 | v >> 2) as u8
}

/// Convert `0rrrrrgg gggbbbbb` pixels (the top bit ignored) to RGB888.
/// `out` holds three bytes per pixel.
pub(crate) fn hicolor_to_rgb888(pixels: &[u16], out: &mut [u8]) {
    hicolor_to_rgb888_at(Level::new(), pixels, out);
}

fn hicolor_to_rgb888_at(level: Level, pixels: &[u16], out: &mut [u8]) {
    if level.is_fallback() {
        hicolor_scalar(pixels, out);
    } else {
        dispatch!(level, simd => hicolor_kernel(simd, pixels, out));
    }
}

#[inline(always)]
fn hicolor_scalar(pixels: &[u16], out: &mut [u8]) {
    for (rgb, &p) in out.as_chunks_mut::<3>().0.iter_mut().zip(pixels) {
        *rgb = [scale5(p >> 10 & 31), scale5(p >> 5 & 31), scale5(p & 31)];
    }
}

/// 16 pixels per iteration: expand the channels on 16-bit lanes, narrow
/// each to a byte plane, then shuffle the planes together.
#[inline(always)]
fn hicolor_kernel<S: Simd>(simd: S, pixels: &[u16], out: &mut [u8]) {
    let (src, src_rest) = pixels.as_chunks::<16>();
    let (dst, dst_rest) = out.as_chunks_mut::<48>();
    for (pixels, out) in src.iter().zip(dst) {
        let p = u16x16::from_slice(simd, pixels);
        // scale5 per channel: the channel's bits moved to 3..=7, and its
        // top three bits to 0..=2
        let r = to_bytes(((p >> 7) & 0xf8) | ((p >> 12) & 0x07));
        let g = to_bytes(((p >> 2) & 0xf8) | ((p >> 7) & 0x07));
        let b = to_bytes(((p << 3) & 0xf8) | ((p >> 2) & 0x07));
        for (out, [mr, mg, mb]) in out.as_chunks_mut::<16>().0.iter_mut().zip(&RGB_SHUFFLE) {
            let rgb = r.swizzle_dyn_precise(u8x16::from_slice(simd, mr))
                | g.swizzle_dyn_precise(u8x16::from_slice(simd, mg))
                | b.swizzle_dyn_precise(u8x16::from_slice(simd, mb));
            rgb.store_slice(out);
        }
    }
    hicolor_scalar(src_rest, dst_rest);
}

/// Narrow 16 lanes holding byte values to bytes.
#[inline(always)]
fn to_bytes<S: Simd>(v: u16x16<S>) -> u8x16<S> {
    let (lo, hi) = v.split();
    lo.narrow(hi)
}

/// Convert palette indices to RGB888; indices with no palette entry come
/// out black. `out` holds three bytes per pixel.
pub(crate) fn indexed_to_rgb888(pixels: &[u8], palette: &[[u8; 3]], out: &mut [u8]) {
    // a full table makes every index valid, with missing entries black
    let mut table = [[0; 3]; 256];
    for (entry, &rgb) in table.iter_mut().zip(palette) {
        *entry = rgb;
    }
    for (rgb, &index) in out.as_chunks_mut::<3>().0.iter_mut().zip(pixels) {
        *rgb = table[usize::from(index)];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The conversion as first written, one pixel at a time.
    fn reference(pixels: &[u16]) -> Vec<u8> {
        pixels
            .iter()
            .flat_map(|&p| {
                let scale = |v: u16| (v << 3 | v >> 2) as u8;
                [scale(p >> 10 & 31), scale(p >> 5 & 31), scale(p & 31)]
            })
            .collect()
    }

    #[test]
    fn hicolor_kernel_matches_reference_on_every_pixel_value() {
        // every 16-bit value, alpha bit included, at the detected level and
        // the baseline one (different instruction sets on x86)
        let pixels: Vec<u16> = (0..=u16::MAX).collect();
        let expected = reference(&pixels);
        for level in [Level::new(), Level::baseline()] {
            let mut out = vec![0; pixels.len() * 3];
            hicolor_to_rgb888_at(level, &pixels, &mut out);
            assert_eq!(out, expected, "{level:?}");
        }
    }

    #[test]
    fn hicolor_kernel_handles_lengths_off_the_vector_width() {
        let pixels: Vec<u16> = (0..50).map(|i| i * 1311 + 7).collect();
        for len in 0..=pixels.len() {
            let mut out = vec![0xaa; len * 3];
            hicolor_to_rgb888(&pixels[..len], &mut out);
            assert_eq!(out, reference(&pixels[..len]), "{len} pixels");
        }
    }

    #[test]
    fn indexed_pixels_outside_the_palette_are_black() {
        let palette = [[1, 2, 3], [4, 5, 6]];
        let mut out = [0xaa; 9];
        indexed_to_rgb888(&[1, 0, 200], &palette, &mut out);
        assert_eq!(out, [4, 5, 6, 1, 2, 3, 0, 0, 0]);
    }
}
