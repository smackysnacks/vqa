//! Pixel conversion from decoded frames to packed RGB888, RGBA8888, and
//! XRGB8888.
//!
//! HiColor conversion is pure lane-wise bit shuffling, so it runs as
//! `fearless_simd` kernels, dispatched at run time to the best instruction
//! set the CPU has. Every SIMD operation must sit in an `#[inline(always)]`
//! function reached from `dispatch!`: closures and other out-of-line code
//! don't inherit the enabled target features, which turns each operation
//! into a function call.

use fearless_simd::prelude::*;
use fearless_simd::{Level, dispatch, u8x16, u16x16, u32x8};

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

// The 4-byte formats need no shuffling: each pixel becomes one 32-bit word
// by shifts and masks. Built for generic x86-64, LLVM vectorizes that only
// at SSE2's width, so they too run as dispatched kernels, on 16 pixels
// widened to two 8-lane halves of words.

/// A `0rrrrrgg gggbbbbb` pixel as `0x00RRGGBB`, each channel expanded as
/// in [`scale5`].
#[inline(always)]
fn hicolor_xrgb(p: u16) -> u32 {
    let p = u32::from(p);
    // each channel's five bits at the top of its byte...
    let rgb = (p & 0x7c00) << 9 | (p & 0x03e0) << 6 | (p & 0x001f) << 3;
    // ...and its top three bits repeated below them
    rgb | (rgb >> 5 & 0x07_0707)
}

/// [`hicolor_xrgb`] on 8 lanes.
#[inline(always)]
fn hicolor_xrgb_x8<S: Simd>(p: u32x8<S>) -> u32x8<S> {
    let rgb = ((p & 0x7c00) << 9) | ((p & 0x03e0) << 6) | ((p & 0x001f) << 3);
    rgb | ((rgb >> 5) & 0x07_0707)
}

/// A `0rrrrrgg gggbbbbb` pixel as the little-endian word of RGBA8888 bytes,
/// alpha opaque.
#[inline(always)]
fn hicolor_rgba(p: u16) -> u32 {
    let p = u32::from(p);
    let bgr = (p & 0x7c00) >> 7 | (p & 0x03e0) << 6 | (p & 0x001f) << 19;
    0xff00_0000 | bgr | (bgr >> 5 & 0x07_0707)
}

/// [`hicolor_rgba`] on 8 lanes.
#[inline(always)]
fn hicolor_rgba_x8<S: Simd>(p: u32x8<S>) -> u32x8<S> {
    let bgr = ((p & 0x7c00) >> 7) | ((p & 0x03e0) << 6) | ((p & 0x001f) << 19);
    bgr | ((bgr >> 5) & 0x07_0707) | 0xff00_0000
}

/// Convert `0rrrrrgg gggbbbbb` pixels (the top bit ignored) to packed
/// `0x00RRGGBB` words.
pub(crate) fn hicolor_to_xrgb8888(pixels: &[u16], out: &mut [u32]) {
    hicolor_to_xrgb8888_at(Level::new(), pixels, out);
}

fn hicolor_to_xrgb8888_at(level: Level, pixels: &[u16], out: &mut [u32]) {
    if level.is_fallback() {
        xrgb_scalar(pixels, out);
    } else {
        dispatch!(level, simd => xrgb_kernel(simd, pixels, out));
    }
}

#[inline(always)]
fn xrgb_scalar(pixels: &[u16], out: &mut [u32]) {
    for (out, &p) in out.iter_mut().zip(pixels) {
        *out = hicolor_xrgb(p);
    }
}

#[inline(always)]
fn xrgb_kernel<S: Simd>(simd: S, pixels: &[u16], out: &mut [u32]) {
    let (src, src_rest) = pixels.as_chunks::<16>();
    let (dst, dst_rest) = out.as_chunks_mut::<16>();
    for (pixels, out) in src.iter().zip(dst) {
        let (lo, hi) = simd.widen_u16x16(u16x16::from_slice(simd, pixels));
        let (out_lo, out_hi) = out.split_at_mut(8);
        hicolor_xrgb_x8(lo).store_slice(out_lo);
        hicolor_xrgb_x8(hi).store_slice(out_hi);
    }
    xrgb_scalar(src_rest, dst_rest);
}

/// Convert `0rrrrrgg gggbbbbb` pixels (the top bit ignored) to RGBA8888.
/// `out` holds four bytes per pixel.
pub(crate) fn hicolor_to_rgba8888(pixels: &[u16], out: &mut [u8]) {
    hicolor_to_rgba8888_at(Level::new(), pixels, out);
}

fn hicolor_to_rgba8888_at(level: Level, pixels: &[u16], out: &mut [u8]) {
    if level.is_fallback() {
        rgba_scalar(pixels, out);
    } else {
        dispatch!(level, simd => rgba_kernel(simd, pixels, out));
    }
}

#[inline(always)]
fn rgba_scalar(pixels: &[u16], out: &mut [u8]) {
    for (out, &p) in out.as_chunks_mut::<4>().0.iter_mut().zip(pixels) {
        *out = hicolor_rgba(p).to_le_bytes();
    }
}

/// The words' bytes land in memory order, which is RGBA on the
/// little-endian CPUs that have a SIMD level.
#[inline(always)]
fn rgba_kernel<S: Simd>(simd: S, pixels: &[u16], out: &mut [u8]) {
    let (src, src_rest) = pixels.as_chunks::<16>();
    let (dst, dst_rest) = out.as_chunks_mut::<64>();
    for (pixels, out) in src.iter().zip(dst) {
        let (lo, hi) = simd.widen_u16x16(u16x16::from_slice(simd, pixels));
        let (out_lo, out_hi) = out.split_at_mut(32);
        hicolor_rgba_x8(lo).to_bytes().store_slice(out_lo);
        hicolor_rgba_x8(hi).to_bytes().store_slice(out_hi);
    }
    rgba_scalar(src_rest, dst_rest);
}

/// Convert palette indices to packed `0x00RRGGBB` words; indices with no
/// palette entry come out black.
pub(crate) fn indexed_to_xrgb8888(pixels: &[u8], palette: &[[u8; 3]], out: &mut [u32]) {
    let mut table = [0; 256];
    for (entry, &[r, g, b]) in table.iter_mut().zip(palette) {
        *entry = u32::from_be_bytes([0, r, g, b]);
    }
    for (out, &index) in out.iter_mut().zip(pixels) {
        *out = table[usize::from(index)];
    }
}

/// Convert palette indices to RGBA8888, alpha opaque; indices with no
/// palette entry come out opaque black. `out` holds four bytes per pixel.
pub(crate) fn indexed_to_rgba8888(pixels: &[u8], palette: &[[u8; 3]], out: &mut [u8]) {
    let mut table = [[0, 0, 0, 0xff]; 256];
    for (entry, &[r, g, b]) in table.iter_mut().zip(palette) {
        *entry = [r, g, b, 0xff];
    }
    for (out, &index) in out.as_chunks_mut::<4>().0.iter_mut().zip(pixels) {
        *out = table[usize::from(index)];
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

    /// `rgb` (RGB888 bytes) as XRGB8888 words and RGBA8888 bytes.
    fn four_byte(rgb: &[u8]) -> (Vec<u32>, Vec<u8>) {
        let rgb = rgb.as_chunks::<3>().0;
        let xrgb = rgb
            .iter()
            .map(|&[r, g, b]| u32::from_be_bytes([0, r, g, b]));
        let rgba = rgb.iter().flat_map(|&[r, g, b]| [r, g, b, 0xff]);
        (xrgb.collect(), rgba.collect())
    }

    #[test]
    fn four_byte_formats_match_rgb888_on_every_pixel_value() {
        // every 16-bit value, at the detected and the baseline level
        let pixels: Vec<u16> = (0..=u16::MAX).collect();
        let (xrgb, rgba) = four_byte(&reference(&pixels));
        for level in [Level::new(), Level::baseline()] {
            let mut out = vec![0; pixels.len()];
            hicolor_to_xrgb8888_at(level, &pixels, &mut out);
            assert!(out == xrgb, "XRGB8888 at {level:?}");
            let mut out = vec![0; pixels.len() * 4];
            hicolor_to_rgba8888_at(level, &pixels, &mut out);
            assert!(out == rgba, "RGBA8888 at {level:?}");
        }
    }

    #[test]
    fn four_byte_kernels_handle_lengths_off_the_vector_width() {
        let pixels: Vec<u16> = (0..50).map(|i| i * 1311 + 7).collect();
        for len in 0..=pixels.len() {
            let (xrgb, rgba) = four_byte(&reference(&pixels[..len]));
            let mut out = vec![0xaaaa_aaaa; len];
            hicolor_to_xrgb8888(&pixels[..len], &mut out);
            assert_eq!(out, xrgb, "{len} pixels");
            let mut out = vec![0xaa; len * 4];
            hicolor_to_rgba8888(&pixels[..len], &mut out);
            assert_eq!(out, rgba, "{len} pixels");
        }
    }

    #[test]
    fn indexed_four_byte_formats_look_up_the_palette() {
        let palette = [[1, 2, 3], [4, 5, 6]];
        let mut xrgb = [0xaaaa_aaaa; 3];
        indexed_to_xrgb8888(&[1, 0, 200], &palette, &mut xrgb);
        assert_eq!(xrgb, [0x0004_0506, 0x0001_0203, 0]);

        let mut rgba = [0xaa; 12];
        indexed_to_rgba8888(&[1, 0, 200], &palette, &mut rgba);
        assert_eq!(rgba, [4, 5, 6, 0xff, 1, 2, 3, 0xff, 0, 0, 0, 0xff]);
    }

    #[test]
    fn indexed_pixels_outside_the_palette_are_black() {
        let palette = [[1, 2, 3], [4, 5, 6]];
        let mut out = [0xaa; 9];
        indexed_to_rgb888(&[1, 0, 200], &palette, &mut out);
        assert_eq!(out, [4, 5, 6, 1, 2, 3, 0, 0, 0]);
    }
}
