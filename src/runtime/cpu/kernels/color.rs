//! CPU kernels of the `ColorOps` YCbCr conversions.
//!
//! Every value is an f32 operation with one rounding, in the order
//! `ColorOps` documents. Rust never fuses `a * b + c`, so these kernels give
//! the same bits as the CUDA kernels in `color.cu`, which call the `_rn`
//! intrinsics.
//!
//! Planes are read and written through their strides. Decode runs one luma
//! row per task. Encode runs one chroma row, so two luma rows, per task.

use crate::dtype::Element;
use crate::ops::common::{ColorCoeffs, PlaneView, YuvFrame};
use crate::runtime::cpu::CpuClient;

/// A stored plane sample: U8, or U16 with the value in the high bits.
pub trait PlaneSample: Copy + Send + Sync + 'static {
    /// Sample value after dropping `shift` low bits.
    fn value(self, shift: u32) -> i32;
    /// Stored sample of an integral `code` in the sample's range.
    fn from_code(code: f32, shift: u32) -> Self;
}

impl PlaneSample for u8 {
    #[inline(always)]
    fn value(self, _shift: u32) -> i32 {
        i32::from(self)
    }
    #[inline(always)]
    fn from_code(code: f32, _shift: u32) -> Self {
        code as u8
    }
}

impl PlaneSample for u16 {
    #[inline(always)]
    fn value(self, shift: u32) -> i32 {
        i32::from(self >> shift)
    }
    #[inline(always)]
    fn from_code(code: f32, shift: u32) -> Self {
        (code as u16) << shift
    }
}

/// `min(max(x, 0), 1)`. A NaN gives 0, as `fminf(fmaxf(x, 0), 1)` does.
#[inline(always)]
#[allow(
    clippy::manual_clamp,
    reason = "f32::clamp keeps NaN; this op must map NaN to 0 like the CUDA kernel"
)]
pub fn clamp01(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

#[inline(always)]
fn at(p: PlaneView, r: usize, c: usize) -> isize {
    r as isize * p.row_stride + c as isize * p.col_stride
}

/// # Safety
///
/// `[r, c]` lies inside the plane view `p` of `P` samples.
#[inline(always)]
unsafe fn read<P: PlaneSample>(p: PlaneView, r: usize, c: usize) -> P {
    // SAFETY: forwarded from this function's contract.
    unsafe { *(p.addr as *const P).offset(at(p, r, c)) }
}

/// # Safety
///
/// `[r, c]` lies inside the plane view `p` of `P` samples, and no other
/// task writes it.
#[inline(always)]
unsafe fn write<P: PlaneSample>(p: PlaneView, r: usize, c: usize, v: P) {
    // SAFETY: forwarded from this function's contract.
    unsafe { *(p.addr as *mut P).offset(at(p, r, c)) = v }
}

/// Chroma indices and weights in quarters for luma index `pos` along an axis
/// of `len` chroma samples. `len` is at least 1.
#[inline(always)]
fn up_taps(pos: usize, len: usize, center: bool) -> ([usize; 2], [i32; 2]) {
    let k = pos / 2;
    let next = (k + 1).min(len - 1);
    match (center, pos.is_multiple_of(2)) {
        (true, true) => ([k.saturating_sub(1), k], [1, 3]),
        (true, false) => ([k, next], [3, 1]),
        (false, true) => ([k, k], [4, 0]),
        (false, false) => ([k, next], [2, 2]),
    }
}

/// `[R, G, B]` of luma value `y` and chroma sums `cb16`, `cr16` in sixteenths.
#[inline(always)]
pub fn decode(k: &ColorCoeffs, y: i32, cb16: i32, cr16: i32) -> [f32; 3] {
    let yn = (y - k.y_off) as f32 / k.y_mul;
    let cb = (cb16 - 16 * k.c_off) as f32 / k.c_div16;
    let cr = (cr16 - 16 * k.c_off) as f32 / k.c_div16;
    let r = yn + k.cr_r * cr;
    let b = yn + k.cb_b * cb;
    let g = (yn + k.g_cb * cb) + k.g_cr * cr;
    [clamp01(r), clamp01(g), clamp01(b)]
}

/// `(Y', Cb', Cr')` of one pixel, after clamping R, G and B to `[0, 1]`.
#[inline(always)]
pub fn encode(k: &ColorCoeffs, [r, g, b]: [f32; 3]) -> (f32, f32, f32) {
    let (r, g, b) = (clamp01(r), clamp01(g), clamp01(b));
    let y = (k.kr * r + k.kg * g) + k.kb * b;
    (y, (b - y) / k.cb_b, (r - y) / k.cr_r)
}

/// `round_ties_even(clamp(v·mul + off, lo, hi))`.
#[inline(always)]
pub fn quantize(v: f32, mul: f32, off: i32, lo: f32, hi: f32) -> f32 {
    (v * mul + off as f32).max(lo).min(hi).round_ties_even()
}

/// Convert planes to RGB, one luma row per task.
///
/// `out_strides` are the `[C, H, W]` strides of the RGB view at `out`.
///
/// # Safety
///
/// - `frame` came from `plan_yuv_frame`, its views hold `P` samples, and
///   every address they reach lies inside its allocation.
/// - `out` points at RGB element `[0, 0, 0]`, every address `out_strides`
///   reach lies inside its allocation, and no two pixels share an element.
/// - `out` shares no element with a plane.
pub unsafe fn yuv_to_rgb_kernel<P: PlaneSample, O: Element>(
    client: &CpuClient,
    frame: &YuvFrame,
    out: *mut O,
    out_strides: [isize; 3],
) {
    if frame.pixels() == 0 {
        return;
    }
    let f = *frame;
    let out_addr = out as usize;
    client.par_for_each(f.height, f.pixels() * 16, move |row| {
        let out = out_addr as *mut O;
        let (rows, wv) = up_taps(row, f.height / 2, true);
        for col in 0..f.width {
            let (cols, wh) = up_taps(col, f.width / 2, f.center);
            let (mut cb16, mut cr16) = (0, 0);
            // SAFETY: the taps lie inside the [H/2, W/2] chroma views, the
            // pixel inside the luma view and the RGB view, by the contract.
            unsafe {
                for a in 0..2 {
                    for b in 0..2 {
                        let wt = wv[a] * wh[b];
                        cb16 += wt * read::<P>(f.u, rows[a], cols[b]).value(f.shift);
                        cr16 += wt * read::<P>(f.v, rows[a], cols[b]).value(f.shift);
                    }
                }
                let y = read::<P>(f.y, row, col).value(f.shift);
                let base = row as isize * out_strides[1] + col as isize * out_strides[2];
                for (c, v) in decode(&f.coeffs, y, cb16, cr16).into_iter().enumerate() {
                    *out.offset(base + c as isize * out_strides[0]) = O::from_f32(v);
                }
            }
        }
    });
}

/// Encode the pixels of luma row `row` at columns `2j - 1` (clamped),
/// `2j` and `2j + 1`: write their luma and return `(Cb', Cr')` of each.
///
/// # Safety
///
/// As [`rgb_to_yuv_kernel`], for one chroma column `j`.
#[inline(always)]
unsafe fn encode_row<I: Element, P: PlaneSample>(
    f: &YuvFrame,
    rgb: *const I,
    rgb_strides: [isize; 3],
    row: usize,
    j: usize,
) -> ([f32; 3], [f32; 3]) {
    let k = &f.coeffs;
    let load = |col: usize| {
        let base = row as isize * rgb_strides[1] + col as isize * rgb_strides[2];
        // SAFETY: `[row, col]` lies inside the RGB view, by the contract.
        let ch = |c: isize| unsafe { (*rgb.offset(base + c * rgb_strides[0])).to_f32() };
        encode(k, [ch(0), ch(1), ch(2)])
    };
    let (mut cb, mut cr) = ([0.0f32; 3], [0.0f32; 3]);
    for (slot, col) in [(1, 2 * j), (2, 2 * j + 1)] {
        let (y, u, v) = load(col);
        let code = quantize(y, k.y_mul, k.y_off, k.y_lo, k.y_hi);
        // SAFETY: `[row, col]` lies inside the luma view, and only this task
        // writes chroma column `j`'s luma pixels.
        unsafe { write(f.y, row, col, P::from_code(code, f.shift)) };
        (cb[slot], cr[slot]) = (u, v);
    }
    if !f.center {
        (cb[0], cr[0]) = if j == 0 {
            (cb[1], cr[1])
        } else {
            let (_, u, v) = load(2 * j - 1);
            (u, v)
        };
    }
    (cb, cr)
}

/// Convert RGB to planes, one chroma row per task.
///
/// `rgb_strides` are the `[C, H, W]` strides of the RGB view at `rgb`.
///
/// # Safety
///
/// - `frame` came from `plan_yuv_frame`, its views hold `P` samples, every
///   address they reach lies inside its allocation, and no plane element is
///   reached twice.
/// - `rgb` points at RGB element `[0, 0, 0]` and every address
///   `rgb_strides` reach lies inside its allocation.
/// - No plane shares a byte with another plane or with `rgb`.
pub unsafe fn rgb_to_yuv_kernel<I: Element, P: PlaneSample>(
    client: &CpuClient,
    frame: &YuvFrame,
    rgb: *const I,
    rgb_strides: [isize; 3],
) {
    if frame.pixels() == 0 {
        return;
    }
    let f = *frame;
    let rgb_addr = rgb as usize;
    let k = f.coeffs;
    client.par_for_each(f.height / 2, f.pixels() * 32, move |i| {
        let rgb = rgb_addr as *const I;
        for j in 0..f.width / 2 {
            // SAFETY: rows 2i and 2i + 1 and columns up to 2j + 1 lie inside
            // the frame, by the contract.
            let (top, bottom) = unsafe {
                (
                    encode_row::<I, P>(&f, rgb, rgb_strides, 2 * i, j),
                    encode_row::<I, P>(&f, rgb, rgb_strides, 2 * i + 1, j),
                )
            };
            let filter = |a: [f32; 3], b: [f32; 3]| {
                if f.center {
                    ((a[1] + a[2]) + (b[1] + b[2])) * 0.25
                } else {
                    let h = |c: [f32; 3]| (c[0] + 2.0 * c[1]) + c[2];
                    (h(a) + h(b)) * 0.125
                }
            };
            let cb = quantize(filter(top.0, bottom.0), k.c_mul, k.c_off, k.c_lo, k.c_hi);
            let cr = quantize(filter(top.1, bottom.1), k.c_mul, k.c_off, k.c_lo, k.c_hi);
            // SAFETY: `[i, j]` lies inside the chroma views, and only this
            // task writes it.
            unsafe {
                write(f.u, i, j, P::from_code(cb, f.shift));
                write(f.v, i, j, P::from_code(cr, f.shift));
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::traits::{ChromaSiting, ColorMatrix, ColorRange, YuvSpec};

    fn coeffs(range: ColorRange, bits: u32) -> ColorCoeffs {
        let spec = YuvSpec::new(ColorMatrix::Bt709, range, ChromaSiting::Left);
        ColorCoeffs::new(&spec, bits)
    }

    #[test]
    fn taps_follow_the_siting() {
        assert_eq!(up_taps(0, 3, true), ([0, 0], [1, 3]));
        assert_eq!(up_taps(3, 3, true), ([1, 2], [3, 1]));
        assert_eq!(up_taps(5, 3, true), ([2, 2], [3, 1]));
        assert_eq!(up_taps(2, 3, false), ([1, 1], [4, 0]));
        assert_eq!(up_taps(3, 3, false), ([1, 2], [2, 2]));
        assert_eq!(up_taps(5, 3, false), ([2, 2], [2, 2]));
    }

    #[test]
    fn black_and_white_decode_exactly() {
        let k = coeffs(ColorRange::Limited, 8);
        assert_eq!(decode(&k, 16, 16 * 128, 16 * 128), [0.0, 0.0, 0.0]);
        assert_eq!(decode(&k, 235, 16 * 128, 16 * 128), [1.0, 1.0, 1.0]);
        let k = coeffs(ColorRange::Full, 10);
        assert_eq!(decode(&k, 1023, 16 * 512, 16 * 512), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn decode_clamps_out_of_gamut_codes() {
        let k = coeffs(ColorRange::Limited, 8);
        assert_eq!(decode(&k, 0, 16 * 128, 16 * 128), [0.0, 0.0, 0.0]);
        assert_eq!(decode(&k, 255, 16 * 128, 16 * 128), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn grey_encodes_to_neutral_chroma() {
        let k = coeffs(ColorRange::Limited, 8);
        let (y, cb, cr) = encode(&k, [0.5, 0.5, 0.5]);
        assert_eq!(quantize(y, k.y_mul, k.y_off, k.y_lo, k.y_hi), 126.0);
        assert_eq!(quantize(cb, k.c_mul, k.c_off, k.c_lo, k.c_hi), 128.0);
        assert_eq!(quantize(cr, k.c_mul, k.c_off, k.c_lo, k.c_hi), 128.0);
    }

    #[test]
    fn p010_samples_keep_the_high_bits() {
        assert_eq!(0xFFFFu16.value(6), 1023);
        assert_eq!(u16::from_code(940.0, 6), 940 << 6);
    }
}
