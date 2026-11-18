//! Correctly rounded binary64 `exp`.

// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/exp/exp.c, commit 1851882b0f85ed581af9185cfa020992cec9e7a1.
// Copyright (c) 2022-2026 Alexei Sibidanov <sibid@uvic.ca>.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use super::double_double::{
    CANONICAL_NAN, add_exponent, fast_sum, fast_two_sum, fma_dw, mul_dd, round_ties_even,
};
use super::path::Path;
use super::tables::{EXP_ACCURATE_POLY, EXP_EXCEPTIONS, EXP_FAST_POLY, EXP_T0, EXP_T1, f};

/// `2^12 / log(2)`, `0x1.71547652b82fep+12`.
const SCALE: f64 = f64::from_bits(0x40b7_1547_652b_82fe);
/// High part of `log(2) / 2^12` on 29 bits, `0x1.62e42ffp-13`: `L2H * t` is exact for `|t| < 2^23`.
const L2H: f64 = f64::from_bits(0x3f26_2e42_ff00_0000);
/// Middle part of `log(2) / 2^12`, `0x1.718432a1b0e26p-47`.
const L2L: f64 = f64::from_bits(0x3d07_1843_2a1b_0e26);
/// Low part of `log(2) / 2^12`, `0x1.9ff0342542fc3p-102`.
const L2LL: f64 = f64::from_bits(0x3999_ff03_4254_2fc3);
/// Fast-path error bound relative to the `[1, 2)` scale, `0x1.27p-62`.
const FAST_EPS: f64 = f64::from_bits(0x3c12_7000_0000_0000);

/// `|x| <= 2^-54`: `exp(x)` rounds to `1`.
const TINY_ABS_BITS: u64 = 0x3c90_0000_0000_0000;
/// `|x| >= 0x1.62e42fefa39fp+9`: overflow for positive `x`, the underflow checks for negative `x`.
const LARGE_ABS_BITS: u64 = 0x4086_2e42_fefa_39f0;
/// `x <= -0x1.74910d52d3052p+9`: `exp(x)` rounds to `+0`.
const ZERO_ABS_BITS: u64 = 0x4087_4910_d52d_3052;
/// `x < -0x1.6232bdd7abcd2p+9`: `exp(x)` is below `2^-1022`, so the result is subnormal.
const SUBNORMAL_BITS: u64 = 0xc086_232b_dd7a_bcd2;
const INF_BITS: u64 = 0x7ff0_0000_0000_0000;

/// `e^x`, correctly rounded to nearest, ties to even.
///
/// - NaN gives the canonical quiet NaN `0x7ff8000000000000`.
/// - `+inf` gives `+inf`, and `-inf` gives `+0`.
/// - `|x| <= 2^-54`, zeros and subnormals included, gives exactly `1.0`.
/// - `x >= 0x1.62e42fefa39fp+9` gives `+inf`.
/// - `x <= -0x1.74910d52d3052p+9` gives `+0`.
/// - A result below `2^-1022` is rounded once, on the subnormal grid.
///
/// The bits do not depend on the platform libm. The caller keeps round-to-nearest-even
/// and gradual underflow active.
pub fn exp(x: f64) -> f64 {
    exp_traced(x).0
}

/// `exp(x)` and the path that produced it.
#[doc(hidden)]
#[inline(always)]
pub fn exp_traced(x: f64) -> (f64, Path) {
    let ix = x.to_bits();
    let aix = ix & (u64::MAX >> 1);
    if aix <= TINY_ABS_BITS {
        return (1.0 + x, Path::Special);
    }
    if aix >= LARGE_ABS_BITS {
        if aix > INF_BITS {
            return (CANONICAL_NAN, Path::Special);
        }
        if aix == INF_BITS {
            let value = if ix >> 63 == 1 { 0.0 } else { f64::INFINITY };
            return (value, Path::Special);
        }
        if ix >> 63 == 0 {
            return (f64::INFINITY, Path::Special);
        }
        if aix >= ZERO_ABS_BITS {
            return (0.0, Path::Special);
        }
    }
    let (t, ie, th, tl) = reduce(x);
    let dx = (x - L2H * t) + L2L * t;
    let dx2 = dx * dx;
    // 1 + dx*(c0 + c1*dx + c2*dx^2 + c3*dx^3) approximates exp(dx) on
    // [-log(2)/2^13, log(2)/2^13] with absolute error below 2^-76.173.
    let c = |i| f(&EXP_FAST_POLY, i);
    let p = (c(0) + dx * c(1)) + dx2 * (c(2) + dx * c(3));
    let mut fh = th;
    let tx = th * dx;
    let mut fl = tl + tx * p;
    if ix > SUBNORMAL_BITS {
        let (sh, e) = fast_two_sum(subnormal_offset(ie), fh);
        fh = sh;
        fl += e;
        let ub = fh + (fl + FAST_EPS);
        let lb = fh + (fl - FAST_EPS);
        if ub != lb {
            return exp_accurate(x);
        }
        (clear_exponent(lb), Path::Fast)
    } else {
        let ub = fh + (fl + FAST_EPS);
        let lb = fh + (fl - FAST_EPS);
        if ub != lb {
            return exp_accurate(x);
        }
        (add_exponent(lb, ie), Path::Fast)
    }
}

/// `(t, ie, th, tl)` with `t = round(x * 2^12 / log 2)`, `2^ie` the power of two of `t`,
/// and `th + tl` the double-double `2^((t mod 2^12) / 2^12)`.
#[inline(always)]
fn reduce(x: f64) -> (f64, i64, f64, f64) {
    let t = round_ties_even(x * SCALE);
    let jt = t as i64;
    let i0 = ((jt >> 6) & 0x3f) as usize;
    let i1 = (jt & 0x3f) as usize;
    let ie = jt >> 12;
    let t0h = f(&EXP_T0, 2 * i0 + 1);
    let t0l = f(&EXP_T0, 2 * i0);
    let t1h = f(&EXP_T1, 2 * i1 + 1);
    let t1l = f(&EXP_T1, 2 * i1);
    let (th, tl) = mul_dd(t0h, t0l, t1h, t1l);
    (t, ie, th, tl)
}

/// `2^(-1022 - ie)`. Adding it to a value in `[1, 2)` puts the rounding point of the sum on
/// the subnormal grid of the result scaled by `2^ie`. Needs `-1075 <= ie <= -1022`.
#[inline(always)]
fn subnormal_offset(ie: i64) -> f64 {
    f64::from_bits(((1 - ie) as u64) << 52)
}

/// Clear the exponent field: `2^E * (1 + m)` with `E = -1022 - ie` becomes the subnormal
/// `m * 2^-1022`, which is exactly the rounded result scaled by `2^ie`.
#[inline(always)]
fn clear_exponent(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & (u64::MAX >> 12))
}

/// Accurate path. Assumes `|x| > 2^-54`.
#[inline(never)]
#[cold]
fn exp_accurate(x: f64) -> (f64, Path) {
    let ix = x.to_bits();
    let (t, ie, th, tl) = reduce(x);
    // Cody-Waite reduction: |t| < 2^23 and L2H has 29 bits, so L2H * t is exact.
    let dx = x - L2H * t;
    let dxl = L2L * t;
    let dxll = L2LL * t + L2L.mul_add(t, -dxl);
    let (dxh, dxl) = fast_two_sum(dx, dxl);
    let dxl = dxl + dxll;
    // Q = ch[0]*x + ... + ch[6]*x^7 approximates exp(x) - 1 for |x| < 2^-13.528.
    let (mut fh, mut fl) = poly_dd(dxh, dxl);
    (fh, fl) = mul_dd(dxh, dxl, fh, fl);
    (fh, fl) = mul_dd(fh, fl, th, tl);
    (fh, fl) = fast_sum(th, tl, fh, fl);
    if ix > SUBNORMAL_BITS {
        let (sh, e) = fast_two_sum(subnormal_offset(ie), fh);
        fl += e;
        return (clear_exponent(sh + fl), Path::Accurate);
    }
    let (sh, sl) = fast_two_sum(fh, fl);
    // d <= 4 only for listed exceptions and the single input x = -0x1.07e05e6e03b83p-2.
    let d = sl.to_bits().wrapping_add(4) & (u64::MAX >> 12);
    let (value, path) = if d <= 4 {
        exception(x, sh)
    } else {
        (sh, Path::Accurate)
    };
    (add_exponent(value, ie), path)
}

/// Horner evaluation of `EXP_ACCURATE_POLY` in double-double, highest degree first.
#[inline(always)]
fn poly_dd(xh: f64, xl: f64) -> (f64, f64) {
    let n = EXP_ACCURATE_POLY.len() / 2;
    let mut ch = f(&EXP_ACCURATE_POLY, 2 * (n - 1));
    let mut cl = f(&EXP_ACCURATE_POLY, 2 * (n - 1) + 1);
    for i in (0..n - 1).rev() {
        (ch, cl) = fma_dw(
            xh,
            xl,
            ch,
            cl,
            f(&EXP_ACCURATE_POLY, 2 * i),
            f(&EXP_ACCURATE_POLY, 2 * i + 1),
        );
    }
    (ch, cl)
}

/// The listed result for `x` before scaling by `2^ie`, or `fallback` when `x` is not listed.
fn exception(x: f64, fallback: f64) -> (f64, Path) {
    let entry = |i: usize| f(&EXP_EXCEPTIONS, 3 * i);
    let (mut a, mut b) = (0, EXP_EXCEPTIONS.len() / 3);
    // Invariant: entry(a) <= x < entry(b).
    while a + 1 < b {
        let m = (a + b) / 2;
        if x < entry(m) {
            b = m;
        } else {
            a = m;
        }
    }
    if x == entry(a) {
        let value = f(&EXP_EXCEPTIONS, 3 * a + 1) + f(&EXP_EXCEPTIONS, 3 * a + 2);
        return (value, Path::Exception);
    }
    (fallback, Path::Accurate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_special_values() {
        assert_eq!(exp(f64::NAN).to_bits(), CANONICAL_NAN.to_bits());
        assert_eq!(
            exp(f64::from_bits(0xfff0_0000_0000_0001)).to_bits(),
            CANONICAL_NAN.to_bits()
        );
        assert_eq!(exp(f64::INFINITY), f64::INFINITY);
        assert_eq!(exp(f64::NEG_INFINITY).to_bits(), 0);
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(-0.0), 1.0);
        assert_eq!(exp(f64::from_bits(1)), 1.0);
        assert_eq!(exp(f64::MAX), f64::INFINITY);
        assert_eq!(exp(f64::MIN).to_bits(), 0);
        assert_eq!(exp(1.0), std::f64::consts::E);
    }

    #[test]
    fn test_clear_exponent_rounds_on_subnormal_grid() {
        for ie in -1075..=-1022i64 {
            let offset = subnormal_offset(ie);
            for m in [0u64, 1, 2, 3, 0x000f_ffff_ffff_ffff] {
                let sum = f64::from_bits(offset.to_bits() | m);
                assert_eq!(clear_exponent(sum).to_bits(), m, "ie {ie} m {m}");
            }
        }
    }

    #[test]
    fn test_listed_exception_takes_exception_path() {
        let x = f(&EXP_EXCEPTIONS, 3 * 2);
        assert_eq!(x.to_bits(), 0xbfd0_7e05_e6e0_3b83);
        assert_eq!(exp_traced(x).1, Path::Exception);
    }

    #[test]
    fn test_fast_path_serves_ordinary_inputs() {
        assert_eq!(exp_traced(0.5).1, Path::Fast);
        assert_eq!(exp_traced(-745.0).1, Path::Fast);
    }
}
