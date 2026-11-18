//! Correctly rounded binary64 natural logarithm.

// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/log/log.c, commit 1851882b0f85ed581af9185cfa020992cec9e7a1.
// Copyright (c) 2022-2026 INRIA and CERN. Authors: Paul Zimmermann and Tom Hubrecht.
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

use super::double_double::{CANONICAL_NAN, fast_two_sum};
use super::log_accurate::log_accurate;
use super::path::Path;
use super::tables::{LOG_FAST_POLY, LOG_INVERSE, LOG_LOG_INV, LOG_TINY_POLY, f};

/// First index of `LOG_INVERSE`: entry `k` serves `(k + 362) * 2^-9 <= y < (k + 363) * 2^-9`.
const OFFSET: usize = 362;
/// `log(2)` high part, a multiple of `2^-42`, `0x1.62e42fefa38p-1`: `e * LOG2_H` is exact.
const LOG2_H: f64 = f64::from_bits(0x3fe6_2e42_fefa_3800);
/// `log(2)` low part, `0x1.ef35793c7673p-45`.
const LOG2_L: f64 = f64::from_bits(0x3d2e_f357_93c7_6730);
/// Fast-path absolute error bound proved with Gappa, `0x1.b6p-69`.
const FAST_ERR: f64 = f64::from_bits(0x3bab_6000_0000_0000);
/// Tiny-path relative error bound, `0x1.d1p-61`.
const TINY_EPS: f64 = f64::from_bits(0x3c2d_1000_0000_0000);
/// `2^52`, the exact scale applied to subnormal inputs.
const TWO_POW_52: f64 = f64::from_bits(0x4330_0000_0000_0000);
/// Significand of `sqrt(2)` with the hidden bit: above it the argument is halved.
const SQRT2_SIGNIFICAND: u64 = 0x0016_a09e_667f_3bcd;
/// `1 - 2^-9 <= x < 1 + 2^-9` takes the tiny path.
const TINY_LOW_BITS: u64 = 0x3fef_f000_0000_0000;
const TINY_HIGH_BITS: u64 = 0x3ff0_0800_0000_0000;
const FRACTION_MASK: u64 = 0x000f_ffff_ffff_ffff;

/// Natural logarithm of `x`, correctly rounded to nearest, ties to even.
///
/// - NaN, `-inf` and any `x < 0` give the canonical quiet NaN `0x7ff8000000000000`.
/// - `+0` and `-0` give `-inf`.
/// - `+inf` gives `+inf`.
/// - `1` gives `+0`.
/// - A subnormal `x` is scaled by `2^52` exactly, then takes the normal path.
///
/// The bits do not depend on the platform libm. The caller keeps round-to-nearest-even
/// and gradual underflow active.
pub fn log(x: f64) -> f64 {
    log_traced(x).0
}

/// `log(x)` and the path that produced it.
#[doc(hidden)]
#[inline(always)]
pub fn log_traced(x: f64) -> (f64, Path) {
    let mut v = x.to_bits();
    // The sign bit joins the exponent here, so every negative x lands above 0x400.
    let mut e = (v >> 52) as i32 - 0x3ff;
    if e >= 0x400 || e == -0x3ff {
        if e == 0x400 || (e == 0xc00 && x != f64::NEG_INFINITY) {
            // +inf, or NaN of either sign.
            let value = if x.is_nan() { CANONICAL_NAN } else { x };
            return (value, Path::Special);
        }
        if x <= 0.0 {
            let value = if x < 0.0 {
                CANONICAL_NAN
            } else {
                f64::NEG_INFINITY
            };
            return (value, Path::Special);
        }
        // Positive subnormal.
        v = (x * TWO_POW_52).to_bits();
        e = (v >> 52) as i32 - 0x3ff - 52;
    }
    if (TINY_LOW_BITS..TINY_HIGH_BITS).contains(&v) {
        return log_tiny(x);
    }
    // Normalize v into [1, 2): x = m * 2^e with -1074 <= e <= 1023.
    let m = f64::from_bits((0x3ffu64 << 52) | (v & FRACTION_MASK));
    let (h, l) = log_fast(e, m);
    let left = h + (l - FAST_ERR);
    let right = h + (l + FAST_ERR);
    if left == right {
        return (left, Path::Fast);
    }
    (log_accurate(x), Path::Accurate)
}

/// `log(2^e * m)` for `1 <= m < 2`, as a double-double with absolute error below `0x1.b6p-69`.
#[inline(always)]
fn log_fast(e: i32, m: f64) -> (f64, f64) {
    let bits = 0x0010_0000_0000_0000 | (m.to_bits() & FRACTION_MASK);
    // Above sqrt(2) the argument is halved to avoid cancellation.
    let c = usize::from(bits >= SQRT2_SIGNIFICAND);
    let e = e + c as i32;
    let i = (bits >> [43, 44][c]) as usize;
    let y = m * [1.0, 0.5][c];
    let r = f(&LOG_INVERSE, i - OFFSET);
    let l1 = f(&LOG_LOG_INV, 2 * (i - OFFSET));
    let l2 = f(&LOG_LOG_INV, 2 * (i - OFFSET) + 1);
    // Exact by the table construction: |z| < 0.00212097167968735.
    let z = r.mul_add(y, -1.0);
    let p = |k| f(&LOG_FAST_POLY, k);
    let z2 = z * z;
    let p45 = p(5).mul_add(z, p(4));
    let p23 = p(3).mul_add(z, p(2));
    let mut ph = p45.mul_add(z2, p23);
    ph = ph.mul_add(z, p(1));
    ph *= z2;
    let ee = e as f64;
    // e * LOG2_H + l1 is an exact multiple of 2^-42 below 2^52 * 2^-42.
    let (h, l) = fast_two_sum(ee.mul_add(LOG2_H, l1), z);
    let l = ph + (l + l2);
    (h, ee.mul_add(LOG2_L, l))
}

/// `log(x)` for `1 - 2^-9 <= x < 1 + 2^-9`.
#[inline(always)]
fn log_tiny(x: f64) -> (f64, Path) {
    // log(1) = +0 exactly.
    if x == 1.0 {
        return (0.0, Path::Special);
    }
    // r + c0*r^2 + ... + c5*r^7 approximates log(1 + r) for |r| <= 2^-9 with relative
    // error below 2^-71.173.
    let c = |k| f(&LOG_TINY_POLY, k);
    let r = x - 1.0;
    let r2 = r * r;
    let p6 = c(4) + r * c(5);
    let mut p4 = c(2) + r * c(3);
    let mut p2 = c(0) + r * c(1);
    p4 += r2 * p6;
    p2 += r2 * p4;
    let (h, l) = fast_two_sum(r, r2 * p2);
    let lb = h + (l - TINY_EPS * r);
    let ub = h + (l + TINY_EPS * r);
    if lb == ub {
        return (lb, Path::Fast);
    }
    (log_accurate(x), Path::Accurate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_special_values() {
        let nan = CANONICAL_NAN.to_bits();
        assert_eq!(log(f64::NAN).to_bits(), nan);
        assert_eq!(log(-f64::NAN).to_bits(), nan);
        assert_eq!(log(f64::from_bits(0x7ff0_0000_0000_0001)).to_bits(), nan);
        assert_eq!(log(-1.0).to_bits(), nan);
        assert_eq!(log(f64::NEG_INFINITY).to_bits(), nan);
        assert_eq!(log(-f64::from_bits(1)).to_bits(), nan);
        assert_eq!(log(0.0), f64::NEG_INFINITY);
        assert_eq!(log(-0.0), f64::NEG_INFINITY);
        assert_eq!(log(f64::INFINITY), f64::INFINITY);
        assert_eq!(log(1.0).to_bits(), 0);
        assert_eq!(log(2.0), std::f64::consts::LN_2);
    }

    #[test]
    fn test_log_subnormal_input_uses_exact_scaling() {
        // log(2^-1074) = -1074 log(2).
        let expected = log_accurate(f64::from_bits(1));
        assert_eq!(log(f64::from_bits(1)), expected);
        assert!((expected + 1074.0 * std::f64::consts::LN_2).abs() < 1e-12);
    }

    #[test]
    fn test_log_paths() {
        assert_eq!(log_traced(3.0).1, Path::Fast);
        assert_eq!(log_traced(1.001).1, Path::Fast);
    }
}
