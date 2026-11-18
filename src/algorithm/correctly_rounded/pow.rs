//! Correctly rounded binary64 `pow`.

// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/pow/pow.c, commit 1851882b0f85ed581af9185cfa020992cec9e7a1.
// Copyright (c) 2022-2025 CERN and Inria. Authors: Tom Hubrecht and Paul Zimmermann.
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

use super::double_double::CANONICAL_NAN;
use super::path::Path;
use super::pow_accurate::{pow_2, rounding_is_certain};
use super::pow_dint::to_f64;
use super::pow_exact::{exact_pow, is_exact, is_int, is_odd_int};
use super::pow_extended::pow_3;
use super::pow_fast::{FAST_ERR, FAST_ERR_CANCEL, exp_1, log_1, s_mul};

const INF_BITS: u64 = 0x7ff0_0000_0000_0000;
const NEG_INF_BITS: u64 = 0xfff0_0000_0000_0000;

/// `x^y`, correctly rounded to nearest, ties to even, for every input pair whose 256-bit
/// rounding test succeeds.
///
/// No known input fails that test, but no proof covers it: CORE-MATH proves correct
/// rounding of `pow` modulo the hardest-to-round cases, which are unknown for binary64.
/// If the test fails, the result is the 256-bit approximation rounded to nearest.
///
/// Special values follow C Annex F (F.10.4.4), with every NaN result canonical:
/// - `pow(x, +-0)` is `1` for every `x`, NaN included.
/// - `pow(1, y)` is `1` for every `y`, NaN included. `pow(-1, +-inf)` is `1`.
/// - Any other NaN operand gives the canonical quiet NaN `0x7ff8000000000000`.
/// - A finite `x < 0` with a finite non-integer `y` gives the canonical NaN.
/// - A negative `x` with an integer `y` gives `|x|^y`, negated for odd `y`.
/// - `pow(+-0, y)` for an odd integer `y < 0` is `+-inf`, and for other `y < 0` is `+inf`.
///   For an odd integer `y > 0` it is `+-0`, and for other `y > 0` it is `+0`.
/// - `pow(x, -inf)` is `+inf` for `|x| < 1` and `+0` for `|x| > 1`. `pow(x, +inf)` is
///   `+0` for `|x| < 1` and `+inf` for `|x| > 1`.
/// - `pow(-inf, y)` is `-0` for an odd integer `y < 0`, `+0` for other `y < 0`, `-inf` for
///   an odd integer `y > 0` and `+inf` for other `y > 0`.
/// - `pow(+inf, y)` is `+0` for `y < 0` and `+inf` for `y > 0`.
///
/// A result below `2^-1022` is rounded once, on the subnormal grid. The bits do not depend
/// on the platform libm. The caller keeps round-to-nearest-even and gradual underflow active.
pub fn pow(x: f64, y: f64) -> f64 {
    pow_traced(x, y).0
}

/// `pow(x, y)` and the path that produced it.
#[doc(hidden)]
#[inline(always)]
pub fn pow_traced(x: f64, y: f64) -> (f64, Path) {
    let (ix, iy) = (x.to_bits(), y.to_bits());
    // The unsigned compare also admits every negative operand; finite ones fall through.
    if (ix >= INF_BITS || iy >= INF_BITS)
        && let Some(value) = non_finite(x, y)
    {
        return (value, Path::Special);
    }
    let mut s = 1.0;
    let mut ax = x;
    if x <= 0.0 {
        if let Some(value) = non_positive_base(x, y) {
            return (value, Path::Special);
        }
        // x < 0 and y is a nonzero integer: |x|^y with the parity sign.
        if y.abs() < 9_007_199_254_740_992.0 && (y as i64) & 1 == 1 {
            s = -1.0;
        }
        ax = -x;
    }
    pow_finite(x, ax, y, s)
}

/// Results for a NaN or infinite operand. `None` for finite operands.
#[inline(always)]
fn non_finite(x: f64, y: f64) -> Option<f64> {
    if x.is_nan() {
        // Annex F: pow(x, +-0) = 1 even for a NaN x. MPFR agrees for signalling NaNs too.
        return Some(if y == 0.0 { 1.0 } else { CANONICAL_NAN });
    }
    if y.is_nan() {
        return Some(if x == 1.0 { 1.0 } else { CANONICAL_NAN });
    }
    let (ix, iy) = (x.to_bits(), y.to_bits());
    if ix == INF_BITS {
        return Some(if y == 0.0 {
            1.0
        } else if y < 0.0 {
            0.0
        } else {
            f64::INFINITY
        });
    }
    if ix == NEG_INF_BITS {
        if iy == INF_BITS {
            return Some(f64::INFINITY);
        }
        if iy == NEG_INF_BITS {
            return Some(0.0);
        }
        // y == 0 falls through to the x <= 0 branch, which returns 1.
        if y != 0.0 {
            let odd = is_odd_int(y);
            return Some(match (y < 0.0, odd) {
                (true, true) => -0.0,
                (true, false) => 0.0,
                (false, true) => f64::NEG_INFINITY,
                (false, false) => f64::INFINITY,
            });
        }
    }
    if iy == INF_BITS || iy == NEG_INF_BITS {
        let ax = x.abs();
        let grows = (ax > 1.0) == (iy == INF_BITS);
        return Some(if ax == 1.0 {
            1.0
        } else if grows {
            f64::INFINITY
        } else {
            0.0
        });
    }
    None
}

/// Results for a finite `x <= 0`, or `None` when `x < 0` and `y` is a nonzero integer.
#[inline(always)]
fn non_positive_base(x: f64, y: f64) -> Option<f64> {
    if y == 0.0 {
        return Some(1.0);
    }
    if x == 0.0 {
        let negative_zero = x.to_bits() != 0;
        let odd = is_odd_int(y);
        return Some(match (y < 0.0, odd && negative_zero) {
            (true, true) => f64::NEG_INFINITY,
            (true, false) => f64::INFINITY,
            (false, true) => -0.0,
            (false, false) => 0.0,
        });
    }
    if !is_int(y) {
        return Some(CANONICAL_NAN);
    }
    None
}

/// `s * |x|^y` for finite nonzero `x` and `y`, `ax = |x|`.
#[inline(always)]
fn pow_finite(x: f64, ax: f64, y: f64, s: f64) -> (f64, Path) {
    let (lh, ll, cancel) = log_1(ax);
    // A NaN logarithm sends |y| < 2^-969 and |y| >= 2^1014 to the accurate path, which
    // avoids spurious underflow and overflow in y * log|x|.
    let ey = (y.to_bits() >> 52) & 0x7ff;
    let (lh, ll) = if !(0x36..0x7f5).contains(&ey) {
        (f64::NAN, f64::NAN)
    } else {
        (lh, ll)
    };
    let (rh, rl) = s_mul(y, lh, ll);
    let (res_h, res_l) = exp_1(rh, rl, s);
    let err = if cancel { FAST_ERR_CANCEL } else { FAST_ERR };
    let res_min = res_h + err.mul_add(-res_h, res_l);
    let res_max = res_h + err.mul_add(res_h, res_l);
    if res_min == res_max {
        // res_max is +-0 in the underflow case.
        return (res_max, Path::Fast);
    }
    if y == 1.0 {
        return (s * ax, Path::Exact);
    }
    if y == 2.0 {
        return (ax * ax, Path::Exact);
    }
    if y == 0.0 {
        return (1.0, Path::Special);
    }
    pow_slow(x, ax, y, s)
}

/// The accurate, exact-case and extended paths, after the fast rounding test failed.
#[inline(never)]
#[cold]
fn pow_slow(x: f64, ax: f64, y: f64, s: f64) -> (f64, Path) {
    let mut r = pow_2(ax, y);
    if r.ex < -1075 {
        // |x|^y < 2^-1075 rounds to a zero of the result's sign.
        return (0.5 * (s * f64::from_bits(1)), Path::Accurate);
    }
    r.sgn = u64::from(s == -1.0);
    if rounding_is_certain(r) {
        return (to_f64(r), Path::Accurate);
    }
    if let Some(value) = exact_pow(x, y, r, is_exact(ax, y)) {
        return (value, Path::Exact);
    }
    match pow_3(ax, y, s) {
        (value, true) => (value, Path::Extended),
        (value, false) => (value, Path::Uncertified),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAN: u64 = 0x7ff8_0000_0000_0000;

    #[test]
    fn test_pow_annex_f_special_values() {
        let snan = f64::from_bits(0x7ff0_0000_0000_0001);
        let inf = f64::INFINITY;
        for x in [f64::NAN, snan, -inf, -0.0, 2.0] {
            assert_eq!(pow(x, 0.0), 1.0);
            assert_eq!(pow(x, -0.0), 1.0);
        }
        for y in [f64::NAN, snan, inf, -inf, 0.3] {
            assert_eq!(pow(1.0, y), 1.0);
        }
        assert_eq!(pow(-1.0, inf), 1.0);
        assert_eq!(pow(-1.0, -inf), 1.0);
        assert_eq!(pow(f64::NAN, 1.0).to_bits(), NAN);
        assert_eq!(pow(2.0, snan).to_bits(), NAN);
        assert_eq!(pow(-2.0, 0.5).to_bits(), NAN);
        assert_eq!(pow(-0.0, -3.0), -inf);
        assert_eq!(pow(0.0, -3.0), inf);
        assert_eq!(pow(-0.0, -2.0), inf);
        assert_eq!(pow(-0.0, 3.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(pow(-0.0, 2.5).to_bits(), 0);
        assert_eq!(pow(-inf, -3.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(pow(-inf, 3.0), -inf);
        assert_eq!(pow(-inf, 2.5), inf);
        assert_eq!(pow(0.5, -inf), inf);
        assert_eq!(pow(-2.0, -inf).to_bits(), 0);
        assert_eq!(pow(inf, -1.0).to_bits(), 0);
    }

    #[test]
    fn test_pow_signs_and_exact_values() {
        assert_eq!(pow(-2.0, 3.0), -8.0);
        assert_eq!(pow(-2.0, 4.0), 16.0);
        assert_eq!(pow(-2.0, 1.0e300), f64::INFINITY);
        assert_eq!(pow(2.0, -1074.0).to_bits(), 1);
        assert_eq!(pow(2.0, -1075.0).to_bits(), 0);
        assert_eq!(pow(-2.0, -1075.0).to_bits(), 1 << 63);
        assert_eq!(pow(4.0, 0.5), 2.0);
        assert_eq!(pow(10.0, 2.0), 100.0);
        assert_eq!(pow(2.0, 0.5), std::f64::consts::SQRT_2);
    }

    #[test]
    fn test_paths_are_reached() {
        assert_eq!(pow_traced(1.5, 0.3).1, Path::Fast);
        assert_eq!(pow_traced(3.0, 34.0).1, Path::Exact);
        assert_eq!(pow_traced(3.0, 2.0f64.powi(-1000)).1, Path::Extended);
        assert_eq!(pow_traced(f64::NAN, 2.0).1, Path::Special);
    }
}
