//! Correctly rounded binary64 `log1p`.

// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/log1p/log1p.c, commit 1851882b0f85ed581af9185cfa020992cec9e7a1.
// Copyright (c) 2024-2026 Alexei Sibidanov <sibid@uvic.ca>.
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

use super::double_double::{CANONICAL_NAN, fast_sum, fast_two_sum};
use super::log1p_refine::{log1p_refine, tiny};
use super::path::Path;
use super::tables::{
    LOG1P_LARGE_POLY, LOG1P_LF, LOG1P_MEDIUM_POLY, LOG1P_RF, LOG1P_SMALL_POLY, LOG1P_TINY_POLY, f,
};

/// `ax = bits << 1` thresholds: `|x| < 2^-4`, `2^-53`, `2^-12` and `0x1.19bp-21`.
const AX_SMALL: u64 = 0x7f60_0000_0000_0000;
const AX_TINY: u64 = 0x7940_0000_0000_0000;
const AX_SERIES: u64 = 0x7e60_0000_0000_0000;
const AX_SHORT_SERIES: u64 = 0x7d43_3600_0000_0000;
/// Error bound factor for `|x| < 2^-12`, `0x1.ap-64`.
const EPS_SERIES: f64 = f64::from_bits(0x3bfa_0000_0000_0000);
/// Error bound factor for `2^-12 <= |x| < 2^-4`, `0x1.b6p-52`.
const EPS_MEDIUM: f64 = f64::from_bits(0x3cbb_6000_0000_0000);
/// Error bound for `|x| >= 2^-4`, `0x1.ap-65`.
const EPS_LARGE: f64 = f64::from_bits(0x3bea_0000_0000_0000);
/// `log(2)` high part, `0x1.62e42fefa4p-1`.
const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fefa_4000);
/// `log(2)` low part, `-0x1.8432a1b0e2634p-43`.
const LN2_LO: f64 = f64::from_bits(0xbd48_432a_1b0e_2634);
const INF_BITS: u64 = 0x7ff0_0000_0000_0000;
const MINUS_ONE_BITS: u64 = 0xbff0_0000_0000_0000;

/// `ln(1 + x)`, correctly rounded to nearest, ties to even, without forming `1 + x`.
///
/// - NaN, `-inf` and any `x < -1` give the canonical quiet NaN `0x7ff8000000000000`.
/// - `-1` gives `-inf`.
/// - `+0` and `-0` give themselves, sign kept.
/// - `0 < |x| < 2^-53`, subnormals included, gives `fma(|x|, -2^-54, x)`, rounded once.
/// - `+inf` gives `+inf`.
///
/// The bits do not depend on the platform libm. The caller keeps round-to-nearest-even
/// and gradual underflow active.
pub fn log1p(x: f64) -> f64 {
    log1p_traced(x).0
}

/// `log1p(x)` and the path that produced it.
#[doc(hidden)]
#[inline(always)]
pub fn log1p_traced(x: f64) -> (f64, Path) {
    let ix = x.to_bits();
    let ax = ix << 1;
    let (ln1, ln0, eps);
    if ax < AX_SMALL {
        // Check tiny x first to avoid a spurious underflow in x * x.
        if ax < AX_TINY {
            return (tiny(x, ax), Path::Special);
        }
        let x2 = x * x;
        if ax < AX_SERIES {
            ln1 = x;
            eps = EPS_SERIES * x;
            ln0 = if ax < AX_SHORT_SERIES {
                let c = |k| f(&LOG1P_TINY_POLY, k);
                x2 * (c(0) + x * c(1))
            } else {
                let c = |k| f(&LOG1P_SMALL_POLY, k);
                x2 * ((c(0) + x * c(1)) + x2 * (c(2) + x * c(3)))
            };
        } else {
            let c = |k| f(&LOG1P_MEDIUM_POLY, k);
            let x3 = x2 * x;
            let x4 = x2 * x2;
            let hx = -0.5 * x;
            ln1 = hx.mul_add(x, x);
            let low = hx.mul_add(x, x - ln1);
            let series = ((c(0) + x * c(1)) + x2 * (c(2) + x * c(3)))
                + x4 * (((c(4) + x * c(5)) + x2 * (c(6) + x * c(7)))
                    + x4 * ((c(8) + x * c(9)) + x2 * (c(10) + x * c(11))));
            ln0 = low + x3 * series;
            eps = x3 * EPS_MEDIUM;
        }
    } else {
        let (t, dt) = if (ix as i64) < 0x4340_0000_0000_0000 && ix < MINUS_ONE_BITS {
            // -1 < x <= -2^-4 or 2^-4 <= x < 2^53. For 1 < x < 2^53 the fast two-sum
            // stays exact: s - 1 is exact and the last step is exact by Sterbenz.
            fast_two_sum(1.0, x)
        } else if ix < 0x4690_0000_0000_0000 {
            (x, 1.0)
        } else if ix < INF_BITS {
            (x, 0.0)
        } else {
            return (special(ix, ax), Path::Special);
        };
        let (value, low) = large(t, dt);
        ln1 = value;
        ln0 = low;
        eps = EPS_LARGE;
    }
    let lb = ln1 + (ln0 - eps);
    let ub = ln1 + (ln0 + eps);
    if lb != ub {
        return (log1p_refine(x, lb), Path::Accurate);
    }
    (lb, Path::Fast)
}

/// NaN, `+inf`, `-1`, `-inf` and `x < -1`.
fn special(ix: u64, ax: u64) -> f64 {
    if ax > 0xffe0_0000_0000_0000 {
        return CANONICAL_NAN;
    }
    if ix == INF_BITS {
        return f64::INFINITY;
    }
    if ix == MINUS_ONE_BITS {
        return f64::NEG_INFINITY;
    }
    CANONICAL_NAN
}

/// `log(t + dt)` as a double-double for `|x| >= 2^-4`, where `t + dt == 1 + x` exactly or
/// up to a term below the result's precision.
#[inline(always)]
fn large(mut t: f64, mut dt: f64) -> (f64, f64) {
    let c = |k| f(&LOG1P_LARGE_POLY, k);
    let j = (t.to_bits() as i64).wrapping_sub(0x3fe6_a000_0000_0000);
    let j1 = ((j >> (52 - 6)) & 0x3f) as usize;
    let je = j >> 52;
    let eoff = (je as u64) << 52;
    let mut rs = LOG1P_RF[j1];
    if je < 1022 {
        rs = rs.wrapping_sub(eoff);
    } else {
        // Scale t and dt instead of rs, so no subnormal reciprocal appears.
        rs = rs.wrapping_sub(1021u64 << 52);
        let sc = [0.5, 0.25, 0.125][(je - 1022) as usize];
        t *= sc;
        dt *= sc;
    }
    let rs = f64::from_bits(rs);
    let dh = rs * t;
    let dl = rs.mul_add(t, -dh) + rs * dt;
    let (xh, mut xl) = fast_two_sum(dh - 1.0, dl);
    let x2 = xh * xh;
    xl += x2 * ((c(0) + xh * c(1)) + x2 * ((c(2) + xh * c(3)) + x2 * (c(4) + xh * c(5))));
    let jef = je as f64;
    let l1 = LN2_HI * jef;
    let l0 = LN2_LO * jef;
    let ln1 = f(&LOG1P_LF, 2 * j1 + 1) + l1;
    let ln0 = f(&LOG1P_LF, 2 * j1) + l0;
    fast_sum(ln1, ln0, xh, xl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log1p_special_values() {
        let nan = CANONICAL_NAN.to_bits();
        assert_eq!(log1p(f64::NAN).to_bits(), nan);
        assert_eq!(log1p(-f64::NAN).to_bits(), nan);
        assert_eq!(log1p(-2.0).to_bits(), nan);
        assert_eq!(log1p(f64::NEG_INFINITY).to_bits(), nan);
        assert_eq!(log1p(-1.0), f64::NEG_INFINITY);
        assert_eq!(log1p(0.0).to_bits(), 0);
        assert_eq!(log1p(-0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(log1p(f64::INFINITY), f64::INFINITY);
        assert_eq!(log1p(f64::MAX), 709.782_712_893_384);
        assert_eq!(log1p(f64::from_bits(1)), f64::from_bits(1));
        assert_eq!(log1p(-f64::from_bits(1)), -f64::from_bits(1));
    }

    #[test]
    fn test_log1p_paths() {
        assert_eq!(log1p_traced(1e-30).1, Path::Special);
        assert_eq!(log1p_traced(1e-8).1, Path::Fast);
        assert_eq!(log1p_traced(0.01).1, Path::Fast);
        assert_eq!(log1p_traced(3.0).1, Path::Fast);
        assert_eq!(log1p_traced(-0.5).1, Path::Fast);
    }
}
