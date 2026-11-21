//! Correctly rounded binary64 `acos`.

// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/acos/acos.c, commit 1851882b0f85ed581af9185cfa020992cec9e7a1.
// Copyright (c) 2024-2025 Alexei Sibidanov.
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

use super::acos_refine::acos_refine;
use super::double_double::{CANONICAL_NAN, fast_sum, fast_two_sum, mul_dd, round_ties_even};
use super::path::Path;
use super::tables::{ACOS_CC, f};

/// High part of `pi`, `0x1.921fb54442d18p+1`.
pub(super) const PI_HI: f64 = f64::from_bits(0x4009_21fb_5444_2d18);
/// Low part of `pi`, `0x1.1a62633145c07p-53`.
const PI_LO: f64 = f64::from_bits(0x3ca1_a626_3314_5c07);
/// High part of `pi/2`, `0x1.921fb54442d18p+0`.
pub(super) const HALF_PI_HI: f64 = f64::from_bits(0x3ff9_21fb_5444_2d18);
/// Low part of `pi/2`, `0x1.1a62633145c07p-54`.
const HALF_PI_LO: f64 = f64::from_bits(0x3c91_a626_3314_5c07);

/// `ax = bits << 1` of `|x| = 0.5`. Above it the double-angle reduction runs.
const AX_HALF: u64 = 0x7fc0_0000_0000_0000;
/// `ax` of `|x| = 1`. From it up the input is special.
const AX_ONE: u64 = 0x7fe0_0000_0000_0000;
/// `ax` of `|x| = 2^-15`. Up to it the tiny path runs.
const AX_TINY: u64 = 0x7e00_0000_0000_0000;
/// `ax` of `|x| = 0x1.cb3b3869747f4p-55`. Up to it the cubic term is left out: it cannot
/// change the result, and computing it would underflow.
const AX_NO_CUBE: u64 = 0x7919_6767_0d2e_8fe8;

/// Cubic coefficient of the tiny path, `-0x1.5555555555555p-3`.
const TINY_CUBE: f64 = f64::from_bits(0xbfc5_5555_5555_5555);
/// Tiny-path error bound, `0x1.34p-79`.
const TINY_EPS: f64 = f64::from_bits(0x3b03_4000_0000_0000);
/// Table-path relative error bound, `0x1.81p-52`.
const TABLE_EPS: f64 = f64::from_bits(0x3cb8_1000_0000_0000);
/// Double-angle relative error bound, `0x1.8cp-52`.
const DOUBLE_ANGLE_EPS: f64 = f64::from_bits(0x3cb8_c000_0000_0000);
/// Double-angle absolute error bound, `2^-105`.
const DOUBLE_ANGLE_EPS_ABS: f64 = f64::from_bits(0x3960_0000_0000_0000);
/// `2^-7`, the spacing of the `ACOS_CC` rows in `x^2`.
const ROW_STEP: f64 = f64::from_bits(0x3f80_0000_0000_0000);

/// `acos(x)` in radians, correctly rounded to nearest, ties to even.
///
/// Correct rounding on `[-1, -2^-4]` and `[2^-4, 1]` is proven: CORE-MATH ran an
/// exhaustive search there, with and without FMA contraction. For `|x| < 2^-4` no proof
/// or exhaustive search covers it. There numr checks the result against MPFR on
/// CORE-MATH's hard cases and on fixed-seed samples, and no input is known to misround.
///
/// - NaN, `+-inf` and `|x| > 1` give the canonical quiet NaN `0x7ff8000000000000`.
/// - `acos(1)` is `+0`, and `acos(-1)` is `pi` rounded, `0x1.921fb54442d18p+1`.
/// - `+-0` and every subnormal give `pi/2` rounded, `0x1.921fb54442d18p+0`.
/// - No result is subnormal: the smallest positive one is `acos(1 - 2^-53)`, near `1.5e-8`.
///
/// The bits do not depend on the platform libm. The caller keeps round-to-nearest-even
/// and gradual underflow active.
pub fn acos(x: f64) -> f64 {
    acos_traced(x).0
}

/// `acos(x)` and the path that produced it.
#[doc(hidden)]
#[inline(always)]
pub fn acos_traced(x: f64) -> (f64, Path) {
    let ix = x.to_bits();
    let ax = ix << 1;
    let (z, zl, t, jd, f0h, f0l, eps);
    if ax > AX_HALF {
        let (oh, ol) = if ix >> 63 == 1 {
            (PI_HI, PI_LO)
        } else {
            (0.0, 0.0)
        };
        if ax >= AX_ONE {
            if ax == AX_ONE {
                return (oh + ol, Path::Special);
            }
            // NaN, infinity or |x| > 1.
            return (CANONICAL_NAN, Path::Special);
        }
        // acos(x) = 2 asin(sqrt((1 - x)/2)) for x > 0.5, and
        // acos(x) = pi - 2 asin(sqrt((1 + x)/2)) for x < -0.5.
        let w = 2.0 - 2.0 * x.abs();
        jd = round_ties_even(w * 32.0);
        z = w.sqrt().copysign(x);
        zl = z.mul_add(z, -w) * ((-0.5 / w) * z);
        t = 0.25 * w - jd * ROW_STEP;
        eps = (z * t).abs() * DOUBLE_ANGLE_EPS + DOUBLE_ANGLE_EPS_ABS;
        (f0h, f0l) = (oh, ol);
    } else {
        if ax <= AX_TINY {
            return tiny(x, ax);
        }
        // acos(x) = pi/2 - asin(x), on the same asin tables as the double-angle branch.
        let w = x * x;
        jd = round_ties_even(w * 128.0);
        t = x.mul_add(x, -ROW_STEP * jd);
        z = -x;
        zl = 0.0;
        // eps is negative for x > 0, and the rounding test still holds.
        eps = (z * t) * TABLE_EPS;
        (f0h, f0l) = (HALF_PI_HI, HALF_PI_LO);
    }
    // asin(z + zl) = (z + zl) * (c0 + (c1 + t * P(t))) with t = z^2 - j/128.
    let row = 8 * jd as usize;
    let c = |k| f(&ACOS_CC, row + k);
    let t2 = t * t;
    let d = t * ((c(2) + t * c(3)) + t2 * ((c(4) + t * c(5)) + t2 * (c(6) + t * c(7))));
    let (fh, fl) = mul_dd(z, zl, c(0), c(1) + d);
    let (fh, fl) = fast_sum(f0h, f0l, fh, fl);
    let lb = fh + (fl - eps);
    let ub = fh + (fl + eps);
    if lb != ub {
        return acos_refine(x, lb);
    }
    (lb, Path::Fast)
}

/// `|x| <= 2^-15`: `acos(x) = pi/2 - x - x^3/6`, with the cubic term left out where it
/// cannot change the result.
#[inline(always)]
fn tiny(x: f64, ax: u64) -> (f64, Path) {
    let v = if ax <= AX_NO_CUBE {
        0.0
    } else {
        (x * x) * (TINY_CUBE * x)
    };
    let (h, w) = fast_two_sum(HALF_PI_HI, -x);
    let l = v + (w + HALF_PI_LO);
    let lb = h + (l - TINY_EPS);
    let ub = h + (l + TINY_EPS);
    if lb != ub {
        return acos_refine(x, lb);
    }
    (lb, Path::Fast)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, PI};

    #[test]
    fn test_acos_special_values() {
        let nan = CANONICAL_NAN.to_bits();
        for bits in [
            0x7ff8_0000_0000_0000u64,
            0xfff8_0000_0000_0000,
            0x7ff0_0000_0000_0001,
            0xfff4_0000_0000_dead,
            0x7ff8_0000_0000_beef,
        ] {
            assert_eq!(acos(f64::from_bits(bits)).to_bits(), nan, "{bits:#x}");
        }
        for x in [
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.0 + f64::EPSILON,
            -2.0,
            f64::MAX,
        ] {
            assert_eq!(acos(x).to_bits(), nan, "{x:e}");
            assert_eq!(acos_traced(x).1, Path::Special);
        }
        assert_eq!(acos(1.0).to_bits(), 0);
        assert_eq!(acos(-1.0), PI);
        assert_eq!(acos_traced(-1.0).1, Path::Special);
    }

    #[test]
    fn test_acos_of_zero_and_subnormals_is_half_pi() {
        for bits in [0u64, 1, 0x000f_ffff_ffff_ffff, 0x0010_0000_0000_0000] {
            for x in [f64::from_bits(bits), -f64::from_bits(bits)] {
                assert_eq!(acos_traced(x), (FRAC_PI_2, Path::Fast), "{x:e}");
            }
        }
    }

    #[test]
    fn test_acos_anchor_values() {
        assert_eq!(acos(0.5), FRAC_PI_3);
        assert_eq!(acos(-0.5), f64::from_bits(0x4000_c152_382d_7366));
        assert_eq!(acos_traced(0.5).1, Path::Fast);
        // 1 - 2^-53 gives sqrt(2^-52) * (1 + 2^-55 + ...), which rounds to 2^-26.
        assert_eq!(
            acos(1.0 - f64::EPSILON / 2.0),
            f64::from_bits(0x3e50_0000_0000_0000)
        );
    }

    #[test]
    fn test_each_branch_serves_ordinary_inputs_on_the_fast_path() {
        // Tiny, table and double-angle branches.
        for x in [1.0e-300, 3.0e-5, -0.25, 0.3, 0.75, -0.9] {
            assert_eq!(acos_traced(x).1, Path::Fast, "{x:e}");
        }
    }
}
