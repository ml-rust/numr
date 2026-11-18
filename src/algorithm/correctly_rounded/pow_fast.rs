//! Fast path of `pow`: `log|x|` and `exp` in double-double arithmetic.
//!
//! Lemma numbers refer to Hubrecht, Jeannerod and Zimmermann, "Towards a correctly-rounded
//! and fast power function in binary64 arithmetic", ARITH 2023, detailed version
//! <https://inria.hal.science/hal-04159652>.

// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/pow/pow.c and src/binary64/pow/pow.h,
// commit 1851882b0f85ed581af9185cfa020992cec9e7a1.
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

use super::double_double::{fast_two_sum, round_ties_even, two_prod};
use super::tables::{
    POW_FAST_EXP_POLY, POW_FAST_LOG_POLY, POW_INVERSE, POW_LOG_INV, POW_T1, POW_T2, f,
};

/// High part of `log(2)`, `0x1.62e42fefa38p-1`: `E * LOG2_H` is exact for `|E| <= 1075`.
const LOG2_H: f64 = f64::from_bits(0x3fe6_2e42_fefa_3800);
/// Low part of `log(2)`, `0x1.ef35793c7673p-45`.
const LOG2_L: f64 = f64::from_bits(0x3d2e_f357_93c7_6730);
/// `2^52 * sqrt(2)` rounded up, as a significand: `0x16a09e667f3bcd`.
const SQRT2_SIGNIFICAND: u64 = 0x16_a09e_667f_3bcd;

/// `exp_1` gives `+-0` below `RHO0`, `-0x1.74910ee4e8a27p+9`.
const RHO0: f64 = f64::from_bits(0xc087_4910_ee4e_8a27);
/// `exp_1` defers below `RHO1`, `-0x1.483b8cca421afp+9`, where the result can be subnormal.
const RHO1: f64 = f64::from_bits(0xc084_83b8_cca4_21af);
/// `exp_1` defers above `RHO2`, `0x1.62e42e709a95bp+9`, where the result can overflow.
const RHO2: f64 = f64::from_bits(0x4086_2e42_e709_a95b);
/// `exp_1` gives `+-DBL_MAX` pairs above `RHO3`, `0x1.62e4316ea5df9p+9`, which round to `+-inf`.
const RHO3: f64 = f64::from_bits(0x4086_2e43_16ea_5df9);
/// `2^12 / log(2)`, `0x1.71547652b82fep+12`.
const INVLOG2: f64 = f64::from_bits(0x40b7_1547_652b_82fe);
/// High part of `log(2) / 2^12`, `0x1.62e42fefa39efp-13`.
const LOG2H: f64 = f64::from_bits(0x3f26_2e42_fefa_39ef);
/// Low part of `log(2) / 2^12`, `0x1.abc9e3b39803fp-68`.
const LOG2L: f64 = f64::from_bits(0x3bba_bc9e_3b39_803f);

/// Fast-path error bound of the result, `0x1.27p-64 > 2^-63.797`.
pub const FAST_ERR: f64 = f64::from_bits(0x3bf2_7000_0000_0000);
/// Fast-path error bound when `log|x|` cancels near `x = 1`, `0x1.57p-58 > 2^-57.579`.
pub const FAST_ERR_CANCEL: f64 = f64::from_bits(0x3c55_7000_0000_0000);

/// `(hi, lo)` of `a + (bh + bl)`. Needs `|a| >= |bh|`.
#[inline(always)]
fn fast_sum(a: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (hi, lo) = fast_two_sum(a, bh);
    (hi, lo + bl)
}

/// `a * (bh + bl)`: the high product is exact, `a * bl` is fused into the low part.
#[inline(always)]
pub fn s_mul(a: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (hi, s) = two_prod(a, bh);
    (hi, a.mul_add(bl, s))
}

/// `(ah + al) * (bh + bl) - al * bl`.
#[inline(always)]
fn d_mul(ah: f64, al: f64, bh: f64, bl: f64) -> (f64, f64) {
    let (hi, s) = two_prod(ah, bh);
    let t = al.mul_add(bh, s);
    (hi, ah.mul_add(bl, t))
}

/// `log(1 + z) - z` for `|z| <= 33 * 2^-13`, error below `2^-75.492` (Lemma 2).
#[inline(always)]
fn p_1(z: f64) -> (f64, f64) {
    let c = |i| f(&POW_FAST_LOG_POLY, i);
    let (wh, wl) = two_prod(z, z);
    let t = c(5).mul_add(z, c(4));
    let u = c(3).mul_add(z, c(2));
    let v = c(1).mul_add(z, c(0));
    let u = t.mul_add(wh, u);
    let v = u.mul_add(wh, v);
    let u = v * wh;
    (-0.5 * wh, u.mul_add(z, -0.5 * wl))
}

/// `log(x)` as `(h, l, cancel)` for `2^-1074 <= x < 2^1024` (Lemma 4).
///
/// The relative error is below `2^-73.527`, or `2^-67.0544` for `1/sqrt(2) < x < sqrt(2)`.
/// `cancel` selects the larger fast-path bound.
pub fn log_1(x: f64) -> (f64, f64, bool) {
    let bits = x.to_bits();
    let mut m = bits & (u64::MAX >> 12);
    let mut e = ((bits >> 52) & 0x7ff) as i64;
    let t = if e != 0 {
        let t = f64::from_bits(m | (0x3ff << 52));
        m += 1 << 52;
        e -= 0x3ff;
        t
    } else {
        // Subnormal x: normalize the significand.
        let k = m.leading_zeros() - 11;
        e = -0x3fe - k as i64;
        m <<= k;
        f64::from_bits(m | (0x3ff << 52))
    };
    // Now x = 2^e * t with 1 <= t < 2, and 2^52 <= m < 2^53.
    let c = m >= SQRT2_SIGNIFICAND;
    e += i64::from(c);
    let (i, t) = if c {
        ((m >> 45) as usize, t * 0.5)
    } else {
        ((m >> 44) as usize, t)
    };
    // Now 1/sqrt(2) <= t < sqrt(2) and log(x) = e * log(2) + log(t).
    let big_e = e as f64;
    let r = f(&POW_INVERSE, i - 181);
    let l1 = f(&POW_LOG_INV, 2 * (i - 181));
    let l2 = f(&POW_LOG_INV, 2 * (i - 181) + 1);
    let z = r.mul_add(t, -1.0);
    let th = big_e.mul_add(LOG2_H, l1);
    let tl = big_e.mul_add(LOG2_L, l2);
    let (h, l) = fast_sum(th, z, tl);
    let (ph, pl) = p_1(z);
    let (h, l) = fast_sum(h, ph, l + pl);
    if e == 0 && l.abs() > h.abs() * f64::from_bits(0x3e70_0000_0000_0000) {
        let (h, l) = fast_two_sum(h, l);
        return (h, l, true);
    }
    (h, l, false)
}

/// `exp(z)` for `|z| < 2^-12.905`, relative error below `2^-64.902632` (Lemma 6).
#[inline(always)]
fn q_1(z: f64) -> (f64, f64) {
    let c = |i| f(&POW_FAST_EXP_POLY, i);
    let q = c(4).mul_add(z, c(3));
    let q = q.mul_add(z, c(2));
    let h0 = q.mul_add(z, c(1));
    let (h1, l1) = two_prod(z, h0);
    fast_sum(c(0), h1, l1)
}

/// `s * exp(rh + rl)` with relative error below `2^-63.78597` (Lemma 7).
///
/// Returns a NaN pair where the result can overflow or be subnormal: the rounding test
/// then fails and the accurate path decides.
pub fn exp_1(rh: f64, rl: f64, s: f64) -> (f64, f64) {
    if rh.is_nan() || rh > RHO2 {
        if !rh.is_nan() && rh > RHO3 {
            return (f64::MAX * s, f64::MAX * s);
        }
        return (f64::NAN, f64::NAN);
    }
    if rh < RHO1 {
        if rh < RHO0 {
            // Both parts round to a zero of the result's sign.
            return (0.0 * s, f64::from_bits(1) * (0.5 * s));
        }
        return (f64::NAN, f64::NAN);
    }
    let k = round_ties_even(rh * INVLOG2);
    let zh = LOG2H.mul_add(-k, rh);
    let zl = LOG2L.mul_add(-k, rl);
    let ki = k as i64;
    let scale = ((ki >> 12) + 0x3ff) as u64;
    let i2 = ((ki >> 6) & 0x3f) as usize;
    let i1 = (ki & 0x3f) as usize;
    let (eh, el) = d_mul(
        f(&POW_T2, 2 * i1),
        f(&POW_T2, 2 * i1 + 1),
        f(&POW_T1, 2 * i2),
        f(&POW_T1, 2 * i2 + 1),
    );
    let (qh, ql) = q_1(zh + zl);
    let (eh, el) = d_mul(eh, el, qh, ql);
    // 1 < scale < 2047: the range checks above exclude overflow and underflow.
    let d = f64::from_bits(scale << 52) * s;
    (eh * d, el * d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_1_is_close_to_log() {
        for x in [
            0.5,
            2.0,
            10.0,
            1.0e-300,
            1.0e300,
            f64::from_bits(3),
            1.25,
            0.8,
        ] {
            let (h, l, _) = log_1(x);
            let want = x.ln();
            assert!(
                ((h + l) - want).abs() <= 4.0 * f64::EPSILON * want.abs(),
                "{x}"
            );
        }
        assert_eq!(log_1(1.0).0, 0.0);
    }

    #[test]
    fn test_exp_1_is_close_to_exp() {
        for r in [0.0, 1.0, -1.0, 700.0, -600.0, 1.0e-10] {
            let (h, l) = exp_1(r, 0.0, 1.0);
            let want = r.exp();
            assert!(((h + l) - want).abs() <= 4.0 * f64::EPSILON * want, "{r}");
        }
        let (h, l) = exp_1(1.0, 0.0, -1.0);
        assert!(h + l < 0.0);
        assert!(exp_1(-720.0, 0.0, 1.0).0.is_nan());
        assert!(exp_1(0.5 * (RHO2 + RHO3), 0.0, 1.0).0.is_nan());
        assert_eq!(exp_1(711.0, 0.0, 1.0).0, f64::MAX);
        assert_eq!(exp_1(-746.0, 0.0, -1.0).0.to_bits(), (-0.0f64).to_bits());
    }
}
