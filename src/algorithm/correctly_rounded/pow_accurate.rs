//! Accurate path of `pow`: `log|x|` and `exp` with 128-bit significands.

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

use super::dint::Dint;
use super::pow_dint::{add, add_11, mul, mul_11, mul_21, mul_int64, to_i64};
use super::tables::{
    LOG_DINT_LN2, LOG_DINT_MINUS_ONE, POW_DINT_EXP_POLY, POW_DINT_INVERSE_1, POW_DINT_INVERSE_2,
    POW_DINT_LOG_INV_1, POW_DINT_LOG_INV_2, POW_DINT_LOG_POLY, POW_DINT_LOG2_INV, POW_DINT_T1,
    POW_DINT_T2,
};

/// Top 64 bits of `sqrt(2)` as a significand.
pub const SQRT2_HI: u64 = 0xb504_f333_f9de_6484;

/// `log(1 + z)` for `|z| <= 2^-13` with a zero low word: relative error below `2^-124.82`,
/// absolute error below `2^-137.95`.
fn p_2(z: Dint) -> Dint {
    let c = |k| Dint::from_table(&POW_DINT_LOG_POLY, k);
    // The degree 6 to 9 coefficients have 64 bits.
    let mut r = mul_11(z, c(0));
    r = add_11(c(1), r);
    for k in 2..4 {
        r = mul_11(z, r);
        r = add_11(c(k), r);
    }
    r = mul_11(z, r);
    r = add(c(4), r);
    for k in 5..9 {
        r = mul_21(r, z);
        r = add(c(k), r);
    }
    mul_21(r, z)
}

/// `log(x)` for positive `x` with 53 significant bits, relative error below `2^-122.88`.
pub fn log_2(x: Dint) -> Dint {
    let mut e = x.ex;
    // i = floor(x * 2^7) after scaling x into (sqrt(2)/2, sqrt(2)).
    let i = if x.hi > SQRT2_HI {
        e += 1;
        (x.hi >> 57) as usize
    } else {
        (x.hi >> 56) as usize
    };
    let x = Dint { ex: x.ex - e, ..x };
    // 0.9921875 <= z <= 1.0078125, exact: 53 bits times 9 bits fit in 64.
    let z = mul_11(x, Dint::from_table(&POW_DINT_INVERSE_1, i - 90));
    let j = (z.hi >> (50 - z.ex)) as usize;
    let z = mul_11(z, Dint::from_table(&POW_DINT_INVERSE_2, j - 8128));
    // |z - 1| <= 2^-13, and the subtraction is exact.
    let z = add(Dint::from_table(&LOG_DINT_MINUS_ONE, 0), z);
    let r = mul_int64(Dint::from_table(&LOG_DINT_LN2, 0), e);
    let p = p_2(z);
    let p = add(Dint::from_table(&POW_DINT_LOG_INV_2, j - 8128), p);
    let p = add(Dint::from_table(&POW_DINT_LOG_INV_1, i - 90), p);
    add(p, r)
}

/// `exp(y)` for `|y| < 0.00016923`, relative error below `2^-122.29`.
fn q_2(y: Dint) -> Dint {
    let c = |k| Dint::from_table(&POW_DINT_EXP_POLY, k);
    let mut r = mul_11(y, c(0));
    r = add_11(c(1), r);
    r = mul_11(y, r);
    r = add_11(c(2), r);
    r = mul_11(y, r);
    r = add(c(3), r);
    for k in 4..8 {
        r = mul(y, r);
        r = add(c(k), r);
    }
    r
}

/// `exp(x)` for `|x| < 744.45`, relative error below `2^-121.70`.
///
/// For `|x| >= 2^10` the result is a marker with exponent `1025` (overflow) or `-1076`
/// (underflow), which rounds to `inf` or `0`.
pub fn exp_2(x: Dint) -> Dint {
    if x.ex >= 10 {
        return Dint {
            ex: if x.sgn == 1 { -1076 } else { 1025 },
            sgn: 0,
            ..x
        };
    }
    // Only used to round x * 2^12 / log(2) to an integer.
    let k = to_i64(mul_11(x, Dint::from_table(&POW_DINT_LOG2_INV, 0)));
    // |k| <= 4399162. x - k * log(2) / 2^12 must be accurate: full product, exact sum.
    let mut kl = mul_int64(Dint::from_table(&LOG_DINT_LN2, 0), k);
    kl.ex -= 12;
    kl.sgn = u64::from(kl.sgn == 0);
    let y = add(x, kl);
    let i2 = ((k >> 6) & 0x3f) as usize;
    let i1 = (k & 0x3f) as usize;
    let r = q_2(y);
    let r = mul(Dint::from_table(&POW_DINT_T1, i2), r);
    let r = mul(Dint::from_table(&POW_DINT_T2, i1), r);
    Dint {
        ex: r.ex + (k >> 12),
        ..r
    }
}

/// `|x|^y` with relative error below `2^-113.17`, for finite nonzero `x` and `y`.
pub fn pow_2(x: f64, y: f64) -> Dint {
    let mut xd = Dint::from_f64(x);
    xd.sgn = 0;
    exp_2(mul_21(log_2(xd), Dint::from_f64(y)))
}

/// Whether the rounding of `r` to binary64 is certain under the accurate-path error bound.
///
/// `r` lies within 29126 ulps of 128 bits from `|x|^y`. In the normal range the 64 bits
/// after the round bit must keep more than 28 units from 0 and from `2^64`. In the
/// subnormal range the round bit moves up by `ex` bits, and the margin is 14 units.
pub fn rounding_is_certain(r: Dint) -> bool {
    if r.ex < -1022 {
        // -1075 <= r.ex <= -1023, so 1 <= ex <= 53.
        let ex = (-(1022 + r.ex)) as u32;
        let m = (r.lo >> (10 + ex)) | (r.hi << (54 - ex));
        m.wrapping_add(14) > 28
    } else {
        let lo = (r.lo >> 10) | (r.hi << 54);
        lo.wrapping_add(28) > 56
    }
}

#[cfg(test)]
mod tests {
    use super::super::pow_dint::to_f64;
    use super::*;

    #[test]
    fn test_log_2_matches_known_values() {
        let ln = |x: f64| to_f64(log_2(Dint::from_f64(x)));
        assert_eq!(ln(2.0), std::f64::consts::LN_2);
        assert_eq!(ln(10.0), std::f64::consts::LN_10);
        assert_eq!(ln(0.5), -std::f64::consts::LN_2);
        assert_eq!(ln(1.0), 0.0);
    }

    #[test]
    fn test_exp_2_matches_known_values() {
        let e = |x: f64| to_f64(exp_2(Dint::from_f64(x)));
        assert_eq!(e(1.0), std::f64::consts::E);
        assert_eq!(e(std::f64::consts::LN_2), 2.0);
        assert_eq!(e(2000.0), f64::INFINITY);
        assert_eq!(e(-2000.0), 0.0);
    }

    #[test]
    fn test_pow_2_is_accurate() {
        assert_eq!(to_f64(pow_2(2.0, 0.5)), std::f64::consts::SQRT_2);
        assert_eq!(to_f64(pow_2(10.0, 3.0)), 1000.0);
        assert!(!rounding_is_certain(pow_2(10.0, 3.0)));
        assert!(rounding_is_certain(pow_2(2.0, 0.5)));
    }
}
