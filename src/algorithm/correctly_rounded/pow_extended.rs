//! Extended path of `pow`: `log|x|` and `exp` with 256-bit significands.

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
use super::pow_accurate::SQRT2_HI;
use super::qint::Qint;
use super::tables::{
    LOG_DINT_MINUS_ONE, POW_DINT_INVERSE_1, POW_DINT_INVERSE_2, POW_DINT_LOG2_INV,
    POW_QINT_EXP_POLY, POW_QINT_LN2, POW_QINT_LOG_INV_1, POW_QINT_LOG_INV_2, POW_QINT_LOG_POLY,
    POW_QINT_T1, POW_QINT_T2,
};

/// Margin of the extended rounding test: `floor(62433 / 2^10)` units of the 64 bits after
/// the third limb's round position.
const ERR_BND_3: u64 = 60;

/// A 128-bit table entry as a [`Qint`]: the 256-bit reciprocal tables equal the 128-bit ones.
#[inline(always)]
fn widened(table: &[u64], index: usize) -> Qint {
    Qint::from_dint(Dint::from_table(table, index))
}

/// `log(1 + z)` for `|z| <= 2^-13` with only the top limb nonzero: relative error below
/// `2^-252.66`.
fn p_3(z: Qint) -> Qint {
    let c = |k| Qint::from_table(&POW_QINT_LOG_POLY, k);
    let mut r = Qint::mul_11(c(0), z);
    r = Qint::add_22(c(1), r);
    for k in 2..4 {
        r = Qint::mul_11(r, z);
        r = Qint::add_22(c(k), r);
    }
    for k in 4..8 {
        r = Qint::mul_21(r, z);
        r = Qint::add_22(c(k), r);
    }
    for k in 8..14 {
        r = Qint::mul_31(r, z);
        r = Qint::add(c(k), r);
    }
    for k in 14..18 {
        r = Qint::mul_41(r, z);
        r = Qint::add(c(k), r);
    }
    Qint::mul_41(r, z)
}

/// `log(x)` for positive `x` with 53 significant bits, relative error below `2^-250.74`.
fn log_3(x: Qint) -> Qint {
    let mut e = x.ex;
    let i = if x.hh > SQRT2_HI {
        e += 1;
        (x.hh >> 57) as usize
    } else {
        (x.hh >> 56) as usize
    };
    let x = Qint { ex: x.ex - e, ..x };
    let z = Qint::mul(x, widened(&POW_DINT_INVERSE_1, i - 90));
    let j = (z.hh >> (50 - z.ex)) as usize;
    let z = Qint::mul(z, widened(&POW_DINT_INVERSE_2, j - 8128));
    // |z - 1| <= 2^-13 fits in 64 bits, and the subtraction is exact.
    let z = Qint::add(widened(&LOG_DINT_MINUS_ONE, 0), z);
    let r = Qint::mul_int(e, Qint::from_table(&POW_QINT_LN2, 0));
    let p = p_3(z);
    // Smallest terms first.
    let p = Qint::add(Qint::from_table(&POW_QINT_LOG_INV_2, j - 8128), p);
    let p = Qint::add(Qint::from_table(&POW_QINT_LOG_INV_1, i - 90), p);
    Qint::add(p, r)
}

/// `exp(y)` for `|y| < 0.00016923`, absolute and relative error below `2^-241.11`.
fn q_3(y: Qint) -> Qint {
    let c = |k| Qint::from_table(&POW_QINT_EXP_POLY, k);
    // Degrees 13 and 14 have 64 bits, degrees 8 to 12 have 128 bits.
    let mut r = Qint::mul_11(y, c(0));
    r = Qint::add_22(c(1), r);
    for k in 2..7 {
        r = Qint::mul_22(y, r);
        r = Qint::add_22(c(k), r);
    }
    for k in 7..12 {
        r = Qint::mul_33(y, r);
        r = Qint::add(c(k), r);
    }
    for k in 12..15 {
        r = Qint::mul(y, r);
        r = Qint::add(c(k), r);
    }
    r
}

/// `exp(x)` for `|x| < 744.45`, relative error below `2^-241.10`.
///
/// For `|x| >= 2^10` the result is the overflow or underflow marker of the accurate path.
/// Upstream has no such check here and would shift by 64 or more for `|x| >= 2^64`.
fn exp_3(x: Qint) -> Qint {
    if x.ex >= 10 {
        return Qint {
            ex: if x.sgn == 1 { -1076 } else { 1025 },
            sgn: 0,
            ..x
        };
    }
    let k = Qint::mul_11(x, widened(&POW_DINT_LOG2_INV, 0)).to_i64();
    let mut kl = Qint::mul_int(k, Qint::from_table(&POW_QINT_LN2, 0));
    kl.ex -= 12;
    kl.sgn = u64::from(kl.sgn == 0);
    let y = Qint::add(x, kl);
    let i2 = ((k >> 6) & 0x3f) as usize;
    let i1 = (k & 0x3f) as usize;
    let r = q_3(y);
    let r = Qint::mul(Qint::from_table(&POW_QINT_T1, i2), r);
    let r = Qint::mul(Qint::from_table(&POW_QINT_T2, i1), r);
    Qint {
        ex: r.ex + (k >> 12),
        ..r
    }
}

/// The extended result for finite nonzero `x` and `y`, with sign `s`.
///
/// Returns the value and whether the rounding test certified it. `|x|^y` lies within
/// 62433 ulps of 256 bits from the approximation.
pub fn pow_3(x: f64, y: f64, s: f64) -> (f64, bool) {
    let mut qx = Qint::from_f64(x);
    qx.sgn = 0;
    let qr = Qint::mul_41(log_3(qx), Qint::from_f64(y));
    let mut qz = exp_3(qr);
    let r1 = (qz.hh << 54) | (qz.hl >> 10);
    let r2 = (qz.hl << 54) | (qz.lh >> 10);
    let r3 = (qz.lh << 54) | (qz.ll >> 10);
    // Upstream tests the all-ones side as r3 + 120 <= 60 modulo 2^64, which misses
    // r3 > 2^64 - 60. The symmetric test sends those to the uncertified branch, which
    // returns the same bits.
    let near_boundary =
        (r1 == 0 && r2 == 0 && r3 <= ERR_BND_3) || (!r1 == 0 && !r2 == 0 && !r3 <= ERR_BND_3);
    qz.sgn = u64::from(s == -1.0);
    qz.ll &= u64::MAX << 10;
    if !near_boundary {
        return (qz.to_f64(), true);
    }
    if qr.ex < -56 {
        // |y log|x|| < 2^-55: 1 - 2^-54 < |x|^y < 1 + 2^-53 rounds to 1, and s = 1 here.
        return (1.0, true);
    }
    (qz.to_f64(), false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow_3_matches_known_values() {
        assert_eq!(pow_3(2.0, 0.5, 1.0), (std::f64::consts::SQRT_2, true));
        assert_eq!(pow_3(std::f64::consts::E, 1.0, 1.0).0, std::f64::consts::E);
        assert_eq!(pow_3(3.0, 3.0, -1.0).0, -27.0);
        assert_eq!(pow_3(2.0, 2000.0, 1.0).0, f64::INFINITY);
        assert_eq!(pow_3(2.0, -2000.0, 1.0).0, 0.0);
        // 2^(2^-60) is within 2^-60 of 1.
        assert_eq!(pow_3(2.0, 2f64.powi(-60), 1.0), (1.0, true));
    }

    #[test]
    fn test_log_3_of_one_is_zero() {
        let r = log_3(Qint::from_f64(1.0));
        assert_eq!(r.rh() | r.rl(), 0);
    }
}
