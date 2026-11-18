//! Products of [`Qint`] values for the extended `pow` path.
//!
//! `mul_ab` uses the upper `a` limbs of the first operand and the upper `b` limbs of the
//! second. Each drops the same partial products as CORE-MATH, so its error bound holds.

// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/pow/qint.h, commit 1851882b0f85ed581af9185cfa020992cec9e7a1.
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

use super::qint::Qint;

const LOW64: u128 = u64::MAX as u128;

#[inline(always)]
fn p(x: u64, y: u64) -> u128 {
    (x as u128) * (y as u128)
}

#[inline(always)]
fn carry(sum: (u128, bool)) -> (u128, u128) {
    (sum.0, u128::from(sum.1))
}

/// Normalize the upper 256 bits `t6 : low` of a product so the top bit is set.
#[inline(always)]
fn finish(t6: u128, low: u128, a: Qint, b: Qint) -> Qint {
    let ex = u64::from(t6 >> 127 == 0);
    let (rh, rl) = if ex == 1 {
        ((t6 << 1) | (low >> 127), low << 1)
    } else {
        (t6, low)
    };
    Qint::from_halves(rh, rl, a.ex + b.ex + 1 - ex as i64, a.sgn ^ b.sgn)
}

/// Sum the terms of "degree" 4 and 5 above `t4`, then assemble the top 256 bits.
#[inline(always)]
fn assemble(r33: u128, t5: u128, c5: u128, t4: u128, c4: u128, a: Qint, b: Qint) -> Qint {
    let t6 = r33.wrapping_add((c5 << 64) | (t5 >> 64)).wrapping_add(c4);
    finish(t6, (t5 << 64) | (t4 & LOW64), a, b)
}

impl Qint {
    /// `a * b` with error below 14 ulps.
    pub fn mul(a: Qint, b: Qint) -> Qint {
        let r33 = p(a.hh, b.hh);
        let r32 = p(a.hh, b.hl);
        let r23 = p(a.hl, b.hh);
        let r31 = p(a.hh, b.lh);
        let r13 = p(a.lh, b.hh);
        let r22 = p(a.hl, b.hl);
        let r30 = p(a.hh, b.ll);
        let r03 = p(a.ll, b.hh);
        let r21 = p(a.hl, b.lh);
        let r12 = p(a.lh, b.hl);
        let t3 = (r12 >> 64) + (r21 >> 64) + (r03 >> 64) + (r30 >> 64);
        let (t4, mut c4) = carry(r22.overflowing_add(t3));
        let (t4, c) = carry(r13.overflowing_add(t4));
        c4 += c;
        let (t4, c) = carry(r31.overflowing_add(t4));
        c4 += c;
        let (t5, mut c5) = carry(r23.overflowing_add(t4 >> 64));
        let (t5, c) = carry(r32.overflowing_add(t5));
        c5 += c;
        assemble(r33, t5, c5, t4, c4, a, b)
    }

    /// `a * b` on the upper three limbs of each, with error below 6 ulps.
    pub fn mul_33(a: Qint, b: Qint) -> Qint {
        let r33 = p(a.hh, b.hh);
        let r32 = p(a.hh, b.hl);
        let r23 = p(a.hl, b.hh);
        let r31 = p(a.hh, b.lh);
        let r13 = p(a.lh, b.hh);
        let r22 = p(a.hl, b.hl);
        let r21 = p(a.hl, b.lh);
        let r12 = p(a.lh, b.hl);
        let t3 = (r12 >> 64) + (r21 >> 64);
        let (t4, mut c4) = carry(r22.overflowing_add(t3));
        let (t4, c) = carry(r13.overflowing_add(t4));
        c4 += c;
        let (t4, c) = carry(r31.overflowing_add(t4));
        c4 += c;
        let (t5, mut c5) = carry(r23.overflowing_add(t4 >> 64));
        let (t5, c) = carry(r32.overflowing_add(t5));
        c5 += c;
        assemble(r33, t5, c5, t4, c4, a, b)
    }

    /// `a * b` on all limbs of `a` and the top limb of `b`, with error below 2 ulps.
    pub fn mul_41(a: Qint, b: Qint) -> Qint {
        let r33 = p(a.hh, b.hh);
        let r23 = p(a.hl, b.hh);
        let r13 = p(a.lh, b.hh);
        let r03 = p(a.ll, b.hh);
        let (t4, c4) = carry(r13.overflowing_add(r03 >> 64));
        let (t5, c5) = carry(r23.overflowing_add(t4 >> 64));
        assemble(r33, t5, c5, t4, c4, a, b)
    }

    /// The exact product of the upper three limbs of `a` and the top limb of `b`.
    pub fn mul_31(a: Qint, b: Qint) -> Qint {
        let r33 = p(a.hh, b.hh);
        let r23 = p(a.hl, b.hh);
        let t4 = p(a.lh, b.hh);
        let (t5, c5) = carry(r23.overflowing_add(t4 >> 64));
        assemble(r33, t5, c5, t4, 0, a, b)
    }

    /// The exact product of the upper two limbs of `a` and `b`.
    pub fn mul_22(a: Qint, b: Qint) -> Qint {
        let r33 = p(a.hh, b.hh);
        let r32 = p(a.hh, b.hl);
        let r23 = p(a.hl, b.hh);
        let t4 = p(a.hl, b.hl);
        let (t5, mut c5) = carry(r23.overflowing_add(t4 >> 64));
        let (t5, c) = carry(r32.overflowing_add(t5));
        c5 += c;
        assemble(r33, t5, c5, t4, 0, a, b)
    }

    /// The exact product of the upper two limbs of `a` and the top limb of `b`.
    pub fn mul_21(a: Qint, b: Qint) -> Qint {
        let r33 = p(a.hh, b.hh);
        let r23 = p(a.hl, b.hh);
        finish(r33.wrapping_add(r23 >> 64), r23 << 64, a, b)
    }

    /// The exact product of the top limbs of `a` and `b`.
    pub fn mul_11(a: Qint, b: Qint) -> Qint {
        finish(p(a.hh, b.hh), 0, a, b)
    }

    /// `b * a` for an integer `b`, with error below 2 ulps.
    pub fn mul_int(b: i64, a: Qint) -> Qint {
        if b == 0 {
            return Qint::ZERO;
        }
        let sgn = u64::from(b < 0) ^ a.sgn;
        let mut c = b.unsigned_abs();
        if c == 1 {
            return Qint { sgn, ..a };
        }
        // Scale c so that 2^63 <= c < 2^64.
        let k = c.leading_zeros();
        c <<= k;
        let mut ex = a.ex + 64 - k as i64;
        let mut t3 = p(a.hh, c);
        let t2 = p(a.hl, c);
        let t1 = p(a.lh, c);
        let t0 = p(a.ll, c);
        let (t1, cy) = carry((t0 >> 64).overflowing_add(t1));
        let (t2, cy) = carry(((cy << 64) | (t1 >> 64)).overflowing_add(t2));
        t3 = t3.wrapping_add((cy << 64) | (t2 >> 64));
        let low = (t2 << 64) | (t1 & LOW64);
        // The top bit of t3 is clear in at most one position, as a and c are normalized.
        let (rh, rl) = if (t3 >> 64) as u64 >> 63 == 0 {
            ex -= 1;
            ((t3 << 1) | (low >> 127), low << 1)
        } else {
            (t3, low)
        };
        Qint::from_halves(rh, rl, ex, sgn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(x: f64) -> Qint {
        Qint::from_f64(x)
    }

    #[test]
    fn test_products_match_binary64_on_short_operands() {
        for (x, y) in [(1.5, -0.25), (3.0, 7.0), (0.1, 10.0), (-1.0e100, 1.0e-200)] {
            for product in [
                Qint::mul(q(x), q(y)),
                Qint::mul_33(q(x), q(y)),
                Qint::mul_41(q(x), q(y)),
                Qint::mul_31(q(x), q(y)),
                Qint::mul_22(q(x), q(y)),
                Qint::mul_21(q(x), q(y)),
                Qint::mul_11(q(x), q(y)),
            ] {
                assert_eq!(product.to_f64(), x * y, "{x} * {y}");
            }
        }
        assert_eq!(Qint::mul_int(-1074, q(1.5)).to_f64(), -1611.0);
        assert_eq!(Qint::mul_int(1, q(-0.75)).to_f64(), -0.75);
        assert_eq!(Qint::mul_int(4_399_162, q(0.5)).to_f64(), 2_199_581.0);
        assert_eq!(Qint::mul_int(0, q(2.0)), Qint::ZERO);
    }

    #[test]
    fn test_full_product_keeps_low_limbs() {
        // (1 + 2^-192)^2 = 1 + 2^-191 + 2^-384: the 2^-191 bit sits in the third limb.
        let a = Qint::from_halves(1 << 127, 1 << 63, 0, 0);
        let r = Qint::mul(a, a);
        assert_eq!((r.rh(), r.rl(), r.ex), (1 << 127, 1 << 64, 0));
    }
}
