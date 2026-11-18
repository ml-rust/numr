//! 128-bit significand arithmetic for the accurate `log` path.
//!
//! A [`Dint`] is `(-1)^sgn * (hi + lo / 2^64) * 2^(ex - 63)` with the top bit of `hi` set.
//! Every operation is integer arithmetic, so it is exact or truncates in a fixed way.

// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/log/dint.h and src/binary64/log/log.c,
// commit 1851882b0f85ed581af9185cfa020992cec9e7a1.
// Copyright (c) 2022 CERN. Author: Tom Hubrecht.
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

use std::cmp::Ordering;

/// A sign, a 128-bit significand and an exponent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dint {
    /// High 64 significand bits.
    pub hi: u64,
    /// Low 64 significand bits.
    pub lo: u64,
    /// Exponent of the leading significand bit.
    pub ex: i64,
    /// `1` for negative, `0` for positive.
    pub sgn: u64,
}

impl Dint {
    /// Zero.
    pub const ZERO: Dint = Dint {
        hi: 0,
        lo: 0,
        ex: 0,
        sgn: 0,
    };

    /// Entry `index` of a table of `(hi, lo, ex, sgn)` words.
    #[inline(always)]
    pub fn from_table(table: &[u64], index: usize) -> Dint {
        Dint {
            hi: table[4 * index],
            lo: table[4 * index + 1],
            ex: table[4 * index + 2] as i64,
            sgn: table[4 * index + 3],
        }
    }

    #[inline(always)]
    fn significand(self) -> u128 {
        ((self.hi as u128) << 64) | self.lo as u128
    }

    #[inline(always)]
    fn is_zero(self) -> bool {
        self.hi | self.lo == 0
    }

    /// Magnitude order: exponent first, then the significand.
    #[inline(always)]
    fn cmp_magnitude(self, other: Dint) -> Ordering {
        self.ex
            .cmp(&other.ex)
            .then(self.hi.cmp(&other.hi))
            .then(self.lo.cmp(&other.lo))
    }

    /// `a + b`, with the smaller operand rounded to nearest when shifted.
    pub fn add(a: Dint, b: Dint) -> Dint {
        if a.is_zero() {
            return b;
        }
        if b.is_zero() {
            return a;
        }
        match a.cmp_magnitude(b) {
            Ordering::Equal => {
                if a.sgn ^ b.sgn != 0 {
                    return Dint::ZERO;
                }
                return Dint { ex: a.ex + 1, ..a };
            }
            Ordering::Less => return Dint::add(b, a),
            Ordering::Greater => {}
        }
        // From here |a| > |b|.
        let big = a.significand();
        let mut small = b.significand();
        let mut m_ex = a.ex;
        if a.ex > b.ex {
            let sh = a.ex - b.ex;
            if sh <= 128 {
                small = small.wrapping_add(1 & (small >> (sh - 1)));
            }
            small = if sh < 128 { small >> sh } else { 0 };
        }
        let mut c;
        if a.sgn ^ b.sgn != 0 {
            c = big.wrapping_sub(small);
        } else {
            let (sum, overflow) = big.overflowing_add(small);
            c = sum;
            if overflow {
                c = c.wrapping_add(c & 1);
                c = (1u128 << 127) | (c >> 1);
                m_ex += 1;
            }
        }
        if c == 0 {
            // Only the zero significand is read downstream.
            return Dint {
                hi: 0,
                lo: 0,
                ex: m_ex.wrapping_sub(64i64.wrapping_add(a.ex)),
                sgn: a.sgn,
            };
        }
        let shift = c.leading_zeros();
        c <<= shift;
        Dint {
            hi: (c >> 64) as u64,
            lo: c as u64,
            ex: m_ex - shift as i64,
            sgn: a.sgn,
        }
    }

    /// `a * b` with 126 correct bits; `lo * lo` is dropped.
    pub fn mul(a: Dint, b: Dint) -> Dint {
        let mut t = (a.hi as u128) * (b.hi as u128);
        let m1 = (a.hi as u128) * (b.lo as u128);
        let m2 = (a.lo as u128) * (b.hi as u128);
        let (m, carry) = m1.overflowing_add(m2);
        if carry {
            t = t.wrapping_add(1u128 << 64);
        }
        t = t.wrapping_add(m >> 64);
        // Ensure that the high word starts with a 1.
        let ex = i64::from(t >> 127 == 0);
        if ex == 1 {
            t <<= 1;
        }
        t = t.wrapping_add(((m as u64) >> 63) as u128);
        Dint {
            hi: (t >> 64) as u64,
            lo: t as u64,
            ex: a.ex + b.ex - ex + 1,
            sgn: a.sgn ^ b.sgn,
        }
    }

    /// `b * a` for a small integer `b` (`|b| < 2^11`).
    pub fn mul_int(b: i64, a: Dint) -> Dint {
        if b == 0 {
            return Dint::ZERO;
        }
        let c = b.unsigned_abs();
        let sgn = if b < 0 { u64::from(a.sgn == 0) } else { a.sgn };
        let mut t = (a.hi as u128) * (c as u128);
        let high = (t >> 64) as u64;
        let mut m = if high != 0 { high.leading_zeros() } else { 64 };
        t <<= m;
        let l = ((a.lo as u128) * (c as u128)) << (m - 1) >> 63;
        let (sum, overflow) = l.overflowing_add(t);
        t = sum;
        if overflow {
            t = t.wrapping_add(t & 1);
            t = (1u128 << 127) | (t >> 1);
            m -= 1;
        }
        Dint {
            hi: (t >> 64) as u64,
            lo: t as u64,
            ex: a.ex + 64 - m as i64,
            sgn,
        }
    }

    /// The exact value of a finite nonzero binary64 `x`, subnormals included.
    pub fn from_f64(x: f64) -> Dint {
        let bits = x.to_bits();
        let biased = ((bits >> 52) & 0x7ff) as i64;
        let m = (bits & (u64::MAX >> 12)) + if biased != 0 { 1u64 << 52 } else { 0 };
        let t = m.leading_zeros();
        Dint {
            hi: m << t,
            lo: 0,
            ex: biased - 0x3ff - if t > 11 { t as i64 - 12 } else { 0 },
            sgn: u64::from(x < 0.0),
        }
    }

    /// Round to the nearest binary64. Needs a result in the normal range whose bits past
    /// the round bit are not all zero, which holds for every `log` result.
    pub fn to_f64(self) -> f64 {
        let mut r = (self.hi >> 11) | (0x3ffu64 << 52);
        // 1 <= r < 2 holds the upper 53 bits. The sticky bits are never all zero.
        let mut rd = f64::from_bits(0x3c90_0000_0000_0000);
        if (self.hi >> 10) & 1 == 1 {
            rd += f64::from_bits(0x3ca0_0000_0000_0000);
        }
        r |= self.sgn << 63;
        let rounded = f64::from_bits(r) + if self.sgn == 0 { rd } else { -rd };
        let scale = f64::from_bits(((self.ex + 1023) as u64 & 0x7ff) << 52);
        rounded * scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(d: Dint) -> f64 {
        // Exact for significands of at most 53 bits.
        let mut magnitude = (d.hi as f64) * 2f64.powi(-63);
        let mut e = d.ex as i32;
        while e.abs() > 500 {
            let step = 500 * e.signum();
            magnitude *= 2f64.powi(step);
            e -= step;
        }
        magnitude *= 2f64.powi(e);
        if d.sgn == 1 { -magnitude } else { magnitude }
    }

    #[test]
    fn test_from_f64_holds_exact_value() {
        for x in [1.0, 1.5, -3.25, f64::MIN_POSITIVE, f64::from_bits(1), 1e300] {
            let d = Dint::from_f64(x);
            assert_eq!(d.hi >> 63, 1);
            assert_eq!(value(d), x, "{x}");
        }
    }

    #[test]
    fn test_add_and_mul_on_small_values_are_exact() {
        let a = Dint::from_f64(1.5);
        let b = Dint::from_f64(-0.25);
        assert_eq!(value(Dint::add(a, b)), 1.25);
        assert_eq!(value(Dint::add(b, a)), 1.25);
        assert_eq!(value(Dint::add(a, a)), 3.0);
        assert!(Dint::add(a, Dint { sgn: 1, ..a }).is_zero());
        assert_eq!(value(Dint::mul(a, b)), -0.375);
        assert_eq!(value(Dint::mul_int(-12, a)), -18.0);
        assert_eq!(value(Dint::mul_int(1, b)), -0.25);
    }

    #[test]
    fn test_to_f64_rounds_to_nearest() {
        let one_and_half_ulp = Dint {
            hi: (1u64 << 63) | (1 << 11) | (1 << 10) | 1,
            lo: 0,
            ex: 0,
            sgn: 0,
        };
        assert_eq!(one_and_half_ulp.to_f64(), 1.0 + 2.0 * f64::EPSILON);
        let below_half = Dint {
            hi: (1u64 << 63) | (1 << 9),
            lo: 0,
            ex: 3,
            sgn: 1,
        };
        assert_eq!(below_half.to_f64(), -8.0);
    }
}
