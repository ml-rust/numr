//! 256-bit significand arithmetic for the extended `pow` path.
//!
//! A [`Qint`] is `(-1)^sgn * (hh + hl / 2^64 + lh / 2^128 + ll / 2^192) * 2^(ex - 63)` with
//! the top bit of `hh` set. Every operation is integer arithmetic, so it is exact or loses
//! a fixed set of bits. The products live in `qint_mul.rs`.

// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/pow/qint.h and src/binary64/pow/pow.h,
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

use std::cmp::Ordering;

use super::dint::Dint;
use super::pow_dint::round_to_f64;

/// A sign, a 256-bit significand and an exponent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qint {
    /// Highest 64 significand bits.
    pub hh: u64,
    /// Second 64 significand bits.
    pub hl: u64,
    /// Third 64 significand bits.
    pub lh: u64,
    /// Lowest 64 significand bits.
    pub ll: u64,
    /// Exponent of the leading significand bit.
    pub ex: i64,
    /// `1` for negative, `0` for positive.
    pub sgn: u64,
}

impl Qint {
    /// Zero.
    pub const ZERO: Qint = Qint {
        hh: 0,
        hl: 0,
        lh: 0,
        ll: 0,
        ex: 0,
        sgn: 0,
    };

    /// Entry `index` of a table of `(hh, hl, lh, ll, ex, sgn)` words.
    #[inline(always)]
    pub fn from_table(table: &[u64], index: usize) -> Qint {
        let w = &table[6 * index..6 * index + 6];
        Qint {
            hh: w[0],
            hl: w[1],
            lh: w[2],
            ll: w[3],
            ex: w[4] as i64,
            sgn: w[5],
        }
    }

    /// The value of `d`, with zero low limbs.
    #[inline(always)]
    pub fn from_dint(d: Dint) -> Qint {
        Qint {
            hh: d.hi,
            hl: d.lo,
            lh: 0,
            ll: 0,
            ex: d.ex,
            sgn: d.sgn,
        }
    }

    /// The exact value of a finite nonzero binary64 `x`, subnormals included.
    #[inline(always)]
    pub fn from_f64(x: f64) -> Qint {
        Qint::from_dint(Dint::from_f64(x))
    }

    /// The upper 128 significand bits.
    #[inline(always)]
    pub fn rh(self) -> u128 {
        ((self.hh as u128) << 64) | self.hl as u128
    }

    /// The lower 128 significand bits.
    #[inline(always)]
    pub fn rl(self) -> u128 {
        ((self.lh as u128) << 64) | self.ll as u128
    }

    /// A value from its two 128-bit halves.
    #[inline(always)]
    pub fn from_halves(rh: u128, rl: u128, ex: i64, sgn: u64) -> Qint {
        Qint {
            hh: (rh >> 64) as u64,
            hl: rh as u64,
            lh: (rl >> 64) as u64,
            ll: rl as u64,
            ex,
            sgn,
        }
    }

    /// Magnitude order: exponent first, then the significand.
    #[inline(always)]
    fn cmp_magnitude(self, other: Qint) -> Ordering {
        self.ex
            .cmp(&other.ex)
            .then(self.rh().cmp(&other.rh()))
            .then(self.rl().cmp(&other.rl()))
    }

    /// `a + b` with error below 2 ulps of 256 bits. Exact under Sterbenz's condition.
    pub fn add(a: Qint, b: Qint) -> Qint {
        if a.rh() == 0 && a.rl() == 0 {
            return b;
        }
        if b.rh() == 0 && b.rl() == 0 {
            return a;
        }
        match a.cmp_magnitude(b) {
            Ordering::Equal => {
                if a.sgn ^ b.sgn != 0 {
                    return Qint::ZERO;
                }
                return Qint { ex: a.ex + 1, ..a };
            }
            Ordering::Less => return Qint::add(b, a),
            Ordering::Greater => {}
        }
        // From here |a| > |b|.
        let (mut ah, mut al) = (a.rh(), a.rl());
        let k = a.ex - b.ex;
        let (mut bh, mut bl) = shift_right(b.rh(), b.rl(), k);
        let mut ex = a.ex;
        let (ch, cl) = if a.sgn ^ b.sgn != 0 {
            let (mut ch, mut cl) = sub256(ah, al, bh, bl);
            // C != 0 because |a| > |b|.
            let mut shift = leading_zeros256(ch, cl);
            if shift > 0 {
                (ah, al) = shift_left(ah, al, shift as i64);
                // Shift b left by shift - k from its unshifted value, so no bit is lost
                // when the cancellation exceeds the alignment.
                let sh = shift as i64 - k;
                (bh, bl) = if sh >= 0 {
                    shift_left(b.rh(), b.rl(), sh)
                } else {
                    shift_right(b.rh(), b.rl(), -sh)
                };
                ex -= shift as i64;
                (ch, cl) = sub256(ah, al, bh, bl);
                shift = leading_zeros256(ch, cl);
            }
            ex -= shift as i64;
            shift_left(ch, cl, shift as i64)
        } else {
            let (mut ch, mut carry) = ah.overflowing_add(bh);
            let (cl, low_carry) = al.overflowing_add(bl);
            if low_carry {
                ch = ch.wrapping_add(1);
                carry |= ch == 0;
            }
            if carry {
                ex += 1;
                ((1u128 << 127) | (ch >> 1), (ch << 127) | (cl >> 1))
            } else {
                (ch, cl)
            }
        };
        Qint::from_halves(ch, cl, ex, a.sgn)
    }

    /// `a + b` on the upper two limbs only, with error below 2 ulps of 128 bits.
    /// The lower two limbs of a computed sum are zero.
    pub fn add_22(a: Qint, b: Qint) -> Qint {
        if a.rh() == 0 {
            return b;
        }
        if b.rh() == 0 {
            return a;
        }
        match a.ex.cmp(&b.ex).then(a.rh().cmp(&b.rh())) {
            Ordering::Equal => {
                if a.sgn ^ b.sgn != 0 {
                    return Qint::ZERO;
                }
                return Qint { ex: a.ex + 1, ..a };
            }
            Ordering::Less => return Qint::add_22(b, a),
            Ordering::Greater => {}
        }
        let mut ah = a.rh();
        let k = (a.ex - b.ex) as u64;
        let mut bh = if k >= 128 { 0 } else { b.rh() >> k };
        let mut ex = a.ex;
        let ch = if a.sgn ^ b.sgn != 0 {
            let mut ch = ah.wrapping_sub(bh);
            let mut shift = ch.leading_zeros();
            if shift > 0 {
                // k < 128 here: otherwise bh = 0 and ch = ah has no leading zero.
                ah <<= shift;
                bh = if shift as u64 >= k {
                    b.rh() << (shift as u64 - k)
                } else {
                    b.rh() >> (k - shift as u64)
                };
                ex -= shift as i64;
                ch = ah.wrapping_sub(bh);
                shift = ch.leading_zeros();
            }
            ex -= shift as i64;
            ch << shift
        } else {
            let (ch, carry) = ah.overflowing_add(bh);
            if carry {
                ex += 1;
                (1u128 << 127) | (ch >> 1)
            } else {
                ch
            }
        };
        Qint::from_halves(ch, 0, ex, a.sgn)
    }

    /// `self` rounded toward zero to an integer. Needs `self.ex < 64`.
    #[inline(always)]
    pub fn to_i64(self) -> i64 {
        if self.ex < 0 {
            return 0;
        }
        let r = (self.hh >> (63 - self.ex)) as i64;
        if self.sgn != 0 { -r } else { r }
    }

    /// `self` rounded to nearest binary64, once in every range.
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        round_to_f64(self.hh, self.hl | self.lh | self.ll != 0, self.ex, self.sgn)
    }
}

/// `(h, l) - (bh, bl)` modulo `2^256`.
#[inline(always)]
fn sub256(h: u128, l: u128, bh: u128, bl: u128) -> (u128, u128) {
    let (cl, borrow) = l.overflowing_sub(bl);
    (h.wrapping_sub(bh).wrapping_sub(u128::from(borrow)), cl)
}

/// Leading zero bits of a nonzero 256-bit value.
#[inline(always)]
fn leading_zeros256(h: u128, l: u128) -> u32 {
    if h != 0 {
        h.leading_zeros()
    } else {
        128 + l.leading_zeros()
    }
}

/// `(h, l) >> k` for `k >= 0`, zero from `k >= 256`.
#[inline(always)]
fn shift_right(h: u128, l: u128, k: i64) -> (u128, u128) {
    match k {
        0 => (h, l),
        1..128 => (h >> k, (l >> k) | (h << (128 - k))),
        128..256 => (0, h >> (k - 128)),
        _ => (0, 0),
    }
}

/// `(h, l) << k` modulo `2^256`, for `0 <= k < 256`.
#[inline(always)]
fn shift_left(h: u128, l: u128, k: i64) -> (u128, u128) {
    match k {
        0 => (h, l),
        1..128 => ((h << k) | (l >> (128 - k)), l << k),
        _ => (l << (k - 128), 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(x: f64) -> Qint {
        Qint::from_f64(x)
    }

    #[test]
    fn test_add_matches_binary64_on_short_operands() {
        let values = [1.5, -0.25, 3.0, -3.0, 1.0e-3, -7.75, 2.0f64.powi(-200)];
        for &x in &values {
            for &y in values.iter().filter(|&&y| x + y != 0.0) {
                assert_eq!(Qint::add(q(x), q(y)).to_f64(), x + y, "{x} + {y}");
                assert_eq!(Qint::add_22(q(x), q(y)).to_f64(), x + y, "{x} + {y}");
            }
        }
        assert_eq!(Qint::add(q(2.5), q(-2.5)), Qint::ZERO);
    }

    #[test]
    fn test_add_keeps_bits_across_all_four_limbs() {
        // 1 - (1 - 2^-256) = 2^-256.
        let b = Qint::from_halves(u128::MAX, u128::MAX, -1, 1);
        let r = Qint::add(q(1.0), b);
        assert_eq!((r.rh(), r.rl(), r.ex, r.sgn), (1 << 127, 0, -256, 0));
        // (2 - 2^-255) + 2^-255 carries into a new top bit.
        let a = Qint::from_halves(u128::MAX, u128::MAX, 0, 0);
        let tiny = Qint::from_halves(1 << 127, 0, -255, 0);
        let r = Qint::add(a, tiny);
        assert_eq!((r.rh(), r.rl(), r.ex), (1 << 127, 0, 1));
    }

    #[test]
    fn test_shifts_cover_every_range() {
        let (h, l) = (0x1234u128 << 100, 0xabcdu128);
        assert_eq!(shift_right(h, l, 0), (h, l));
        assert_eq!(shift_right(h, l, 128), (0, h));
        assert_eq!(shift_right(h, l, 300), (0, 0));
        assert_eq!(shift_left(h, l, 128), (l, 0));
        let (sh, sl) = shift_left(h, l, 20);
        assert_eq!(shift_right(sh, sl, 20), (h & (u128::MAX >> 20), l));
        assert_eq!(leading_zeros256(0, 1), 255);
    }

    #[test]
    fn test_to_i64_truncates() {
        assert_eq!(q(-12.9).to_i64(), -12);
        assert_eq!(q(0.75).to_i64(), 0);
    }
}
