//! 128-bit [`Dint`] operations of the accurate `pow` path.
//!
//! They are CORE-MATH's `pow/dint.h` versions, which differ from the `log` ones in
//! `dint.rs`: a shifted operand is truncated, a subtraction under Sterbenz's condition is
//! exact, and each product drops a fixed set of partial products. The `pow` error proof
//! counts exactly these losses, so the two sets are not interchangeable.

// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/pow/dint.h and src/binary64/pow/pow.h,
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

/// Zero, with the exponent that makes [`to_f64`] return `+0`.
pub const ZERO: Dint = Dint {
    hi: 0,
    lo: 0,
    ex: -1076,
    sgn: 0,
};

#[inline(always)]
fn significand(a: Dint) -> u128 {
    ((a.hi as u128) << 64) | a.lo as u128
}

#[inline(always)]
fn from_parts(c: u128, ex: i64, sgn: u64) -> Dint {
    Dint {
        hi: (c >> 64) as u64,
        lo: c as u64,
        ex,
        sgn,
    }
}

/// Order of `|a|` and `|b|`. A value with a zero high word counts as zero.
#[inline(always)]
pub fn cmp_abs(a: Dint, b: Dint) -> Ordering {
    if a.hi == 0 {
        return if b.hi == 0 {
            Ordering::Equal
        } else {
            Ordering::Less
        };
    }
    if b.hi == 0 {
        return Ordering::Greater;
    }
    a.ex.cmp(&b.ex).then(significand(a).cmp(&significand(b)))
}

/// `a + b` with error below 2 ulps of the 128-bit result. Exact under Sterbenz's condition.
pub fn add(a: Dint, b: Dint) -> Dint {
    if a.hi | a.lo == 0 {
        return b;
    }
    let (a, b) = match cmp_abs(a, b) {
        Ordering::Equal => {
            if a.sgn ^ b.sgn != 0 {
                return ZERO;
            }
            return Dint { ex: a.ex + 1, ..a };
        }
        Ordering::Less => (b, a),
        Ordering::Greater => (a, b),
    };
    // From here |a| > |b|, so a.ex >= b.ex.
    let big = significand(a);
    let mut small = significand(b);
    let k = a.ex.wrapping_sub(b.ex) as u64;
    if k > 0 {
        small = if k < 128 { small >> k } else { 0 };
    }
    let mut ex = a.ex;
    let c = if a.sgn ^ b.sgn != 0 {
        let mut c = big.wrapping_sub(small);
        // c != 0 because |a| > |b|.
        let mut shift = c.leading_zeros();
        if shift > 0 {
            c = if k == 1 {
                // Sterbenz case: no bit of b is lost.
                (big << shift).wrapping_sub(significand(b) << (shift - 1))
            } else {
                (big << shift).wrapping_sub(small << shift)
            };
            ex -= shift as i64;
            shift = ((c >> 64) as u64).leading_zeros();
        }
        ex -= shift as i64;
        c << shift
    } else {
        let (c, carry) = big.overflowing_add(small);
        if carry {
            ex += 1;
            (1u128 << 127) | (c >> 1)
        } else {
            c
        }
    };
    from_parts(c, ex, a.sgn)
}

/// `a + b` on the high words only, with error below 2 ulps of 64 bits.
///
/// The low word of a computed sum is zero. Upstream leaves the destination's previous low
/// word there; every caller feeds the sum to [`mul_11`], which reads the high word only,
/// so both give the same bits.
pub fn add_11(a: Dint, b: Dint) -> Dint {
    if a.hi == 0 {
        return b;
    }
    if b.hi == 0 {
        return a;
    }
    let (a, b) = match a.ex.cmp(&b.ex).then(a.hi.cmp(&b.hi)) {
        Ordering::Equal => {
            if a.sgn ^ b.sgn != 0 {
                return ZERO;
            }
            return Dint { ex: a.ex + 1, ..a };
        }
        Ordering::Less => (b, a),
        Ordering::Greater => (a, b),
    };
    let big = a.hi;
    let mut small = b.hi;
    if a.ex > b.ex {
        let k = (a.ex - b.ex) as u64;
        small = if k < 64 { small >> k } else { 0 };
    }
    let mut ex = a.ex;
    let c = if a.sgn ^ b.sgn != 0 {
        let mut c = big - small;
        let mut shift = c.leading_zeros();
        if shift > 0 {
            c = (big << shift).wrapping_sub(small << shift);
            ex -= shift as i64;
            shift = c.leading_zeros();
        }
        ex -= shift as i64;
        c << shift
    } else {
        let (c, carry) = big.overflowing_add(small);
        if carry {
            ex += 1;
            (1u64 << 63) | (c >> 1)
        } else {
            c
        }
    };
    Dint {
        hi: c,
        lo: 0,
        ex,
        sgn: a.sgn,
    }
}

/// Shift a product so its top bit is set, and attach exponent and sign.
#[inline(always)]
fn normalize(product: u128, a: Dint, b: Dint) -> Dint {
    let top = (product >> 127) as u64;
    from_parts(
        product << (1 - top),
        a.ex + b.ex + top as i64,
        a.sgn ^ b.sgn,
    )
}

/// `a * b` with error below 6 ulps: drops `lo * lo` and the low halves of the cross terms.
pub fn mul(a: Dint, b: Dint) -> Dint {
    let m1 = (a.hi as u128) * (b.lo as u128);
    let m2 = (a.lo as u128) * (b.hi as u128);
    let product = ((a.hi as u128) * (b.hi as u128))
        .wrapping_add(m1 >> 64)
        .wrapping_add(m2 >> 64);
    normalize(product, a, b)
}

/// `a * b` ignoring the low word of `b`, with error below 2 ulps.
pub fn mul_21(a: Dint, b: Dint) -> Dint {
    let hi = (a.hi as u128) * (b.hi as u128);
    let lo = (a.lo as u128) * (b.hi as u128);
    normalize(hi.wrapping_add(lo >> 64), a, b)
}

/// The exact product of the high words of `a` and `b`.
#[inline(always)]
pub fn mul_11(a: Dint, b: Dint) -> Dint {
    normalize((a.hi as u128) * (b.hi as u128), a, b)
}

/// `b * a` for an integer `b` with `|b| < 2^63`, with error below 1 ulp.
pub fn mul_int64(a: Dint, b: i64) -> Dint {
    if b == 0 {
        return ZERO;
    }
    let c = b.unsigned_abs() as u128;
    let sgn = if b < 0 { u64::from(a.sgn == 0) } else { a.sgn };
    let mut r = (a.hi as u128) * c;
    let high = (r >> 64) as u64;
    let m = if high != 0 { high.leading_zeros() } else { 64 };
    r <<= m;
    let mut ex = a.ex + 64 - m as i64;
    // hi * c < 2^(128 - m) with hi >= 2^63 gives c < 2^(65 - m), so no bit is lost.
    let l = (((a.lo as u128) * c) << (m - 1)) >> 63;
    r = r.wrapping_add(l);
    if r < l {
        r = (1u128 << 127) | (r >> 1);
        ex += 1;
    }
    from_parts(r, ex, sgn)
}

/// `a` rounded toward zero to an integer. Needs `a.ex < 64`.
#[inline(always)]
pub fn to_i64(a: Dint) -> i64 {
    if a.ex < 0 {
        return 0;
    }
    let r = (a.hi >> (63 - a.ex)) as i64;
    if a.sgn != 0 { -r } else { r }
}

/// `2^-53` and `2^-54`: the round and sticky increments on a significand in `[1, 2)`.
const HALF_ULP: f64 = f64::from_bits(0x3ca0_0000_0000_0000);
const QUARTER_ULP: f64 = f64::from_bits(0x3c90_0000_0000_0000);

/// Round `(-1)^sgn * hi * 2^(ex - 63)` (plus `sticky` below `hi`) to nearest, ties to even.
///
/// Rounds once in every range: a result below `2^-1022` is rounded on the subnormal grid,
/// a result from `2^1024` up is `inf`.
pub fn round_to_f64(hi: u64, sticky: bool, ex: i64, sgn: u64) -> f64 {
    if ex < -1022 {
        return round_subnormal(hi, sticky, ex, sgn);
    }
    let r = f64::from_bits((hi >> 11) | (0x3ff << 52) | (sgn << 63));
    let mut rd = 0.0;
    if (hi >> 10) & 1 == 1 {
        rd += HALF_ULP;
    }
    if hi & 0x3ff != 0 || sticky {
        rd += QUARTER_ULP;
    }
    if sgn != 0 {
        rd = -rd;
    }
    let r = r + rd;
    if ex > 1023 {
        if ex == 1024 {
            return (r * 2.0) * f64::from_bits(0x7fe0_0000_0000_0000);
        }
        return f64::MAX * f64::MAX;
    }
    r * f64::from_bits(((ex + 1023) as u64 & 0x7ff) << 52)
}

/// Round to the subnormal grid. Needs `ex <= -1023`.
fn round_subnormal(hi: u64, sticky: bool, ex: i64, sgn: u64) -> f64 {
    // The number of bits of hi below 2^-1074.
    let shift = (-(1011 + ex)) as u64;
    let q = if shift >= 64 {
        // |a| < 2^-1074: the round bit is the top bit of hi only when shift = 64.
        let round = hi >> 63 == 1;
        let rest = (hi << 1) != 0 || sticky;
        u64::from(shift == 64 && round && rest)
    } else {
        let q = hi >> shift;
        let round = (hi >> (shift - 1)) & 1;
        let rest = (hi << (65 - shift)) != 0 || sticky;
        q + if rest { round } else { q & round }
    };
    // q <= 2^52; q = 2^52 is the bit pattern of 2^-1022.
    f64::from_bits(q | (sgn << 63))
}

/// `a` rounded to nearest binary64.
#[inline(always)]
pub fn to_f64(a: Dint) -> f64 {
    round_to_f64(a.hi, a.lo != 0, a.ex, a.sgn)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(x: f64) -> Dint {
        Dint::from_f64(x)
    }

    #[test]
    fn test_add_matches_binary64_on_short_operands() {
        let values = [1.5, -0.25, 3.0, -3.0, 1.0e-3, -7.75, 2.0f64.powi(-60)];
        for &x in &values {
            for &y in &values {
                let sum = add(d(x), d(y));
                assert_eq!(to_f64(sum), x + y, "{x} + {y}");
                if x.abs() > y.abs() || x.abs() < y.abs() {
                    let sum11 = add_11(d(x), d(y));
                    assert_eq!(to_f64(sum11), x + y, "{x} + {y}");
                }
            }
        }
        assert_eq!(add(d(2.5), d(-2.5)), ZERO);
    }

    #[test]
    fn test_sterbenz_subtraction_is_exact() {
        // 1 - (1 - 2^-128): the low bit of b is shifted out unless the Sterbenz branch runs.
        let one = Dint::from_f64(1.0);
        let b = Dint {
            hi: u64::MAX,
            lo: u64::MAX,
            ex: -1,
            sgn: 1,
        };
        let r = add(one, b);
        assert_eq!((r.hi, r.lo, r.ex, r.sgn), (1 << 63, 0, -128, 0));
    }

    #[test]
    fn test_products_match_binary64_on_short_operands() {
        for (x, y) in [(1.5, -0.25), (3.0, 7.0), (0.1, 10.0), (-1.0e100, 1.0e-200)] {
            assert_eq!(to_f64(mul(d(x), d(y))), x * y);
            assert_eq!(to_f64(mul_21(d(x), d(y))), x * y);
            assert_eq!(to_f64(mul_11(d(x), d(y))), x * y);
        }
        assert_eq!(to_f64(mul_int64(d(1.5), -1074)), -1611.0);
        assert_eq!(to_f64(mul_int64(d(-0.75), 1)), -0.75);
        assert_eq!(mul_int64(d(2.0), 0), ZERO);
    }

    #[test]
    fn test_to_i64_truncates() {
        assert_eq!(to_i64(d(3.75)), 3);
        assert_eq!(to_i64(d(-4_399_162.9)), -4_399_162);
        assert_eq!(to_i64(d(0.5)), 0);
    }

    #[test]
    fn test_round_to_f64_rounds_once_in_every_range() {
        let tie = (1u64 << 63) | (1 << 10);
        assert_eq!(round_to_f64(tie, false, 0, 0), 1.0);
        assert_eq!(round_to_f64(tie, true, 0, 0), 1.0 + f64::EPSILON);
        assert_eq!(
            round_to_f64(tie | (1 << 11), false, 0, 1),
            -1.0 - 2.0 * f64::EPSILON
        );
        assert_eq!(round_to_f64(1 << 63, false, 1024, 0), f64::INFINITY);
        assert_eq!(round_to_f64(1 << 63, false, 1023, 0), 2f64.powi(1023));
        // 2^-1075 is a tie between 0 and 2^-1074; anything above it rounds up.
        assert_eq!(round_to_f64(1 << 63, false, -1075, 0).to_bits(), 0);
        assert_eq!(round_to_f64(1 << 63, true, -1075, 0).to_bits(), 1);
        assert_eq!(round_to_f64(u64::MAX, true, -1076, 1).to_bits(), 1 << 63);
        // 1.5 * 2^-1074 is a tie that rounds to 2 * 2^-1074.
        assert_eq!(round_to_f64(3 << 62, false, -1074, 0).to_bits(), 2);
        // Just below 2^-1022 rounds up into the normal range.
        assert_eq!(round_to_f64(u64::MAX, false, -1023, 0), f64::MIN_POSITIVE);
    }
}
