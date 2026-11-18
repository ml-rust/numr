//! Exact and midpoint cases of `pow`, and the integer tests on binary64 values.
//!
//! [`exact_pow`] is Algorithm detectRoundingBoundaryCase of Lauter and Lefèvre, "An
//! efficient rounding boundary test for pow(x,y) in double precision", IEEE Transactions
//! on Computers 58(2), 2009. It runs after the accurate path's rounding test failed.

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

use std::cmp::Ordering;

use super::dint::Dint;
use super::pow_dint::{add, cmp_abs};

const FRACTION: u64 = (1 << 52) - 1;

/// `xmax[y]` for `1 <= y <= 33`: the largest odd `m` with `m^y` below `2^53`.
const XMAX: [u64; 34] = [
    0,
    u64::MAX,
    94_906_265,
    208_063,
    9741,
    1551,
    455,
    189,
    97,
    59,
    39,
    27,
    21,
    15,
    13,
    11,
    9,
    7,
    7,
    5,
    5,
    5,
    5,
    4,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
];

/// Whether the finite `x` is an integer, read from its bits.
#[inline(always)]
pub fn is_int(x: f64) -> bool {
    let bits = x.to_bits() & (u64::MAX >> 1);
    let biased = bits >> 52;
    if biased < 0x3ff {
        return bits == 0;
    }
    if biased >= 0x433 {
        return true;
    }
    bits & ((1u64 << (0x433 - biased)) - 1) == 0
}

/// Whether the finite `y` is an odd integer.
#[inline(always)]
pub fn is_odd_int(y: f64) -> bool {
    is_int(y) && !is_int(y * 0.5)
}

/// `(e, m)` with `|x| = 2^e * m` and `m` odd. Needs a finite nonzero `x`.
#[inline(always)]
fn extract(x: f64) -> (i64, u64) {
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i64;
    let m = (bits & FRACTION) + if biased != 0 { 1 << 52 } else { 0 };
    let t = m.trailing_zeros();
    (biased + t as i64 - (0x433 - i64::from(biased == 0)), m >> t)
}

/// `|x| = m * 2^e` with `m` odd, from the integer layout of `is_exact`.
#[inline(always)]
fn odd_split(bits: u64) -> (u64, i64) {
    let mut m = bits & FRACTION;
    let mut e = ((bits << 1) >> 53) as i64 - 0x433;
    if e >= -1074 {
        m |= 1 << 52;
    } else {
        e += 1;
    }
    let t = m.trailing_zeros();
    (m >> t, e + t as i64)
}

/// `x * 2^e`, rounded once. Exact while the result is normal; needs `-2044 <= e <= 2047`.
#[inline(always)]
fn pow2(x: f64, e: i64) -> f64 {
    let x = if e & 1 != 0 { x * 2.0 } else { x };
    let e2 = f64::from_bits((((e >> 1) + 0x3ff) as u64 & 0x7ff) << 52);
    (x * e2) * e2
}

/// Whether `x^y` is exactly representable, for finite `x > 0` and finite `y`.
///
/// Only the case of a non-integer `y > 0` changes a result: it lets [`exact_pow`] skip its
/// closeness check. The other branches set exception flags upstream and are kept as they are.
pub fn is_exact(x: f64, y: f64) -> bool {
    let (v, w) = (x.to_bits(), y.to_bits());
    // Every exact case has |x| = 1 or the low 42 bits of y zero.
    if (v << 1) != 0x7fe0_0000_0000_0000 && (w << 22) != 0 {
        return false;
    }
    if (v << 1) == 0x7fe0_0000_0000_0000 {
        return true;
    }
    let (mut m, mut e) = odd_split(v);
    if y >= 0.0 && is_int(y) {
        if y == 0.0 || y == 1.0 {
            return true;
        }
        if m == 1 {
            let ye = y * e as f64;
            return (-1074.0..1024.0).contains(&ye);
        }
        if !(0.0..=33.0).contains(&y) {
            return false;
        }
        let yi = y as usize;
        if m > XMAX[yi] {
            return false;
        }
        // Upstream sizes the result from m, not m^y. The flag-only result keeps it.
        let t = 64 - m.leading_zeros() as i64;
        let ez = e * yi as i64 + t;
        if ez <= -1074 || 1024 < ez {
            return false;
        }
        return e * yi as i64 >= -1074;
    }
    let (n, mut f) = odd_split(w);
    if y < 0.0 {
        if m != 1 {
            return false;
        }
        // e != 0 here, since |x| = 1 returned above.
        let ez = if f >= 0 {
            if f < 12 {
                (-(n as i64)).wrapping_mul(e).wrapping_shl(f as u32)
            } else {
                1024
            }
        } else {
            if -f > e.trailing_zeros() as i64 {
                return false;
            }
            (-e >> -f) * n as i64
        };
        return (-1074..1024).contains(&ez);
    }
    // y > 0 is not an integer: y = n * 2^f with n odd and f < 0, so x must be a
    // (2^-f)-th power. Each step takes one exact square root of m * 2^e.
    while f != 0 {
        f += 1;
        if e & 1 != 0 {
            return false;
        }
        e /= 2;
        let s = m.isqrt();
        if s * s != m {
            return false;
        }
        m = s;
    }
    if m > 1 && (33 < n || m > XMAX[n as usize]) {
        return false;
    }
    // n <= 2^11 here, as y has at most 11 significant bits.
    let mut my = m;
    for _ in 1..n {
        my = my.wrapping_mul(m);
    }
    let t = 64 - my.leading_zeros() as i64;
    let en = e * n as i64;
    -1074 <= en && en + t <= 1024
}

/// The value of `x^y` if it is exact or a midpoint, from the accurate approximation `z`.
///
/// Needs `|z / x^y - 1| < 2^-117`. `exact` is [`is_exact`] of `(|x|, y)`.
pub fn exact_pow(x: f64, y: f64, z: Dint, exact: bool) -> Option<f64> {
    let sign: i64 = if z.sgn != 0 { -1 } else { 1 };
    let (e, m) = extract(x);
    if m == 1 {
        // x is a power of two.
        let g = e as f64 * y;
        if !is_int(g) {
            return None;
        }
        // Upstream converts g unclamped. Every |g| > 1100 gives inf or 0, as here.
        let g = (g as i64).clamp(-1100, 1100);
        return Some(pow2(sign as f64, g));
    }
    if !(0.0..=34.0).contains(&y) {
        return None;
    }
    let (f, n) = extract(y);
    if n > 34 || f < -5 {
        return None;
    }
    if f < 0 {
        // y = n * 2^f with -5 <= f < 0: e must be a multiple of 2^-f.
        if (e as u64) & (u64::MAX >> (64 + f)) != 0 {
            return None;
        }
        let g = (e >> -f) * n as i64;
        // z rounded to 54 bits is k * 2^big_g.
        let big_g = z.ex - 53;
        let k = ((z.hi >> 10) + ((z.hi >> 9) & 1)) as i64;
        if !exact {
            // Reject unless |k * 2^big_g - z| < 2^-116 * |z|.
            let cnt = (k as u64).leading_zeros();
            let rounded = Dint {
                hi: (k as u64) << cnt,
                lo: 0,
                ex: big_g + 63 - cnt as i64,
                sgn: 1 - z.sgn,
            };
            let mut d = add(z, rounded);
            d.ex += 116;
            if cmp_abs(d, z) != Ordering::Less {
                return None;
            }
        }
        if big_g > g {
            return None;
        }
        // k must be an odd number times 2^(g - big_g); g - big_g <= 53 as m^y > 1.
        let shift = (g - big_g) as u32;
        if shift >= 64 {
            return None;
        }
        let k = k as u64;
        if k & !(!1u64 << shift) == 1u64 << shift {
            return Some(pow2(((k >> shift) as i64 * sign) as f64, g));
        }
        return None;
    }
    // y is an integer with 2 <= y <= 34: compute m^y exactly if it fits in 54 bits.
    let mut t = n << f;
    let mut k: i64 = 1;
    let mut base = m;
    while t != 0 {
        if t & 1 != 0 {
            k = i64::try_from(base as i128 * k as i128).ok()?;
        }
        t >>= 1;
        if t != 0 {
            base = base.checked_mul(base)?;
        }
    }
    if k >> 54 != 0 {
        return None;
    }
    Some(pow2((k * sign) as f64, e * (n << f) as i64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_int_reads_bits() {
        for (x, want) in [
            (0.0, true),
            (-0.0, true),
            (1.0, true),
            (-3.0, true),
            (0.5, false),
            (2.5, false),
            (4_503_599_627_370_495.5, false),
            (4_503_599_627_370_496.0, true),
            (1.0e300, true),
            (f64::from_bits(1), false),
            (f64::MAX, true),
        ] {
            assert_eq!(is_int(x), want, "{x}");
        }
        assert!(is_odd_int(-3.0));
        assert!(is_odd_int(9_007_199_254_740_991.0));
        assert!(!is_odd_int(9_007_199_254_740_992.0));
        assert!(!is_odd_int(1.0e300));
        assert!(!is_odd_int(0.5));
    }

    #[test]
    fn test_extract_returns_odd_significand() {
        assert_eq!(extract(12.0), (2, 3));
        assert_eq!(extract(-0.375), (-3, 3));
        assert_eq!(extract(f64::from_bits(1)), (-1074, 1));
        assert_eq!(extract(f64::from_bits(6)), (-1073, 3));
        assert_eq!(extract(1.0), (0, 1));
    }

    #[test]
    fn test_pow2_rounds_once() {
        assert_eq!(pow2(3.0, 5), 96.0);
        assert_eq!(pow2(1.0, -1074).to_bits(), 1);
        assert_eq!(pow2(1.0, -1075).to_bits(), 0);
        assert_eq!(
            pow2(-3.0, -1075).to_bits(),
            (-2.0 * f64::from_bits(1)).to_bits()
        );
        assert_eq!(pow2(1.0, 1024), f64::INFINITY);
        assert_eq!(pow2(-1.0, 1100), f64::NEG_INFINITY);
        assert_eq!(pow2(1.0, -1100).to_bits(), 0);
    }

    #[test]
    fn test_is_exact_classifies_cases() {
        assert!(is_exact(1.0, 0.3));
        assert!(is_exact(3.0, 2.0));
        assert!(is_exact(9.0, 0.5));
        assert!(is_exact(0.25, 1.5));
        assert!(is_exact(4.0, -0.5));
        assert!(!is_exact(3.0, 0.5));
        assert!(!is_exact(2.0, 0.5));
        assert!(!is_exact(3.0, -1.0));
        assert!(!is_exact(3.0, 4.0 / 3.0));
    }

    #[test]
    fn test_exact_pow_finds_exact_and_midpoint_results() {
        let z = |v: f64| Dint::from_f64(v);
        assert_eq!(exact_pow(4.0, 1.5, z(8.0), true), Some(8.0));
        assert_eq!(exact_pow(2.0, -3.0, z(0.125), true), Some(0.125));
        assert_eq!(
            exact_pow(-2.0, 3.0, Dint { sgn: 1, ..z(8.0) }, true),
            Some(-8.0)
        );
        assert_eq!(exact_pow(3.0, 2.0, z(9.0), true), Some(9.0));
        // 3^34 has 54 bits: a midpoint that rounds to even.
        let r = exact_pow(3.0, 34.0, z(16_677_181_699_666_569.0), false);
        assert_eq!(r, Some(16_677_181_699_666_568.0));
        assert_eq!(exact_pow(3.0, 0.5, z(1.732_050_807_568_877_2), false), None);
        assert_eq!(exact_pow(3.0, 40.0, z(1.0e19), false), None);
    }
}
