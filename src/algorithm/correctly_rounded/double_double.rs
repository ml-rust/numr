//! Error-free transformations and double-double steps shared by the ported kernels.
//!
//! Every step uses only IEEE add, subtract, multiply and `mul_add`, in the order written.
//! Each result is exact or carries the rounding of one named operation, under
//! round-to-nearest-even with gradual underflow.

// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/{exp/exp.c,log/log.c,log1p/log1p.c}.
// Copyright (c) 2022-2026 Alexei Sibidanov <sibid@uvic.ca>.
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

/// The canonical quiet NaN that every NaN result returns.
pub const CANONICAL_NAN: f64 = f64::from_bits(0x7ff8_0000_0000_0000);

/// `1.5 * 2^52`: adding and subtracting it rounds `|y| < 2^51` to the nearest integer, ties to even.
const ROUND_MAGIC: f64 = 6_755_399_441_055_744.0;

/// `(s, e)` with `s + e == a + b` exactly. Needs `|a| >= |b|` or `a == 0`.
#[inline(always)]
pub fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let z = s - a;
    (s, b - z)
}

/// `(s, e)` with `s + e == a + b` exactly, for any order of magnitude.
///
/// The low part is `(b - d) + (a + (d - s))`, CORE-MATH's operation order.
#[inline(always)]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let d = s - a;
    (s, (b - d) + (a + (d - s)))
}

/// `(p, e)` with `p + e == a * b` exactly, unless the product underflows.
#[inline(always)]
pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

/// Double-double sum with a fast two-sum on the high parts: needs `|xh| >= |yh|`.
#[inline(always)]
pub fn fast_sum(xh: f64, xl: f64, yh: f64, yl: f64) -> (f64, f64) {
    let (sh, sl) = fast_two_sum(xh, yh);
    (sh, (xl + yl) + sl)
}

/// Double-double sum with a full two-sum on the high parts.
#[inline(always)]
pub fn sum(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let (sh, sl) = two_sum(xh, ch);
    (sh, (xl + cl) + sl)
}

/// Double-double product `(xh + xl) * (ch + cl)`, dropping `xl * cl`.
///
/// The low part is `(cl * xh + ch * xl) + fma(ch, xh, -ch * xh)`.
/// IEEE addition commutes, so this equals CORE-MATH's `muldd` in every source file.
#[inline(always)]
pub fn mul_dd(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let (ahhh, err) = two_prod(ch, xh);
    (ahhh, (cl * xh + ch * xl) + err)
}

/// Product of a double `x` and a double-double `ch + cl`.
#[inline(always)]
pub fn mul_d_dd(x: f64, ch: f64, cl: f64) -> (f64, f64) {
    let (ahhh, err) = two_prod(ch, x);
    (ahhh, cl * x + err)
}

/// `(ah + al) * (bh + bl) + (ch + cl)` as a double-double.
///
/// Algorithm 4 of Jeannerod, Joldes, Louvet and Muller, "Extended-Precision FMA under
/// Parameterized Double-Word Overlap", with CORE-MATH's two unfused final steps.
#[inline(always)]
pub fn fma_dw(ah: f64, al: f64, bh: f64, bl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let dh = ah.mul_add(bh, ch);
    let t = ch - dh;
    let e = ah.mul_add(bh, t);
    let f = e + cl;
    let g = ah * bl + f;
    (dh, al * bh + g)
}

/// Round `y` to the nearest integer, ties to even. Needs `|y| < 2^51`.
#[inline(always)]
pub fn round_ties_even(y: f64) -> f64 {
    (y + ROUND_MAGIC) - ROUND_MAGIC
}

/// Add `e` to the biased exponent field of `x`. Exact while the result stays normal.
#[inline(always)]
pub fn add_exponent(x: f64, e: i64) -> f64 {
    f64::from_bits(x.to_bits().wrapping_add((e as u64) << 52))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `x` as `m * 2^e` with an odd or zero integer `m`.
    fn dyadic(x: f64) -> (i128, i32) {
        if x == 0.0 {
            return (0, 0);
        }
        let bits = x.to_bits();
        let biased = ((bits >> 52) & 0x7ff) as i32;
        let frac = (bits & ((1u64 << 52) - 1)) as i128;
        let (mut m, mut e) = if biased == 0 {
            (frac, -1074)
        } else {
            (frac | (1i128 << 52), biased - 1075)
        };
        while m & 1 == 0 {
            m >>= 1;
            e += 1;
        }
        if bits >> 63 == 1 { (-m, e) } else { (m, e) }
    }

    /// Exact sum of dyadic values, as a numerator over `2^min_e`.
    fn exact_sum(values: &[f64]) -> (i128, i32) {
        let parts: Vec<(i128, i32)> = values.iter().map(|&v| dyadic(v)).collect();
        let base = parts
            .iter()
            .filter(|p| p.0 != 0)
            .map(|p| p.1)
            .min()
            .unwrap_or(0);
        let total = parts
            .iter()
            .map(|&(m, e)| if m == 0 { 0 } else { m << (e - base) })
            .sum();
        (total, base)
    }

    fn same_value(a: (i128, i32), b: (i128, i32)) -> bool {
        if a.0 == 0 || b.0 == 0 {
            return a.0 == b.0;
        }
        let base = a.1.min(b.1);
        (a.0 << (a.1 - base)) == (b.0 << (b.1 - base))
    }

    const SAMPLES: [f64; 10] = [
        1.0,
        -3.5,
        0.1,
        1e-10,
        -7.25e3,
        1.0 + f64::EPSILON,
        0.333_333_333_333_333_3,
        -2.0e-5,
        6.02e2,
        1.25e-3,
    ];

    #[test]
    fn test_two_sum_and_fast_two_sum_are_exact() {
        for &a in &SAMPLES {
            for &b in &SAMPLES {
                let (s, e) = two_sum(a, b);
                assert!(
                    same_value(exact_sum(&[s, e]), exact_sum(&[a, b])),
                    "{a} {b}"
                );
                if a.abs() >= b.abs() {
                    let (s, e) = fast_two_sum(a, b);
                    assert!(
                        same_value(exact_sum(&[s, e]), exact_sum(&[a, b])),
                        "{a} {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_two_prod_is_exact() {
        for &a in &SAMPLES {
            for &b in &SAMPLES {
                let (p, e) = two_prod(a, b);
                let (ma, ea) = dyadic(a);
                let (mb, eb) = dyadic(b);
                assert!(
                    same_value(exact_sum(&[p, e]), (ma * mb, ea + eb)),
                    "{a} {b}"
                );
            }
        }
    }

    #[test]
    fn test_round_ties_even_matches_integer_rounding() {
        let cases = [
            (0.5, 0.0),
            (1.5, 2.0),
            (2.5, 2.0),
            (-0.5, 0.0),
            (-1.5, -2.0),
            (-2.5, -2.0),
            (3.499_999_999_999_999_6, 3.0),
            (4_194_303.5, 4_194_304.0),
            (-4_398_046_511_103.5, -4_398_046_511_104.0),
        ];
        for (y, expected) in cases {
            assert_eq!(round_ties_even(y), expected, "{y}");
        }
        let mut y = -1.0e6;
        while y < 1.0e6 {
            assert_eq!(round_ties_even(y), y.round_ties_even(), "{y}");
            y += 0.123_456_789;
        }
    }

    #[test]
    fn test_add_exponent_scales_exactly() {
        assert_eq!(add_exponent(1.5, 10), 1536.0);
        assert_eq!(add_exponent(1.5, -1022), 1.5 * f64::MIN_POSITIVE);
        assert_eq!(add_exponent(1.0, 1023), 2f64.powi(1023));
    }
}
