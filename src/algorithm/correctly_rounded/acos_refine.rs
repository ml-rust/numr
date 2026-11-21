//! Accurate path of the correctly rounded binary64 `acos`: angle rotation by a multiple of
//! `pi/64`, an asin series near zero, and a triple-double sum.

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

use super::acos::HALF_PI_HI;
use super::double_double::{fast_two_sum, horner_dd, mul_dd, round_ties_even, two_prod};
use super::path::Path;
use super::tables::{ACOS_C, ACOS_CT, ACOS_EXCEPTIONS, ACOS_S, f};

/// `64/pi`, `0x1.45f306dc9c883p+4`.
const SIXTY_FOUR_OVER_PI: f64 = f64::from_bits(0x4034_5f30_6dc9_c883);
/// `0x1.8p-4`: adding and subtracting it rounds a product below `2^-5` to a multiple of `2^-56`.
const MAGIC: f64 = f64::from_bits(0x3fb8_0000_0000_0000);
/// High part of `pi/64` on 46 bits, `0x1.921fb54442dp-5`: `jt * PI64_H` is exact for `jt <= 64`.
const PI64_H: f64 = f64::from_bits(0x3fa9_21fb_5444_2d00);
/// Middle part of `pi/64`, `0x1.8469898cc518p-53`: `jt * PI64_M` is exact for `jt <= 64`.
const PI64_M: f64 = f64::from_bits(0x3ca8_4698_98cc_5180);
/// Low part of `pi/64`, `-0x1.fc8f8cbb5bf6cp-102`. The three parts are within `2^-155` of `pi/64`.
const PI64_L: f64 = f64::from_bits(0xb99f_c8f8_cbb5_bf6c);

/// Accurate `acos(x)` for `|x| < 1`. `phi` is the fast-path lower bound.
#[inline(never)]
#[cold]
pub fn acos_refine(x: f64, phi: f64) -> (f64, Path) {
    // With x = cos(y), ch + cl approximates sin(y) = sqrt(1 - x^2).
    let (s2, dx2) = two_prod(x, x);
    let (c2h, c2l) = fast_two_sum(1.0, -s2);
    let (c2h, c2l) = fast_two_sum(c2h, c2l - dx2);
    let ch = c2h.sqrt();
    let cl = (c2l - ch.mul_add(ch, -c2h)) * (0.5 / ch);

    // y = jt * pi/64 - delta with jt = 32 -+ jf and |delta| < pi/128, so
    // sin(delta) = sgn(x) * [cos(jf pi/64) |x| - sin(jf pi/64) sqrt(1 - x^2)].
    let jf = round_ties_even((phi - HALF_PI_HI).abs() * SIXTY_FOUR_OVER_PI) as usize;
    let s = |j: usize, m: usize| f(&ACOS_S, 2 * j + m);
    let (cos_h, cos_l) = (s(32 - jf, 1), s(32 - jf, 0));
    let (sin_h, sin_l) = (s(jf, 1), s(jf, 0));
    let dsh = x.abs() - sin_h;
    let dsl = -sin_l;
    let dch = ch - cos_h;
    let dcl = cl - cos_l;
    // sc + dsc is sin_h * dch and cs + dcs is cos_h * dsh, with sc and cs multiples of
    // 2^-56 below 2^-5, so cs - sc is exact.
    let sc = sin_h.mul_add(dch, MAGIC) - MAGIC;
    let dsc = sin_h.mul_add(dch, -sc);
    let cs = cos_h.mul_add(dsh, MAGIC) - MAGIC;
    let dcs = cos_h.mul_add(dsh, -cs);
    let v = cs - sc;
    let dv = ((cos_h * dsl + cos_l * dsh) - (sin_h * dcl + sin_l * dch)) - (dsc - dcs);
    let (v, dv) = fast_two_sum(v, dv);
    let sgn = 1.0f64.copysign(x);
    let jt = 32.0 - (jf as f64) * sgn;

    // fh + fl = asin(-delta), an odd series in v + dv.
    let (v2, dv2) = mul_dd(v, dv, v, dv);
    let v = v * -sgn;
    let dv = dv * -sgn;
    let ct = |k| f(&ACOS_CT, k);
    let low = v2 * (ct(0) + v2 * (ct(1) + v2 * ct(2)));
    let (fh, fl) = horner_dd(v2, dv2, &ACOS_C, low);
    let (fh, fl) = mul_dd(v, dv, fh, fl);

    // ph + pl + ps = jt * pi/64 - delta. jt * PI64_H and jt * PI64_M are exact.
    let ph = jt * PI64_H;
    let pl = PI64_M * jt;
    let ps = PI64_L * jt;
    let (pl, ps) = sum(fh, fl, pl, ps);
    let (ph, pl) = fast_two_sum(ph, pl);
    let (pl, ps) = fast_two_sum(pl, ps);
    let (ph, pl) = fast_two_sum(ph, pl);
    let (mut pl, ps) = fast_two_sum(pl, ps);

    if near_rounding_boundary(pl) {
        if let Some(value) = exception(x) {
            return (value, Path::Exception);
        }
        // Move pl one ulp toward ps, so the final add sees the sticky information of ps.
        let t = pl.to_bits();
        let t = if (ps.to_bits() ^ t) >> 63 != 0 {
            t.wrapping_sub(1)
        } else {
            t.wrapping_add(1)
        };
        pl = f64::from_bits(t);
    }
    (ph + pl, Path::Accurate)
}

/// True when the bits of `pl` that lie below `2^-107` relative to the result's unit
/// round to zero, so `ph + pl` may sit on a midpoint.
#[inline(always)]
fn near_rounding_boundary(pl: f64) -> bool {
    let t = pl.to_bits();
    let e = ((t >> 52) & 0x7ff) as i64 - 1023;
    let e = (52 - (107 + e)).clamp(0, 52);
    let m = (1u64 << 52) - (1u64 << e);
    let e = if e == 0 { 64 } else { e };
    t.wrapping_add(1u64 << (e - 1)) & m == 0
}

/// The listed result for `x`, or `None` when `x` is not listed.
#[inline(always)]
fn exception(x: f64) -> Option<f64> {
    ACOS_EXCEPTIONS
        .as_chunks::<3>()
        .0
        .iter()
        .find(|entry| f64::from_bits(entry[0]) == x)
        .map(|entry| f64::from_bits(entry[1]) + f64::from_bits(entry[2]))
}

/// `(s, e)` with `s + e == a + b` exactly: Knuth's two-sum in `acos.c`'s operation order.
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let a_prime = s - b;
    let b_prime = s - a_prime;
    (s, (a - a_prime) + (b - b_prime))
}

/// Double-double sum with a full two-sum on the high parts, in `acos.c`'s operation order.
#[inline(always)]
fn sum(xh: f64, xl: f64, ch: f64, cl: f64) -> (f64, f64) {
    let (sh, sl) = two_sum(xh, ch);
    (sh, (xl + cl) + sl)
}

#[cfg(test)]
mod tests {
    use super::super::acos::acos_traced;
    use super::*;

    #[test]
    fn test_listed_exceptions_take_the_exception_path() {
        let mut paths = Vec::new();
        for entry in ACOS_EXCEPTIONS.as_chunks::<3>().0 {
            let x = f64::from_bits(entry[0]);
            let (fast, path) = acos_traced(x);
            paths.push(path);
            let listed = f64::from_bits(entry[1]) + f64::from_bits(entry[2]);
            assert_eq!(fast.to_bits(), listed.to_bits(), "{x:e}");
            assert_eq!(acos_refine(x, fast), (fast, Path::Exception), "{x:e}");
        }
        // Under round-to-nearest without contraction, only the second listed input fails
        // the fast-path test. The refine path still returns the listed result for all five.
        use Path::{Exception, Fast};
        assert_eq!(paths, [Fast, Exception, Fast, Fast, Fast]);
    }

    #[test]
    fn test_refine_agrees_with_certified_fast_results() {
        // Every branch: tiny, table, double-angle, both signs.
        let mut x = 1.0e-6;
        while x < 1.0 {
            for v in [x, -x] {
                let (fast, path) = acos_traced(v);
                if path == Path::Fast {
                    assert_eq!(acos_refine(v, fast).0.to_bits(), fast.to_bits(), "{v:e}");
                }
            }
            x *= 1.0137;
        }
    }

    #[test]
    fn test_two_sum_is_exact() {
        for (a, b) in [(1.0, 1.0e-20), (1.0e-20, -1.0), (0.1, 0.2), (-3.5, 3.5)] {
            let (s, e) = two_sum(a, b);
            let (s2, e2) = super::super::double_double::two_sum(a, b);
            assert_eq!((s, e), (s2, e2), "{a} {b}");
        }
    }

    #[test]
    fn test_rounding_boundary_test_reads_the_low_bits() {
        // pl = 2^-60 relative to ph near 1: the low part below 2^-107 is zero.
        assert!(near_rounding_boundary(f64::from_bits(
            0x3c30_0000_0000_0000
        )));
        assert!(!near_rounding_boundary(f64::from_bits(
            0x3c30_0000_0000_0123
        )));
        assert!(near_rounding_boundary(0.0));
    }
}
