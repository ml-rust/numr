//! Accurate path of the correctly rounded binary64 `log1p`, in triple-double arithmetic.

// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/log1p/log1p.c, commit 1851882b0f85ed581af9185cfa020992cec9e7a1.
// Copyright (c) 2024-2026 Alexei Sibidanov <sibid@uvic.ca>.
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

use super::double_double::{
    fast_sum, fast_two_sum, mul_d_dd, mul_dd, round_ties_even, sum, two_sum,
};
use super::tables::{
    LOG1P_LN, LOG1P_REFINE_CL, LOG1P_REFINE_CY, LOG1P_REFINE_CZ, LOG1P_REFINE_CZL, LOG1P_RT, f,
};

/// `ax = bits << 1` threshold for `|x| < 2^-53`.
const AX_TINY: u64 = 0x7940_0000_0000_0000;
/// `ax` threshold for `|x| < 2^-10`, where the series in `x` runs.
const AX_SERIES: u64 = 0x7ea0_0000_0000_0000;
/// `-2^-54`.
const MINUS_TWO_POW_54: f64 = f64::from_bits(0xbc90_0000_0000_0000);
/// `2^16 / log(2)`, `0x1.71547652b82fep+16`.
const SCALE: f64 = f64::from_bits(0x40f7_1547_652b_82fe);
/// Offset that makes the table index of `round(a * SCALE)` non-negative.
const INDEX_OFFSET: i64 = 34952;
/// `log(2)` as a triple-double: `0x1.62e42fefa4p-1`, `-0x1.8432a1b0e28p-43`, `0x1.cc01f97b57a08p-87`.
const LN2_2: f64 = f64::from_bits(0x3fe6_2e42_fefa_4000);
const LN2_1: f64 = f64::from_bits(0xbd48_432a_1b0e_2800);
const LN2_0: f64 = f64::from_bits(0x3a8c_c01f_97b5_7a08);

/// `log1p(x)` for `0 < |x| < 2^-53`: the cubic term cannot change the rounding, and one
/// fused operation rounds once. A subnormal `x` gives `x` exactly.
#[inline(always)]
pub fn tiny(x: f64, ax: u64) -> f64 {
    if ax == 0 {
        return x;
    }
    x.abs().mul_add(MINUS_TWO_POW_54, x)
}

/// Accurate `log1p(x)`; `a` is the fast-path lower bound.
#[inline(never)]
#[cold]
pub fn log1p_refine(x: f64, a: f64) -> f64 {
    let ix = x.to_bits();
    let ax = ix << 1;
    let (mut ln22, mut ln21, mut ln20);
    if ax < AX_SERIES {
        if ax < AX_TINY {
            return tiny(x, ax);
        }
        let x2h = x * x;
        let x2l = x.mul_add(x, -x2h);
        let (x3h, x3l) = mul_d_dd(x, x2h, x2l);
        let czl = |k| f(&LOG1P_REFINE_CZL, k);
        let sl = x * ((czl(0) + x * czl(1)) + x2h * (czl(2) + x * czl(3)));
        let (sh, sl) = poly_d(x, &LOG1P_REFINE_CZ, sl);
        let (sh, sl) = mul_dd(sh, sl, x3h, x3l);
        let x2h = x2h * -0.5;
        let x2l = x2l * -0.5;
        ln22 = x;
        (ln21, ln20) = fast_sum(x2h, x2l, sh, sl);
    } else {
        let j = round_ties_even(a * SCALE) as i64;
        let i = j + INDEX_OFFSET;
        let j1 = ((i >> 12) & 0xf) as usize;
        let j2 = ((i >> 8) & 0xf) as usize;
        let j3 = ((i >> 4) & 0xf) as usize;
        let j4 = (i & 0xf) as usize;
        let je = i >> 16;
        let ln = |k: usize, j: usize, m: usize| f(&LOG1P_LN, 3 * (16 * k + j) + m);
        let part = |m| (ln(0, j1, m) + ln(1, j2, m)) + (ln(2, j3, m) + ln(3, j4, m));
        let (l0, l1, l2) = (part(0), part(1), part(2));

        let (mut t, mut dt) = if (ix as i64) < 0x4690_0000_0000_0000 && ix < 0xbfe0_0000_0000_0000 {
            let (t, dt) = two_sum(1.0, x);
            (t, if dt.to_bits() << 1 == 0 { 0.0 } else { dt })
        } else if (ix as i64) >= 0x4690_0000_0000_0000 {
            (x, 0.0)
        } else {
            (1.0 + x, 0.0)
        };
        t = f64::from_bits(t.to_bits().wrapping_sub((je as u64) << 52));

        let rt = |k: usize, j: usize| f(&LOG1P_RT, 16 * k + j);
        let t12 = rt(0, j1) * rt(1, j2);
        let t34 = rt(2, j3) * rt(3, j4);
        let th = t12 * t34;
        let tl = t12.mul_add(t34, -th);
        let dh = th * t;
        let dl = th.mul_add(t, -dh);
        let sh = tl * t;
        let sl = tl.mul_add(t, -sh);
        let (mut xh, mut xl) = fast_two_sum(dh - 1.0, dl);
        (xh, xl) = fast_sum(xh, xl, sh, sl);
        if dt.to_bits() != 0 {
            dt = f64::from_bits(dt.to_bits().wrapping_sub((je as u64) << 52));
            let ddh = th * dt;
            let ddl = th.mul_add(dt, -ddh) + tl * dt;
            (xh, xl) = fast_sum(xh, xl, ddh, ddl);
        }
        let cl = |k| f(&LOG1P_REFINE_CL, k);
        let sl = xh * ((cl(0) + xh * cl(1)) + (xh * xh) * (cl(2) + xh * cl(3)));
        let (sh, sl) = poly_dd(xh, xl, &LOG1P_REFINE_CY, sl);
        let (sh, sl) = mul_dd(xh, xl, sh, sl);
        if je != 0 {
            let jef = je as f64;
            ln22 = LN2_2 * jef;
            ln21 = LN2_1 * jef;
            ln20 = LN2_0 * jef;
            ln22 += l2;
            (ln21, ln20) = fast_sum(ln21, ln20, l1, l0);
        } else {
            ln22 = l2;
            ln21 = l1;
            ln20 = l0;
        }
        (ln21, ln20) = sum(ln21, ln20, sh, sl);
    }
    (ln22, ln21) = fast_two_sum(ln22, ln21);
    (ln21, ln20) = fast_two_sum(ln21, ln20);
    // A power-of-two middle term is moved one ulp toward the low term, so the final add
    // sees the sticky information of ln20.
    let mut t = ln21.to_bits();
    if t & (u64::MAX >> 12) == 0 {
        let w = ln20.to_bits();
        t = if (w ^ t) >> 63 != 0 {
            t.wrapping_sub(1)
        } else {
            t.wrapping_add(1)
        };
        ln21 = f64::from_bits(t);
    }
    ln22 + ln21
}

/// Horner evaluation in double-double of `c` (`(hi, lo)` pairs) at the double-double
/// `xh + xl`. `low` is added to the leading coefficient first.
#[inline(always)]
fn poly_dd(xh: f64, xl: f64, c: &[u64], low: f64) -> (f64, f64) {
    let n = c.len() / 2;
    let (mut ch, l) = fast_two_sum(f(c, 2 * (n - 1)), low);
    let mut cl = f(c, 2 * (n - 1) + 1) + l;
    for i in (0..n - 1).rev() {
        (ch, cl) = mul_dd(xh, xl, ch, cl);
        (ch, cl) = fast_sum(f(c, 2 * i), f(c, 2 * i + 1), ch, cl);
    }
    (ch, cl)
}

/// Horner evaluation in double-double of `c` at the double `x`.
#[inline(always)]
fn poly_d(x: f64, c: &[u64], low: f64) -> (f64, f64) {
    let n = c.len() / 2;
    let (mut ch, l) = fast_two_sum(f(c, 2 * (n - 1)), low);
    let mut cl = f(c, 2 * (n - 1) + 1) + l;
    for i in (0..n - 1).rev() {
        (ch, cl) = mul_d_dd(x, ch, cl);
        (ch, cl) = fast_sum(f(c, 2 * i), f(c, 2 * i + 1), ch, cl);
    }
    (ch, cl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiny_branch_rounds_once() {
        assert_eq!(tiny(0.0, 0).to_bits(), 0);
        let x = f64::from_bits(0x3c80_0000_0000_0001);
        // log1p(x) = x - x^2/2 + ..., and x^2/2 is far below half an ulp of x.
        assert_eq!(tiny(x, x.to_bits() << 1), x);
        let sub = f64::from_bits(5);
        assert_eq!(tiny(sub, sub.to_bits() << 1), sub);
    }

    #[test]
    fn test_refine_matches_anchor_values() {
        assert_eq!(
            log1p_refine(1.0, std::f64::consts::LN_2),
            std::f64::consts::LN_2
        );
        assert_eq!(
            log1p_refine(9.0, 2.302_585_092_994_045),
            std::f64::consts::LN_10
        );
    }
}
