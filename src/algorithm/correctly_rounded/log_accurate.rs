//! Accurate path of the correctly rounded binary64 `log`, in 128-bit dint arithmetic.

// Ported from the CORE-MATH project (https://core-math.gitlabpages.inria.fr/),
// src/binary64/log/log.c, commit 1851882b0f85ed581af9185cfa020992cec9e7a1.
// The code was copied there from Tom Hubrecht's correctly rounded pow for CORE-MATH.
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

use super::dint::Dint;
use super::tables::{
    LOG_DINT_INVERSE, LOG_DINT_LN2, LOG_DINT_LOG_INV, LOG_DINT_MINUS_ONE, LOG_DINT_POLY,
};

/// Significand of `sqrt(2)`: above it the argument is halved to avoid cancellation.
const SQRT2_SIGNIFICAND: u64 = 0xb504_f333_f9de_6484;

/// `log(x)` for a positive finite `x != 1`, correctly rounded.
#[inline(never)]
#[cold]
pub fn log_accurate(x: f64) -> f64 {
    log_dint(Dint::from_f64(x)).to_f64()
}

/// `log(x)` in dint arithmetic: `E * log(2) - log(r) + log(1 + z)` with `z = x * r - 1`.
fn log_dint(mut x: Dint) -> Dint {
    let mut e = x.ex;
    // Lookup index from the leading 9 significand bits.
    let mut i = (x.hi >> 55) as usize;
    if x.hi > SQRT2_SIGNIFICAND {
        e += 1;
        i >>= 1;
    }
    x.ex -= e;
    let z = Dint::mul(x, Dint::from_table(&LOG_DINT_INVERSE, i - 128));
    let z = Dint::add(Dint::from_table(&LOG_DINT_MINUS_ONE, 0), z);
    let r = Dint::mul_int(e, Dint::from_table(&LOG_DINT_LN2, 0));
    let p = poly(z);
    let p = Dint::add(Dint::from_table(&LOG_DINT_LOG_INV, i - 128), p);
    Dint::add(p, r)
}

/// `log(1 + z)` by Horner's rule over `LOG_DINT_POLY`, then a final product with `z`.
fn poly(z: Dint) -> Dint {
    let n = LOG_DINT_POLY.len() / 4;
    let mut r = Dint::from_table(&LOG_DINT_POLY, 0);
    for k in 1..n {
        r = Dint::mul(r, z);
        r = Dint::add(Dint::from_table(&LOG_DINT_POLY, k), r);
    }
    Dint::mul(r, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_accurate_on_exact_anchors() {
        assert_eq!(log_accurate(2.0), std::f64::consts::LN_2);
        assert_eq!(log_accurate(10.0), std::f64::consts::LN_10);
        assert_eq!(log_accurate(0.5), -std::f64::consts::LN_2);
        assert_eq!(log_accurate(std::f64::consts::E), 1.0);
    }
}
