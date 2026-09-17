//! Log family (log, log2, log10, log1p) f64 dispatch.
//! Same threshold/ISA-selection logic as every unary family (`dispatch.rs`).

use super::super::{SimdLevel, detect_simd};
use super::dispatch::{SIMD_THRESHOLD, is_simd_supported};
use super::unary_scalar_f64;
use crate::ops::UnaryOp;

/// Dispatch a log-family op for f64.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub(super) unsafe fn log_f64(op: UnaryOp, a: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();
    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar || !is_simd_supported(op) {
        return unsafe { unary_scalar_f64(op, a, out, len) };
    }
    unsafe {
        #[cfg(target_arch = "x86_64")]
        match level {
            SimdLevel::Avx512 => super::x86_64::avx512::unary_f64(op, a, out, len),
            SimdLevel::Avx2Fma => super::x86_64::avx2::unary_f64(op, a, out, len),
            _ => unary_scalar_f64(op, a, out, len),
        }
        #[cfg(target_arch = "aarch64")]
        match level {
            SimdLevel::Neon | SimdLevel::NeonFp16 => {
                super::aarch64::neon::unary_f64(op, a, out, len)
            }
            _ => unary_scalar_f64(op, a, out, len),
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        unary_scalar_f64(op, a, out, len);
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{fill_range_f64, log_probe_points_f64, rel_err_f64};
    use super::*;

    #[test]
    fn test_unary_log_f64_double_precision() {
        const LEN: usize = 2048;
        let a = log_probe_points_f64(LEN);

        let mut out = vec![0.0f64; LEN];
        unsafe { log_f64(UnaryOp::Log, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].ln();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "log({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_log2_f64_double_precision() {
        const LEN: usize = 2048;
        let a = log_probe_points_f64(LEN);

        let mut out = vec![0.0f64; LEN];
        unsafe { log_f64(UnaryOp::Log2, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].log2();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "log2({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_log10_f64_double_precision() {
        const LEN: usize = 2048;
        let a = log_probe_points_f64(LEN);

        let mut out = vec![0.0f64; LEN];
        unsafe { log_f64(UnaryOp::Log10, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].log10();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "log10({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_log1p_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        // Tiny |x|, where 1 + x rounds x away entirely and log1p must fall back
        // on log1p(x) == x. This is the whole reason log1p exists separately.
        for k in 1i32..=300 {
            a.push(10.0f64.powi(-k));
            a.push(-10.0f64.powi(-k));
        }

        // Around -0.5, far enough from 0 that a low-degree series in x diverges
        // from log(1+x) but still inside any |x| <= 0.5 fast path.
        for k in -60i32..=60 {
            a.push(-0.5 + (k as f64) * 1e-3);
        }

        // Approaching -1 from above, where log1p(x) -> -inf. 2^-53 is the last
        // offset that still rounds to something other than -1 itself.
        for k in 1i32..=53 {
            a.push(-1.0 + 2.0f64.powi(-k));
        }

        // Both sides of |x| = 1, the Fast2Sum branch point.
        for k in -20i32..=20 {
            a.push(1.0 + (k as f64) * 1e-12);
        }

        fill_range_f64(&mut a, LEN, -0.9, 10.0);

        let mut out = vec![0.0f64; LEN];
        unsafe { log_f64(UnaryOp::Log1p, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].ln_1p();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "log1p({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_log_f64_domain_edges() {
        const LEN: usize = 2048;

        // log is undefined at and below zero, and 1 must come out exactly zero.
        let log_edges = [
            0.0f64,
            -0.0,
            1.0,
            -1.0,
            -1e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let a: Vec<f64> = (0..LEN).map(|i| log_edges[i % log_edges.len()]).collect();
        let mut out = vec![0.0f64; LEN];

        for (op, reference) in [
            (UnaryOp::Log, f64::ln as fn(f64) -> f64),
            (UnaryOp::Log2, f64::log2 as fn(f64) -> f64),
            (UnaryOp::Log10, f64::log10 as fn(f64) -> f64),
        ] {
            unsafe { log_f64(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
            for i in 0..LEN {
                let expected = reference(a[i]);
                if expected.is_nan() {
                    assert!(
                        out[i].is_nan(),
                        "{:?}({}) = {}, expected NaN",
                        op,
                        a[i],
                        out[i]
                    );
                } else {
                    assert_eq!(out[i], expected, "{:?}({}) mismatch", op, a[i]);
                }
            }
        }

        // log1p(-1) = -inf and log1p(x < -1) = NaN; log1p(0) keeps its sign.
        let log1p_edges = [
            -1.0f64,
            -1.5,
            0.0,
            -0.0,
            -1e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let a: Vec<f64> = (0..LEN)
            .map(|i| log1p_edges[i % log1p_edges.len()])
            .collect();
        unsafe { log_f64(UnaryOp::Log1p, a.as_ptr(), out.as_mut_ptr(), LEN) }
        for i in 0..LEN {
            let expected = a[i].ln_1p();
            if expected.is_nan() {
                assert!(
                    out[i].is_nan(),
                    "log1p({}) = {}, expected NaN",
                    a[i],
                    out[i]
                );
            } else {
                assert_eq!(out[i], expected, "log1p({}) mismatch", a[i]);
            }
        }
    }
}
