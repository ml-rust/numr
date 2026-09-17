//! Exp family (exp, exp2, expm1) f64 dispatch.
//! Same threshold/ISA-selection logic as every unary family (`dispatch.rs`).

use super::super::{SimdLevel, detect_simd};
use super::dispatch::{SIMD_THRESHOLD, is_simd_supported};
use super::unary_scalar_f64;
use crate::ops::UnaryOp;

/// Dispatch an exp-family op for f64.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub(super) unsafe fn exp_f64(op: UnaryOp, a: *const f64, out: *mut f64, len: usize) {
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
    use super::super::test_support::{fill_range_f64, rel_err_f64};
    use super::*;

    /// f64 exp must reach double precision, not merely f32 precision.
    ///
    /// The length forces the SIMD path (>= SIMD_THRESHOLD, and a whole number of
    /// AVX2/AVX-512/NEON f64 lanes), so no element falls through to the exact
    /// scalar fallback. A degree-6 Taylor polynomial leaves a truncation error
    /// near 1.2e-7 here and fails this bound by nine orders of magnitude.
    #[test]
    fn test_unary_exp_f64_double_precision() {
        const LEN: usize = 2048;
        let a: Vec<f64> = (0..LEN)
            .map(|i| -700.0 + 1400.0 * (i as f64) / (LEN as f64 - 1.0))
            .collect();
        let mut out = vec![0.0f64; LEN];

        unsafe { exp_f64(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].exp();
            let rel_err = (out[i] - expected).abs() / expected;
            assert!(
                rel_err < 1e-14,
                "exp({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// One f64 subnormal step, the whole precision available below
    /// `f64::MIN_POSITIVE`. Relative error is meaningless there, so the
    /// subnormal test below bounds the absolute error in these units instead.
    /// One step is also the ulp of every normal number in the last binade, so
    /// the bound stays meaningful across the whole tail.
    const SUBNORMAL_STEP_F64: f64 = 5e-324;

    /// f64 exp must cover its whole representable range, not the ±709 the
    /// clamp used to allow.
    ///
    /// ln(f64::MAX) = 709.7827, so every input in [709, 709.7827] has a
    /// perfectly representable result that a clamp at 709 replaces with
    /// exp(709). At 709.78 that is 8.2184074616e307 against a true
    /// 1.7928227944e308 — 54% low.
    ///
    /// The length forces the SIMD path (>= SIMD_THRESHOLD, and a whole number
    /// of AVX2, AVX-512 and NEON f64 lanes), so no element falls through to
    /// the exact scalar fallback.
    #[test]
    fn test_unary_exp_f64_upper_band() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        // The band the ±709 clamp used to swallow, up to ln(f64::MAX).
        for k in 0i32..1024 {
            a.push(709.0 + 0.782_712_893_383 * (k as f64) / 1023.0);
        }
        fill_range_f64(&mut a, LEN, 700.0, 709.782_712_893_383);

        let mut out = vec![0.0f64; LEN];
        unsafe { exp_f64(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].exp();
            assert!(
                expected.is_finite(),
                "probe {} is outside the representable range at index {}",
                a[i],
                i
            );
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "exp({}) = {:e}, expected {:e}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// exp keeps producing subnormals down to -745. A clamp at -709 returns
    /// 1.2e-308 for every one of them, sixteen orders of magnitude high at the
    /// bottom of the range.
    ///
    /// Every result in this range has an ulp of one subnormal step or less, so
    /// the absolute bound below is a bound of two ulps throughout.
    #[test]
    fn test_unary_exp_f64_subnormal_tail() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);
        fill_range_f64(&mut a, LEN, -745.0, -708.0);

        let mut out = vec![0.0f64; LEN];
        unsafe { exp_f64(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].exp();
            let abs_err = (out[i] - expected).abs();
            assert!(
                abs_err <= 2.0 * SUBNORMAL_STEP_F64,
                "exp({}) = {:e}, expected {:e}, off by {} subnormal steps at index {}",
                a[i],
                out[i],
                expected,
                abs_err / SUBNORMAL_STEP_F64,
                i
            );
        }
    }

    /// Past ln(f64::MAX) the result genuinely overflows and below -745.13 it
    /// genuinely underflows, so the clamp bounds must sit outside both: a bound
    /// that is itself representable turns an infinity into a finite number and
    /// a zero into a subnormal.
    #[test]
    fn test_unary_exp_f64_range_ends() {
        const LEN: usize = 2048;
        let probes: [f64; 8] = [
            // Past ln(f64::MAX) = 709.782712893384, so exp is +inf.
            709.782_712_893_4,
            710.0,
            745.0,
            1.0e300,
            // Past ln(2^-1075) = -745.1332, so exp rounds to zero.
            -746.0,
            -750.0,
            -1.0e300,
            f64::NEG_INFINITY,
        ];
        let a: Vec<f64> = (0..LEN).map(|i| probes[i % probes.len()]).collect();
        let mut out = vec![0.0f64; LEN];

        unsafe { exp_f64(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = if a[i] > 0.0 { f64::INFINITY } else { 0.0 };
            assert_eq!(
                out[i], expected,
                "exp({}) = {:e}, expected {:e} at index {}",
                a[i], out[i], expected, i
            );
        }
    }

    /// exp(NaN) must be NaN. On x86 maxpd/minpd return their second operand
    /// when either input is NaN, so a clamp written as `max(x, MIN)` replaces
    /// NaN with the clamp bound and returns approximately zero; NEON's
    /// FMAX/FMIN propagate it, so the same source diverges across ISAs.
    #[test]
    fn test_unary_exp_f64_domain_edges() {
        const LEN: usize = 2048;
        let probes: [f64; 8] = [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            1.0,
            -1.0,
            709.0,
        ];
        let a: Vec<f64> = (0..LEN).map(|i| probes[i % probes.len()]).collect();
        let mut out = vec![0.0f64; LEN];

        unsafe { exp_f64(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].exp();
            assert!(
                rel_err_f64(out[i], expected) < 1e-14,
                "exp({}) = {}, expected {} at index {}",
                a[i],
                out[i],
                expected,
                i
            );
        }
    }

    /// Arguments that break a naive expm1: the ±0.5 boundary of the old
    /// degree-4 Taylor branch, where the dropped `x⁵/120` term is 2.6e-4, and
    /// arguments small enough that `exp(x) - 1` keeps none of the result.
    #[test]
    fn test_unary_expm1_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        for k in -60i32..=60 {
            a.push(0.5 + (k as f64) * 1e-3);
            a.push(-0.5 + (k as f64) * 1e-3);
        }
        for k in 1i32..=300 {
            a.push(10.0f64.powi(-k));
            a.push(-10.0f64.powi(-k));
        }
        fill_range_f64(&mut a, LEN, -700.0, 700.0);

        let mut out = vec![0.0f64; LEN];
        unsafe { exp_f64(UnaryOp::Expm1, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].exp_m1();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "expm1({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// exp2 must not borrow `exp(x * ln2)`. That premultiply rounds a value as
    /// large as 710 to one ulp, and the exponential turns the absolute error
    /// into a relative one, so the failure only shows at large |x|.
    #[test]
    fn test_unary_exp2_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        for k in -100i32..=100 {
            a.push(978.022 + (k as f64) * 1e-3);
            a.push(-978.022 + (k as f64) * 1e-3);
        }

        // Exact integers and half-integers are the ends of the reduction
        // interval, where `r = x - n` is largest.
        for k in -40i32..=40 {
            a.push(k as f64);
            a.push(k as f64 + 0.5);
        }
        fill_range_f64(&mut a, LEN, -1000.0, 1000.0);

        let mut out = vec![0.0f64; LEN];
        unsafe { exp_f64(UnaryOp::Exp2, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].exp2();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "exp2({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }
}
