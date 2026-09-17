//! Exp family (exp, exp2, expm1) f32 dispatch.
//! Same threshold/ISA-selection logic as every unary family (`dispatch.rs`).

use super::super::{SimdLevel, detect_simd};
use super::dispatch::{SIMD_THRESHOLD, is_simd_supported};
use super::unary_scalar_f32;
use crate::ops::UnaryOp;

/// Dispatch an exp-family op for f32.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub(super) unsafe fn exp_f32(op: UnaryOp, a: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();
    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar || !is_simd_supported(op) {
        return unsafe { unary_scalar_f32(op, a, out, len) };
    }
    unsafe {
        #[cfg(target_arch = "x86_64")]
        match level {
            SimdLevel::Avx512 => super::x86_64::avx512::unary_f32(op, a, out, len),
            SimdLevel::Avx2Fma => super::x86_64::avx2::unary_f32(op, a, out, len),
            _ => unary_scalar_f32(op, a, out, len),
        }
        #[cfg(target_arch = "aarch64")]
        match level {
            SimdLevel::Neon | SimdLevel::NeonFp16 => {
                super::aarch64::neon::unary_f32(op, a, out, len)
            }
            _ => unary_scalar_f32(op, a, out, len),
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        unary_scalar_f32(op, a, out, len);
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{SUBNORMAL_STEP_F32, fill_range_f32, rel_err_f32};
    use super::*;

    #[test]
    fn test_unary_exp_f32() {
        let a: Vec<f32> = (0..100).map(|x| (x as f32 - 50.0) * 0.1).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { exp_f32(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].exp();
            let diff = (out[i] - expected).abs();
            assert!(
                diff < 1e-5 * expected.abs().max(1.0),
                "exp mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    /// f32 exp must reach single precision over its whole representable range.
    ///
    /// Two defects show up here. Clamping the input to ±88 replaces every
    /// result in [88, ln(f32::MAX)] with exp(88), a factor of two out at the
    /// top. Reducing as `(x*log2(e) - n) * ln(2)` instead of Cody-Waite leaves
    /// an absolute error in r proportional to |x|, worth 3e-6 relative near
    /// |x| = 88 — thirty ulps — which the sweep below reaches everywhere.
    ///
    /// The length forces the SIMD path (>= SIMD_THRESHOLD, and a whole number
    /// of AVX2, AVX-512 and NEON f32 lanes), so no element falls through to
    /// the exact scalar fallback.
    #[test]
    fn test_unary_exp_f32_single_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);

        // The band the ±88 clamp used to swallow, up to ln(f32::MAX).
        for k in 0i32..400 {
            a.push(88.0 + 0.722 * (k as f32) / 399.0);
        }
        fill_range_f32(&mut a, LEN, -87.3, 88.72);

        let mut out = vec![0.0f32; LEN];
        unsafe { exp_f32(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).exp() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "exp({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// exp keeps producing subnormals down to -104. A clamp at -88 returns
    /// 6e-39 for every one of them, twenty-eight orders of magnitude high at
    /// the bottom of the range.
    #[test]
    fn test_unary_exp_f32_subnormal_tail() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);
        fill_range_f32(&mut a, LEN, -104.0, -87.4);

        let mut out = vec![0.0f32; LEN];
        unsafe { exp_f32(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).exp() as f32;
            let abs_err = (out[i] - expected).abs();
            assert!(
                abs_err <= 2.0 * SUBNORMAL_STEP_F32,
                "exp({}) = {:e}, expected {:e}, off by {} subnormal steps at index {}",
                a[i],
                out[i],
                expected,
                abs_err / SUBNORMAL_STEP_F32,
                i
            );
        }
    }

    /// f32 exp2 must not borrow `exp(x * ln2)`. That premultiply rounds a value
    /// as large as 128 before the exponential, worth 5e-6 relative — forty
    /// ulps. The ±88 clamp of `exp` also truncates the top of the range, where
    /// 2^127 is still perfectly representable.
    #[test]
    fn test_unary_exp2_f32_single_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);

        // The last binade, which the exp clamp used to cut off at 2^88.
        for k in 0i32..400 {
            a.push(127.0 + (k as f32) / 400.0);
        }

        // Exact integers and half-integers are the ends of the reduction
        // interval, where `r = x - n` is largest.
        for k in -126i32..=127 {
            a.push(k as f32);
            if k < 127 {
                a.push(k as f32 + 0.5);
            }
        }
        fill_range_f32(&mut a, LEN, -126.0, 127.9);

        let mut out = vec![0.0f32; LEN];
        unsafe { exp_f32(UnaryOp::Exp2, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).exp2() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "exp2({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// exp2 keeps producing subnormals down to -149, the whole range a clamp at
    /// -88 discards.
    #[test]
    fn test_unary_exp2_f32_subnormal_tail() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);
        fill_range_f32(&mut a, LEN, -149.0, -126.5);

        let mut out = vec![0.0f32; LEN];
        unsafe { exp_f32(UnaryOp::Exp2, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).exp2() as f32;
            let abs_err = (out[i] - expected).abs();
            assert!(
                abs_err <= 2.0 * SUBNORMAL_STEP_F32,
                "exp2({}) = {:e}, expected {:e}, off by {} subnormal steps at index {}",
                a[i],
                out[i],
                expected,
                abs_err / SUBNORMAL_STEP_F32,
                i
            );
        }
    }

    /// Arguments that break a naive f32 expm1: the ±0.5 boundary of the old
    /// degree-4 Taylor branch, where the dropped `x⁵/120` term is 2.6e-4, and
    /// arguments small enough that `exp(x) - 1` keeps none of the result.
    #[test]
    fn test_unary_expm1_f32_single_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);

        for k in -60i32..=60 {
            a.push(0.5 + (k as f32) * 1e-3);
            a.push(-0.5 + (k as f32) * 1e-3);
        }
        for k in 1i32..=30 {
            a.push(10.0f32.powi(-k));
            a.push(-10.0f32.powi(-k));
        }
        fill_range_f32(&mut a, LEN, -87.0, 88.5);

        let mut out = vec![0.0f32; LEN];
        unsafe { exp_f32(UnaryOp::Expm1, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).exp_m1() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "expm1({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// The clamp that keeps the reduction in range must not eat the domain
    /// edges: on x86 `maxps`/`minps` return their second operand for NaN, so a
    /// clamp written the wrong way round turns NaN into the clamp bound.
    #[test]
    fn test_unary_exp_family_f32_domain_edges() {
        const LEN: usize = 2048;
        let probes: [f32; 8] = [
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            0.0,
            -0.0,
            1.0,
            -1.0,
            120.0,
        ];
        let a: Vec<f32> = (0..LEN).map(|i| probes[i % probes.len()]).collect();
        let mut out = vec![0.0f32; LEN];

        for (op, reference) in [
            (UnaryOp::Exp, f64::exp as fn(f64) -> f64),
            (UnaryOp::Exp2, f64::exp2 as fn(f64) -> f64),
            (UnaryOp::Expm1, f64::exp_m1 as fn(f64) -> f64),
            (UnaryOp::Cbrt, f64::cbrt as fn(f64) -> f64),
        ] {
            unsafe { exp_f32(op, a.as_ptr(), out.as_mut_ptr(), LEN) }

            for i in 0..LEN {
                let expected = reference(a[i] as f64) as f32;
                assert!(
                    rel_err_f32(out[i], expected) < 1e-6,
                    "{:?}({}) = {}, expected {} at index {}",
                    op,
                    a[i],
                    out[i],
                    expected,
                    i
                );
            }
        }
    }

    /// expm1 and cbrt are odd, so they carry the sign of zero, which they can
    /// only do by restoring the sign bit rather than negating.
    #[test]
    fn test_unary_exp_family_f32_signed_zero() {
        const LEN: usize = 2048;
        let a: Vec<f32> = (0..LEN)
            .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
            .collect();
        let mut out = vec![0.0f32; LEN];

        for op in [UnaryOp::Expm1, UnaryOp::Cbrt] {
            unsafe { exp_f32(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
            for i in 0..LEN {
                assert_eq!(
                    out[i].to_bits(),
                    a[i].to_bits(),
                    "{:?} lost the sign of zero at index {}",
                    op,
                    i
                );
            }
        }
    }
}
