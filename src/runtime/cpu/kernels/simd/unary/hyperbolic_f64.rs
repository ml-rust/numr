//! Hyperbolic family (sinh, cosh, tanh, asinh, acosh, atanh) f64 dispatch.
//! Same threshold/ISA-selection logic as every unary family (`dispatch.rs`).

use super::super::{SimdLevel, detect_simd};
use super::dispatch::{SIMD_THRESHOLD, is_simd_supported};
use super::unary_scalar_f64;
use crate::ops::UnaryOp;

/// Dispatch a hyperbolic-family op for f64.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub(super) unsafe fn hyperbolic_f64(op: UnaryOp, a: *const f64, out: *mut f64, len: usize) {
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
    use super::super::test_support::{
        atanh_reference, fill_range_f64, fill_sweep_f64, rel_err_f64,
    };
    use super::*;

    /// sinh and cosh stay finite up to 710.4758, past the 709.7827 where exp
    /// and expm1 overflow, so the band between the two has to be reached by
    /// squaring `exp(|x|/2)` rather than by scaling `exp(|x|)`.
    #[test]
    fn test_unary_hyperbolic_f64_upper_band() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        for k in 0i32..1024 {
            a.push(709.0 + 1.475 * (k as f64) / 1023.0);
        }
        fill_range_f64(&mut a, LEN, -710.475, -709.0);

        let mut out = vec![0.0f64; LEN];

        for (op, reference) in [
            (UnaryOp::Sinh, f64::sinh as fn(f64) -> f64),
            (UnaryOp::Cosh, f64::cosh as fn(f64) -> f64),
        ] {
            unsafe { hyperbolic_f64(op, a.as_ptr(), out.as_mut_ptr(), LEN) }

            for i in 0..LEN {
                let expected = reference(a[i]);
                assert!(
                    expected.is_finite(),
                    "probe {} is outside the representable range at index {}",
                    a[i],
                    i
                );
                let rel_err = rel_err_f64(out[i], expected);
                assert!(
                    rel_err < 1e-14,
                    "{:?}({}) = {:e}, expected {:e}, rel_err = {} at index {}",
                    op,
                    a[i],
                    out[i],
                    expected,
                    rel_err,
                    i
                );
            }
        }
    }

    /// Small |x| for the two hyperbolic functions. `(e^x - e^-x)/2` and
    /// `(e^2x - 1)/(e^2x + 1)` both subtract quantities that approach each
    /// other as x approaches zero, so they lose the result they are computing.
    #[test]
    fn test_unary_sinh_tanh_f64_double_precision() {
        const LEN: usize = 2048;
        let mut probes: Vec<f64> = Vec::with_capacity(LEN);

        for k in 1i32..=300 {
            probes.push(10.0f64.powi(-k));
            probes.push(-10.0f64.powi(-k));
        }
        for k in -100i32..=100 {
            probes.push(-0.0024 + (k as f64) * 1e-5);
        }

        let mut out = vec![0.0f64; LEN];

        for (op, reference, hi) in [
            (UnaryOp::Sinh, f64::sinh as fn(f64) -> f64, 700.0f64),
            (UnaryOp::Tanh, f64::tanh as fn(f64) -> f64, 30.0f64),
        ] {
            let mut a = probes.clone();
            fill_range_f64(&mut a, LEN, -hi, hi);

            unsafe { hyperbolic_f64(op, a.as_ptr(), out.as_mut_ptr(), LEN) }

            for i in 0..LEN {
                let expected = reference(a[i]);
                let rel_err = rel_err_f64(out[i], expected);
                assert!(
                    rel_err < 1e-14,
                    "{:?}({}) = {}, expected {}, rel_err = {} at index {}",
                    op,
                    a[i],
                    out[i],
                    expected,
                    rel_err,
                    i
                );
            }
        }
    }

    /// asinh at negative arguments, where `log(x + sqrt(x²+1))` cancels: at
    /// x = -49.6 the two addends agree to twelve digits and the sum keeps only
    /// the top three. Small |x| exercises the same cancellation from the other
    /// side, where asinh(x) == x.
    #[test]
    fn test_unary_asinh_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        for k in -100i32..=100 {
            a.push(-49.6093 + (k as f64) * 1e-6);
        }
        for k in 1i32..=300 {
            a.push(10.0f64.powi(-k));
            a.push(-10.0f64.powi(-k));
        }

        // Both sides of the 2 and 2^28 branch points, and the far tail.
        for k in -20i32..=20 {
            a.push(2.0 + (k as f64) * 1e-12);
            a.push(268_435_456.0 + (k as f64) * 1e-3);
        }
        for k in 1i32..=30 {
            a.push(10.0f64.powi(k));
            a.push(-10.0f64.powi(k));
        }
        fill_sweep_f64(&mut a, LEN, 3.0);

        let mut out = vec![0.0f64; LEN];
        unsafe { hyperbolic_f64(UnaryOp::Asinh, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].asinh();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "asinh({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// acosh near 1, where `x² - 1` throws away half the significant bits of
    /// `x - 1` — and `x - 1` is the whole result there.
    #[test]
    fn test_unary_acosh_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        for k in -100i32..=100 {
            a.push(1.01 + (k as f64) * 1e-6);
        }
        for k in 1i32..=300 {
            a.push(1.0 + 10.0f64.powi(-k));
        }
        for k in -20i32..=20 {
            a.push(2.0 + (k as f64) * 1e-12);
            a.push(268_435_456.0 + (k as f64) * 1e-3);
        }
        fill_range_f64(&mut a, LEN, 1.0, 1e6);

        let mut out = vec![0.0f64; LEN];
        unsafe { hyperbolic_f64(UnaryOp::Acosh, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].acosh();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "acosh({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// atanh at small |x|, where forming `(1+x)/(1-x)` rounds the ratio to
    /// one ulp of 1 and the log of it keeps nothing: at x = 7e-4 that is a
    /// relative error of 1.4e-13, 600 ulps.
    #[test]
    fn test_unary_atanh_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        for k in -100i32..=100 {
            a.push(0.0007 + (k as f64) * 1e-9);
            a.push(-0.0007 + (k as f64) * 1e-9);
        }
        for k in 1i32..=300 {
            a.push(10.0f64.powi(-k));
            a.push(-10.0f64.powi(-k));
        }

        // Approaching ±1, where atanh -> ±inf, and both sides of the 0.5 split.
        for k in 1i32..=40 {
            a.push(1.0 - 2.0f64.powi(-k));
            a.push(-1.0 + 2.0f64.powi(-k));
        }
        for k in -20i32..=20 {
            a.push(0.5 + (k as f64) * 1e-12);
        }
        fill_sweep_f64(&mut a, LEN, 0.9999);

        let mut out = vec![0.0f64; LEN];
        unsafe { hyperbolic_f64(UnaryOp::Atanh, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = atanh_reference(a[i]);
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "atanh({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }

        // The reference only exercises std on x >= 0, so the sign is pinned
        // separately: atanh is odd, and this kernel is odd bit for bit because
        // it works on |x| and restores the sign bit.
        let neg: Vec<f64> = a.iter().map(|v| -v).collect();
        let mut neg_out = vec![0.0f64; LEN];
        unsafe { hyperbolic_f64(UnaryOp::Atanh, neg.as_ptr(), neg_out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            if out[i].is_nan() {
                assert!(neg_out[i].is_nan(), "atanh({}) lost NaN", neg[i]);
            } else {
                assert_eq!(
                    neg_out[i].to_bits(),
                    (-out[i]).to_bits(),
                    "atanh({}) and atanh({}) are not exact negatives",
                    neg[i],
                    a[i]
                );
            }
        }
    }

    #[test]
    fn test_unary_hyperbolic_f64_domain_edges() {
        const LEN: usize = 2048;

        let edges = [
            0.0f64,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
            -2.0,
            1.5,
            -1.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            5e-324,
            -5e-324,
            1e-310,
        ];
        let a: Vec<f64> = (0..LEN).map(|i| edges[i % edges.len()]).collect();
        let mut out = vec![0.0f64; LEN];

        for (op, reference) in [
            (UnaryOp::Expm1, f64::exp_m1 as fn(f64) -> f64),
            (UnaryOp::Exp2, f64::exp2 as fn(f64) -> f64),
            (UnaryOp::Sinh, f64::sinh as fn(f64) -> f64),
            (UnaryOp::Tanh, f64::tanh as fn(f64) -> f64),
            (UnaryOp::Asinh, f64::asinh as fn(f64) -> f64),
            (UnaryOp::Acosh, f64::acosh as fn(f64) -> f64),
            (UnaryOp::Atanh, f64::atanh as fn(f64) -> f64),
        ] {
            unsafe { hyperbolic_f64(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
            for i in 0..LEN {
                let expected = reference(a[i]);
                let rel_err = rel_err_f64(out[i], expected);
                assert!(
                    rel_err < 1e-14,
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

    /// The odd functions carry the sign of zero, which they can only do by
    /// working on |x| and restoring the sign bit rather than negating.
    #[test]
    fn test_unary_hyperbolic_f64_signed_zero() {
        const LEN: usize = 2048;
        let a: Vec<f64> = (0..LEN)
            .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
            .collect();
        let mut out = vec![0.0f64; LEN];

        for op in [
            UnaryOp::Expm1,
            UnaryOp::Sinh,
            UnaryOp::Tanh,
            UnaryOp::Asinh,
            UnaryOp::Atanh,
        ] {
            unsafe { hyperbolic_f64(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
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
