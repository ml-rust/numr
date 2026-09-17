//! Hyperbolic family (sinh, cosh, tanh, asinh, acosh, atanh) f32 dispatch.
//! Same threshold/ISA-selection logic as every unary family (`dispatch.rs`).

use super::super::{SimdLevel, detect_simd};
use super::dispatch::{SIMD_THRESHOLD, is_simd_supported};
use super::unary_scalar_f32;
use crate::ops::UnaryOp;

/// Dispatch a hyperbolic-family op for f32.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub(super) unsafe fn hyperbolic_f32(op: UnaryOp, a: *const f32, out: *mut f32, len: usize) {
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
    use super::super::test_support::{
        SUBNORMAL_STEP_F32, atanh_reference, fill_range_f32, nudge_f32, rel_err_f32,
    };
    use super::*;

    #[test]
    fn test_unary_tanh_f32() {
        let a: Vec<f32> = (0..100).map(|x| (x as f32 - 50.0) * 0.1).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { hyperbolic_f32(UnaryOp::Tanh, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].tanh();
            let diff = (out[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "tanh mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    /// f32 sinh and tanh at tiny |x|, where `(e^x - e^-x)/2` and
    /// `(e^2x - 1)/(e^2x + 1)` subtract two values that both approach 1 and
    /// keep none of the result.
    #[test]
    fn test_unary_sinh_tanh_f32_single_precision() {
        const LEN: usize = 2048;
        let mut probes: Vec<f32> = Vec::with_capacity(LEN);

        for k in 1i32..=30 {
            probes.push(10.0f32.powi(-k));
            probes.push(-10.0f32.powi(-k));
        }
        for k in -100i32..=100 {
            probes.push(0.0024420025 + (k as f32) * 1e-8);
            probes.push(-0.021489622 + (k as f32) * 1e-7);
        }

        let mut out = vec![0.0f32; LEN];

        for (op, reference, hi) in [
            (UnaryOp::Sinh, f64::sinh as fn(f64) -> f64, 88.0f32),
            (UnaryOp::Tanh, f64::tanh as fn(f64) -> f64, 20.0f32),
        ] {
            let mut a = probes.clone();
            fill_range_f32(&mut a, LEN, -hi, hi);

            unsafe { hyperbolic_f32(op, a.as_ptr(), out.as_mut_ptr(), LEN) }

            for i in 0..LEN {
                let expected = reference(a[i] as f64) as f32;
                let rel_err = rel_err_f32(out[i], expected);
                assert!(
                    rel_err < 1e-6,
                    "{:?}({:e}) = {:e}, expected {:e}, rel_err = {} at index {}",
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

    /// sinh and cosh stay finite in f32 up to 89.4159, but `exp` overflows at
    /// ln(f32::MAX) = 88.7228. Composing them from `exp` returns infinity over
    /// that whole band, where every true result is an ordinary float.
    #[test]
    fn test_unary_hyperbolic_f32_upper_band() {
        const LEN: usize = 2048;
        let mut out = vec![0.0f32; LEN];

        for (op, reference) in [
            (UnaryOp::Sinh, f64::sinh as fn(f64) -> f64),
            (UnaryOp::Cosh, f64::cosh as fn(f64) -> f64),
        ] {
            for sign in [1.0f32, -1.0] {
                let mut a: Vec<f32> = Vec::with_capacity(LEN);
                fill_range_f32(&mut a, LEN, sign * 88.73, sign * 89.40);

                unsafe { hyperbolic_f32(op, a.as_ptr(), out.as_mut_ptr(), LEN) }

                for i in 0..LEN {
                    let expected = reference(a[i] as f64) as f32;
                    assert!(
                        expected.is_finite(),
                        "{:?}({}) reference is not finite",
                        op,
                        a[i]
                    );
                    let rel_err = rel_err_f32(out[i], expected);
                    assert!(
                        rel_err < 1e-6,
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
    }

    /// f32 asinh at negative arguments, where `log(x + sqrt(x²+1))` cancels,
    /// and past 1.8e19, where `x²` overflows f32 outright.
    #[test]
    fn test_unary_asinh_f32_single_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);

        for k in -100i32..=100 {
            a.push(-0.35409036 + (k as f32) * 1e-7);
        }
        for k in 1i32..=30 {
            a.push(10.0f32.powi(-k));
            a.push(-10.0f32.powi(-k));
        }
        // Both sides of the 2 and 2^12 branch points, and the far tail, where
        // the middle branch would square into infinity.
        for k in -20i32..=20 {
            a.push(nudge_f32(2.0, k));
            a.push(nudge_f32(4096.0, k));
        }
        for k in 1i32..=38 {
            a.push(10.0f32.powi(k));
            a.push(-10.0f32.powi(k));
        }
        fill_range_f32(&mut a, LEN, -5.0, 5.0);

        let mut out = vec![0.0f32; LEN];
        unsafe { hyperbolic_f32(UnaryOp::Asinh, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).asinh() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "asinh({:e}) = {:e}, expected {:e}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// f32 acosh near 1, where `x² - 1` throws away half the significant bits
    /// of `x - 1`, and at very large x, where `x²` overflows f32 and the old
    /// kernel returned `log(f32::INFINITY)` — 88.72 for every input above
    /// 1.8e19, whatever the true answer.
    #[test]
    fn test_unary_acosh_f32_single_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);

        for k in -100i32..=100 {
            a.push(1.0019791 + (k as f32) * 1e-7);
        }
        for k in 1i32..=23 {
            a.push(1.0 + 2.0f32.powi(-k));
        }
        for k in -20i32..=20 {
            a.push(nudge_f32(2.0, k));
            a.push(nudge_f32(4096.0, k));
        }
        for k in 1i32..=38 {
            a.push(10.0f32.powi(k));
        }
        a.push(1.0);
        a.push(2.442e33);
        a.push(f32::MAX);
        fill_range_f32(&mut a, LEN, 1.0, 1e6);

        let mut out = vec![0.0f32; LEN];
        unsafe { hyperbolic_f32(UnaryOp::Acosh, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).acosh() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "acosh({:e}) = {:e}, expected {:e}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// f32 atanh at small |x|, where forming `(1+x)/(1-x)` rounds the ratio to
    /// 1 and discards every bit the result is made of.
    #[test]
    fn test_unary_atanh_f32_single_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);

        for k in -100i32..=100 {
            a.push(0.1714066 + (k as f32) * 1e-7);
            a.push(-0.1714066 + (k as f32) * 1e-7);
        }
        for k in 1i32..=30 {
            a.push(10.0f32.powi(-k));
            a.push(-10.0f32.powi(-k));
        }
        // Approaching ±1, where atanh -> ±inf, and both sides of the 0.5 split.
        for k in 1i32..=23 {
            a.push(1.0 - 2.0f32.powi(-k));
            a.push(-1.0 + 2.0f32.powi(-k));
        }
        for k in -20i32..=20 {
            a.push(nudge_f32(0.5, k));
        }
        fill_range_f32(&mut a, LEN, -0.9999, 0.9999);

        let mut out = vec![0.0f32; LEN];
        unsafe { hyperbolic_f32(UnaryOp::Atanh, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            // `f64::atanh` is accurate on x >= 0 only; see `atanh_reference`.
            let expected = atanh_reference(a[i] as f64) as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "atanh({:e}) = {:e}, expected {:e}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// The f32 hyperbolic domain edges: tanh(±inf) = ±1, acosh(x < 1) = NaN,
    /// atanh(±1) = ±inf and atanh(|x| > 1) = NaN. The old kernels had none of
    /// them — atanh(1) formed 2/0 and reported the logarithm of infinity.
    #[test]
    fn test_unary_hyperbolic_f32_domain_edges() {
        const LEN: usize = 2048;

        let edges = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
            -2.0,
            1.5,
            -1.5,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            SUBNORMAL_STEP_F32,
            -SUBNORMAL_STEP_F32,
            f32::MIN_POSITIVE,
        ];
        let a: Vec<f32> = (0..LEN).map(|i| edges[i % edges.len()]).collect();
        let mut out = vec![0.0f32; LEN];

        for (op, reference) in [
            (UnaryOp::Sinh, f64::sinh as fn(f64) -> f64),
            (UnaryOp::Cosh, f64::cosh as fn(f64) -> f64),
            (UnaryOp::Tanh, f64::tanh as fn(f64) -> f64),
            (UnaryOp::Asinh, f64::asinh as fn(f64) -> f64),
            (UnaryOp::Acosh, f64::acosh as fn(f64) -> f64),
            (UnaryOp::Atanh, f64::atanh as fn(f64) -> f64),
            (UnaryOp::Log1p, f64::ln_1p as fn(f64) -> f64),
        ] {
            unsafe { hyperbolic_f32(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
            for i in 0..LEN {
                let expected = reference(a[i] as f64) as f32;
                let rel_err = rel_err_f32(out[i], expected);
                assert!(
                    rel_err < 1e-6,
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

    /// The odd f32 functions carry the sign of zero, which they can only do by
    /// working on |x| and restoring the sign bit rather than negating.
    #[test]
    fn test_unary_hyperbolic_f32_signed_zero() {
        const LEN: usize = 2048;
        let a: Vec<f32> = (0..LEN)
            .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
            .collect();
        let mut out = vec![0.0f32; LEN];

        for op in [
            UnaryOp::Sinh,
            UnaryOp::Tanh,
            UnaryOp::Asinh,
            UnaryOp::Atanh,
            UnaryOp::Log1p,
        ] {
            unsafe { hyperbolic_f32(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
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
