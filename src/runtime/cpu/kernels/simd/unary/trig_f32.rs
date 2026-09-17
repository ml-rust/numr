//! Trig family (sin, cos, tan) f32 dispatch.
//! Same threshold/ISA-selection logic as every unary family (`dispatch.rs`).

use super::super::{SimdLevel, detect_simd};
use super::dispatch::{SIMD_THRESHOLD, is_simd_supported};
use super::unary_scalar_f32;
use crate::ops::UnaryOp;

/// Dispatch a trig op (Sin/Cos/Tan) for f32.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub(super) unsafe fn trig_f32(op: UnaryOp, a: *const f32, out: *mut f32, len: usize) {
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
    use super::super::test_support::{fill_range_f32, rel_err_f32};
    use super::*;

    #[test]
    fn test_unary_sin_f32() {
        let a: Vec<f32> = (0..100).map(|x| (x as f32 - 50.0) * 0.1).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { trig_f32(UnaryOp::Sin, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].sin();
            let diff = (out[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "sin mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_unary_cos_f32() {
        let a: Vec<f32> = (0..100).map(|x| (x as f32 - 50.0) * 0.1).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { trig_f32(UnaryOp::Cos, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].cos();
            let diff = (out[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "cos mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_unary_tan_f32() {
        // Avoid values near π/2 where tan approaches infinity
        let a: Vec<f32> = (0..100).map(|x| (x as f32 - 50.0) * 0.02).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { trig_f32(UnaryOp::Tan, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].tan();
            let diff = (out[i] - expected).abs();
            // Relative error tolerance of ~2e-4 is acceptable for f32 SIMD tan approximations
            assert!(
                diff < 2e-4 * expected.abs().max(1.0),
                "tan mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    /// Probe points for f32 sin/cos: every multiple of π/2 the reduction has to
    /// survive, plus magnitudes far past the sweep.
    fn trig_probe_points_f32(len: usize) -> Vec<f32> {
        let mut a: Vec<f32> = Vec::with_capacity(len);

        // Every multiple of π/2 out to |x| = 100, and its immediate
        // neighbourhood. One of sin or cos crosses zero at each of them, so the
        // reduced argument carries the whole result.
        for k in -64i32..=64 {
            let c = (k as f32) * std::f32::consts::FRAC_PI_2;
            a.push(c);
            for step in 1i32..=3 {
                a.push(c + (step as f32) * 1e-6);
                a.push(c - (step as f32) * 1e-6);
            }
        }

        // Multiples of π/2 out to |x| = 1.2e5. Reducing with a single rounded
        // π/2 leaves an absolute phase error of |j| * 4.4e-8 here — over 3e-3,
        // which is the whole answer.
        for k in 0i32..120 {
            let c = ((20_000 + k * 500) as f32) * std::f32::consts::FRAC_PI_2;
            a.push(c);
            a.push(-c);
        }

        // Magnitudes on a stride that lines up with no multiple of π/2.
        for k in 0i32..200 {
            let d = 1.0e3 + (k as f32) * 601.0;
            a.push(d);
            a.push(-d);
        }

        fill_range_f32(&mut a, len, -100.0, 100.0);
        a
    }

    /// f32 sin must reach single precision across the reduction range.
    ///
    /// Reducing with a single rounded π/2 costs |j| * 4.4e-8 of absolute phase,
    /// so this fails by more than the answer itself past |x| ~ 1e4. The
    /// degree-6 cos Taylor series it paired with is separately worth 3.6e-6 at
    /// y = π/4, thirty ulps.
    ///
    /// The length forces the SIMD path (>= SIMD_THRESHOLD, and a whole number
    /// of AVX2, AVX-512 and NEON f32 lanes), so no element falls through to the
    /// exact scalar fallback.
    #[test]
    fn test_unary_sin_f32_single_precision() {
        const LEN: usize = 2048;
        let a = trig_probe_points_f32(LEN);

        let mut out = vec![0.0f32; LEN];
        unsafe { trig_f32(UnaryOp::Sin, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).sin() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "sin({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// f32 cos must reach single precision across the reduction range.
    ///
    /// Building cos as `sin(x + π/2)` rounds the sum before reduction, which
    /// costs an ulp of x — already the whole answer near a zero of cos.
    #[test]
    fn test_unary_cos_f32_single_precision() {
        const LEN: usize = 2048;
        let a = trig_probe_points_f32(LEN);

        let mut out = vec![0.0f32; LEN];
        unsafe { trig_f32(UnaryOp::Cos, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).cos() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "cos({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// f32 tan must reach single precision away from its poles.
    ///
    /// The truncated Taylor series this replaced dropped a term worth 1.5e-4 at
    /// y = ±π/4, which the dense band there reaches on every point.
    #[test]
    fn test_unary_tan_f32_single_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);

        // ±π/4 is the edge of the reduction interval and the worst point of any
        // fixed-degree polynomial in the reduced argument.
        for &b in &[std::f32::consts::FRAC_PI_4, 0.5, 1.0] {
            for k in -60i32..=60 {
                let d = b + (k as f32) * 1e-6;
                a.push(d);
                a.push(-d);
            }
        }

        // Multiples of π, where tan crosses zero and the reduction cancels.
        for k in -30i32..=30 {
            a.push((k as f32) * std::f32::consts::PI);
        }

        // Large |x|, where reduction error dominates. Points near a pole are
        // dropped: there the result itself is ill-conditioned, not the kernel.
        for k in 0i32..200 {
            let d = 1.0e3 + (k as f32) * 601.0 + 0.37;
            if (d as f64).cos().abs() > 1e-3 {
                a.push(d);
                a.push(-d);
            }
        }

        fill_range_f32(&mut a, LEN, -1.5, 1.5);

        let mut out = vec![0.0f32; LEN];
        unsafe { trig_f32(UnaryOp::Tan, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).tan() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "tan({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_trig_f32_domain_edges() {
        const LEN: usize = 2048;

        // ±inf and NaN have no finite reduction, so all three are NaN there.
        // ±0 must keep its own sign, which `x - j*π/2` destroys for j = -0.
        let edges = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            std::f32::consts::FRAC_PI_2,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let a: Vec<f32> = (0..LEN).map(|i| edges[i % edges.len()]).collect();
        let mut out = vec![0.0f32; LEN];

        for (op, reference) in [
            (UnaryOp::Sin, f64::sin as fn(f64) -> f64),
            (UnaryOp::Cos, f64::cos as fn(f64) -> f64),
            (UnaryOp::Tan, f64::tan as fn(f64) -> f64),
        ] {
            unsafe { trig_f32(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
            for i in 0..LEN {
                if !a[i].is_finite() {
                    assert!(
                        out[i].is_nan(),
                        "{:?}({}) = {}, expected NaN",
                        op,
                        a[i],
                        out[i]
                    );
                    continue;
                }
                let expected = reference(a[i] as f64) as f32;
                assert!(
                    rel_err_f32(out[i], expected) < 1e-6,
                    "{:?}({}) = {}, expected {}",
                    op,
                    a[i],
                    out[i],
                    expected
                );
                if expected == 0.0 {
                    assert_eq!(
                        out[i].is_sign_negative(),
                        expected.is_sign_negative(),
                        "{:?}({}) lost the sign of zero",
                        op,
                        a[i]
                    );
                }
            }
        }
    }
}
