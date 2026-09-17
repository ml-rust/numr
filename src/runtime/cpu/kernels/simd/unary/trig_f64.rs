//! Trig family (sin, cos, tan) f64 dispatch.
//! Same threshold/ISA-selection logic as every unary family (`dispatch.rs`).

use super::super::{SimdLevel, detect_simd};
use super::dispatch::{SIMD_THRESHOLD, is_simd_supported};
use super::unary_scalar_f64;
use crate::ops::UnaryOp;

/// Dispatch a trig op (Sin/Cos/Tan) for f64.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub(super) unsafe fn trig_f64(op: UnaryOp, a: *const f64, out: *mut f64, len: usize) {
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
    use super::super::test_support::{fill_range_f64, fill_sweep_f64, rel_err_f64};
    use super::*;

    /// Arguments that expose every weak point of a π/2 range reduction: the
    /// multiples of π/2, where the reduced argument cancels down to almost
    /// nothing, and large |x|, where a single rounded π/2 leaves an absolute
    /// error proportional to |x|. A uniform sweep alone hits neither.
    fn trig_probe_points_f64(len: usize) -> Vec<f64> {
        let mut a: Vec<f64> = Vec::with_capacity(len);

        // Every multiple of π/2 out to |x| = 100, and its immediate
        // neighbourhood. One of sin or cos crosses zero at each of them, so the
        // reduced argument carries the whole result.
        for k in -64i32..=64 {
            let c = (k as f64) * std::f64::consts::FRAC_PI_2;
            a.push(c);
            for step in 1i32..=3 {
                a.push(c + (step as f64) * 1e-13);
                a.push(c - (step as f64) * 1e-13);
            }
        }

        // Magnitudes far past the test sweep, where reduction error dominates
        // polynomial error. The stride is prime-ish so the points do not line
        // up with any multiple of π/2.
        for k in 0i32..200 {
            let d = 1e3 + (k as f64) * 977.0;
            a.push(d);
            a.push(-d);
        }

        fill_sweep_f64(&mut a, len, 100.0);
        a
    }

    /// f64 sin must reach double precision across the reduction range.
    ///
    /// The degree-9 Taylor series this replaced left ~2.6e-8 of truncation
    /// error near the ends of [-π/4, π/4], and reducing with a single rounded
    /// π/2 left ~7e-12 at |x| = 2e5. Both fail this bound outright.
    #[test]
    fn test_unary_sin_f64_double_precision() {
        const LEN: usize = 2048;
        let a = trig_probe_points_f64(LEN);

        let mut out = vec![0.0f64; LEN];
        unsafe { trig_f64(UnaryOp::Sin, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].sin();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "sin({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// f64 cos must reach double precision across the reduction range.
    ///
    /// Building cos as `sin(x + π/2)` rounds the sum before reduction, so this
    /// also fails for any x large enough that the addition is inexact.
    #[test]
    fn test_unary_cos_f64_double_precision() {
        const LEN: usize = 2048;
        let a = trig_probe_points_f64(LEN);

        let mut out = vec![0.0f64; LEN];
        unsafe { trig_f64(UnaryOp::Cos, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].cos();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "cos({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// f64 tan must reach double precision away from its poles.
    ///
    /// The truncated Taylor series this replaced left ~5e-5 of relative error
    /// at the edge of the reduction interval, which ±π/4 targets directly.
    #[test]
    fn test_unary_tan_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        // ±π/4 is the edge of the reduction interval and the worst point of any
        // fixed-degree polynomial in the reduced argument.
        for &b in &[std::f64::consts::FRAC_PI_4, 0.5, 1.0] {
            for k in -60i32..=60 {
                let d = b + (k as f64) * 1e-13;
                a.push(d);
                a.push(-d);
            }
        }

        // Multiples of π, where tan crosses zero and the reduction cancels.
        for k in -30i32..=30 {
            a.push((k as f64) * std::f64::consts::PI);
        }

        // Large |x|, where reduction error dominates. Points near a pole are
        // dropped: there the result itself is ill-conditioned, not the kernel.
        for k in 0i32..200 {
            let d = 1e3 + (k as f64) * 977.0 + 0.37;
            if d.cos().abs() > 1e-3 {
                a.push(d);
                a.push(-d);
            }
        }

        fill_range_f64(&mut a, LEN, -1.5, 1.5);

        let mut out = vec![0.0f64; LEN];
        unsafe { trig_f64(UnaryOp::Tan, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].tan();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
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
    fn test_unary_trig_f64_domain_edges() {
        const LEN: usize = 2048;

        // ±inf and NaN have no finite reduction, so all three are NaN there.
        // ±0 must keep its own sign, which `x - j*π/2` destroys for j = -0.
        let edges = [
            0.0f64,
            -0.0,
            1.0,
            -1.0,
            std::f64::consts::FRAC_PI_2,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let a: Vec<f64> = (0..LEN).map(|i| edges[i % edges.len()]).collect();
        let mut out = vec![0.0f64; LEN];

        for (op, reference) in [
            (UnaryOp::Sin, f64::sin as fn(f64) -> f64),
            (UnaryOp::Cos, f64::cos as fn(f64) -> f64),
            (UnaryOp::Tan, f64::tan as fn(f64) -> f64),
        ] {
            unsafe { trig_f64(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
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
                let expected = reference(a[i]);
                assert!(
                    rel_err_f64(out[i], expected) < 1e-14,
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
