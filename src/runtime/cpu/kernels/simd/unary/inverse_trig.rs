//! Inverse trig family (asin, acos, atan) dispatch.
//! Same threshold/ISA-selection logic as every unary family (`dispatch.rs`).

use super::super::{SimdLevel, detect_simd};
use super::dispatch::{SIMD_THRESHOLD, is_simd_supported};
use super::{unary_scalar_f32, unary_scalar_f64};
use crate::ops::UnaryOp;

/// Dispatch an inverse trig op (Asin/Acos/Atan) for f32.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub(super) unsafe fn inverse_trig_f32(op: UnaryOp, a: *const f32, out: *mut f32, len: usize) {
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

/// f64 counterpart of [`inverse_trig_f32`].
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub(super) unsafe fn inverse_trig_f64(op: UnaryOp, a: *const f64, out: *mut f64, len: usize) {
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
    use super::super::test_support::{fill_range_f32, fill_sweep_f64, rel_err_f32, rel_err_f64};
    use super::*;

    #[test]
    fn test_unary_asin_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        // 1/sqrt(2) is where asin crosses from the direct series to the
        // reflection in every naive atan-based formulation, so a wrong branch
        // shows up here first. 0.5 is this implementation's own branch point.
        for &b in &[std::f64::consts::FRAC_1_SQRT_2, 0.5, 1.0] {
            for k in -100i32..=100 {
                let d = b + (k as f64) * 1e-12;
                if d <= 1.0 {
                    a.push(d);
                    a.push(-d);
                }
            }
        }
        fill_sweep_f64(&mut a, LEN, 1.0);

        let mut out = vec![0.0f64; LEN];
        unsafe { inverse_trig_f64(UnaryOp::Asin, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].asin();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "asin({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_acos_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        for &b in &[std::f64::consts::FRAC_1_SQRT_2, 0.5, 1.0] {
            for k in -100i32..=100 {
                let d = b + (k as f64) * 1e-12;
                if d <= 1.0 {
                    a.push(d);
                    a.push(-d);
                }
            }
        }
        fill_sweep_f64(&mut a, LEN, 1.0);

        let mut out = vec![0.0f64; LEN];
        unsafe { inverse_trig_f64(UnaryOp::Acos, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].acos();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "acos({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_atan_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        // Every reduction breakpoint, from both sides. |x| = 1 is the boundary
        // of the naive reciprocal reduction and the worst point of a truncated
        // Gregory series.
        for &b in &[0.4375f64, 0.6875, 1.0, 1.1875, 2.4375] {
            for k in -40i32..=40 {
                let d = b + (k as f64) * 1e-12;
                a.push(d);
                a.push(-d);
            }
        }
        // Magnitudes far past the last breakpoint, where t = -1/|x| is used.
        for k in 0..100 {
            let d = 10.0f64.powi(k % 25 + 2);
            a.push(d);
            a.push(-d);
        }
        fill_sweep_f64(&mut a, LEN, 8.0);

        let mut out = vec![0.0f64; LEN];
        unsafe { inverse_trig_f64(UnaryOp::Atan, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].atan();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "atan({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_inverse_trig_f64_domain_edges() {
        const LEN: usize = 2048;

        // ±1 are exact endpoints; beyond them asin/acos are undefined.
        let asin_edges = [1.0f64, -1.0, 0.0, -0.0, 1.5, -1.5, f64::INFINITY, f64::NAN];
        let a: Vec<f64> = (0..LEN).map(|i| asin_edges[i % asin_edges.len()]).collect();

        let mut out = vec![0.0f64; LEN];
        unsafe { inverse_trig_f64(UnaryOp::Asin, a.as_ptr(), out.as_mut_ptr(), LEN) }
        for i in 0..LEN {
            let expected = a[i].asin();
            if expected.is_nan() {
                assert!(out[i].is_nan(), "asin({}) = {}, expected NaN", a[i], out[i]);
            } else {
                assert!(
                    rel_err_f64(out[i], expected) < 1e-14,
                    "asin({}) = {}, expected {}",
                    a[i],
                    out[i],
                    expected
                );
            }
        }

        unsafe { inverse_trig_f64(UnaryOp::Acos, a.as_ptr(), out.as_mut_ptr(), LEN) }
        for i in 0..LEN {
            let expected = a[i].acos();
            if expected.is_nan() {
                assert!(out[i].is_nan(), "acos({}) = {}, expected NaN", a[i], out[i]);
            } else {
                assert!(
                    rel_err_f64(out[i], expected) < 1e-14,
                    "acos({}) = {}, expected {}",
                    a[i],
                    out[i],
                    expected
                );
            }
        }

        // atan is defined everywhere; ±inf must saturate to ±pi/2.
        let atan_edges = [
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0f64,
            1.0,
            -1.0,
            1e308,
            -1e308,
            f64::NAN,
        ];
        let a: Vec<f64> = (0..LEN).map(|i| atan_edges[i % atan_edges.len()]).collect();
        unsafe { inverse_trig_f64(UnaryOp::Atan, a.as_ptr(), out.as_mut_ptr(), LEN) }
        for i in 0..LEN {
            if a[i].is_nan() {
                assert!(out[i].is_nan(), "atan(NaN) = {}", out[i]);
            } else {
                let expected = a[i].atan();
                assert!(
                    rel_err_f64(out[i], expected) < 1e-14,
                    "atan({}) = {}, expected {}",
                    a[i],
                    out[i],
                    expected
                );
            }
        }
    }

    /// Probe points for f32 asin/acos: the branch points, the endpoints, and a
    /// sweep of the whole domain.
    fn inverse_trig_probe_points_f32(len: usize) -> Vec<f32> {
        let mut a: Vec<f32> = Vec::with_capacity(len);

        // 1/sqrt(2) is where asin crosses from the direct series to the
        // reflection in any atan-based formulation, because atan's argument
        // reaches exactly 1 there. 0.5 is this implementation's own branch
        // point, and ±1 is the endpoint the reflection has to land on exactly.
        for &b in &[std::f32::consts::FRAC_1_SQRT_2, 0.5, 1.0] {
            for k in -80i32..=80 {
                let d = b + (k as f32) * 1e-7;
                if d <= 1.0 {
                    a.push(d);
                    a.push(-d);
                }
            }
        }

        fill_range_f32(&mut a, len, -1.0, 1.0);
        a
    }

    /// f32 asin must reach single precision over [-1, 1].
    ///
    /// Composing it as `atan(x / sqrt(1 - x²))` sends atan's argument to 1 at
    /// |x| = 1/sqrt(2), the slowest-converging point of the Gregory series it
    /// used to call, worth 4.5e-2 relative.
    #[test]
    fn test_unary_asin_f32_single_precision() {
        const LEN: usize = 2048;
        let a = inverse_trig_probe_points_f32(LEN);

        let mut out = vec![0.0f32; LEN];
        unsafe { inverse_trig_f32(UnaryOp::Asin, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).asin() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "asin({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// f32 acos must reach single precision over [-1, 1].
    ///
    /// `π/2 - asin(x)` inherits every defect of asin and adds cancellation as
    /// x approaches 1, where the result is the difference that vanishes.
    #[test]
    fn test_unary_acos_f32_single_precision() {
        const LEN: usize = 2048;
        let a = inverse_trig_probe_points_f32(LEN);

        let mut out = vec![0.0f32; LEN];
        unsafe { inverse_trig_f32(UnaryOp::Acos, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).acos() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "acos({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// f32 atan must reach single precision for every finite input.
    ///
    /// The Gregory series it used to evaluate on [0, 1] converges like
    /// 1/(2n+3) at the boundary: seven terms leave ~4e-2 relative error, which
    /// is where the old peak sat.
    #[test]
    fn test_unary_atan_f32_single_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);

        // |x| = 1 is the old reduction boundary; the two tan(π/8) and tan(3π/8)
        // values are the new ones.
        for &b in &[1.0f32, 0.414_213_56, 2.414_213_6, 0.989_011] {
            for k in -60i32..=60 {
                let d = b * (1.0 + (k as f32) * 1e-6);
                a.push(d);
                a.push(-d);
            }
        }

        // Many magnitudes: atan has to hold from the smallest normal up to the
        // point where the result is π/2 to the last bit.
        for k in -35i32..=35 {
            let d = 10f32.powi(k);
            a.push(d);
            a.push(-d);
        }

        fill_range_f32(&mut a, LEN, -20.0, 20.0);

        let mut out = vec![0.0f32; LEN];
        unsafe { inverse_trig_f32(UnaryOp::Atan, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).atan() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "atan({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_inverse_trig_f32_domain_edges() {
        const LEN: usize = 2048;

        // asin/acos are NaN outside [-1, 1] and exact at the endpoints; atan
        // saturates to ±π/2 at infinity.
        let edges = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            1.000_001,
            -1.000_001,
            2.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let a: Vec<f32> = (0..LEN).map(|i| edges[i % edges.len()]).collect();
        let mut out = vec![0.0f32; LEN];

        for (op, reference) in [
            (UnaryOp::Asin, f64::asin as fn(f64) -> f64),
            (UnaryOp::Acos, f64::acos as fn(f64) -> f64),
            (UnaryOp::Atan, f64::atan as fn(f64) -> f64),
        ] {
            unsafe { inverse_trig_f32(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
            for i in 0..LEN {
                let expected = reference(a[i] as f64) as f32;
                if expected.is_nan() {
                    assert!(
                        out[i].is_nan(),
                        "{:?}({}) = {}, expected NaN",
                        op,
                        a[i],
                        out[i]
                    );
                    continue;
                }
                assert!(
                    rel_err_f32(out[i], expected) < 1e-6,
                    "{:?}({}) = {}, expected {}",
                    op,
                    a[i],
                    out[i],
                    expected
                );
            }
        }
    }
}
