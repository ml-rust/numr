//! Entry points and SIMD-level dispatch for comparison operations

use super::scalar::{compare_scalar_f32, compare_scalar_f64};
use crate::ops::CompareOp;
use crate::runtime::cpu::kernels::simd::{SimdLevel, detect_simd};

/// Minimum length to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

/// SIMD comparison for f32
///
/// # Safety
/// - `a`, `b`, and `out` must point to `len` elements
#[inline]
pub unsafe fn compare_f32(op: CompareOp, a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        compare_scalar_f32(op, a, b, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => super::avx512::compare_f32(op, a, b, out, len),
        SimdLevel::Avx2Fma => super::avx2::compare_f32(op, a, b, out, len),
        _ => compare_scalar_f32(op, a, b, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            super::aarch64::neon::compare_f32(op, a, b, out, len)
        }
        _ => compare_scalar_f32(op, a, b, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    compare_scalar_f32(op, a, b, out, len);
}

/// SIMD comparison for f64
///
/// # Safety
/// - `a`, `b`, and `out` must point to `len` elements
#[inline]
pub unsafe fn compare_f64(op: CompareOp, a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        compare_scalar_f64(op, a, b, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => super::avx512::compare_f64(op, a, b, out, len),
        SimdLevel::Avx2Fma => super::avx2::compare_f64(op, a, b, out, len),
        _ => compare_scalar_f64(op, a, b, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            super::aarch64::neon::compare_f64(op, a, b, out, len)
        }
        _ => compare_scalar_f64(op, a, b, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    compare_scalar_f64(op, a, b, out, len);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_eq_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0f32, 3.0, 3.0, 5.0, 5.0, 6.0, 8.0, 8.0];
        let mut out = [0.0f32; 8];
        let mut out_ref = [0.0f32; 8];

        unsafe {
            compare_f32(CompareOp::Eq, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 8);
            compare_scalar_f32(
                CompareOp::Eq,
                a.as_ptr(),
                b.as_ptr(),
                out_ref.as_mut_ptr(),
                8,
            );
        }

        assert_eq!(out, out_ref);
        assert_eq!(out, [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_compare_lt_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0f32, 2.0, 2.0, 5.0, 4.0, 7.0, 7.0, 9.0];
        let mut out = [0.0f32; 8];
        let mut out_ref = [0.0f32; 8];

        unsafe {
            compare_f32(CompareOp::Lt, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 8);
            compare_scalar_f32(
                CompareOp::Lt,
                a.as_ptr(),
                b.as_ptr(),
                out_ref.as_mut_ptr(),
                8,
            );
        }

        assert_eq!(out, out_ref);
        assert_eq!(out, [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_compare_large_f32() {
        let len = 256;
        let a: Vec<f32> = (0..len).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..len).map(|x| (x as f32) - 0.5).collect();
        let mut out = vec![0.0f32; len];
        let mut out_ref = vec![0.0f32; len];

        unsafe {
            compare_f32(CompareOp::Gt, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), len);
            compare_scalar_f32(
                CompareOp::Gt,
                a.as_ptr(),
                b.as_ptr(),
                out_ref.as_mut_ptr(),
                len,
            );
        }

        assert_eq!(out, out_ref);
    }
}
