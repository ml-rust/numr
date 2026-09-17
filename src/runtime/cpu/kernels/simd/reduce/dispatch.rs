//! SIMD dispatch entry points for reductions.

use super::super::{SimdLevel, detect_simd};
use super::scalar::{reduce_scalar_f32, reduce_scalar_f64};
use crate::ops::ReduceOp;

#[cfg(target_arch = "x86_64")]
use super::{avx2, avx512};

#[cfg(target_arch = "aarch64")]
use super::aarch64;

/// Minimum elements to justify SIMD overhead for reductions
const SIMD_THRESHOLD: usize = 64;

/// Check if operation has SIMD support
#[inline]
const fn is_simd_supported(op: ReduceOp) -> bool {
    matches!(
        op,
        ReduceOp::Sum | ReduceOp::Max | ReduceOp::Min | ReduceOp::Prod
    )
}

/// SIMD reduction for f32
///
/// # Safety
/// - `a` must point to `reduce_size * outer_size` elements
/// - `out` must point to `outer_size` elements
#[inline]
pub unsafe fn reduce_f32(
    op: ReduceOp,
    a: *const f32,
    out: *mut f32,
    reduce_size: usize,
    outer_size: usize,
) {
    let level = detect_simd();

    if reduce_size < SIMD_THRESHOLD || level == SimdLevel::Scalar || !is_simd_supported(op) {
        reduce_scalar_f32(op, a, out, reduce_size, outer_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::reduce_f32(op, a, out, reduce_size, outer_size),
        SimdLevel::Avx2Fma => avx2::reduce_f32(op, a, out, reduce_size, outer_size),
        _ => reduce_scalar_f32(op, a, out, reduce_size, outer_size),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::reduce_f32(op, a, out, reduce_size, outer_size)
        }
        _ => reduce_scalar_f32(op, a, out, reduce_size, outer_size),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    reduce_scalar_f32(op, a, out, reduce_size, outer_size);
}

/// SIMD reduction for f64
///
/// # Safety
/// - `a` must point to `reduce_size * outer_size` elements
/// - `out` must point to `outer_size` elements
#[inline]
pub unsafe fn reduce_f64(
    op: ReduceOp,
    a: *const f64,
    out: *mut f64,
    reduce_size: usize,
    outer_size: usize,
) {
    let level = detect_simd();

    if reduce_size < SIMD_THRESHOLD || level == SimdLevel::Scalar || !is_simd_supported(op) {
        reduce_scalar_f64(op, a, out, reduce_size, outer_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::reduce_f64(op, a, out, reduce_size, outer_size),
        SimdLevel::Avx2Fma => avx2::reduce_f64(op, a, out, reduce_size, outer_size),
        _ => reduce_scalar_f64(op, a, out, reduce_size, outer_size),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::reduce_f64(op, a, out, reduce_size, outer_size)
        }
        _ => reduce_scalar_f64(op, a, out, reduce_size, outer_size),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    reduce_scalar_f64(op, a, out, reduce_size, outer_size);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_sum_f32() {
        let a: Vec<f32> = (0..1000).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 1];

        unsafe { reduce_f32(ReduceOp::Sum, a.as_ptr(), out.as_mut_ptr(), 1000, 1) }

        // Sum of 0..999 = 999 * 1000 / 2 = 499500
        assert!((out[0] - 499500.0).abs() < 1e-3, "got {}", out[0]);
    }

    #[test]
    fn test_reduce_max_f32() {
        let a: Vec<f32> = (0..1000).map(|x| (x as f32) - 500.0).collect();
        let mut out = vec![0.0f32; 1];

        unsafe { reduce_f32(ReduceOp::Max, a.as_ptr(), out.as_mut_ptr(), 1000, 1) }

        assert_eq!(out[0], 499.0);
    }

    #[test]
    fn test_reduce_min_f32() {
        let a: Vec<f32> = (0..1000).map(|x| (x as f32) - 500.0).collect();
        let mut out = vec![0.0f32; 1];

        unsafe { reduce_f32(ReduceOp::Min, a.as_ptr(), out.as_mut_ptr(), 1000, 1) }

        assert_eq!(out[0], -500.0);
    }

    #[test]
    fn test_reduce_prod_f32() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0f32; 1];

        // Use scalar path for small arrays
        unsafe { reduce_scalar_f32(ReduceOp::Prod, a.as_ptr(), out.as_mut_ptr(), 5, 1) }

        assert_eq!(out[0], 120.0); // 5! = 120
    }

    #[test]
    fn test_reduce_multiple_outer() {
        // Two reductions of 500 elements each
        let a: Vec<f32> = (0..1000).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 2];

        unsafe { reduce_f32(ReduceOp::Sum, a.as_ptr(), out.as_mut_ptr(), 500, 2) }

        // First: 0..499 = 499*500/2 = 124750
        // Second: 500..999 = sum(500..999) = 374750
        assert!((out[0] - 124750.0).abs() < 1e-3, "first got {}", out[0]);
        assert!((out[1] - 374750.0).abs() < 1e-3, "second got {}", out[1]);
    }
}
