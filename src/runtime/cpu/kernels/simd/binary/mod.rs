//! SIMD-accelerated binary operations
//!
//! This module provides AVX2 and AVX-512 implementations for element-wise
//! binary operations (add, sub, mul, div, max, min).

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

use super::{SimdLevel, detect_simd};
use crate::ops::BinaryOp;

// Import scalar fallbacks from kernels module (single source of truth)
pub use crate::runtime::cpu::kernels::binary::{binary_scalar_f32, binary_scalar_f64};

/// Minimum elements to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

/// SIMD binary operation for f32
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn binary_f32(op: BinaryOp, a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        binary_scalar_f32(op, a, b, out, len);
        return;
    }

    match level {
        SimdLevel::Avx512 => avx512::binary_f32(op, a, b, out, len),
        SimdLevel::Avx2Fma => avx2::binary_f32(op, a, b, out, len),
        SimdLevel::Scalar => unreachable!(),
    }
}

/// SIMD binary operation for f64
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn binary_f64(op: BinaryOp, a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        binary_scalar_f64(op, a, b, out, len);
        return;
    }

    match level {
        SimdLevel::Avx512 => avx512::binary_f64(op, a, b, out, len),
        SimdLevel::Avx2Fma => avx2::binary_f64(op, a, b, out, len),
        SimdLevel::Scalar => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_add_f32() {
        let a: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..100).map(|x| (x * 2) as f32).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { binary_f32(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert_eq!(out[i], a[i] + b[i], "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_binary_mul_f64() {
        let a: Vec<f64> = (1..101).map(|x| x as f64).collect();
        let b: Vec<f64> = (1..101).map(|x| (x * 2) as f64).collect();
        let mut out = vec![0.0f64; 100];

        unsafe { binary_f64(BinaryOp::Mul, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert_eq!(out[i], a[i] * b[i], "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_small_array_uses_scalar() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];

        unsafe { binary_f32(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4) }

        assert_eq!(out, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_non_aligned_length() {
        let a: Vec<f32> = (0..67).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..67).map(|x| (x * 2) as f32).collect();
        let mut out = vec![0.0f32; 67];

        unsafe { binary_f32(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 67) }

        for i in 0..67 {
            assert_eq!(out[i], a[i] + b[i], "mismatch at index {}", i);
        }
    }
}
