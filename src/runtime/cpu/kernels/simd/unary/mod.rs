//! SIMD-accelerated unary operations
//!
//! This module provides AVX2 and AVX-512 implementations for element-wise
//! unary operations.
//!
//! # SIMD Support
//!
//! Operations with direct SIMD instructions (fast path):
//! - Neg, Abs, Sqrt, Square, Recip, Floor, Ceil, Round
//! - ReLU (critical for ML)
//!
//! Operations requiring scalar fallback:
//! - Exp, Log, Sin, Cos, Tan, Tanh, Sign (complex math)

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

use super::{SimdLevel, detect_simd};
use crate::ops::UnaryOp;

// Import scalar fallbacks from kernels module (single source of truth)
pub use crate::runtime::cpu::kernels::unary::{
    relu_scalar_f32, relu_scalar_f64, unary_scalar_f32, unary_scalar_f64,
};

/// Minimum elements to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

/// Check if operation has SIMD support
#[inline]
const fn is_simd_supported(op: UnaryOp) -> bool {
    matches!(
        op,
        UnaryOp::Neg
            | UnaryOp::Abs
            | UnaryOp::Sqrt
            | UnaryOp::Square
            | UnaryOp::Recip
            | UnaryOp::Floor
            | UnaryOp::Ceil
            | UnaryOp::Round
    )
}

/// SIMD unary operation for f32
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn unary_f32(op: UnaryOp, a: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar || !is_simd_supported(op) {
        unary_scalar_f32(op, a, out, len);
        return;
    }

    match level {
        SimdLevel::Avx512 => avx512::unary_f32(op, a, out, len),
        SimdLevel::Avx2Fma => avx2::unary_f32(op, a, out, len),
        SimdLevel::Scalar => unreachable!(),
    }
}

/// SIMD unary operation for f64
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn unary_f64(op: UnaryOp, a: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar || !is_simd_supported(op) {
        unary_scalar_f64(op, a, out, len);
        return;
    }

    match level {
        SimdLevel::Avx512 => avx512::unary_f64(op, a, out, len),
        SimdLevel::Avx2Fma => avx2::unary_f64(op, a, out, len),
        SimdLevel::Scalar => unreachable!(),
    }
}

/// SIMD ReLU for f32
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn relu_f32(a: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        relu_scalar_f32(a, out, len);
        return;
    }

    match level {
        SimdLevel::Avx512 => avx512::relu_f32(a, out, len),
        SimdLevel::Avx2Fma => avx2::relu_f32(a, out, len),
        SimdLevel::Scalar => unreachable!(),
    }
}

/// SIMD ReLU for f64
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn relu_f64(a: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        relu_scalar_f64(a, out, len);
        return;
    }

    match level {
        SimdLevel::Avx512 => avx512::relu_f64(a, out, len),
        SimdLevel::Avx2Fma => avx2::relu_f64(a, out, len),
        SimdLevel::Scalar => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unary_neg_f32() {
        let a: Vec<f32> = (0..100).map(|x| x as f32 - 50.0).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { unary_f32(UnaryOp::Neg, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert_eq!(out[i], -a[i], "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_unary_abs_f32() {
        let a: Vec<f32> = (0..100).map(|x| x as f32 - 50.0).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { unary_f32(UnaryOp::Abs, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert_eq!(out[i], a[i].abs(), "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_relu_f32() {
        let a: Vec<f32> = (0..100).map(|x| x as f32 - 50.0).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { relu_f32(a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = if a[i] > 0.0 { a[i] } else { 0.0 };
            assert_eq!(out[i], expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_relu_f64() {
        let a: Vec<f64> = (0..100).map(|x| x as f64 - 50.0).collect();
        let mut out = vec![0.0f64; 100];

        unsafe { relu_f64(a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = if a[i] > 0.0 { a[i] } else { 0.0 };
            assert_eq!(out[i], expected, "mismatch at index {}", i);
        }
    }
}
