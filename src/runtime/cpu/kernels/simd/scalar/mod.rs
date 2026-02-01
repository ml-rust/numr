//! SIMD-accelerated scalar operations
//!
//! This module provides AVX2 and AVX-512 implementations for tensor-scalar operations.
//!
//! # SIMD Support
//!
//! Operations with SIMD fast paths:
//! - Add, Sub, Mul, Div, Max, Min (with scalar)
//!
//! Operations using scalar fallback:
//! - Pow (requires libm, no direct SIMD instruction)

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

use super::{SimdLevel, detect_simd};
use crate::ops::BinaryOp;

/// Minimum elements to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

/// Check if operation has SIMD support
#[inline]
const fn is_simd_supported(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::Div
            | BinaryOp::Max
            | BinaryOp::Min
    )
}

/// SIMD scalar operation for f32
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn scalar_f32(op: BinaryOp, a: *const f32, scalar: f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar || !is_simd_supported(op) {
        scalar_scalar_f32(op, a, scalar, out, len);
        return;
    }

    match level {
        SimdLevel::Avx512 => avx512::scalar_f32(op, a, scalar, out, len),
        SimdLevel::Avx2Fma => avx2::scalar_f32(op, a, scalar, out, len),
        SimdLevel::Scalar => unreachable!(),
    }
}

/// SIMD scalar operation for f64
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn scalar_f64(op: BinaryOp, a: *const f64, scalar: f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar || !is_simd_supported(op) {
        scalar_scalar_f64(op, a, scalar, out, len);
        return;
    }

    match level {
        SimdLevel::Avx512 => avx512::scalar_f64(op, a, scalar, out, len),
        SimdLevel::Avx2Fma => avx2::scalar_f64(op, a, scalar, out, len),
        SimdLevel::Scalar => unreachable!(),
    }
}

/// Scalar fallback for f32
#[inline]
pub unsafe fn scalar_scalar_f32(
    op: BinaryOp,
    a: *const f32,
    scalar: f32,
    out: *mut f32,
    len: usize,
) {
    match op {
        BinaryOp::Add => {
            for i in 0..len {
                *out.add(i) = *a.add(i) + scalar;
            }
        }
        BinaryOp::Sub => {
            for i in 0..len {
                *out.add(i) = *a.add(i) - scalar;
            }
        }
        BinaryOp::Mul => {
            for i in 0..len {
                *out.add(i) = *a.add(i) * scalar;
            }
        }
        BinaryOp::Div => {
            for i in 0..len {
                *out.add(i) = *a.add(i) / scalar;
            }
        }
        BinaryOp::Max => {
            for i in 0..len {
                let v = *a.add(i);
                *out.add(i) = if v > scalar { v } else { scalar };
            }
        }
        BinaryOp::Min => {
            for i in 0..len {
                let v = *a.add(i);
                *out.add(i) = if v < scalar { v } else { scalar };
            }
        }
        BinaryOp::Pow => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).powf(scalar);
            }
        }
        BinaryOp::Atan2 => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).atan2(scalar);
            }
        }
    }
}

/// Scalar fallback for f64
#[inline]
pub unsafe fn scalar_scalar_f64(
    op: BinaryOp,
    a: *const f64,
    scalar: f64,
    out: *mut f64,
    len: usize,
) {
    match op {
        BinaryOp::Add => {
            for i in 0..len {
                *out.add(i) = *a.add(i) + scalar;
            }
        }
        BinaryOp::Sub => {
            for i in 0..len {
                *out.add(i) = *a.add(i) - scalar;
            }
        }
        BinaryOp::Mul => {
            for i in 0..len {
                *out.add(i) = *a.add(i) * scalar;
            }
        }
        BinaryOp::Div => {
            for i in 0..len {
                *out.add(i) = *a.add(i) / scalar;
            }
        }
        BinaryOp::Max => {
            for i in 0..len {
                let v = *a.add(i);
                *out.add(i) = if v > scalar { v } else { scalar };
            }
        }
        BinaryOp::Min => {
            for i in 0..len {
                let v = *a.add(i);
                *out.add(i) = if v < scalar { v } else { scalar };
            }
        }
        BinaryOp::Pow => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).powf(scalar);
            }
        }
        BinaryOp::Atan2 => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).atan2(scalar);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_add_f32() {
        let a: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { scalar_f32(BinaryOp::Add, a.as_ptr(), 10.0, out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert_eq!(out[i], a[i] + 10.0, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_scalar_mul_f32() {
        let a: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { scalar_f32(BinaryOp::Mul, a.as_ptr(), 2.5, out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert!(
                (out[i] - a[i] * 2.5).abs() < 1e-6,
                "mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_scalar_max_f32() {
        let a: Vec<f32> = (0..100).map(|x| (x as f32) - 50.0).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { scalar_f32(BinaryOp::Max, a.as_ptr(), 0.0, out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = if a[i] > 0.0 { a[i] } else { 0.0 };
            assert_eq!(out[i], expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_scalar_div_f64() {
        let a: Vec<f64> = (1..101).map(|x| x as f64).collect();
        let mut out = vec![0.0f64; 100];

        unsafe { scalar_f64(BinaryOp::Div, a.as_ptr(), 2.0, out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert!(
                (out[i] - a[i] / 2.0).abs() < 1e-10,
                "mismatch at index {}",
                i
            );
        }
    }
}
