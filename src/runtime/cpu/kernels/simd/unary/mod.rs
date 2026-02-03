//! SIMD-accelerated unary operations
//!
//! This module provides multi-architecture SIMD implementations for element-wise
//! unary operations.
//!
//! # SIMD Support
//!
//! ALL operations now have SIMD implementations:
//! - Neg, Abs, Sqrt, Square, Recip, Floor, Ceil, Round, Trunc (direct SIMD)
//! - Exp, Log, Sin, Cos, Tan, Atan, Tanh (polynomial approximations from math module)
//! - Sign (comparison-based)
//! - ReLU (critical for ML)
//!
//! # Architecture Support
//!
//! | Architecture | Instruction Set | Vector Width | f32 lanes | f64 lanes |
//! |--------------|-----------------|--------------|-----------|-----------|
//! | x86-64       | AVX-512         | 512 bits     | 16        | 8         |
//! | x86-64       | AVX2 + FMA      | 256 bits     | 8         | 4         |
//! | ARM64        | NEON            | 128 bits     | 4         | 2         |

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

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
            | UnaryOp::Rsqrt
            | UnaryOp::Cbrt
            | UnaryOp::Exp
            | UnaryOp::Exp2
            | UnaryOp::Expm1
            | UnaryOp::Log
            | UnaryOp::Log2
            | UnaryOp::Log10
            | UnaryOp::Log1p
            | UnaryOp::Sin
            | UnaryOp::Cos
            | UnaryOp::Tan
            | UnaryOp::Asin
            | UnaryOp::Acos
            | UnaryOp::Atan
            | UnaryOp::Sinh
            | UnaryOp::Cosh
            | UnaryOp::Tanh
            | UnaryOp::Asinh
            | UnaryOp::Acosh
            | UnaryOp::Atanh
            | UnaryOp::Square
            | UnaryOp::Recip
            | UnaryOp::Floor
            | UnaryOp::Ceil
            | UnaryOp::Round
            | UnaryOp::Trunc
            | UnaryOp::Sign
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

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512::unary_f32(op, a, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2::unary_f32(op, a, out, len),
        _ => unary_scalar_f32(op, a, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::unary_f32(op, a, out, len),
        _ => unary_scalar_f32(op, a, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    unary_scalar_f32(op, a, out, len);
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

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512::unary_f64(op, a, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2::unary_f64(op, a, out, len),
        _ => unary_scalar_f64(op, a, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::unary_f64(op, a, out, len),
        _ => unary_scalar_f64(op, a, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    unary_scalar_f64(op, a, out, len);
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

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512::relu_f32(a, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2::relu_f32(a, out, len),
        _ => relu_scalar_f32(a, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::relu_f32(a, out, len),
        _ => relu_scalar_f32(a, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    relu_scalar_f32(a, out, len);
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

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512::relu_f64(a, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2::relu_f64(a, out, len),
        _ => relu_scalar_f64(a, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::relu_f64(a, out, len),
        _ => relu_scalar_f64(a, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    relu_scalar_f64(a, out, len);
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
    fn test_unary_exp_f32() {
        let a: Vec<f32> = (0..100).map(|x| (x as f32 - 50.0) * 0.1).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { unary_f32(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].exp();
            let diff = (out[i] - expected).abs();
            assert!(
                diff < 1e-5 * expected.abs().max(1.0),
                "exp mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_unary_tanh_f32() {
        let a: Vec<f32> = (0..100).map(|x| (x as f32 - 50.0) * 0.1).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { unary_f32(UnaryOp::Tanh, a.as_ptr(), out.as_mut_ptr(), 100) }

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

    #[test]
    fn test_unary_sign_f32() {
        let a: Vec<f32> = (0..100).map(|x| x as f32 - 50.0).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { unary_f32(UnaryOp::Sign, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = if a[i] > 0.0 {
                1.0
            } else if a[i] < 0.0 {
                -1.0
            } else {
                0.0
            };
            assert_eq!(out[i], expected, "sign mismatch at index {}", i);
        }
    }

    #[test]
    fn test_unary_log_f32() {
        let a: Vec<f32> = (1..101).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { unary_f32(UnaryOp::Log, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].ln();
            let diff = (out[i] - expected).abs();
            // Relative error tolerance of ~1e-4 is acceptable for f32 SIMD approximations
            assert!(
                diff < 5e-5 * expected.abs().max(1.0),
                "log mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_unary_sin_f32() {
        let a: Vec<f32> = (0..100).map(|x| (x as f32 - 50.0) * 0.1).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { unary_f32(UnaryOp::Sin, a.as_ptr(), out.as_mut_ptr(), 100) }

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

        unsafe { unary_f32(UnaryOp::Cos, a.as_ptr(), out.as_mut_ptr(), 100) }

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

        unsafe { unary_f32(UnaryOp::Tan, a.as_ptr(), out.as_mut_ptr(), 100) }

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
