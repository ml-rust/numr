//! SIMD unary op dispatch: threshold/support checks, family routing, ReLU,
//! and f16/bf16 wrappers.
//!
//! `unary_f32`/`unary_f64` route each [`UnaryOp`] to the family module that
//! owns its SIMD dispatch (`basic_ops`, `exp_f32`/`exp_f64`, `log_f32`/
//! `log_f64`, `trig_f32`/`trig_f64`, `inverse_trig`, `hyperbolic_f32`/
//! `hyperbolic_f64`). Each family function is a full copy of the
//! threshold-check/ISA-selection logic below, so it behaves identically
//! whether reached through this router or called directly (as the family's
//! own tests do).

use super::super::{SimdLevel, detect_simd};
use super::{relu_scalar_f32, relu_scalar_f64};
use crate::ops::UnaryOp;

/// Minimum elements to justify SIMD overhead
pub(super) const SIMD_THRESHOLD: usize = 32;

/// Check if operation has SIMD support
#[inline]
pub(super) const fn is_simd_supported(op: UnaryOp) -> bool {
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
            | UnaryOp::RoundTiesEven
            | UnaryOp::Trunc
            | UnaryOp::Sign
    )
}

/// SIMD unary operation for f32: routes to the family module that owns `op`.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn unary_f32(op: UnaryOp, a: *const f32, out: *mut f32, len: usize) {
    unsafe {
        match op {
            UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Tan => {
                super::trig_f32::trig_f32(op, a, out, len)
            }
            UnaryOp::Asin | UnaryOp::Acos | UnaryOp::Atan => {
                super::inverse_trig::inverse_trig_f32(op, a, out, len)
            }
            UnaryOp::Exp | UnaryOp::Exp2 | UnaryOp::Expm1 => {
                super::exp_f32::exp_f32(op, a, out, len)
            }
            UnaryOp::Log | UnaryOp::Log2 | UnaryOp::Log10 | UnaryOp::Log1p => {
                super::log_f32::log_f32(op, a, out, len)
            }
            UnaryOp::Sinh
            | UnaryOp::Cosh
            | UnaryOp::Tanh
            | UnaryOp::Asinh
            | UnaryOp::Acosh
            | UnaryOp::Atanh => super::hyperbolic_f32::hyperbolic_f32(op, a, out, len),
            _ => super::basic_ops::basic_f32(op, a, out, len),
        }
    }
}

/// SIMD unary operation for f64: routes to the family module that owns `op`.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn unary_f64(op: UnaryOp, a: *const f64, out: *mut f64, len: usize) {
    unsafe {
        match op {
            UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Tan => {
                super::trig_f64::trig_f64(op, a, out, len)
            }
            UnaryOp::Asin | UnaryOp::Acos | UnaryOp::Atan => {
                super::inverse_trig::inverse_trig_f64(op, a, out, len)
            }
            UnaryOp::Exp | UnaryOp::Exp2 | UnaryOp::Expm1 => {
                super::exp_f64::exp_f64(op, a, out, len)
            }
            UnaryOp::Log | UnaryOp::Log2 | UnaryOp::Log10 | UnaryOp::Log1p => {
                super::log_f64::log_f64(op, a, out, len)
            }
            UnaryOp::Sinh
            | UnaryOp::Cosh
            | UnaryOp::Tanh
            | UnaryOp::Asinh
            | UnaryOp::Acosh
            | UnaryOp::Atanh => super::hyperbolic_f64::hyperbolic_f64(op, a, out, len),
            _ => super::basic_ops::basic_f64(op, a, out, len),
        }
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

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => super::x86_64::avx512::relu_f32(a, out, len),
        SimdLevel::Avx2Fma => super::x86_64::avx2::relu_f32(a, out, len),
        _ => relu_scalar_f32(a, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => super::aarch64::neon::relu_f32(a, out, len),
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
        SimdLevel::Avx512 => super::x86_64::avx512::relu_f64(a, out, len),
        SimdLevel::Avx2Fma => super::x86_64::avx2::relu_f64(a, out, len),
        _ => relu_scalar_f64(a, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => super::aarch64::neon::relu_f64(a, out, len),
        _ => relu_scalar_f64(a, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    relu_scalar_f64(a, out, len);
}

// ---------------------------------------------------------------------------
// f16/bf16 via f32 block-convert-compute
// ---------------------------------------------------------------------------

half_unary_op!(unary, unary_f32, UnaryOp);
half_unary!(relu, relu_f32);

#[cfg(test)]
mod tests {
    use super::*;

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
