//! SIMD-accelerated fused activation-multiplication operations
//!
//! Provides vectorized implementations of fused activation * multiplication:
//! - silu_mul: (x / (1 + exp(-x))) * y
//! - gelu_mul: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) * y
//! - relu_mul: max(0, x) * y
//! - sigmoid_mul: (1 / (1 + exp(-x))) * y
//!
//! These operations take TWO inputs (a, b) and compute `activation(a) * b` in one pass,
//! reducing memory bandwidth compared to separate operations.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

use super::{SimdLevel, detect_simd};

/// Minimum length to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

/// SIMD silu_mul for f32
///
/// Computes: (a / (1 + exp(-a))) * b
///
/// # Safety
/// - `a`, `b`, and `out` must point to `len` elements
/// - Elements must not overlap
#[inline]
pub unsafe fn silu_mul_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        silu_mul_scalar_f32(a, b, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::silu_mul_f32(a, b, out, len),
        SimdLevel::Avx2Fma => avx2::silu_mul_f32(a, b, out, len),
        _ => silu_mul_scalar_f32(a, b, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::silu_mul_f32(a, b, out, len),
        _ => silu_mul_scalar_f32(a, b, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    silu_mul_scalar_f32(a, b, out, len);
}

/// SIMD silu_mul for f64
///
/// Computes: (a / (1 + exp(-a))) * b
///
/// # Safety
/// - `a`, `b`, and `out` must point to `len` elements
/// - Elements must not overlap
#[inline]
pub unsafe fn silu_mul_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        silu_mul_scalar_f64(a, b, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::silu_mul_f64(a, b, out, len),
        SimdLevel::Avx2Fma => avx2::silu_mul_f64(a, b, out, len),
        _ => silu_mul_scalar_f64(a, b, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::silu_mul_f64(a, b, out, len),
        _ => silu_mul_scalar_f64(a, b, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    silu_mul_scalar_f64(a, b, out, len);
}

/// SIMD gelu_mul for f32
///
/// Computes: 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3))) * b
///
/// # Safety
/// - `a`, `b`, and `out` must point to `len` elements
/// - Elements must not overlap
#[inline]
pub unsafe fn gelu_mul_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        gelu_mul_scalar_f32(a, b, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::gelu_mul_f32(a, b, out, len),
        SimdLevel::Avx2Fma => avx2::gelu_mul_f32(a, b, out, len),
        _ => gelu_mul_scalar_f32(a, b, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::gelu_mul_f32(a, b, out, len),
        _ => gelu_mul_scalar_f32(a, b, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    gelu_mul_scalar_f32(a, b, out, len);
}

/// SIMD gelu_mul for f64
///
/// Computes: 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3))) * b
///
/// # Safety
/// - `a`, `b`, and `out` must point to `len` elements
/// - Elements must not overlap
#[inline]
pub unsafe fn gelu_mul_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        gelu_mul_scalar_f64(a, b, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::gelu_mul_f64(a, b, out, len),
        SimdLevel::Avx2Fma => avx2::gelu_mul_f64(a, b, out, len),
        _ => gelu_mul_scalar_f64(a, b, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::gelu_mul_f64(a, b, out, len),
        _ => gelu_mul_scalar_f64(a, b, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    gelu_mul_scalar_f64(a, b, out, len);
}

/// SIMD relu_mul for f32
///
/// Computes: max(0, a) * b
///
/// # Safety
/// - `a`, `b`, and `out` must point to `len` elements
/// - Elements must not overlap
#[inline]
pub unsafe fn relu_mul_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        relu_mul_scalar_f32(a, b, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::relu_mul_f32(a, b, out, len),
        SimdLevel::Avx2Fma => avx2::relu_mul_f32(a, b, out, len),
        _ => relu_mul_scalar_f32(a, b, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::relu_mul_f32(a, b, out, len),
        _ => relu_mul_scalar_f32(a, b, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    relu_mul_scalar_f32(a, b, out, len);
}

/// SIMD relu_mul for f64
///
/// Computes: max(0, a) * b
///
/// # Safety
/// - `a`, `b`, and `out` must point to `len` elements
/// - Elements must not overlap
#[inline]
pub unsafe fn relu_mul_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        relu_mul_scalar_f64(a, b, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::relu_mul_f64(a, b, out, len),
        SimdLevel::Avx2Fma => avx2::relu_mul_f64(a, b, out, len),
        _ => relu_mul_scalar_f64(a, b, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::relu_mul_f64(a, b, out, len),
        _ => relu_mul_scalar_f64(a, b, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    relu_mul_scalar_f64(a, b, out, len);
}

/// SIMD sigmoid_mul for f32
///
/// Computes: (1 / (1 + exp(-a))) * b
///
/// # Safety
/// - `a`, `b`, and `out` must point to `len` elements
/// - Elements must not overlap
#[inline]
pub unsafe fn sigmoid_mul_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        sigmoid_mul_scalar_f32(a, b, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::sigmoid_mul_f32(a, b, out, len),
        SimdLevel::Avx2Fma => avx2::sigmoid_mul_f32(a, b, out, len),
        _ => sigmoid_mul_scalar_f32(a, b, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::sigmoid_mul_f32(a, b, out, len),
        _ => sigmoid_mul_scalar_f32(a, b, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    sigmoid_mul_scalar_f32(a, b, out, len);
}

/// SIMD sigmoid_mul for f64
///
/// Computes: (1 / (1 + exp(-a))) * b
///
/// # Safety
/// - `a`, `b`, and `out` must point to `len` elements
/// - Elements must not overlap
#[inline]
pub unsafe fn sigmoid_mul_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        sigmoid_mul_scalar_f64(a, b, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::sigmoid_mul_f64(a, b, out, len),
        SimdLevel::Avx2Fma => avx2::sigmoid_mul_f64(a, b, out, len),
        _ => sigmoid_mul_scalar_f64(a, b, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::sigmoid_mul_f64(a, b, out, len),
        _ => sigmoid_mul_scalar_f64(a, b, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    sigmoid_mul_scalar_f64(a, b, out, len);
}

// ============================================================================
// Scalar fallbacks
// ============================================================================

/// Scalar silu_mul for f32
#[inline]
pub unsafe fn silu_mul_scalar_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    for i in 0..len {
        let x = *a.add(i);
        let y = *b.add(i);
        *out.add(i) = (x / (1.0 + (-x).exp())) * y;
    }
}

/// Scalar silu_mul for f64
#[inline]
pub unsafe fn silu_mul_scalar_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    for i in 0..len {
        let x = *a.add(i);
        let y = *b.add(i);
        *out.add(i) = (x / (1.0 + (-x).exp())) * y;
    }
}

/// Scalar gelu_mul for f32
#[inline]
pub unsafe fn gelu_mul_scalar_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    const TANH_COEF: f32 = 0.044715;

    for i in 0..len {
        let x = *a.add(i);
        let y = *b.add(i);
        let inner = SQRT_2_OVER_PI * (x + TANH_COEF * x * x * x);
        *out.add(i) = 0.5 * x * (1.0 + inner.tanh()) * y;
    }
}

/// Scalar gelu_mul for f64
#[inline]
pub unsafe fn gelu_mul_scalar_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
    const TANH_COEF: f64 = 0.044715;

    for i in 0..len {
        let x = *a.add(i);
        let y = *b.add(i);
        let inner = SQRT_2_OVER_PI * (x + TANH_COEF * x * x * x);
        *out.add(i) = 0.5 * x * (1.0 + inner.tanh()) * y;
    }
}

/// Scalar relu_mul for f32
#[inline]
pub unsafe fn relu_mul_scalar_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    for i in 0..len {
        let x = *a.add(i);
        let y = *b.add(i);
        *out.add(i) = if x > 0.0 { x * y } else { 0.0 };
    }
}

/// Scalar relu_mul for f64
#[inline]
pub unsafe fn relu_mul_scalar_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    for i in 0..len {
        let x = *a.add(i);
        let y = *b.add(i);
        *out.add(i) = if x > 0.0 { x * y } else { 0.0 };
    }
}

/// Scalar sigmoid_mul for f32
#[inline]
pub unsafe fn sigmoid_mul_scalar_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    for i in 0..len {
        let x = *a.add(i);
        let y = *b.add(i);
        *out.add(i) = (1.0 / (1.0 + (-x).exp())) * y;
    }
}

/// Scalar sigmoid_mul for f64
#[inline]
pub unsafe fn sigmoid_mul_scalar_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    for i in 0..len {
        let x = *a.add(i);
        let y = *b.add(i);
        *out.add(i) = (1.0 / (1.0 + (-x).exp())) * y;
    }
}

// ============================================================================
// f16/bf16 block-convert-compute wrappers
// ============================================================================

/// Generate f16/bf16 wrappers for binary fused ops: `fn(a, b, out, len)`
macro_rules! _half_binary_fused {
    ($fn_name:ident, $half_ty:ty, $to_f32:path, $from_f32:path, $f32_fn:path) => {
        #[cfg(feature = "f16")]
        #[inline]
        pub unsafe fn $fn_name(
            a: *const $half_ty,
            b: *const $half_ty,
            out: *mut $half_ty,
            len: usize,
        ) {
            use super::half_convert_utils::HALF_BLOCK;
            let mut a_buf = [0.0f32; HALF_BLOCK];
            let mut b_buf = [0.0f32; HALF_BLOCK];
            let mut out_buf = [0.0f32; HALF_BLOCK];
            let mut offset = 0;
            while offset < len {
                let chunk = (len - offset).min(HALF_BLOCK);
                $to_f32(a.add(offset) as *const u16, a_buf.as_mut_ptr(), chunk);
                $to_f32(b.add(offset) as *const u16, b_buf.as_mut_ptr(), chunk);
                $f32_fn(a_buf.as_ptr(), b_buf.as_ptr(), out_buf.as_mut_ptr(), chunk);
                $from_f32(out_buf.as_ptr(), out.add(offset) as *mut u16, chunk);
                offset += chunk;
            }
        }
    };
}

macro_rules! half_binary_fused {
    ($name:ident, $f32_fn:path) => {
        paste::paste! {
            _half_binary_fused!([<$name _f16>], half::f16,
                super::half_convert_utils::convert_f16_to_f32,
                super::half_convert_utils::convert_f32_to_f16, $f32_fn);
            _half_binary_fused!([<$name _bf16>], half::bf16,
                super::half_convert_utils::convert_bf16_to_f32,
                super::half_convert_utils::convert_f32_to_bf16, $f32_fn);
        }
    };
}

half_binary_fused!(silu_mul, silu_mul_f32);
half_binary_fused!(gelu_mul, gelu_mul_f32);
half_binary_fused!(relu_mul, relu_mul_f32);
half_binary_fused!(sigmoid_mul, sigmoid_mul_f32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_mul_f32() {
        let len = 128;
        let a: Vec<f32> = (0..len).map(|x| (x as f32) / 32.0 - 2.0).collect();
        let b: Vec<f32> = (0..len).map(|x| (x as f32) / 64.0 + 0.5).collect();
        let mut out = vec![0.0f32; len];
        let mut out_ref = vec![0.0f32; len];

        unsafe {
            silu_mul_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), len);
            silu_mul_scalar_f32(a.as_ptr(), b.as_ptr(), out_ref.as_mut_ptr(), len);
        }

        for i in 0..len {
            let diff = (out[i] - out_ref[i]).abs();
            let denom = out_ref[i].abs().max(1e-6);
            let rel_err = diff / denom;
            assert!(
                rel_err < 0.01,
                "silu_mul mismatch at {}: {} vs {} (err: {})",
                i,
                out[i],
                out_ref[i],
                rel_err
            );
        }
    }

    #[test]
    fn test_gelu_mul_f32() {
        let len = 128;
        let a: Vec<f32> = (0..len).map(|x| (x as f32) / 32.0 - 2.0).collect();
        let b: Vec<f32> = (0..len).map(|x| (x as f32) / 64.0 + 0.5).collect();
        let mut out = vec![0.0f32; len];
        let mut out_ref = vec![0.0f32; len];

        unsafe {
            gelu_mul_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), len);
            gelu_mul_scalar_f32(a.as_ptr(), b.as_ptr(), out_ref.as_mut_ptr(), len);
        }

        for i in 0..len {
            let diff = (out[i] - out_ref[i]).abs();
            let denom = out_ref[i].abs().max(1e-6);
            let rel_err = diff / denom;
            assert!(
                rel_err < 0.02,
                "gelu_mul mismatch at {}: {} vs {} (err: {})",
                i,
                out[i],
                out_ref[i],
                rel_err
            );
        }
    }

    #[test]
    fn test_relu_mul_f32() {
        let len = 128;
        let a: Vec<f32> = (0..len).map(|x| (x as f32) - 64.0).collect();
        let b: Vec<f32> = (0..len).map(|x| (x as f32) / 64.0 + 0.5).collect();
        let mut out = vec![0.0f32; len];
        let mut out_ref = vec![0.0f32; len];

        unsafe {
            relu_mul_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), len);
            relu_mul_scalar_f32(a.as_ptr(), b.as_ptr(), out_ref.as_mut_ptr(), len);
        }

        assert_eq!(out, out_ref);
    }

    #[test]
    fn test_sigmoid_mul_f32() {
        let len = 128;
        let a: Vec<f32> = (0..len).map(|x| (x as f32) / 32.0 - 2.0).collect();
        let b: Vec<f32> = (0..len).map(|x| (x as f32) / 64.0 + 0.5).collect();
        let mut out = vec![0.0f32; len];
        let mut out_ref = vec![0.0f32; len];

        unsafe {
            sigmoid_mul_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), len);
            sigmoid_mul_scalar_f32(a.as_ptr(), b.as_ptr(), out_ref.as_mut_ptr(), len);
        }

        for i in 0..len {
            let diff = (out[i] - out_ref[i]).abs();
            let denom = out_ref[i].abs().max(1e-6);
            let rel_err = diff / denom;
            assert!(
                rel_err < 0.01,
                "sigmoid_mul mismatch at {}: {} vs {} (err: {})",
                i,
                out[i],
                out_ref[i],
                rel_err
            );
        }
    }
}
