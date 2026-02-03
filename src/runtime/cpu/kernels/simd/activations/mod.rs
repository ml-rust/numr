//! SIMD-accelerated activation functions
//!
//! Provides vectorized implementations of common neural network activations:
//! - Sigmoid: 1 / (1 + exp(-x))
//! - SiLU (Swish): x * sigmoid(x)
//! - GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//! - Leaky ReLU: max(negative_slope * x, x)
//! - ELU: x if x > 0, else alpha * (exp(x) - 1)
//!
//! # SIMD Approach
//!
//! Uses polynomial approximations for exp and tanh:
//! - exp(x): Range reduction + Taylor series
//! - tanh(x): Based on exp via tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

use super::{SimdLevel, detect_simd};

/// Minimum length to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

/// SIMD sigmoid for f32
///
/// # Safety
/// - `a` and `out` must point to `len` elements
#[inline]
pub unsafe fn sigmoid_f32(a: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        sigmoid_scalar_f32(a, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::sigmoid_f32(a, out, len),
        SimdLevel::Avx2Fma => avx2::sigmoid_f32(a, out, len),
        _ => sigmoid_scalar_f32(a, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::sigmoid_f32(a, out, len),
        _ => sigmoid_scalar_f32(a, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    sigmoid_scalar_f32(a, out, len);
}

/// SIMD sigmoid for f64
///
/// # Safety
/// - `a` and `out` must point to `len` elements
#[inline]
pub unsafe fn sigmoid_f64(a: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        sigmoid_scalar_f64(a, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::sigmoid_f64(a, out, len),
        SimdLevel::Avx2Fma => avx2::sigmoid_f64(a, out, len),
        _ => sigmoid_scalar_f64(a, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::sigmoid_f64(a, out, len),
        _ => sigmoid_scalar_f64(a, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    sigmoid_scalar_f64(a, out, len);
}

/// SIMD SiLU for f32
///
/// # Safety
/// - `a` and `out` must point to `len` elements
#[inline]
pub unsafe fn silu_f32(a: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        silu_scalar_f32(a, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::silu_f32(a, out, len),
        SimdLevel::Avx2Fma => avx2::silu_f32(a, out, len),
        _ => silu_scalar_f32(a, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::silu_f32(a, out, len),
        _ => silu_scalar_f32(a, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    silu_scalar_f32(a, out, len);
}

/// SIMD SiLU for f64
///
/// # Safety
/// - `a` and `out` must point to `len` elements
#[inline]
pub unsafe fn silu_f64(a: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        silu_scalar_f64(a, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::silu_f64(a, out, len),
        SimdLevel::Avx2Fma => avx2::silu_f64(a, out, len),
        _ => silu_scalar_f64(a, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::silu_f64(a, out, len),
        _ => silu_scalar_f64(a, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    silu_scalar_f64(a, out, len);
}

/// SIMD GELU for f32
///
/// # Safety
/// - `a` and `out` must point to `len` elements
#[inline]
pub unsafe fn gelu_f32(a: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        gelu_scalar_f32(a, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::gelu_f32(a, out, len),
        SimdLevel::Avx2Fma => avx2::gelu_f32(a, out, len),
        _ => gelu_scalar_f32(a, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::gelu_f32(a, out, len),
        _ => gelu_scalar_f32(a, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    gelu_scalar_f32(a, out, len);
}

/// SIMD GELU for f64
///
/// # Safety
/// - `a` and `out` must point to `len` elements
#[inline]
pub unsafe fn gelu_f64(a: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        gelu_scalar_f64(a, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::gelu_f64(a, out, len),
        SimdLevel::Avx2Fma => avx2::gelu_f64(a, out, len),
        _ => gelu_scalar_f64(a, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::gelu_f64(a, out, len),
        _ => gelu_scalar_f64(a, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    gelu_scalar_f64(a, out, len);
}

/// SIMD Leaky ReLU for f32
///
/// # Safety
/// - `a` and `out` must point to `len` elements
#[inline]
pub unsafe fn leaky_relu_f32(a: *const f32, out: *mut f32, len: usize, negative_slope: f32) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        leaky_relu_scalar_f32(a, out, len, negative_slope);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::leaky_relu_f32(a, out, len, negative_slope),
        SimdLevel::Avx2Fma => avx2::leaky_relu_f32(a, out, len, negative_slope),
        _ => leaky_relu_scalar_f32(a, out, len, negative_slope),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::leaky_relu_f32(a, out, len, negative_slope)
        }
        _ => leaky_relu_scalar_f32(a, out, len, negative_slope),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    leaky_relu_scalar_f32(a, out, len, negative_slope);
}

/// SIMD Leaky ReLU for f64
///
/// # Safety
/// - `a` and `out` must point to `len` elements
#[inline]
pub unsafe fn leaky_relu_f64(a: *const f64, out: *mut f64, len: usize, negative_slope: f64) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        leaky_relu_scalar_f64(a, out, len, negative_slope);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::leaky_relu_f64(a, out, len, negative_slope),
        SimdLevel::Avx2Fma => avx2::leaky_relu_f64(a, out, len, negative_slope),
        _ => leaky_relu_scalar_f64(a, out, len, negative_slope),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::leaky_relu_f64(a, out, len, negative_slope)
        }
        _ => leaky_relu_scalar_f64(a, out, len, negative_slope),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    leaky_relu_scalar_f64(a, out, len, negative_slope);
}

/// SIMD ELU for f32
///
/// # Safety
/// - `a` and `out` must point to `len` elements
#[inline]
pub unsafe fn elu_f32(a: *const f32, out: *mut f32, len: usize, alpha: f32) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        elu_scalar_f32(a, out, len, alpha);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::elu_f32(a, out, len, alpha),
        SimdLevel::Avx2Fma => avx2::elu_f32(a, out, len, alpha),
        _ => elu_scalar_f32(a, out, len, alpha),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::elu_f32(a, out, len, alpha),
        _ => elu_scalar_f32(a, out, len, alpha),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    elu_scalar_f32(a, out, len, alpha);
}

/// SIMD ELU for f64
///
/// # Safety
/// - `a` and `out` must point to `len` elements
#[inline]
pub unsafe fn elu_f64(a: *const f64, out: *mut f64, len: usize, alpha: f64) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        elu_scalar_f64(a, out, len, alpha);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::elu_f64(a, out, len, alpha),
        SimdLevel::Avx2Fma => avx2::elu_f64(a, out, len, alpha),
        _ => elu_scalar_f64(a, out, len, alpha),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::elu_f64(a, out, len, alpha),
        _ => elu_scalar_f64(a, out, len, alpha),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    elu_scalar_f64(a, out, len, alpha);
}

// ============================================================================
// Scalar fallbacks
// ============================================================================

/// Scalar sigmoid for f32
#[inline]
pub unsafe fn sigmoid_scalar_f32(a: *const f32, out: *mut f32, len: usize) {
    for i in 0..len {
        let x = *a.add(i);
        *out.add(i) = 1.0 / (1.0 + (-x).exp());
    }
}

/// Scalar sigmoid for f64
#[inline]
pub unsafe fn sigmoid_scalar_f64(a: *const f64, out: *mut f64, len: usize) {
    for i in 0..len {
        let x = *a.add(i);
        *out.add(i) = 1.0 / (1.0 + (-x).exp());
    }
}

/// Scalar SiLU for f32
#[inline]
pub unsafe fn silu_scalar_f32(a: *const f32, out: *mut f32, len: usize) {
    for i in 0..len {
        let x = *a.add(i);
        *out.add(i) = x / (1.0 + (-x).exp());
    }
}

/// Scalar SiLU for f64
#[inline]
pub unsafe fn silu_scalar_f64(a: *const f64, out: *mut f64, len: usize) {
    for i in 0..len {
        let x = *a.add(i);
        *out.add(i) = x / (1.0 + (-x).exp());
    }
}

/// Scalar GELU for f32
#[inline]
pub unsafe fn gelu_scalar_f32(a: *const f32, out: *mut f32, len: usize) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608; // sqrt(2/pi)
    const TANH_COEF: f32 = 0.044715;

    for i in 0..len {
        let x = *a.add(i);
        let inner = SQRT_2_OVER_PI * (x + TANH_COEF * x * x * x);
        *out.add(i) = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// Scalar GELU for f64
#[inline]
pub unsafe fn gelu_scalar_f64(a: *const f64, out: *mut f64, len: usize) {
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654; // sqrt(2/pi)
    const TANH_COEF: f64 = 0.044715;

    for i in 0..len {
        let x = *a.add(i);
        let inner = SQRT_2_OVER_PI * (x + TANH_COEF * x * x * x);
        *out.add(i) = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// Scalar Leaky ReLU for f32
#[inline]
pub unsafe fn leaky_relu_scalar_f32(a: *const f32, out: *mut f32, len: usize, negative_slope: f32) {
    for i in 0..len {
        let x = *a.add(i);
        *out.add(i) = if x > 0.0 { x } else { negative_slope * x };
    }
}

/// Scalar Leaky ReLU for f64
#[inline]
pub unsafe fn leaky_relu_scalar_f64(a: *const f64, out: *mut f64, len: usize, negative_slope: f64) {
    for i in 0..len {
        let x = *a.add(i);
        *out.add(i) = if x > 0.0 { x } else { negative_slope * x };
    }
}

/// Scalar ELU for f32
#[inline]
pub unsafe fn elu_scalar_f32(a: *const f32, out: *mut f32, len: usize, alpha: f32) {
    for i in 0..len {
        let x = *a.add(i);
        *out.add(i) = if x > 0.0 { x } else { alpha * (x.exp() - 1.0) };
    }
}

/// Scalar ELU for f64
#[inline]
pub unsafe fn elu_scalar_f64(a: *const f64, out: *mut f64, len: usize, alpha: f64) {
    for i in 0..len {
        let x = *a.add(i);
        *out.add(i) = if x > 0.0 { x } else { alpha * (x.exp() - 1.0) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_f32() {
        let len = 128;
        let input: Vec<f32> = (0..len).map(|x| (x as f32) / 32.0 - 2.0).collect();
        let mut out = vec![0.0f32; len];
        let mut out_ref = vec![0.0f32; len];

        unsafe {
            sigmoid_f32(input.as_ptr(), out.as_mut_ptr(), len);
            sigmoid_scalar_f32(input.as_ptr(), out_ref.as_mut_ptr(), len);
        }

        for i in 0..len {
            let diff = (out[i] - out_ref[i]).abs();
            let rel_err = diff / out_ref[i].abs().max(1e-6);
            assert!(
                rel_err < 0.01,
                "sigmoid mismatch at {}: {} vs {}",
                i,
                out[i],
                out_ref[i]
            );
        }
    }

    #[test]
    fn test_silu_f32() {
        let len = 128;
        let input: Vec<f32> = (0..len).map(|x| (x as f32) / 32.0 - 2.0).collect();
        let mut out = vec![0.0f32; len];
        let mut out_ref = vec![0.0f32; len];

        unsafe {
            silu_f32(input.as_ptr(), out.as_mut_ptr(), len);
            silu_scalar_f32(input.as_ptr(), out_ref.as_mut_ptr(), len);
        }

        for i in 0..len {
            let diff = (out[i] - out_ref[i]).abs();
            let denom = out_ref[i].abs().max(1e-6);
            let rel_err = diff / denom;
            assert!(
                rel_err < 0.01,
                "silu mismatch at {}: {} vs {}",
                i,
                out[i],
                out_ref[i]
            );
        }
    }

    #[test]
    fn test_gelu_f32() {
        let len = 128;
        let input: Vec<f32> = (0..len).map(|x| (x as f32) / 32.0 - 2.0).collect();
        let mut out = vec![0.0f32; len];
        let mut out_ref = vec![0.0f32; len];

        unsafe {
            gelu_f32(input.as_ptr(), out.as_mut_ptr(), len);
            gelu_scalar_f32(input.as_ptr(), out_ref.as_mut_ptr(), len);
        }

        for i in 0..len {
            let diff = (out[i] - out_ref[i]).abs();
            let denom = out_ref[i].abs().max(1e-6);
            let rel_err = diff / denom;
            assert!(
                rel_err < 0.02,
                "gelu mismatch at {}: {} vs {}",
                i,
                out[i],
                out_ref[i]
            );
        }
    }

    #[test]
    fn test_leaky_relu_f32() {
        let len = 128;
        let input: Vec<f32> = (0..len).map(|x| (x as f32) - 64.0).collect();
        let mut out = vec![0.0f32; len];
        let mut out_ref = vec![0.0f32; len];
        let negative_slope = 0.1f32;

        unsafe {
            leaky_relu_f32(input.as_ptr(), out.as_mut_ptr(), len, negative_slope);
            leaky_relu_scalar_f32(input.as_ptr(), out_ref.as_mut_ptr(), len, negative_slope);
        }

        assert_eq!(out, out_ref);
    }

    #[test]
    fn test_elu_f32() {
        let len = 128;
        let input: Vec<f32> = (0..len).map(|x| (x as f32) / 32.0 - 2.0).collect();
        let mut out = vec![0.0f32; len];
        let mut out_ref = vec![0.0f32; len];
        let alpha = 1.0f32;

        unsafe {
            elu_f32(input.as_ptr(), out.as_mut_ptr(), len, alpha);
            elu_scalar_f32(input.as_ptr(), out_ref.as_mut_ptr(), len, alpha);
        }

        for i in 0..len {
            let diff = (out[i] - out_ref[i]).abs();
            let denom = out_ref[i].abs().max(1e-6);
            let rel_err = diff / denom;
            assert!(
                rel_err < 0.01,
                "elu mismatch at {}: {} vs {}",
                i,
                out[i],
                out_ref[i]
            );
        }
    }
}
