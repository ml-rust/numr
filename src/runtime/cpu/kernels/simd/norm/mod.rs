//! SIMD-accelerated normalization operations
//!
//! This module provides AVX2 and AVX-512 implementations for normalization
//! operations critical in transformer architectures.
//!
//! # SIMD Optimizations
//!
//! - Sum of squares (for RMS): SIMD vertical accumulation + horizontal reduction
//! - Mean computation: SIMD sum + division
//! - Element-wise operations: SIMD multiply, add

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

use super::{SimdLevel, detect_simd};

/// Minimum hidden_size to justify SIMD overhead
const SIMD_THRESHOLD: usize = 64;

/// SIMD RMS normalization for f32
///
/// # Safety
/// - `input` and `out` must point to `batch_size * hidden_size` elements
/// - `weight` must point to `hidden_size` elements
#[inline]
pub unsafe fn rms_norm_f32(
    input: *const f32,
    weight: *const f32,
    out: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let level = detect_simd();

    if hidden_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        rms_norm_scalar_f32(input, weight, out, batch_size, hidden_size, eps);
        return;
    }

    match level {
        SimdLevel::Avx512 => avx512::rms_norm_f32(input, weight, out, batch_size, hidden_size, eps),
        SimdLevel::Avx2Fma => avx2::rms_norm_f32(input, weight, out, batch_size, hidden_size, eps),
        SimdLevel::Scalar => unreachable!(),
    }
}

/// SIMD RMS normalization for f64
///
/// # Safety
/// - `input` and `out` must point to `batch_size * hidden_size` elements
/// - `weight` must point to `hidden_size` elements
#[inline]
pub unsafe fn rms_norm_f64(
    input: *const f64,
    weight: *const f64,
    out: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    let level = detect_simd();

    if hidden_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        rms_norm_scalar_f64(input, weight, out, batch_size, hidden_size, eps);
        return;
    }

    match level {
        SimdLevel::Avx512 => avx512::rms_norm_f64(input, weight, out, batch_size, hidden_size, eps),
        SimdLevel::Avx2Fma => avx2::rms_norm_f64(input, weight, out, batch_size, hidden_size, eps),
        SimdLevel::Scalar => unreachable!(),
    }
}

/// SIMD Layer normalization for f32
///
/// # Safety
/// - `input` and `out` must point to `batch_size * hidden_size` elements
/// - `weight` and `bias` must point to `hidden_size` elements
#[inline]
pub unsafe fn layer_norm_f32(
    input: *const f32,
    weight: *const f32,
    bias: *const f32,
    out: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let level = detect_simd();

    if hidden_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        layer_norm_scalar_f32(input, weight, bias, out, batch_size, hidden_size, eps);
        return;
    }

    match level {
        SimdLevel::Avx512 => {
            avx512::layer_norm_f32(input, weight, bias, out, batch_size, hidden_size, eps)
        }
        SimdLevel::Avx2Fma => {
            avx2::layer_norm_f32(input, weight, bias, out, batch_size, hidden_size, eps)
        }
        SimdLevel::Scalar => unreachable!(),
    }
}

/// SIMD Layer normalization for f64
///
/// # Safety
/// - `input` and `out` must point to `batch_size * hidden_size` elements
/// - `weight` and `bias` must point to `hidden_size` elements
#[inline]
pub unsafe fn layer_norm_f64(
    input: *const f64,
    weight: *const f64,
    bias: *const f64,
    out: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    let level = detect_simd();

    if hidden_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        layer_norm_scalar_f64(input, weight, bias, out, batch_size, hidden_size, eps);
        return;
    }

    match level {
        SimdLevel::Avx512 => {
            avx512::layer_norm_f64(input, weight, bias, out, batch_size, hidden_size, eps)
        }
        SimdLevel::Avx2Fma => {
            avx2::layer_norm_f64(input, weight, bias, out, batch_size, hidden_size, eps)
        }
        SimdLevel::Scalar => unreachable!(),
    }
}

// ============================================================================
// Scalar fallbacks
// ============================================================================

/// Scalar RMS norm for f32
#[inline]
pub unsafe fn rms_norm_scalar_f32(
    input: *const f32,
    weight: *const f32,
    out: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        // Compute sum of squares
        let mut sum_sq = 0.0f32;
        for i in 0..hidden_size {
            let x = *input.add(row_start + i);
            sum_sq += x * x;
        }

        // Compute inverse RMS
        let inv_rms = 1.0 / (sum_sq / hidden_size as f32 + eps).sqrt();

        // Apply normalization and weight
        for i in 0..hidden_size {
            let x = *input.add(row_start + i);
            let w = *weight.add(i);
            *out.add(row_start + i) = x * inv_rms * w;
        }
    }
}

/// Scalar RMS norm for f64
#[inline]
pub unsafe fn rms_norm_scalar_f64(
    input: *const f64,
    weight: *const f64,
    out: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        let mut sum_sq = 0.0f64;
        for i in 0..hidden_size {
            let x = *input.add(row_start + i);
            sum_sq += x * x;
        }

        let inv_rms = 1.0 / (sum_sq / hidden_size as f64 + eps).sqrt();

        for i in 0..hidden_size {
            let x = *input.add(row_start + i);
            let w = *weight.add(i);
            *out.add(row_start + i) = x * inv_rms * w;
        }
    }
}

/// Scalar layer norm for f32
#[inline]
pub unsafe fn layer_norm_scalar_f32(
    input: *const f32,
    weight: *const f32,
    bias: *const f32,
    out: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        // Compute mean
        let mut sum = 0.0f32;
        for i in 0..hidden_size {
            sum += *input.add(row_start + i);
        }
        let mean = sum / hidden_size as f32;

        // Compute variance
        let mut var_sum = 0.0f32;
        for i in 0..hidden_size {
            let diff = *input.add(row_start + i) - mean;
            var_sum += diff * diff;
        }
        let inv_std = 1.0 / (var_sum / hidden_size as f32 + eps).sqrt();

        // Apply normalization, weight, and bias
        for i in 0..hidden_size {
            let x = *input.add(row_start + i);
            let w = *weight.add(i);
            let b = *bias.add(i);
            *out.add(row_start + i) = (x - mean) * inv_std * w + b;
        }
    }
}

/// Scalar layer norm for f64
#[inline]
pub unsafe fn layer_norm_scalar_f64(
    input: *const f64,
    weight: *const f64,
    bias: *const f64,
    out: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        let mut sum = 0.0f64;
        for i in 0..hidden_size {
            sum += *input.add(row_start + i);
        }
        let mean = sum / hidden_size as f64;

        let mut var_sum = 0.0f64;
        for i in 0..hidden_size {
            let diff = *input.add(row_start + i) - mean;
            var_sum += diff * diff;
        }
        let inv_std = 1.0 / (var_sum / hidden_size as f64 + eps).sqrt();

        for i in 0..hidden_size {
            let x = *input.add(row_start + i);
            let w = *weight.add(i);
            let b = *bias.add(i);
            *out.add(row_start + i) = (x - mean) * inv_std * w + b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_f32() {
        let hidden_size = 128;
        let batch_size = 4;
        let input: Vec<f32> = (0..(batch_size * hidden_size))
            .map(|x| (x as f32) / 100.0 - 2.5)
            .collect();
        let weight: Vec<f32> = vec![1.0f32; hidden_size];
        let mut out = vec![0.0f32; batch_size * hidden_size];
        let mut out_ref = vec![0.0f32; batch_size * hidden_size];

        unsafe {
            rms_norm_f32(
                input.as_ptr(),
                weight.as_ptr(),
                out.as_mut_ptr(),
                batch_size,
                hidden_size,
                1e-5,
            );
            rms_norm_scalar_f32(
                input.as_ptr(),
                weight.as_ptr(),
                out_ref.as_mut_ptr(),
                batch_size,
                hidden_size,
                1e-5,
            );
        }

        for i in 0..(batch_size * hidden_size) {
            assert!(
                (out[i] - out_ref[i]).abs() < 1e-4,
                "mismatch at {}: {} vs {}",
                i,
                out[i],
                out_ref[i]
            );
        }
    }

    #[test]
    fn test_layer_norm_f32() {
        let hidden_size = 128;
        let batch_size = 4;
        let input: Vec<f32> = (0..(batch_size * hidden_size))
            .map(|x| (x as f32) / 100.0 - 2.5)
            .collect();
        let weight: Vec<f32> = vec![1.0f32; hidden_size];
        let bias: Vec<f32> = vec![0.0f32; hidden_size];
        let mut out = vec![0.0f32; batch_size * hidden_size];
        let mut out_ref = vec![0.0f32; batch_size * hidden_size];

        unsafe {
            layer_norm_f32(
                input.as_ptr(),
                weight.as_ptr(),
                bias.as_ptr(),
                out.as_mut_ptr(),
                batch_size,
                hidden_size,
                1e-5,
            );
            layer_norm_scalar_f32(
                input.as_ptr(),
                weight.as_ptr(),
                bias.as_ptr(),
                out_ref.as_mut_ptr(),
                batch_size,
                hidden_size,
                1e-5,
            );
        }

        for i in 0..(batch_size * hidden_size) {
            assert!(
                (out[i] - out_ref[i]).abs() < 1e-4,
                "mismatch at {}: {} vs {}",
                i,
                out[i],
                out_ref[i]
            );
        }
    }
}
