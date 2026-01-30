//! Normalization operation kernels
//!
//! Provides normalization operations with automatic SIMD dispatch.
//! On x86-64, f32 and f64 operations use AVX-512 or AVX2 when available.

use crate::dtype::{DType, Element};

/// RMS Normalization: output = input * rsqrt(mean(input^2) + eps) * weight
///
/// Normalizes over the last dimension. Used in LLaMA and other modern transformers.
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration
/// - AVX2: 8 f32s or 4 f64s per iteration
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Arguments
/// * `input` - Input tensor data, shape [batch_size, hidden_size] flattened
/// * `weight` - Weight tensor, shape [hidden_size]
/// * `out` - Output tensor, same shape as input
/// * `batch_size` - Number of rows
/// * `hidden_size` - Size of each row (dimension to normalize over)
/// * `eps` - Small constant for numerical stability
///
/// # Safety
/// - `input` and `out` must be valid pointers to `batch_size * hidden_size` elements
/// - `weight` must be valid pointer to `hidden_size` elements
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn rms_norm_kernel<T: Element>(
    input: *const T,
    weight: *const T,
    out: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    // Dispatch to SIMD for f32/f64 on x86-64
    #[cfg(target_arch = "x86_64")]
    {
        use super::simd::norm;

        match T::DTYPE {
            DType::F32 => {
                norm::rms_norm_f32(
                    input as *const f32,
                    weight as *const f32,
                    out as *mut f32,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            DType::F64 => {
                norm::rms_norm_f64(
                    input as *const f64,
                    weight as *const f64,
                    out as *mut f64,
                    batch_size,
                    hidden_size,
                    eps as f64,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback
    rms_norm_kernel_scalar(input, weight, out, batch_size, hidden_size, eps);
}

/// Scalar RMS norm for all Element types
#[inline]
unsafe fn rms_norm_kernel_scalar<T: Element>(
    input: *const T,
    weight: *const T,
    out: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let weight_slice = std::slice::from_raw_parts(weight, hidden_size);
    let eps = eps as f64;

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        // Compute sum of squares
        let mut sum_sq = 0.0f64;
        for i in 0..hidden_size {
            let x = (*input.add(row_start + i)).to_f64();
            sum_sq += x * x;
        }

        // Compute RMS: sqrt(mean(x^2) + eps)
        let rms = (sum_sq / hidden_size as f64 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Apply normalization and weight
        for (i, &w) in weight_slice.iter().enumerate() {
            let x = (*input.add(row_start + i)).to_f64();
            let result = x * inv_rms * w.to_f64();
            *out.add(row_start + i) = T::from_f64(result);
        }
    }
}

/// Layer Normalization: output = (input - mean) / sqrt(variance + eps) * weight + bias
///
/// Normalizes over the last dimension.
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration
/// - AVX2: 8 f32s or 4 f64s per iteration
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Arguments
/// * `input` - Input tensor data, shape [batch_size, hidden_size] flattened
/// * `weight` - Weight (gamma) tensor, shape [hidden_size]
/// * `bias` - Bias (beta) tensor, shape [hidden_size]
/// * `out` - Output tensor, same shape as input
/// * `batch_size` - Number of rows
/// * `hidden_size` - Size of each row (dimension to normalize over)
/// * `eps` - Small constant for numerical stability
///
/// # Safety
/// - `input` and `out` must be valid pointers to `batch_size * hidden_size` elements
/// - `weight` and `bias` must be valid pointers to `hidden_size` elements
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn layer_norm_kernel<T: Element>(
    input: *const T,
    weight: *const T,
    bias: *const T,
    out: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    // Dispatch to SIMD for f32/f64 on x86-64
    #[cfg(target_arch = "x86_64")]
    {
        use super::simd::norm;

        match T::DTYPE {
            DType::F32 => {
                norm::layer_norm_f32(
                    input as *const f32,
                    weight as *const f32,
                    bias as *const f32,
                    out as *mut f32,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            DType::F64 => {
                norm::layer_norm_f64(
                    input as *const f64,
                    weight as *const f64,
                    bias as *const f64,
                    out as *mut f64,
                    batch_size,
                    hidden_size,
                    eps as f64,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback
    layer_norm_kernel_scalar(input, weight, bias, out, batch_size, hidden_size, eps);
}

/// Scalar layer norm for all Element types
#[inline]
unsafe fn layer_norm_kernel_scalar<T: Element>(
    input: *const T,
    weight: *const T,
    bias: *const T,
    out: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let weight_slice = std::slice::from_raw_parts(weight, hidden_size);
    let bias_slice = std::slice::from_raw_parts(bias, hidden_size);
    let eps = eps as f64;

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        // Compute mean
        let mut sum = 0.0f64;
        for i in 0..hidden_size {
            sum += (*input.add(row_start + i)).to_f64();
        }
        let mean = sum / hidden_size as f64;

        // Compute variance
        let mut var_sum = 0.0f64;
        for i in 0..hidden_size {
            let x = (*input.add(row_start + i)).to_f64();
            let diff = x - mean;
            var_sum += diff * diff;
        }
        let variance = var_sum / hidden_size as f64;

        // Compute inverse standard deviation
        let inv_std = 1.0 / (variance + eps).sqrt();

        // Apply normalization, weight, and bias
        for i in 0..hidden_size {
            let x = (*input.add(row_start + i)).to_f64();
            let w = weight_slice[i].to_f64();
            let b = bias_slice[i].to_f64();
            let result = (x - mean) * inv_std * w + b;
            *out.add(row_start + i) = T::from_f64(result);
        }
    }
}
