//! f16/bf16 normalization wrappers via bulk f32 conversion
//!
//! Pre-converts all inputs to f32 using a single allocation, runs the f32 SIMD
//! norm kernel, then converts the output back.

use super::super::half_convert_utils::*;

/// f16 wrapper for RMS norm.
///
/// # Safety
/// - `input` and `out` must point to `batch_size * hidden_size` elements
/// - `weight` must point to `hidden_size` elements
#[cfg(feature = "f16")]
pub unsafe fn rms_norm_f16(
    input: *const half::f16,
    weight: *const half::f16,
    out: *mut half::f16,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let total = batch_size * hidden_size;
    let mut buf = vec![0.0f32; total + hidden_size + total];
    let (input_f32, rest) = buf.split_at_mut(total);
    let (weight_f32, out_f32) = rest.split_at_mut(hidden_size);
    convert_f16_to_f32(input as *const u16, input_f32.as_mut_ptr(), total);
    convert_f16_to_f32(weight as *const u16, weight_f32.as_mut_ptr(), hidden_size);
    super::rms_norm_f32(
        input_f32.as_ptr(),
        weight_f32.as_ptr(),
        out_f32.as_mut_ptr(),
        batch_size,
        hidden_size,
        eps,
    );
    convert_f32_to_f16(out_f32.as_ptr(), out as *mut u16, total);
}

/// bf16 wrapper for RMS norm.
///
/// # Safety
/// - `input` and `out` must point to `batch_size * hidden_size` elements
/// - `weight` must point to `hidden_size` elements
#[cfg(feature = "f16")]
pub unsafe fn rms_norm_bf16(
    input: *const half::bf16,
    weight: *const half::bf16,
    out: *mut half::bf16,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let total = batch_size * hidden_size;
    let mut buf = vec![0.0f32; total + hidden_size + total];
    let (input_f32, rest) = buf.split_at_mut(total);
    let (weight_f32, out_f32) = rest.split_at_mut(hidden_size);
    convert_bf16_to_f32(input as *const u16, input_f32.as_mut_ptr(), total);
    convert_bf16_to_f32(weight as *const u16, weight_f32.as_mut_ptr(), hidden_size);
    super::rms_norm_f32(
        input_f32.as_ptr(),
        weight_f32.as_ptr(),
        out_f32.as_mut_ptr(),
        batch_size,
        hidden_size,
        eps,
    );
    convert_f32_to_bf16(out_f32.as_ptr(), out as *mut u16, total);
}

/// f16 wrapper for layer norm.
///
/// # Safety
/// - `input` and `out` must point to `batch_size * hidden_size` elements
/// - `weight` and `bias` must point to `hidden_size` elements
#[cfg(feature = "f16")]
pub unsafe fn layer_norm_f16(
    input: *const half::f16,
    weight: *const half::f16,
    bias: *const half::f16,
    out: *mut half::f16,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let total = batch_size * hidden_size;
    let mut buf = vec![0.0f32; total + hidden_size + hidden_size + total];
    let (input_f32, rest) = buf.split_at_mut(total);
    let (weight_f32, rest) = rest.split_at_mut(hidden_size);
    let (bias_f32, out_f32) = rest.split_at_mut(hidden_size);
    convert_f16_to_f32(input as *const u16, input_f32.as_mut_ptr(), total);
    convert_f16_to_f32(weight as *const u16, weight_f32.as_mut_ptr(), hidden_size);
    convert_f16_to_f32(bias as *const u16, bias_f32.as_mut_ptr(), hidden_size);
    super::layer_norm_f32(
        input_f32.as_ptr(),
        weight_f32.as_ptr(),
        bias_f32.as_ptr(),
        out_f32.as_mut_ptr(),
        batch_size,
        hidden_size,
        eps,
    );
    convert_f32_to_f16(out_f32.as_ptr(), out as *mut u16, total);
}

/// bf16 wrapper for layer norm.
///
/// # Safety
/// - `input` and `out` must point to `batch_size * hidden_size` elements
/// - `weight` and `bias` must point to `hidden_size` elements
#[cfg(feature = "f16")]
pub unsafe fn layer_norm_bf16(
    input: *const half::bf16,
    weight: *const half::bf16,
    bias: *const half::bf16,
    out: *mut half::bf16,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let total = batch_size * hidden_size;
    let mut buf = vec![0.0f32; total + hidden_size + hidden_size + total];
    let (input_f32, rest) = buf.split_at_mut(total);
    let (weight_f32, rest) = rest.split_at_mut(hidden_size);
    let (bias_f32, out_f32) = rest.split_at_mut(hidden_size);
    convert_bf16_to_f32(input as *const u16, input_f32.as_mut_ptr(), total);
    convert_bf16_to_f32(weight as *const u16, weight_f32.as_mut_ptr(), hidden_size);
    convert_bf16_to_f32(bias as *const u16, bias_f32.as_mut_ptr(), hidden_size);
    super::layer_norm_f32(
        input_f32.as_ptr(),
        weight_f32.as_ptr(),
        bias_f32.as_ptr(),
        out_f32.as_mut_ptr(),
        batch_size,
        hidden_size,
        eps,
    );
    convert_f32_to_bf16(out_f32.as_ptr(), out as *mut u16, total);
}
