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

/// f16 wrapper for fused add + RMS norm.
///
/// # Safety
/// - `input`, `residual`, and `out` must point to `batch_size * hidden_size` elements
/// - `weight` must point to `hidden_size` elements
/// - `pre_norm` must point to `batch_size * hidden_size` elements
#[cfg(feature = "f16")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_rms_norm_f16(
    input: *const half::f16,
    residual: *const half::f16,
    weight: *const half::f16,
    out: *mut half::f16,
    pre_norm: *mut half::f16,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let total = batch_size * hidden_size;
    let mut buf = vec![0.0f32; total + total + hidden_size + total + total];
    let (input_f32, rest) = buf.split_at_mut(total);
    let (residual_f32, rest) = rest.split_at_mut(total);
    let (weight_f32, rest) = rest.split_at_mut(hidden_size);
    let (out_f32, pre_norm_f32) = rest.split_at_mut(total);
    convert_f16_to_f32(input as *const u16, input_f32.as_mut_ptr(), total);
    convert_f16_to_f32(residual as *const u16, residual_f32.as_mut_ptr(), total);
    convert_f16_to_f32(weight as *const u16, weight_f32.as_mut_ptr(), hidden_size);
    super::fused_add_rms_norm_f32(
        input_f32.as_ptr(),
        residual_f32.as_ptr(),
        weight_f32.as_ptr(),
        out_f32.as_mut_ptr(),
        pre_norm_f32.as_mut_ptr(),
        batch_size,
        hidden_size,
        eps,
    );
    convert_f32_to_f16(out_f32.as_ptr(), out as *mut u16, total);
    convert_f32_to_f16(pre_norm_f32.as_ptr(), pre_norm as *mut u16, total);
}

/// bf16 wrapper for fused add + RMS norm.
///
/// # Safety
/// - `input`, `residual`, and `out` must point to `batch_size * hidden_size` elements
/// - `weight` must point to `hidden_size` elements
/// - `pre_norm` must point to `batch_size * hidden_size` elements
#[cfg(feature = "f16")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_rms_norm_bf16(
    input: *const half::bf16,
    residual: *const half::bf16,
    weight: *const half::bf16,
    out: *mut half::bf16,
    pre_norm: *mut half::bf16,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let total = batch_size * hidden_size;
    let mut buf = vec![0.0f32; total + total + hidden_size + total + total];
    let (input_f32, rest) = buf.split_at_mut(total);
    let (residual_f32, rest) = rest.split_at_mut(total);
    let (weight_f32, rest) = rest.split_at_mut(hidden_size);
    let (out_f32, pre_norm_f32) = rest.split_at_mut(total);
    convert_bf16_to_f32(input as *const u16, input_f32.as_mut_ptr(), total);
    convert_bf16_to_f32(residual as *const u16, residual_f32.as_mut_ptr(), total);
    convert_bf16_to_f32(weight as *const u16, weight_f32.as_mut_ptr(), hidden_size);
    super::fused_add_rms_norm_f32(
        input_f32.as_ptr(),
        residual_f32.as_ptr(),
        weight_f32.as_ptr(),
        out_f32.as_mut_ptr(),
        pre_norm_f32.as_mut_ptr(),
        batch_size,
        hidden_size,
        eps,
    );
    convert_f32_to_bf16(out_f32.as_ptr(), out as *mut u16, total);
    convert_f32_to_bf16(pre_norm_f32.as_ptr(), pre_norm as *mut u16, total);
}

/// f16 wrapper for backward pass of fused add + RMS norm.
///
/// # Safety
/// - `grad` and `pre_norm` must point to `batch_size * hidden_size` elements
/// - `weight` must point to `hidden_size` elements
/// - `d_input_residual` must point to `batch_size * hidden_size` elements
/// - `d_weight` must point to `hidden_size` elements (pre-zeroed by caller)
#[cfg(feature = "f16")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_rms_norm_bwd_f16(
    grad: *const half::f16,
    pre_norm: *const half::f16,
    weight: *const half::f16,
    d_input_residual: *mut half::f16,
    d_weight: *mut half::f16,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let total = batch_size * hidden_size;
    let mut buf = vec![0.0f32; total + total + hidden_size + total + hidden_size];
    let (grad_f32, rest) = buf.split_at_mut(total);
    let (pre_norm_f32, rest) = rest.split_at_mut(total);
    let (weight_f32, rest) = rest.split_at_mut(hidden_size);
    let (d_ir_f32, d_weight_f32) = rest.split_at_mut(total);
    convert_f16_to_f32(grad as *const u16, grad_f32.as_mut_ptr(), total);
    convert_f16_to_f32(pre_norm as *const u16, pre_norm_f32.as_mut_ptr(), total);
    convert_f16_to_f32(weight as *const u16, weight_f32.as_mut_ptr(), hidden_size);
    super::fused_add_rms_norm_bwd_f32(
        grad_f32.as_ptr(),
        pre_norm_f32.as_ptr(),
        weight_f32.as_ptr(),
        d_ir_f32.as_mut_ptr(),
        d_weight_f32.as_mut_ptr(),
        batch_size,
        hidden_size,
        eps,
    );
    convert_f32_to_f16(d_ir_f32.as_ptr(), d_input_residual as *mut u16, total);
    convert_f32_to_f16(d_weight_f32.as_ptr(), d_weight as *mut u16, hidden_size);
}

/// bf16 wrapper for backward pass of fused add + RMS norm.
///
/// # Safety
/// - `grad` and `pre_norm` must point to `batch_size * hidden_size` elements
/// - `weight` must point to `hidden_size` elements
/// - `d_input_residual` must point to `batch_size * hidden_size` elements
/// - `d_weight` must point to `hidden_size` elements (pre-zeroed by caller)
#[cfg(feature = "f16")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_rms_norm_bwd_bf16(
    grad: *const half::bf16,
    pre_norm: *const half::bf16,
    weight: *const half::bf16,
    d_input_residual: *mut half::bf16,
    d_weight: *mut half::bf16,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let total = batch_size * hidden_size;
    let mut buf = vec![0.0f32; total + total + hidden_size + total + hidden_size];
    let (grad_f32, rest) = buf.split_at_mut(total);
    let (pre_norm_f32, rest) = rest.split_at_mut(total);
    let (weight_f32, rest) = rest.split_at_mut(hidden_size);
    let (d_ir_f32, d_weight_f32) = rest.split_at_mut(total);
    convert_bf16_to_f32(grad as *const u16, grad_f32.as_mut_ptr(), total);
    convert_bf16_to_f32(pre_norm as *const u16, pre_norm_f32.as_mut_ptr(), total);
    convert_bf16_to_f32(weight as *const u16, weight_f32.as_mut_ptr(), hidden_size);
    super::fused_add_rms_norm_bwd_f32(
        grad_f32.as_ptr(),
        pre_norm_f32.as_ptr(),
        weight_f32.as_ptr(),
        d_ir_f32.as_mut_ptr(),
        d_weight_f32.as_mut_ptr(),
        batch_size,
        hidden_size,
        eps,
    );
    convert_f32_to_bf16(d_ir_f32.as_ptr(), d_input_residual as *mut u16, total);
    convert_f32_to_bf16(d_weight_f32.as_ptr(), d_weight as *mut u16, hidden_size);
}

/// f16 wrapper for fused add + layer norm.
///
/// # Safety
/// - `input`, `residual`, and `out` must point to `batch_size * hidden_size` elements
/// - `weight` and `bias` must point to `hidden_size` elements
/// - `pre_norm` must point to `batch_size * hidden_size` elements
#[cfg(feature = "f16")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_layer_norm_f16(
    input: *const half::f16,
    residual: *const half::f16,
    weight: *const half::f16,
    bias: *const half::f16,
    out: *mut half::f16,
    pre_norm: *mut half::f16,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let total = batch_size * hidden_size;
    let mut buf = vec![0.0f32; total + total + hidden_size + hidden_size + total + total];
    let (input_f32, rest) = buf.split_at_mut(total);
    let (residual_f32, rest) = rest.split_at_mut(total);
    let (weight_f32, rest) = rest.split_at_mut(hidden_size);
    let (bias_f32, rest) = rest.split_at_mut(hidden_size);
    let (out_f32, pre_norm_f32) = rest.split_at_mut(total);
    convert_f16_to_f32(input as *const u16, input_f32.as_mut_ptr(), total);
    convert_f16_to_f32(residual as *const u16, residual_f32.as_mut_ptr(), total);
    convert_f16_to_f32(weight as *const u16, weight_f32.as_mut_ptr(), hidden_size);
    convert_f16_to_f32(bias as *const u16, bias_f32.as_mut_ptr(), hidden_size);
    super::fused_add_layer_norm_f32(
        input_f32.as_ptr(),
        residual_f32.as_ptr(),
        weight_f32.as_ptr(),
        bias_f32.as_ptr(),
        out_f32.as_mut_ptr(),
        pre_norm_f32.as_mut_ptr(),
        batch_size,
        hidden_size,
        eps,
    );
    convert_f32_to_f16(out_f32.as_ptr(), out as *mut u16, total);
    convert_f32_to_f16(pre_norm_f32.as_ptr(), pre_norm as *mut u16, total);
}

/// bf16 wrapper for fused add + layer norm.
///
/// # Safety
/// - `input`, `residual`, and `out` must point to `batch_size * hidden_size` elements
/// - `weight` and `bias` must point to `hidden_size` elements
/// - `pre_norm` must point to `batch_size * hidden_size` elements
#[cfg(feature = "f16")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_layer_norm_bf16(
    input: *const half::bf16,
    residual: *const half::bf16,
    weight: *const half::bf16,
    bias: *const half::bf16,
    out: *mut half::bf16,
    pre_norm: *mut half::bf16,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let total = batch_size * hidden_size;
    let mut buf = vec![0.0f32; total + total + hidden_size + hidden_size + total + total];
    let (input_f32, rest) = buf.split_at_mut(total);
    let (residual_f32, rest) = rest.split_at_mut(total);
    let (weight_f32, rest) = rest.split_at_mut(hidden_size);
    let (bias_f32, rest) = rest.split_at_mut(hidden_size);
    let (out_f32, pre_norm_f32) = rest.split_at_mut(total);
    convert_bf16_to_f32(input as *const u16, input_f32.as_mut_ptr(), total);
    convert_bf16_to_f32(residual as *const u16, residual_f32.as_mut_ptr(), total);
    convert_bf16_to_f32(weight as *const u16, weight_f32.as_mut_ptr(), hidden_size);
    convert_bf16_to_f32(bias as *const u16, bias_f32.as_mut_ptr(), hidden_size);
    super::fused_add_layer_norm_f32(
        input_f32.as_ptr(),
        residual_f32.as_ptr(),
        weight_f32.as_ptr(),
        bias_f32.as_ptr(),
        out_f32.as_mut_ptr(),
        pre_norm_f32.as_mut_ptr(),
        batch_size,
        hidden_size,
        eps,
    );
    convert_f32_to_bf16(out_f32.as_ptr(), out as *mut u16, total);
    convert_f32_to_bf16(pre_norm_f32.as_ptr(), pre_norm as *mut u16, total);
}

/// f16 wrapper for backward pass of fused add + layer norm.
///
/// # Safety
/// - `grad` and `pre_norm` must point to `batch_size * hidden_size` elements
/// - `weight` must point to `hidden_size` elements
/// - `d_input_residual` must point to `batch_size * hidden_size` elements
/// - `d_weight` and `d_bias` must point to `hidden_size` elements (pre-zeroed by caller)
#[cfg(feature = "f16")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_layer_norm_bwd_f16(
    grad: *const half::f16,
    pre_norm: *const half::f16,
    weight: *const half::f16,
    d_input_residual: *mut half::f16,
    d_weight: *mut half::f16,
    d_bias: *mut half::f16,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let total = batch_size * hidden_size;
    let mut buf = vec![0.0f32; total + total + hidden_size + total + hidden_size + hidden_size];
    let (grad_f32, rest) = buf.split_at_mut(total);
    let (pre_norm_f32, rest) = rest.split_at_mut(total);
    let (weight_f32, rest) = rest.split_at_mut(hidden_size);
    let (d_ir_f32, rest) = rest.split_at_mut(total);
    let (d_weight_f32, d_bias_f32) = rest.split_at_mut(hidden_size);
    convert_f16_to_f32(grad as *const u16, grad_f32.as_mut_ptr(), total);
    convert_f16_to_f32(pre_norm as *const u16, pre_norm_f32.as_mut_ptr(), total);
    convert_f16_to_f32(weight as *const u16, weight_f32.as_mut_ptr(), hidden_size);
    super::fused_add_layer_norm_bwd_f32(
        grad_f32.as_ptr(),
        pre_norm_f32.as_ptr(),
        weight_f32.as_ptr(),
        d_ir_f32.as_mut_ptr(),
        d_weight_f32.as_mut_ptr(),
        d_bias_f32.as_mut_ptr(),
        batch_size,
        hidden_size,
        eps,
    );
    convert_f32_to_f16(d_ir_f32.as_ptr(), d_input_residual as *mut u16, total);
    convert_f32_to_f16(d_weight_f32.as_ptr(), d_weight as *mut u16, hidden_size);
    convert_f32_to_f16(d_bias_f32.as_ptr(), d_bias as *mut u16, hidden_size);
}

/// bf16 wrapper for backward pass of fused add + layer norm.
///
/// # Safety
/// - `grad` and `pre_norm` must point to `batch_size * hidden_size` elements
/// - `weight` must point to `hidden_size` elements
/// - `d_input_residual` must point to `batch_size * hidden_size` elements
/// - `d_weight` and `d_bias` must point to `hidden_size` elements (pre-zeroed by caller)
#[cfg(feature = "f16")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_layer_norm_bwd_bf16(
    grad: *const half::bf16,
    pre_norm: *const half::bf16,
    weight: *const half::bf16,
    d_input_residual: *mut half::bf16,
    d_weight: *mut half::bf16,
    d_bias: *mut half::bf16,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let total = batch_size * hidden_size;
    let mut buf = vec![0.0f32; total + total + hidden_size + total + hidden_size + hidden_size];
    let (grad_f32, rest) = buf.split_at_mut(total);
    let (pre_norm_f32, rest) = rest.split_at_mut(total);
    let (weight_f32, rest) = rest.split_at_mut(hidden_size);
    let (d_ir_f32, rest) = rest.split_at_mut(total);
    let (d_weight_f32, d_bias_f32) = rest.split_at_mut(hidden_size);
    convert_bf16_to_f32(grad as *const u16, grad_f32.as_mut_ptr(), total);
    convert_bf16_to_f32(pre_norm as *const u16, pre_norm_f32.as_mut_ptr(), total);
    convert_bf16_to_f32(weight as *const u16, weight_f32.as_mut_ptr(), hidden_size);
    super::fused_add_layer_norm_bwd_f32(
        grad_f32.as_ptr(),
        pre_norm_f32.as_ptr(),
        weight_f32.as_ptr(),
        d_ir_f32.as_mut_ptr(),
        d_weight_f32.as_mut_ptr(),
        d_bias_f32.as_mut_ptr(),
        batch_size,
        hidden_size,
        eps,
    );
    convert_f32_to_bf16(d_ir_f32.as_ptr(), d_input_residual as *mut u16, total);
    convert_f32_to_bf16(d_weight_f32.as_ptr(), d_weight as *mut u16, hidden_size);
    convert_f32_to_bf16(d_bias_f32.as_ptr(), d_bias as *mut u16, hidden_size);
}
