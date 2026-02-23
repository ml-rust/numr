//! Scalar fallbacks for convolution operations

use crate::ops::conv_common::{Conv1dParams, Conv2dParams};

/// Scalar conv1d for f32
#[inline]
pub unsafe fn conv1d_scalar_f32(
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    output: *mut f32,
    params: Conv1dParams,
) {
    crate::runtime::cpu::kernels::conv::conv1d_kernel(input, weight, bias, output, params);
}

/// Scalar conv1d for f64
#[inline]
pub unsafe fn conv1d_scalar_f64(
    input: *const f64,
    weight: *const f64,
    bias: Option<*const f64>,
    output: *mut f64,
    params: Conv1dParams,
) {
    crate::runtime::cpu::kernels::conv::conv1d_kernel(input, weight, bias, output, params);
}

/// Scalar conv2d for f32
#[inline]
pub unsafe fn conv2d_scalar_f32(
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    output: *mut f32,
    params: Conv2dParams,
) {
    crate::runtime::cpu::kernels::conv::conv2d_kernel(input, weight, bias, output, params);
}

/// Scalar conv2d for f64
#[inline]
pub unsafe fn conv2d_scalar_f64(
    input: *const f64,
    weight: *const f64,
    bias: Option<*const f64>,
    output: *mut f64,
    params: Conv2dParams,
) {
    crate::runtime::cpu::kernels::conv::conv2d_kernel(input, weight, bias, output, params);
}

/// Scalar depthwise conv2d for f32
#[inline]
pub unsafe fn depthwise_conv2d_scalar_f32(
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    output: *mut f32,
    params: Conv2dParams,
) {
    crate::runtime::cpu::kernels::conv::depthwise_conv2d_kernel(
        input, weight, bias, output, params,
    );
}

/// Scalar depthwise conv2d for f64
#[inline]
pub unsafe fn depthwise_conv2d_scalar_f64(
    input: *const f64,
    weight: *const f64,
    bias: Option<*const f64>,
    output: *mut f64,
    params: Conv2dParams,
) {
    crate::runtime::cpu::kernels::conv::depthwise_conv2d_kernel(
        input, weight, bias, output, params,
    );
}
