//! Routing and arguments for the native CUDA `conv2d_fused` kernel.
//!
//! The implicit-GEMM tensor-core kernel (`kernels/conv_implicit_gemm.cu`)
//! runs a call when every condition of [`native_args`] holds. Every other
//! call takes the generic path.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::conv_fused_common::FusedConv2dParams;
use crate::ops::{Conv2dConfig, ConvActivation, ConvLayout, ConvResidual};
use crate::runtime::Device;
use crate::runtime::cuda::kernels::{
    ConvEpilogueActivation, ConvImplicitGemmArgs, NhwcStrides, launch_conv_implicit_gemm,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Largest `C_out` routed native. It keeps the channel grid extent under the
/// CUDA limit of 65535 blocks for every tile.
const MAX_NATIVE_C_OUT: usize = 65_535;

/// `(n, h, w)` element strides of a 4D NHWC view the kernel can address.
///
/// The channel stride must be 1, unless there is one channel. No stride can
/// be negative. A dimension of size 1 reports stride 0, so its stride never
/// blocks the 128-bit staging path.
fn nhwc_strides(t: &Tensor<CudaRuntime>) -> Option<NhwcStrides> {
    let shape = t.shape();
    let strides = t.strides();
    if shape.len() != 4 || strides.iter().any(|&s| s < 0) {
        return None;
    }
    if shape[3] > 1 && strides[3] != 1 {
        return None;
    }
    let dim = |d: usize| -> Option<i64> {
        if shape[d] == 1 {
            Some(0)
        } else {
            i64::try_from(strides[d]).ok()
        }
    };
    Some([dim(0)?, dim(1)?, dim(2)?])
}

/// Contiguous NHWC strides of `shape`.
fn contiguous_nhwc_strides(shape: [usize; 4]) -> Option<NhwcStrides> {
    let w = shape[3];
    let h = shape[2].checked_mul(w)?;
    let n = shape[1].checked_mul(h)?;
    Some([
        i64::try_from(n).ok()?,
        i64::try_from(h).ok()?,
        i64::try_from(w).ok()?,
    ])
}

/// True when every geometry value fits the kernel's `i32` arguments.
fn fits_i32(params: &FusedConv2dParams) -> bool {
    let c = &params.conv;
    let pixels = c
        .batch
        .checked_mul(c.output_h)
        .and_then(|v| v.checked_mul(c.output_w));
    let taps = c
        .kernel_h
        .checked_mul(c.kernel_w)
        .and_then(|v| v.checked_mul(c.c_in));
    let (Some(pixels), Some(taps)) = (pixels, taps) else {
        return false;
    };
    [
        pixels,
        taps,
        c.height,
        c.width,
        c.output_h,
        c.output_w,
        c.stride_h,
        c.stride_w,
        c.dilation_h,
        c.dilation_w,
        c.pad_top,
        c.pad_left,
    ]
    .iter()
    .all(|&v| i32::try_from(v).is_ok())
}

/// Launch arguments when the native kernel can run this call, else `None`.
///
/// Conditions, all required:
/// - F16 on a device with `caps.f16_mma`, or BF16 with `caps.bf16`.
/// - `Nhwc` layout and one group.
/// - A non-empty output and `Kh * Kw * C_in >= 1`.
/// - Input, output and residual views with channel stride 1 and no negative
///   stride. Other strides are free, so channel slices work.
/// - A contiguous weight, bias and PReLU weight.
/// - Geometry within the kernel's `i32` arguments and the grid limit.
///
/// `out` is the destination view of `conv2d_fused_into`. `None` means a
/// fresh contiguous output: the returned `out_ptr` is 0 and the caller sets
/// it after allocating.
#[allow(clippy::too_many_arguments)]
pub(super) fn native_args(
    client: &CudaClient,
    params: &FusedConv2dParams,
    config: &Conv2dConfig,
    input: &Tensor<CudaRuntime>,
    weight: &Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
    activation: ConvActivation<'_, CudaRuntime>,
    residual: Option<ConvResidual<'_, CudaRuntime>>,
    out: Option<&Tensor<CudaRuntime>>,
) -> Option<ConvImplicitGemmArgs> {
    let caps = client.device.profile().caps;
    let dtype_ok = match input.dtype() {
        DType::F16 => caps.f16_mma,
        DType::BF16 => caps.bf16,
        _ => false,
    };
    let c = &params.conv;
    if !dtype_ok
        || config.layout != ConvLayout::Nhwc
        || c.groups != 1
        || c.c_out > MAX_NATIVE_C_OUT
        || params.output_shape.iter().product::<usize>() == 0
        || c.c_in == 0
        || c.kernel_h == 0
        || c.kernel_w == 0
        || !fits_i32(params)
    {
        return None;
    }
    if !weight.is_contiguous() || bias.is_some_and(|b| !b.is_contiguous()) {
        return None;
    }

    let (activation, prelu_ptr, prelu_len) = match activation {
        ConvActivation::None => (ConvEpilogueActivation::None, 0, 0),
        ConvActivation::Relu => (ConvEpilogueActivation::Relu, 0, 0),
        ConvActivation::LeakyRelu(slope) => (ConvEpilogueActivation::LeakyRelu(slope as f32), 0, 0),
        ConvActivation::Prelu(w) if w.is_contiguous() => {
            (ConvEpilogueActivation::Prelu, w.ptr(), w.numel())
        }
        ConvActivation::Prelu(_) => return None,
    };
    let (residual_ptr, residual_strides, residual_scale) = match residual {
        Some(r) => (r.tensor.ptr(), nhwc_strides(r.tensor)?, r.scale as f32),
        None => (0, [0; 3], 0.0),
    };
    let (out_ptr, out_strides) = match out {
        Some(o) => (o.ptr(), nhwc_strides(o)?),
        None => (0, contiguous_nhwc_strides(params.output_shape)?),
    };

    Some(ConvImplicitGemmArgs {
        input_ptr: input.ptr(),
        weight_ptr: weight.ptr(),
        bias_ptr: bias.map_or(0, |b| b.ptr()),
        prelu_ptr,
        residual_ptr,
        out_ptr,
        batch: c.batch,
        in_h: c.height,
        in_w: c.width,
        c_in: c.c_in,
        out_h: c.output_h,
        out_w: c.output_w,
        c_out: c.c_out,
        kernel_h: c.kernel_h,
        kernel_w: c.kernel_w,
        stride: (c.stride_h, c.stride_w),
        pad_top: c.pad_top,
        pad_left: c.pad_left,
        dilation: (c.dilation_h, c.dilation_w),
        input_strides: nhwc_strides(input)?,
        out_strides,
        residual_strides,
        activation,
        prelu_len,
        residual_scale,
    })
}

/// Run the native kernel with `args` from [`native_args`].
///
/// The caller has run the fused-conv validators, so every operand has the
/// shape and dtype the arguments describe, and `out` overlaps no read operand.
pub(super) fn launch_native(
    client: &CudaClient,
    dtype: DType,
    args: &ConvImplicitGemmArgs,
) -> Result<()> {
    // SAFETY: `native_args` built every pointer and stride from a live tensor
    // view of this client's device, so each view reaches only its own
    // allocation. The weight, bias and PReLU weight are contiguous. The
    // validators checked every shape and dtype, and for `_into` that `out`
    // repeats no element and is disjoint from every read operand.
    unsafe {
        launch_conv_implicit_gemm(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            args,
        )
    }
}
