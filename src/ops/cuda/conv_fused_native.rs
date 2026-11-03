//! Routing and arguments for the native CUDA implicit-GEMM conv kernel.
//!
//! The implicit-GEMM tensor-core kernel (`kernels/conv_implicit_gemm.cu`)
//! runs a call when every condition of [`native_args`] holds. Two ops reach
//! it:
//! - `FusedConvOps::conv2d_fused(_into)` in either layout
//!   (`ops/cuda/conv_fused.rs`).
//! - `ConvOps::conv2d` through [`conv2d_native`] (`ops/cuda/conv.rs`).
//!
//! Every other call takes the op's generic path.

use super::conv2d_im2col::MIN_C_OUT;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::conv_common::Conv2dParams;
use crate::ops::conv_fused_common::NCHW_TO_NHWC;
use crate::ops::{ConvAccumulation, ConvActivation, ConvLayout, ConvResidual};
use crate::runtime::Device;
use crate::runtime::cuda::kernels::{
    CONV_MAX_KERNEL_EXTENT, ConvEpilogueActivation, ConvImplicitGemmArgs, ConvStrides,
    launch_conv_implicit_gemm,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Largest `C_out` routed native. It keeps the channel grid extent under the
/// CUDA limit of 65535 blocks for every tile.
const MAX_NATIVE_C_OUT: usize = 65_535;

/// `(n, c, h, w)` element strides of a 4D view in `layout` that the kernel
/// can address.
///
/// No stride can be negative. A dimension of size 1 reports stride 0, so
/// its stride never blocks a vector staging or store path. The innermost
/// dimension differs by layout:
/// - NHWC: the channel stride must be 1, unless there is one channel. It
///   reports 1.
/// - NCHW: any W stride. A W of size 1 reports 1.
fn layout_strides(t: &Tensor<CudaRuntime>, layout: ConvLayout) -> Option<ConvStrides> {
    let shape = t.shape();
    let strides = t.strides();
    if shape.len() != 4 || strides.iter().any(|&s| s < 0) {
        return None;
    }
    let dim = |d: usize| -> Option<i64> {
        if shape[d] == 1 {
            Some(0)
        } else {
            i64::try_from(strides[d]).ok()
        }
    };
    match layout {
        ConvLayout::Nhwc => {
            if shape[3] > 1 && strides[3] != 1 {
                return None;
            }
            Some([dim(0)?, 1, dim(1)?, dim(2)?])
        }
        ConvLayout::Nchw => {
            let w = if shape[3] == 1 { 1 } else { dim(3)? };
            Some([dim(0)?, dim(1)?, dim(2)?, w])
        }
    }
}

/// Contiguous `(n, c, h, w)` strides of an output of `shape` in `layout`.
fn contiguous_strides(shape: [usize; 4], layout: ConvLayout) -> Option<ConvStrides> {
    // Strides of the four storage axes, outermost first.
    let s2 = shape[3];
    let s1 = shape[2].checked_mul(s2)?;
    let s0 = shape[1].checked_mul(s1)?;
    let n = i64::try_from(s0).ok()?;
    let a = i64::try_from(s1).ok()?;
    let b = i64::try_from(s2).ok()?;
    Some(match layout {
        ConvLayout::Nhwc => [n, 1, a, b],
        ConvLayout::Nchw => [n, a, b, 1],
    })
}

/// True when every geometry value, and every tap position the kernel forms
/// in `i32`, fits the kernel's `i32` arithmetic.
fn fits_i32(c: &Conv2dParams) -> bool {
    let pixels = c
        .batch
        .checked_mul(c.output_h)
        .and_then(|v| v.checked_mul(c.output_w));
    let taps = c
        .kernel_h
        .checked_mul(c.kernel_w)
        .and_then(|v| v.checked_mul(c.c_in));
    // Last output row and column times the stride, and last tap times the
    // dilation. An empty output or kernel gives `None` here.
    let last_times = |n: usize, step: usize| n.checked_sub(1).and_then(|v| v.checked_mul(step));
    let reach_h = last_times(c.output_h, c.stride_h);
    let reach_w = last_times(c.output_w, c.stride_w);
    let span_h = last_times(c.kernel_h, c.dilation_h);
    let span_w = last_times(c.kernel_w, c.dilation_w);
    let (Some(pixels), Some(taps), Some(reach_h), Some(reach_w), Some(span_h), Some(span_w)) =
        (pixels, taps, reach_h, reach_w, span_h, span_w)
    else {
        return false;
    };
    [
        pixels,
        taps,
        reach_h,
        reach_w,
        span_h,
        span_w,
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

/// Geometry and accumulator of one native call.
#[derive(Clone, Copy)]
pub(super) struct NativeCall<'a> {
    /// Geometry in NCHW terms, from the op's validator.
    pub conv: &'a Conv2dParams,
    /// Layout of the input, output and residual.
    pub layout: ConvLayout,
    /// Output shape in `layout`.
    pub output_shape: [usize; 4],
    pub accumulation: ConvAccumulation,
}

/// The tensors of one native call, all in [`NativeCall::layout`].
#[derive(Clone, Copy)]
pub(super) struct NativeOperands<'a> {
    pub input: &'a Tensor<CudaRuntime>,
    /// OHWI for NHWC, OIHW for NCHW.
    pub weight: &'a Tensor<CudaRuntime>,
    pub bias: Option<&'a Tensor<CudaRuntime>>,
    pub activation: ConvActivation<'a, CudaRuntime>,
    pub residual: Option<ConvResidual<'a, CudaRuntime>>,
    /// The destination view of `conv2d_fused_into`. `None` means a fresh
    /// contiguous output.
    pub out: Option<&'a Tensor<CudaRuntime>>,
}

/// Launch arguments when the native kernel can run this call, else `None`.
///
/// Conditions, all required:
/// - F16 on sm_75+ (`caps.f16_mma` and `caps.int8_mma`), or BF16 on sm_80+
///   (`caps.bf16`). The kernel needs `ldmatrix` and `mma.sync`, which sm_70
///   lacks.
/// - One group.
/// - A non-empty output and `Kh * Kw * C_in >= 1`.
/// - `Kh` and `Kw` at most [`CONV_MAX_KERNEL_EXTENT`].
/// - Input, output and residual views with no negative stride, and in NHWC
///   channel stride 1. Other strides are free, so channel slices work.
/// - A contiguous bias and PReLU weight. In NHWC a contiguous OHWI weight;
///   in NCHW any OIHW weight, which [`launch_native`] relays to a
///   contiguous OHWI copy.
/// - Geometry within the kernel's `i32` arguments and the grid limit.
///
/// The F16 accumulator runs when `accumulation` is
/// [`ConvAccumulation::AllowHalf`] and the input is F16. BF16 and
/// [`ConvAccumulation::F32`] use the F32 accumulator.
///
/// With `ops.out` set to `None`, the returned `out_ptr` is 0 and the caller
/// sets it after allocating. In NCHW, [`launch_native`] replaces
/// `weight_ptr`.
pub(super) fn native_args(
    client: &CudaClient,
    call: &NativeCall<'_>,
    ops: &NativeOperands<'_>,
) -> Option<ConvImplicitGemmArgs> {
    let caps = client.device.profile().caps;
    // `int8_mma` marks sm_75+, the first generation with `ldmatrix` and
    // `mma.sync.m16n8k8`.
    let dtype_ok = match ops.input.dtype() {
        DType::F16 => caps.f16_mma && caps.int8_mma,
        DType::BF16 => caps.bf16,
        _ => false,
    };
    let c = call.conv;
    if !dtype_ok
        || c.groups != 1
        || c.c_out > MAX_NATIVE_C_OUT
        || call.output_shape.iter().product::<usize>() == 0
        || c.c_in == 0
        || c.kernel_h == 0
        || c.kernel_w == 0
        || c.kernel_h > CONV_MAX_KERNEL_EXTENT
        || c.kernel_w > CONV_MAX_KERNEL_EXTENT
        || !fits_i32(c)
    {
        return None;
    }
    let weight_ok = call.layout == ConvLayout::Nchw || ops.weight.is_contiguous();
    if !weight_ok || ops.bias.is_some_and(|b| !b.is_contiguous()) {
        return None;
    }

    let (activation, prelu_ptr, prelu_len) = match ops.activation {
        ConvActivation::None => (ConvEpilogueActivation::None, 0, 0),
        ConvActivation::Relu => (ConvEpilogueActivation::Relu, 0, 0),
        ConvActivation::LeakyRelu(slope) => (ConvEpilogueActivation::LeakyRelu(slope as f32), 0, 0),
        ConvActivation::Prelu(w) if w.is_contiguous() => {
            (ConvEpilogueActivation::Prelu, w.ptr(), w.numel())
        }
        ConvActivation::Prelu(_) => return None,
    };
    let (residual_ptr, residual_strides, residual_scale) = match ops.residual {
        Some(r) => (
            r.tensor.ptr(),
            layout_strides(r.tensor, call.layout)?,
            r.scale as f32,
        ),
        None => (0, [0; 4], 0.0),
    };
    let (out_ptr, out_strides) = match ops.out {
        Some(o) => (o.ptr(), layout_strides(o, call.layout)?),
        None => (0, contiguous_strides(call.output_shape, call.layout)?),
    };
    let half_accumulate =
        call.accumulation == ConvAccumulation::AllowHalf && ops.input.dtype() == DType::F16;

    Some(ConvImplicitGemmArgs {
        input_ptr: ops.input.ptr(),
        weight_ptr: ops.weight.ptr(),
        bias_ptr: ops.bias.map_or(0, |b| b.ptr()),
        prelu_ptr,
        residual_ptr,
        out_ptr,
        layout: call.layout,
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
        input_strides: layout_strides(ops.input, call.layout)?,
        out_strides,
        residual_strides,
        activation,
        prelu_len,
        residual_scale,
        half_accumulate,
    })
}

/// Run the native kernel with `args` from [`native_args`]. `weight` is the
/// call's weight. In NCHW it is relaid to a contiguous OHWI copy first.
///
/// The caller has run the op's validators, so every operand has the shape
/// and dtype the arguments describe, and `out` overlaps no read operand.
pub(super) fn launch_native(
    client: &CudaClient,
    dtype: DType,
    args: &ConvImplicitGemmArgs,
    weight: &Tensor<CudaRuntime>,
) -> Result<()> {
    // OIHW to OHWI is the same axis move as NCHW to NHWC.
    let relaid = match args.layout {
        ConvLayout::Nhwc => None,
        ConvLayout::Nchw => Some(weight.permute(&NCHW_TO_NHWC)?.contiguous()?),
    };
    let mut args = *args;
    if let Some(w) = &relaid {
        args.weight_ptr = w.ptr();
    }
    // SAFETY: `native_args` built every pointer and stride from a live tensor
    // view of this client's device, so each view reaches only its own
    // allocation. The weight is a contiguous OHWI tensor: the caller's view
    // for NHWC, `relaid` for NCHW. The bias and PReLU weight are contiguous.
    // The validators checked every shape and dtype, and for `_into` that
    // `out` repeats no element and is disjoint from every read operand.
    // `relaid` is freed after the launch is queued, on the same stream, as
    // every temporary of this crate.
    unsafe {
        launch_conv_implicit_gemm(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            &args,
        )
    }
}

/// `ConvOps::conv2d` on the native kernel, or `None` when the call does not
/// qualify.
///
/// The call qualifies when [`native_args`] accepts it as NCHW with the F32
/// accumulator and no activation or residual, and `C_out >= MIN_C_OUT`.
/// The floor is the im2col gate's: below it the direct kernel stays the
/// path, which `tests/backend_parity/conv_half_accumulation.rs` relies on.
/// `params` must come from `validate_conv2d`.
pub(super) fn conv2d_native(
    client: &CudaClient,
    params: &Conv2dParams,
    input: &Tensor<CudaRuntime>,
    weight: &Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
) -> Result<Option<Tensor<CudaRuntime>>> {
    if params.c_out < MIN_C_OUT {
        return Ok(None);
    }
    let output_shape = [params.batch, params.c_out, params.output_h, params.output_w];
    let call = NativeCall {
        conv: params,
        layout: ConvLayout::Nchw,
        output_shape,
        accumulation: ConvAccumulation::F32,
    };
    let ops = NativeOperands {
        input,
        weight,
        bias,
        activation: ConvActivation::None,
        residual: None,
        out: None,
    };
    let Some(mut args) = native_args(client, &call, &ops) else {
        return Ok(None);
    };
    let out = Tensor::<CudaRuntime>::empty(&output_shape, input.dtype(), &client.device)?;
    args.out_ptr = out.ptr();
    launch_native(client, input.dtype(), &args, weight)?;
    Ok(Some(out))
}
