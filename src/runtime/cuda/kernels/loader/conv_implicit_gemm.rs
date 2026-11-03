//! Launcher for the implicit-GEMM tensor-core conv2d.
//!
//! The kernels live in `kernels/conv_implicit_gemm.cu`: F16 and BF16, NHWC
//! input, OHWI weight, fused bias, activation and scaled residual. The
//! accumulator is F32, or F16 for F16 inputs when
//! [`ConvImplicitGemmArgs::half_accumulate`] is set. The tile comes from
//! [`super::conv_implicit_gemm_tile::select_conv_tile`] and the staging
//! variant (halo, vector gather or scalar gather) from
//! [`super::conv_implicit_gemm_tile::select_conv_staging`].

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;

use super::conv_implicit_gemm_tile::{
    ConvWindow, conv_kernel_name, conv_launch_config, select_conv_staging, select_conv_tile,
};
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::kernel_names;

/// Activation code of the conv kernel. The values match `CONV_ACT_*` in
/// `conv_implicit_gemm_epilogue.cuh`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum ConvEpilogueActivation {
    /// Identity.
    None,
    /// `max(0, y)`.
    Relu,
    /// `y` if `y > 0`, else `slope * y`.
    LeakyRelu(f32),
    /// `y` if `y > 0`, else `weight[c] * y`. The weight pointer and length
    /// are in [`ConvImplicitGemmArgs`].
    Prelu,
}

impl ConvEpilogueActivation {
    /// Kernel code and LeakyRelu slope.
    fn code_and_slope(self) -> (u32, f32) {
        match self {
            Self::None => (0, 0.0),
            Self::Relu => (1, 0.0),
            Self::LeakyRelu(slope) => (2, slope),
            Self::Prelu => (3, 0.0),
        }
    }
}

/// Largest `Kh` and `Kw` the kernel takes. Each output row keeps one 32-bit
/// mask per kernel axis, one bit per tap.
pub(crate) const CONV_MAX_KERNEL_EXTENT: usize = 32;

/// Element strides `(n, h, w)` of a 4D NHWC view whose channel stride is 1.
pub(crate) type NhwcStrides = [i64; 3];

/// Every argument of one implicit-GEMM conv launch.
///
/// Pointers are device addresses of the first element of each view. A zero
/// `bias_ptr`, `prelu_ptr` or `residual_ptr` means that operand is absent.
/// Geometry is in NHWC terms and every value fits `i32`.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ConvImplicitGemmArgs {
    pub input_ptr: u64,
    pub weight_ptr: u64,
    pub bias_ptr: u64,
    pub prelu_ptr: u64,
    pub residual_ptr: u64,
    pub out_ptr: u64,
    pub batch: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub c_in: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub c_out: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride: (usize, usize),
    pub pad_top: usize,
    pub pad_left: usize,
    pub dilation: (usize, usize),
    pub input_strides: NhwcStrides,
    pub out_strides: NhwcStrides,
    pub residual_strides: NhwcStrides,
    pub activation: ConvEpilogueActivation,
    /// 1 or `c_out` when the activation is PReLU, else 0.
    pub prelu_len: usize,
    pub residual_scale: f32,
    /// Accumulate the K sum in F16. F16 inputs only.
    pub half_accumulate: bool,
}

/// Convert a geometry value to the kernel's `int`.
fn to_i32(value: usize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        Error::Internal(format!(
            "conv2d implicit GEMM: {what} = {value} exceeds the kernel's i32 range"
        ))
    })
}

/// Launch the implicit-GEMM conv2d for F16 or BF16.
///
/// The caller checks the routing conditions first (see
/// `ops/cuda/conv_fused_native.rs`). An empty output launches nothing.
///
/// # Safety
///
/// - Every non-zero pointer addresses a live device allocation of `dtype`
///   that holds every element its view reaches through the given strides.
/// - The weight is a contiguous `[c_out, kernel_h, kernel_w, c_in]` tensor.
/// - The bias holds `c_out` elements and the PReLU weight `prelu_len`.
/// - `out` repeats no element and shares none with a read operand.
pub(crate) unsafe fn launch_conv_implicit_gemm(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    args: &ConvImplicitGemmArgs,
) -> Result<()> {
    let m = args
        .batch
        .checked_mul(args.out_h)
        .and_then(|v| v.checked_mul(args.out_w))
        .ok_or_else(|| Error::Internal("conv2d implicit GEMM: N*H_out*W_out overflows".into()))?;
    let k = args
        .kernel_h
        .checked_mul(args.kernel_w)
        .and_then(|v| v.checked_mul(args.c_in))
        .ok_or_else(|| Error::Internal("conv2d implicit GEMM: Kh*Kw*C_in overflows".into()))?;
    if m == 0 || args.c_out == 0 {
        return Ok(());
    }
    if args.half_accumulate && dtype != DType::F16 {
        return Err(Error::Internal(format!(
            "conv2d implicit GEMM: half_accumulate is set for {dtype:?}, but only F16 has \
             F16-accumulator kernels. Set half_accumulate only for F16 inputs"
        )));
    }
    // The kernel divides by C_in and Kw to place a K index.
    if k == 0 {
        return Err(Error::Internal(
            "conv2d implicit GEMM: Kh*Kw*C_in is 0; route this shape to the generic path".into(),
        ));
    }
    if args.kernel_h > CONV_MAX_KERNEL_EXTENT || args.kernel_w > CONV_MAX_KERNEL_EXTENT {
        return Err(Error::Internal(format!(
            "conv2d implicit GEMM: kernel {}x{} exceeds the {CONV_MAX_KERNEL_EXTENT}x\
             {CONV_MAX_KERNEL_EXTENT} tap masks; route this shape to the generic path",
            args.kernel_h, args.kernel_w
        )));
    }

    let m_i32 = to_i32(m, "N*H_out*W_out")?;
    to_i32(k, "Kh*Kw*C_in")?;
    let in_h = to_i32(args.in_h, "H")?;
    let in_w = to_i32(args.in_w, "W")?;
    let c_in = to_i32(args.c_in, "C_in")?;
    let out_h = to_i32(args.out_h, "H_out")?;
    let out_w = to_i32(args.out_w, "W_out")?;
    let c_out = to_i32(args.c_out, "C_out")?;
    let kernel_h = to_i32(args.kernel_h, "Kh")?;
    let kernel_w = to_i32(args.kernel_w, "Kw")?;
    let stride_h = to_i32(args.stride.0, "stride_h")?;
    let stride_w = to_i32(args.stride.1, "stride_w")?;
    let pad_top = to_i32(args.pad_top, "pad_top")?;
    let pad_left = to_i32(args.pad_left, "pad_left")?;
    let dil_h = to_i32(args.dilation.0, "dilation_h")?;
    let dil_w = to_i32(args.dilation.1, "dilation_w")?;
    let prelu_len = to_i32(args.prelu_len, "prelu_len")?;
    let (act, slope) = args.activation.code_and_slope();
    let [in_sn, in_sh, in_sw] = args.input_strides;
    let [out_sn, out_sh, out_sw] = args.out_strides;
    let [res_sn, res_sh, res_sw] = args.residual_strides;

    let tile = select_conv_tile(args.c_out, args.half_accumulate);
    let window = ConvWindow {
        batch: args.batch,
        out_h: args.out_h,
        out_w: args.out_w,
        kernel: (args.kernel_h, args.kernel_w),
        stride: args.stride,
        dilation: args.dilation,
    };
    let staging = select_conv_staging(
        args.c_in,
        args.input_ptr,
        args.input_strides,
        args.weight_ptr,
        tile,
        &window,
    );
    let cfg = conv_launch_config(&window, args.c_out, tile, staging).ok_or_else(|| {
        Error::Internal(format!(
            "conv2d implicit GEMM: grid for M = {m}, C_out = {} exceeds the CUDA grid limit; \
             route this shape to the generic path",
            args.c_out
        ))
    })?;
    let module = get_or_load_module(
        context,
        device_index,
        kernel_names::CONV_IMPLICIT_GEMM_MODULE,
    )?;
    let name = conv_kernel_name(dtype, args.half_accumulate, tile, staging);
    let func = get_kernel_function(&module, &name)?;

    // SAFETY: the argument list matches DEFINE_CONV_IGEMM in
    // conv_implicit_gemm.cu in order and width. The caller guarantees the
    // pointers and strides address valid memory (see `# Safety`).
    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&args.input_ptr);
        builder.arg(&args.weight_ptr);
        builder.arg(&args.bias_ptr);
        builder.arg(&args.prelu_ptr);
        builder.arg(&args.residual_ptr);
        builder.arg(&args.out_ptr);
        builder.arg(&m_i32);
        builder.arg(&in_h);
        builder.arg(&in_w);
        builder.arg(&c_in);
        builder.arg(&out_h);
        builder.arg(&out_w);
        builder.arg(&c_out);
        builder.arg(&kernel_h);
        builder.arg(&kernel_w);
        builder.arg(&stride_h);
        builder.arg(&stride_w);
        builder.arg(&pad_top);
        builder.arg(&pad_left);
        builder.arg(&dil_h);
        builder.arg(&dil_w);
        builder.arg(&in_sn);
        builder.arg(&in_sh);
        builder.arg(&in_sw);
        builder.arg(&out_sn);
        builder.arg(&out_sh);
        builder.arg(&out_sw);
        builder.arg(&res_sn);
        builder.arg(&res_sh);
        builder.arg(&res_sw);
        builder.arg(&act);
        builder.arg(&slope);
        builder.arg(&prelu_len);
        builder.arg(&args.residual_scale);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA conv2d implicit GEMM: launch of {name} failed: {e:?}; check the \
                 device and driver state"
            ))
        })?;
    }

    Ok(())
}
