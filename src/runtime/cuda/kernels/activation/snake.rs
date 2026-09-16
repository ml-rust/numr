//! Snake activation CUDA kernel launchers (forward, `d_x`, per-channel params).
//!
//! Kernel source: snake.cu

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::loader::{
    BLOCK_SIZE, MAX_GRID_DIM_X, elementwise_launch_config, get_kernel_function, get_or_load_module,
    kernel_name, kernel_names, launch_config,
};

/// Threads per block in `snake_beta_dparams_*`. Must match `SNAKE_BLOCK` in
/// snake.cu, which sizes the shared-memory reduction buffers.
const SNAKE_DPARAMS_BLOCK: u32 = 256;

/// The `[outer, channels, inner]` geometry of one launch.
#[derive(Clone, Copy)]
pub struct SnakeLaunchDims {
    /// Product of the dims before the channel axis.
    pub outer: usize,
    /// Length of the channel axis.
    pub channels: usize,
    /// Product of the dims after the channel axis.
    pub inner: usize,
}

impl SnakeLaunchDims {
    fn numel(&self) -> usize {
        self.outer * self.channels * self.inner
    }

    /// The kernels take `n`, `channels` and `inner` as `unsigned int`.
    fn as_u32(&self) -> Result<(u32, u32, u32, u32)> {
        let check = |name: &'static str, v: usize| -> Result<u32> {
            u32::try_from(v).map_err(|_| Error::InvalidArgument {
                arg: name,
                reason: format!("{name} = {v} exceeds the u32 extent the snake kernels index with"),
            })
        };
        Ok((
            check("numel", self.numel())?,
            check("outer", self.outer)?,
            check("channels", self.channels)?,
            check("inner", self.inner)?,
        ))
    }
}

fn dtype_supported(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 | DType::F16 | DType::BF16 | DType::FP8E4M3 | DType::FP8E5M2 => {
            Ok(())
        }
        other => Err(Error::UnsupportedDType { dtype: other, op }),
    }
}

/// Launch `snake_beta_<dtype>`: `out = x + sin(alpha * x)^2 / (beta + eps)`.
///
/// # Safety
///
/// - `x_ptr` and `out_ptr` hold `dims.numel()` elements of `dtype`.
/// - `alpha_ptr` and `beta_ptr` hold `dims.channels` elements of `dtype`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_snake_beta(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    x_ptr: u64,
    alpha_ptr: u64,
    beta_ptr: u64,
    out_ptr: u64,
    dims: SnakeLaunchDims,
    eps: f64,
) -> Result<()> {
    dtype_supported(dtype, "snake_beta")?;
    let (n, _outer, channels, inner) = dims.as_u32()?;
    if n == 0 {
        return Ok(());
    }
    let module = get_or_load_module(context, device_index, kernel_names::SNAKE_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name("snake_beta", dtype))?;
    let grid = elementwise_launch_config(dims.numel())?;
    let cfg = launch_config(grid, (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    unsafe {
        builder.arg(&x_ptr);
        builder.arg(&alpha_ptr);
        builder.arg(&beta_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&channels);
        builder.arg(&inner);
        builder.arg(&eps);
        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA snake_beta kernel launch failed: {e:?}")))?;
    }
    Ok(())
}

/// Launch `snake_beta_dx_<dtype>`: `d_x = grad * (1 + alpha * sin(2 alpha x) / (beta + eps))`.
///
/// # Safety
///
/// - `grad_ptr`, `x_ptr` and `d_x_ptr` hold `dims.numel()` elements of `dtype`.
/// - `alpha_ptr` and `beta_ptr` hold `dims.channels` elements of `dtype`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_snake_beta_dx(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    grad_ptr: u64,
    x_ptr: u64,
    alpha_ptr: u64,
    beta_ptr: u64,
    d_x_ptr: u64,
    dims: SnakeLaunchDims,
    eps: f64,
) -> Result<()> {
    dtype_supported(dtype, "snake_beta_bwd")?;
    let (n, _outer, channels, inner) = dims.as_u32()?;
    if n == 0 {
        return Ok(());
    }
    let module = get_or_load_module(context, device_index, kernel_names::SNAKE_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name("snake_beta_dx", dtype))?;
    let grid = elementwise_launch_config(dims.numel())?;
    let cfg = launch_config(grid, (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    unsafe {
        builder.arg(&grad_ptr);
        builder.arg(&x_ptr);
        builder.arg(&alpha_ptr);
        builder.arg(&beta_ptr);
        builder.arg(&d_x_ptr);
        builder.arg(&n);
        builder.arg(&channels);
        builder.arg(&inner);
        builder.arg(&eps);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA snake_beta_dx kernel launch failed: {e:?}"))
        })?;
    }
    Ok(())
}

/// Launch `snake_beta_dparams_<dtype>`: one block per channel, writing the
/// per-channel `d_alpha` and `d_beta` sums.
///
/// # Safety
///
/// - `grad_ptr` and `x_ptr` hold `dims.numel()` elements of `dtype`.
/// - `alpha_ptr`, `beta_ptr`, `d_alpha_ptr` and `d_beta_ptr` hold
///   `dims.channels` elements of `dtype`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_snake_beta_dparams(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    grad_ptr: u64,
    x_ptr: u64,
    alpha_ptr: u64,
    beta_ptr: u64,
    d_alpha_ptr: u64,
    d_beta_ptr: u64,
    dims: SnakeLaunchDims,
    eps: f64,
) -> Result<()> {
    dtype_supported(dtype, "snake_beta_bwd")?;
    let (_n, outer, channels, inner) = dims.as_u32()?;
    if channels == 0 {
        return Ok(());
    }
    if channels > MAX_GRID_DIM_X {
        return Err(Error::InvalidArgument {
            arg: "channels",
            reason: format!(
                "{channels} channels need one block each, exceeding the CUDA max grid extent \
                 of {MAX_GRID_DIM_X}"
            ),
        });
    }
    let module = get_or_load_module(context, device_index, kernel_names::SNAKE_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name("snake_beta_dparams", dtype))?;
    let cfg = launch_config((channels, 1, 1), (SNAKE_DPARAMS_BLOCK, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    unsafe {
        builder.arg(&grad_ptr);
        builder.arg(&x_ptr);
        builder.arg(&alpha_ptr);
        builder.arg(&beta_ptr);
        builder.arg(&d_alpha_ptr);
        builder.arg(&d_beta_ptr);
        builder.arg(&outer);
        builder.arg(&channels);
        builder.arg(&inner);
        builder.arg(&eps);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA snake_beta_dparams kernel launch failed: {e:?}"
            ))
        })?;
    }
    Ok(())
}
