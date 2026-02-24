//! Softmax CUDA kernel launchers (forward + backward)
//!
//! Kernel source: softmax.cu

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::loader::{
    get_kernel_function, get_or_load_module, kernel_name, kernel_names, launch_config,
    softmax_launch_config,
};

/// Launch softmax over the last dimension.
///
/// Uses shared memory for parallel reduction of max and sum values.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `input_ptr` must have `outer_size * dim_size` elements
/// - `output_ptr` must have `outer_size * dim_size` elements
pub unsafe fn launch_softmax(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    dim_size: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SOFTMAX_MODULE)?;
        let func_name = kernel_name("softmax", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid_size, block_size, shared_mem) = softmax_launch_config(outer_size, dim_size);
        let outer = outer_size as u32;
        let dim = dim_size as u32;

        let shared_mem = if dtype == DType::F64 {
            shared_mem * 2
        } else {
            shared_mem
        };

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), shared_mem);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&outer);
        builder.arg(&dim);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA softmax kernel launch failed: {:?}", e)))?;

        Ok(())
    }
}

/// Launch softmax over a non-last dimension.
///
/// For shape `[A, B, C]` with softmax over dim=1: outer=A, dim=B, inner=C.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Tensors must have `outer_size * dim_size * inner_size` elements
pub unsafe fn launch_softmax_dim(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SOFTMAX_MODULE)?;
        let func_name = kernel_name("softmax_dim", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = (outer_size as u32, inner_size as u32, 1);
        let outer = outer_size as u32;
        let dim = dim_size as u32;
        let inner = inner_size as u32;

        let cfg = launch_config(grid, (1, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&outer);
        builder.arg(&dim);
        builder.arg(&inner);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA softmax_dim kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch softmax backward kernel (last dimension).
///
/// Computes: d_input = output * (grad - sum(grad * output))
///
/// # Safety
/// - All pointers must be valid device memory of `outer_size * dim_size` elements
pub unsafe fn launch_softmax_bwd(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    grad_ptr: u64,
    output_ptr: u64,
    d_input_ptr: u64,
    outer_size: usize,
    dim_size: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SOFTMAX_MODULE)?;
        let func_name = kernel_name("softmax_bwd", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid_size, block_size, shared_mem) = softmax_launch_config(outer_size, dim_size);
        let outer = outer_size as u32;
        let dim = dim_size as u32;

        let shared_mem = if dtype == DType::F64 {
            shared_mem * 2
        } else {
            shared_mem
        };

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), shared_mem);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&grad_ptr);
        builder.arg(&output_ptr);
        builder.arg(&d_input_ptr);
        builder.arg(&outer);
        builder.arg(&dim);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA softmax_bwd kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch softmax backward kernel (non-last dimension).
///
/// # Safety
/// - All pointers must be valid device memory
pub unsafe fn launch_softmax_bwd_dim(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    grad_ptr: u64,
    output_ptr: u64,
    d_input_ptr: u64,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SOFTMAX_MODULE)?;
        let func_name = kernel_name("softmax_bwd_dim", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = (outer_size as u32, inner_size as u32, 1);
        let outer = outer_size as u32;
        let dim = dim_size as u32;
        let inner = inner_size as u32;

        let cfg = launch_config(grid, (1, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&grad_ptr);
        builder.arg(&output_ptr);
        builder.arg(&d_input_ptr);
        builder.arg(&outer);
        builder.arg(&dim);
        builder.arg(&inner);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA softmax_bwd_dim kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}
