//! Scatter kernel launchers (scatter, copy, scatter_reduce)

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    launch_config,
};
use super::gather::INDEX_MODULE;
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch scatter kernel.
///
/// Scatters values from src to output at positions specified by indices.
/// `output[i][indices[i][j][k]][k] = src[i][j][k]` (when dim=1)
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Output must be pre-initialized (typically a copy of input)
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_scatter(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    indices_ptr: u64,
    src_ptr: u64,
    output_ptr: u64,
    ndim: usize,
    dim: usize,
    output_shape_ptr: u64,
    output_strides_ptr: u64,
    src_shape_ptr: u64,
    src_strides_ptr: u64,
    src_total: usize,
) -> Result<()> {
    if src_total == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = kernel_name("scatter", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(src_total);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let ndim_u32 = ndim as u32;
        let dim_u32 = dim as u32;
        let src_total_u32 = src_total as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&indices_ptr);
        builder.arg(&src_ptr);
        builder.arg(&output_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&dim_u32);
        builder.arg(&output_shape_ptr);
        builder.arg(&output_strides_ptr);
        builder.arg(&src_shape_ptr);
        builder.arg(&src_strides_ptr);
        builder.arg(&src_total_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA scatter kernel launch failed: {:?}", e)))?;

        Ok(())
    }
}

/// Launch copy kernel for scatter initialization.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - dst must have space for n elements
pub unsafe fn launch_copy(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    src_ptr: u64,
    dst_ptr: u64,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = kernel_name("copy", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&dst_ptr);
        builder.arg(&n_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA copy kernel launch failed: {:?}", e)))?;

        Ok(())
    }
}

/// Scatter reduce operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterReduceOpCuda {
    /// Sum reduction: accumulate values by addition.
    Sum,
    /// Max reduction: keep the maximum value.
    Max,
    /// Min reduction: keep the minimum value.
    Min,
    /// Product reduction: accumulate values by multiplication.
    Prod,
}

/// Launch scatter_reduce kernel.
///
/// Scatters values from src to dst at positions specified by indices with a
/// reduction operation.
///
/// # Safety
///
/// All pointers must be valid device memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_scatter_reduce(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    src_ptr: u64,
    indices_ptr: u64,
    dst_ptr: u64,
    dim: usize,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
    src_dim_size: usize,
    op: ScatterReduceOpCuda,
) -> Result<()> {
    let total = outer_size * src_dim_size * inner_size;
    if total == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;

        let func_name = match (dtype, op) {
            (DType::F32, ScatterReduceOpCuda::Sum) => "scatter_reduce_sum_f32",
            (DType::F32, ScatterReduceOpCuda::Max) => "scatter_reduce_max_f32",
            (DType::F32, ScatterReduceOpCuda::Min) => "scatter_reduce_min_f32",
            (DType::F32, ScatterReduceOpCuda::Prod) => "scatter_reduce_prod_f32",
            (DType::F64, ScatterReduceOpCuda::Sum) => "scatter_reduce_sum_f64",
            (DType::F64, ScatterReduceOpCuda::Max) => "scatter_reduce_max_f64",
            (DType::F64, ScatterReduceOpCuda::Min) => "scatter_reduce_min_f64",
            (DType::F64, ScatterReduceOpCuda::Prod) => "scatter_reduce_prod_f64",
            (DType::I32, ScatterReduceOpCuda::Sum) => "scatter_reduce_sum_i32",
            (DType::I32, ScatterReduceOpCuda::Max) => "scatter_reduce_max_i32",
            (DType::I32, ScatterReduceOpCuda::Min) => "scatter_reduce_min_i32",
            (DType::I32, ScatterReduceOpCuda::Prod) => "scatter_reduce_prod_i32",
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype,
                    op: "scatter_reduce",
                });
            }
        };

        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(total);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let dim_u32 = dim as u32;
        let outer_size_u32 = outer_size as u32;
        let dim_size_u32 = dim_size as u32;
        let inner_size_u32 = inner_size as u32;
        let src_dim_size_u32 = src_dim_size as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&indices_ptr);
        builder.arg(&dst_ptr);
        builder.arg(&dim_u32);
        builder.arg(&outer_size_u32);
        builder.arg(&dim_size_u32);
        builder.arg(&inner_size_u32);
        builder.arg(&src_dim_size_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA scatter_reduce kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch scatter_reduce_count kernel.
///
/// Atomically increments count buffer at scattered positions.
/// Used as part of scatter_reduce mean: sum / count.
///
/// # Safety
///
/// All pointers must be valid device memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_scatter_reduce_count(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    indices_ptr: u64,
    count_ptr: u64,
    dim: usize,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
    src_dim_size: usize,
) -> Result<()> {
    let total = outer_size * src_dim_size * inner_size;
    if total == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;

        let func_name = match dtype {
            DType::F32 => "scatter_reduce_count_f32",
            DType::F64 => "scatter_reduce_count_f64",
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype,
                    op: "scatter_reduce_count",
                });
            }
        };

        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(total);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let dim_u32 = dim as u32;
        let outer_size_u32 = outer_size as u32;
        let dim_size_u32 = dim_size as u32;
        let inner_size_u32 = inner_size as u32;
        let src_dim_size_u32 = src_dim_size as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&indices_ptr);
        builder.arg(&count_ptr);
        builder.arg(&dim_u32);
        builder.arg(&outer_size_u32);
        builder.arg(&dim_size_u32);
        builder.arg(&inner_size_u32);
        builder.arg(&src_dim_size_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA scatter_reduce_count kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch scatter_reduce_mean_div kernel.
///
/// Element-wise: output[i] = sum[i] / count[i].
/// If count[i] == 0, output[i] = 0.
///
/// # Safety
///
/// All pointers must be valid device memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_scatter_reduce_mean_div(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    sum_ptr: u64,
    count_ptr: u64,
    output_ptr: u64,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;

        let func_name = match dtype {
            DType::F32 => "scatter_reduce_mean_div_f32",
            DType::F64 => "scatter_reduce_mean_div_f64",
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype,
                    op: "scatter_reduce_mean_div",
                });
            }
        };

        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&sum_ptr);
        builder.arg(&count_ptr);
        builder.arg(&output_ptr);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA scatter_reduce_mean_div kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}
