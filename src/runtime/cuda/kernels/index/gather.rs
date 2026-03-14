//! Gather kernel launchers (gather, gather_nd, gather_2d)

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Module name for indexing operations
pub const INDEX_MODULE: &str = "index";

/// Launch gather kernel.
///
/// Gathers values from input along a dimension specified by indices.
/// `output[i][j][k] = input[i][indices[i][j][k]][k]` (when dim=1)
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Shape and stride arrays must be valid device memory with `ndim` u32 elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gather(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    indices_ptr: u64,
    output_ptr: u64,
    ndim: usize,
    dim: usize,
    input_shape_ptr: u64,
    input_strides_ptr: u64,
    output_shape_ptr: u64,
    output_strides_ptr: u64,
    total_elements: usize,
) -> Result<()> {
    if total_elements == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = kernel_name("gather", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total_elements);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let ndim_u32 = ndim as u32;
        let dim_u32 = dim as u32;
        let total_u32 = total_elements as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&indices_ptr);
        builder.arg(&output_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&dim_u32);
        builder.arg(&input_shape_ptr);
        builder.arg(&input_strides_ptr);
        builder.arg(&output_shape_ptr);
        builder.arg(&output_strides_ptr);
        builder.arg(&total_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA gather kernel launch failed: {:?}", e)))?;

        Ok(())
    }
}

/// Launch gather_nd kernel.
///
/// Gathers slices from input at positions specified by indices tensor.
///
/// # Safety
///
/// All pointers must be valid device memory with sufficient size.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gather_nd(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    indices_ptr: u64,
    output_ptr: u64,
    input_shape_ptr: u64,
    input_strides_ptr: u64,
    num_slices: usize,
    slice_size: usize,
    index_depth: usize,
    ndim: usize,
) -> Result<()> {
    let total = num_slices * slice_size;
    if total == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = kernel_name("gather_nd", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let num_slices_u32 = num_slices as u32;
        let slice_size_u32 = slice_size as u32;
        let index_depth_u32 = index_depth as u32;
        let ndim_u32 = ndim as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&indices_ptr);
        builder.arg(&output_ptr);
        builder.arg(&input_shape_ptr);
        builder.arg(&input_strides_ptr);
        builder.arg(&num_slices_u32);
        builder.arg(&slice_size_u32);
        builder.arg(&index_depth_u32);
        builder.arg(&ndim_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA gather_nd kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch gather_2d kernel.
///
/// Gathers elements from a 2D matrix at specific (row, col) positions.
/// For each index i: output[i] = input[rows[i], cols[i]]
///
/// # Safety
///
/// All pointers must be valid device memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gather_2d(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    rows_ptr: u64,
    cols_ptr: u64,
    output_ptr: u64,
    nrows: usize,
    ncols: usize,
    num_indices: usize,
) -> Result<()> {
    if num_indices == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = kernel_name("gather_2d", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(num_indices);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let nrows_u32 = nrows as u32;
        let ncols_u32 = ncols as u32;
        let num_indices_u32 = num_indices as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&rows_ptr);
        builder.arg(&cols_ptr);
        builder.arg(&output_ptr);
        builder.arg(&nrows_u32);
        builder.arg(&ncols_u32);
        builder.arg(&num_indices_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA gather_2d kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}
