//! Indexing CUDA kernel launchers
//!
//! Provides launchers for indexing operations: gather, scatter, index_select,
//! masked_select, and masked_fill.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Module name for indexing operations
pub const INDEX_MODULE: &str = "index";

// ============================================================================
// Gather
// ============================================================================

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

// ============================================================================
// Scatter
// ============================================================================

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

// ============================================================================
// Index Select
// ============================================================================

/// Launch index_select kernel.
///
/// Selects elements along a dimension using a 1D index tensor.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - indices must be a 1D tensor of i64 values
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_index_select(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    indices_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
    index_len: usize,
) -> Result<()> {
    let total = outer_size * index_len * inner_size;
    if total == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = kernel_name("index_select", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let outer_u32 = outer_size as u32;
        let dim_u32 = dim_size as u32;
        let inner_u32 = inner_size as u32;
        let index_len_u32 = index_len as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&indices_ptr);
        builder.arg(&output_ptr);
        builder.arg(&outer_u32);
        builder.arg(&dim_u32);
        builder.arg(&inner_u32);
        builder.arg(&index_len_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA index_select kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

// ============================================================================
// Masked Select
// ============================================================================

/// Launch masked_count kernel to count true elements in mask.
///
/// # Safety
///
/// - mask_ptr must be valid device memory with n u8 elements
/// - count_ptr must be valid device memory with 1 u32 element (initialized to 0)
pub unsafe fn launch_masked_count(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    mask_ptr: u64,
    count_ptr: u64,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, "masked_count_kernel")?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&mask_ptr);
        builder.arg(&count_ptr);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA masked_count kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch masked_prefix_sum kernel to compute prefix sum of mask.
///
/// This is a simple sequential kernel for small tensors. For large tensors,
/// a parallel scan algorithm should be used instead.
///
/// # Safety
///
/// - mask_ptr must be valid device memory with n u8 elements
/// - prefix_sum_ptr must be valid device memory with n u32 elements
pub unsafe fn launch_masked_prefix_sum(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    mask_ptr: u64,
    prefix_sum_ptr: u64,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, "masked_prefix_sum_kernel")?;

        // This kernel uses a single thread
        let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&mask_ptr);
        builder.arg(&prefix_sum_ptr);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA masked_prefix_sum kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch masked_select kernel.
///
/// Selects elements from input where mask is true, using precomputed prefix sum.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - prefix_sum must be precomputed via launch_masked_prefix_sum
/// - output must have space for at least count_true elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_masked_select(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    mask_ptr: u64,
    output_ptr: u64,
    prefix_sum_ptr: u64,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = kernel_name("masked_select", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&mask_ptr);
        builder.arg(&output_ptr);
        builder.arg(&prefix_sum_ptr);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA masked_select kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

// ============================================================================
// Masked Fill
// ============================================================================

/// Launch masked_fill kernel.
///
/// Fills elements where mask is true with a scalar value.
/// Dispatches to the appropriate dtype-specific kernel.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - input and output must have n elements
pub unsafe fn launch_masked_fill(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    mask_ptr: u64,
    output_ptr: u64,
    fill_value: f64,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    let kernel_name = match dtype {
        DType::F32 => "masked_fill_f32",
        DType::F64 => "masked_fill_f64",
        DType::I32 => "masked_fill_i32",
        DType::I64 => "masked_fill_i64",
        #[cfg(feature = "f16")]
        DType::F16 => "masked_fill_f16",
        #[cfg(feature = "f16")]
        DType::BF16 => "masked_fill_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "masked_fill",
            });
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&mask_ptr);
        builder.arg(&output_ptr);

        // Pass fill_value with appropriate type conversion
        match dtype {
            DType::F32 => {
                let v = fill_value as f32;
                builder.arg(&v);
            }
            DType::F64 => {
                builder.arg(&fill_value);
            }
            DType::I32 => {
                let v = fill_value as i32;
                builder.arg(&v);
            }
            DType::I64 => {
                let v = fill_value as i64;
                builder.arg(&v);
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                let v = half::f16::from_f64(fill_value).to_bits();
                builder.arg(&v);
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                let v = half::bf16::from_f64(fill_value).to_bits();
                builder.arg(&v);
            }
            _ => unreachable!(), // Already handled above
        }

        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA masked_fill kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}
