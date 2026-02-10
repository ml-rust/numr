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

/// Puts values at specified indices along a dimension.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - indices must be a 1D tensor of i64 values
/// - output must already contain a copy of the input tensor
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_index_put(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    indices_ptr: u64,
    src_ptr: u64,
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
        let func_name = kernel_name("index_put", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let outer_u32 = outer_size as u32;
        let dim_u32 = dim_size as u32;
        let inner_u32 = inner_size as u32;
        let index_len_u32 = index_len as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&indices_ptr);
        builder.arg(&src_ptr);
        builder.arg(&output_ptr);
        builder.arg(&outer_u32);
        builder.arg(&dim_u32);
        builder.arg(&inner_u32);
        builder.arg(&index_len_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA index_put kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

// ============================================================================
// Index Bounds Validation
// ============================================================================

/// Launch index bounds validation kernel.
///
/// Validates that all indices are within bounds [0, dim_size).
/// Returns the count of out-of-bounds indices in error_count buffer.
///
/// # Safety
///
/// - indices_ptr must be valid device memory with index_len i64 elements
/// - error_count_ptr must be valid device memory with 1 u32 element (initialized to 0)
pub unsafe fn launch_validate_indices(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    indices_ptr: u64,
    error_count_ptr: u64,
    index_len: usize,
    dim_size: usize,
) -> Result<()> {
    if index_len == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, "validate_indices_kernel")?;

        let grid = elementwise_launch_config(index_len);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let index_len_u32 = index_len as u32;
        let dim_size_u32 = dim_size as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&indices_ptr);
        builder.arg(&error_count_ptr);
        builder.arg(&index_len_u32);
        builder.arg(&dim_size_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA validate_indices kernel launch failed: {:?}",
                e
            ))
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

        // Pre-convert fill_value to all possible types to avoid lifetime issues
        let fill_f32 = fill_value as f32;
        let fill_f64 = fill_value;
        let fill_i32 = fill_value as i32;
        let fill_i64 = fill_value as i64;
        #[cfg(feature = "f16")]
        let fill_f16 = half::f16::from_f64(fill_value).to_bits();
        #[cfg(feature = "f16")]
        let fill_bf16 = half::bf16::from_f64(fill_value).to_bits();

        // Pass fill_value with appropriate type
        match dtype {
            DType::F32 => builder.arg(&fill_f32),
            DType::F64 => builder.arg(&fill_f64),
            DType::I32 => builder.arg(&fill_i32),
            DType::I64 => builder.arg(&fill_i64),
            #[cfg(feature = "f16")]
            DType::F16 => builder.arg(&fill_f16),
            #[cfg(feature = "f16")]
            DType::BF16 => builder.arg(&fill_bf16),
            _ => unreachable!(), // Already handled above
        };

        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA masked_fill kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

// ============================================================================
// Broadcast Masked Operations
// ============================================================================

/// Launch broadcast masked_count kernel.
///
/// Counts true elements in mask when broadcast to output shape.
///
/// # Safety
///
/// - mask_ptr must be valid device memory
/// - count_ptr must be valid device memory with 1 u32 element (initialized to 0)
/// - mask_strides_ptr, out_shape_ptr must be valid device memory with ndim u32 elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_masked_count_broadcast(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    mask_ptr: u64,
    count_ptr: u64,
    mask_strides_ptr: u64,
    out_shape_ptr: u64,
    ndim: usize,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, "masked_count_broadcast_kernel")?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let ndim_u32 = ndim as u32;
        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&mask_ptr);
        builder.arg(&count_ptr);
        builder.arg(&mask_strides_ptr);
        builder.arg(&out_shape_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA masked_count_broadcast kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch broadcast masked_prefix_sum kernel.
///
/// Computes prefix sum of mask values when broadcast to output shape.
///
/// # Safety
///
/// - mask_ptr must be valid device memory
/// - prefix_sum_ptr must be valid device memory with n u32 elements
/// - mask_strides_ptr, out_shape_ptr must be valid device memory with ndim u32 elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_masked_prefix_sum_broadcast(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    mask_ptr: u64,
    prefix_sum_ptr: u64,
    mask_strides_ptr: u64,
    out_shape_ptr: u64,
    ndim: usize,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, "masked_prefix_sum_broadcast_kernel")?;

        // This kernel uses a single thread
        let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);

        let ndim_u32 = ndim as u32;
        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&mask_ptr);
        builder.arg(&prefix_sum_ptr);
        builder.arg(&mask_strides_ptr);
        builder.arg(&out_shape_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA masked_prefix_sum_broadcast kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch broadcast masked_select kernel.
///
/// Selects elements from input where broadcast mask is true.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - prefix_sum must be precomputed via launch_masked_prefix_sum_broadcast
/// - output must have space for at least count_true elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_masked_select_broadcast(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    mask_ptr: u64,
    output_ptr: u64,
    prefix_sum_ptr: u64,
    mask_strides_ptr: u64,
    out_shape_ptr: u64,
    ndim: usize,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = format!("masked_select_broadcast_{}", dtype_suffix(dtype)?);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let ndim_u32 = ndim as u32;
        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&mask_ptr);
        builder.arg(&output_ptr);
        builder.arg(&prefix_sum_ptr);
        builder.arg(&mask_strides_ptr);
        builder.arg(&out_shape_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA masked_select_broadcast kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch broadcast masked_fill kernel.
///
/// Fills elements where broadcast mask is true with a scalar value.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - input and output must have n elements
/// - mask_strides_ptr, out_shape_ptr must be valid device memory with ndim u32 elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_masked_fill_broadcast(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    mask_ptr: u64,
    output_ptr: u64,
    fill_value: f64,
    mask_strides_ptr: u64,
    out_shape_ptr: u64,
    ndim: usize,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    let kernel_name = match dtype {
        DType::F32 => "masked_fill_broadcast_f32",
        DType::F64 => "masked_fill_broadcast_f64",
        DType::I32 => "masked_fill_broadcast_i32",
        DType::I64 => "masked_fill_broadcast_i64",
        #[cfg(feature = "f16")]
        DType::F16 => "masked_fill_broadcast_f16",
        #[cfg(feature = "f16")]
        DType::BF16 => "masked_fill_broadcast_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "masked_fill_broadcast",
            });
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let ndim_u32 = ndim as u32;
        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&mask_ptr);
        builder.arg(&output_ptr);

        // Pre-convert fill_value to all possible types to avoid lifetime issues
        let fill_f32 = fill_value as f32;
        let fill_f64 = fill_value;
        let fill_i32 = fill_value as i32;
        let fill_i64 = fill_value as i64;
        #[cfg(feature = "f16")]
        let fill_f16 = half::f16::from_f64(fill_value).to_bits();
        #[cfg(feature = "f16")]
        let fill_bf16 = half::bf16::from_f64(fill_value).to_bits();

        // Pass fill_value with appropriate type
        match dtype {
            DType::F32 => builder.arg(&fill_f32),
            DType::F64 => builder.arg(&fill_f64),
            DType::I32 => builder.arg(&fill_i32),
            DType::I64 => builder.arg(&fill_i64),
            #[cfg(feature = "f16")]
            DType::F16 => builder.arg(&fill_f16),
            #[cfg(feature = "f16")]
            DType::BF16 => builder.arg(&fill_bf16),
            _ => unreachable!(), // Already handled above
        };

        builder.arg(&mask_strides_ptr);
        builder.arg(&out_shape_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA masked_fill_broadcast kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Helper to get dtype suffix for kernel name
fn dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F64 => Ok("f64"),
        DType::I32 => Ok("i32"),
        DType::I64 => Ok("i64"),
        #[cfg(feature = "f16")]
        DType::F16 => Ok("f16"),
        #[cfg(feature = "f16")]
        DType::BF16 => Ok("bf16"),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "masked_select_broadcast",
        }),
    }
}

// ============================================================================
// Embedding Lookup
// ============================================================================

/// Launch embedding_lookup kernel.
///
/// Looks up embeddings from an embedding table using indices.
/// This is the industry-standard embedding lookup operation used in neural networks.
///
/// # Algorithm
/// For each index i in [0, num_indices):
///   output[i, :] = embeddings[indices[i], :]
///
/// Output shape: [num_indices, embedding_dim]
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - embeddings must be 2D [vocab_size, embedding_dim]
/// - indices must contain values in [0, vocab_size)
/// - output must have space for num_indices * embedding_dim elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_embedding_lookup(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    embeddings_ptr: u64,
    indices_ptr: u64,
    output_ptr: u64,
    num_indices: usize,
    vocab_size: usize,
    embedding_dim: usize,
) -> Result<()> {
    if num_indices == 0 || embedding_dim == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = kernel_name("embedding_lookup", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        // Each thread handles one embedding lookup (one index)
        // More efficient than one thread per element because we copy contiguous rows
        let grid = elementwise_launch_config(num_indices);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let num_indices_u32 = num_indices as u32;
        let vocab_size_u32 = vocab_size as u32;
        let embedding_dim_u32 = embedding_dim as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&embeddings_ptr);
        builder.arg(&indices_ptr);
        builder.arg(&output_ptr);
        builder.arg(&num_indices_u32);
        builder.arg(&vocab_size_u32);
        builder.arg(&embedding_dim_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA embedding_lookup kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

// ============================================================================
// Gather ND
// ============================================================================

/// Launch gather_nd kernel.
///
/// Gathers slices from input at positions specified by indices tensor.
///
/// # Arguments
///
/// * `input_ptr` - Input tensor data
/// * `indices_ptr` - Indices tensor (num_slices, index_depth)
/// * `output_ptr` - Output tensor (num_slices, remaining_dims...)
/// * `input_shape_ptr` - Device pointer to input shape array
/// * `input_strides_ptr` - Device pointer to input strides array
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

// ============================================================================
// Bincount
// ============================================================================

/// Launch bincount kernel.
///
/// Counts occurrences of each value in an integer tensor, optionally with weights.
///
/// # Arguments
///
/// * `input_ptr` - Input tensor of non-negative integers (i32 or i64)
/// * `weights_ptr` - Optional weights tensor
/// * `output_ptr` - Output tensor (initialized to zeros)
/// * `n` - Number of elements in input
/// * `minlength` - Length of output tensor
///
/// # Safety
///
/// All pointers must be valid device memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_bincount_weighted(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    input_dtype: DType,
    weights_dtype: Option<DType>,
    input_ptr: u64,
    weights_ptr: Option<u64>,
    output_ptr: u64,
    n: usize,
    minlength: usize,
) -> Result<()> {
    if n == 0 || minlength == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;

        let func_name = match (input_dtype, weights_ptr, weights_dtype) {
            (DType::I32, None, _) => "bincount_i32",
            (DType::I64, None, _) => "bincount_i64",
            (DType::I32, Some(_), Some(DType::F32)) => "bincount_weighted_f32",
            (DType::I32, Some(_), Some(DType::F64)) => "bincount_weighted_f64",
            (DType::I64, Some(_), Some(DType::F32)) => "bincount_i64_weighted_f32",
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "dtype",
                    reason: format!("bincount requires i32/i64 input, got {:?}", input_dtype),
                });
            }
        };

        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;
        let minlength_u32 = minlength as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);

        // Store weights_ptr value outside the if block to extend its lifetime
        let weights_ptr_val = weights_ptr.unwrap_or(0);
        if weights_ptr.is_some() {
            builder.arg(&weights_ptr_val);
        }

        builder.arg(&output_ptr);
        builder.arg(&n_u32);
        builder.arg(&minlength_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA bincount kernel launch failed: {:?}", e)))?;

        Ok(())
    }
}

// ============================================================================
// Scatter Reduce
// ============================================================================

/// Scatter reduce operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterReduceOpCuda {
    Sum,
    Max,
    Min,
    Prod,
}

/// Launch scatter_reduce kernel.
///
/// Scatters values from src to dst at positions specified by indices with a
/// reduction operation.
///
/// # Arguments
///
/// * `src_ptr` - Source tensor data
/// * `indices_ptr` - Indices tensor (1D)
/// * `dst_ptr` - Destination tensor (must be pre-initialized with appropriate values)
/// * `op` - Reduction operation (sum, max, min)
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

// ============================================================================
// Scatter Reduce Count (for mean)
// ============================================================================

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

// ============================================================================
// Scatter Reduce Mean Divide
// ============================================================================

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

// ============================================================================
// Gather 2D
// ============================================================================

/// Launch gather_2d kernel.
///
/// Gathers elements from a 2D matrix at specific (row, col) positions.
/// For each index i: output[i] = input[rows[i], cols[i]]
///
/// # Arguments
///
/// * `input_ptr` - 2D input tensor data (row-major)
/// * `rows_ptr` - 1D row indices tensor (i64)
/// * `cols_ptr` - 1D column indices tensor (i64)
/// * `output_ptr` - 1D output tensor
/// * `nrows` - Number of rows in input
/// * `ncols` - Number of columns in input
/// * `num_indices` - Number of (row, col) pairs to gather
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
