//! CUDA kernel launchers for sorting and search operations

use super::loader::{
    BLOCK_SIZE, dtype_suffix, elementwise_launch_config, get_kernel_function, get_or_load_module,
    kernel_name, launch_config,
};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Module name for sort kernels
pub const SORT_MODULE: &str = "sort";

/// Calculate shared memory size for sort operations
fn sort_shared_mem_size(sort_size: usize, elem_size: usize) -> u32 {
    // Need space for values and indices
    // Pad to next power of 2 for bitonic sort
    let n = sort_size.next_power_of_two();
    ((n * elem_size) + (n * 8)) as u32 // values + i64 indices
}

/// Launch sort kernel with indices
pub unsafe fn launch_sort(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    indices_ptr: u64,
    outer_size: usize,
    sort_size: usize,
    inner_size: usize,
    descending: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = kernel_name("sort", dtype);
    let func = get_kernel_function(&module, &kname)?;

    let elem_size = dtype.size_in_bytes();
    let shared_mem = sort_shared_mem_size(sort_size, elem_size);

    // 2D grid: (outer, inner)
    let grid = (outer_size as u32, inner_size as u32, 1);
    let block = (BLOCK_SIZE.min(sort_size as u32).max(1), 1, 1);

    let cfg = launch_config(grid, block, shared_mem);

    let outer_u32 = outer_size as u32;
    let sort_u32 = sort_size as u32;
    let inner_u32 = inner_size as u32;
    let desc_u32 = descending as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&indices_ptr);
    builder.arg(&outer_u32);
    builder.arg(&sort_u32);
    builder.arg(&inner_u32);
    builder.arg(&desc_u32);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("CUDA sort kernel launch failed: {:?}", e)))?;

    Ok(())
}

/// Launch sort kernel (values only, no indices)
pub unsafe fn launch_sort_values_only(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    sort_size: usize,
    inner_size: usize,
    descending: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = format!("sort_values_only_{}", dtype_suffix(dtype));
    let func = get_kernel_function(&module, &kname)?;

    let elem_size = dtype.size_in_bytes();
    let shared_mem = sort_shared_mem_size(sort_size, elem_size);

    let grid = (outer_size as u32, inner_size as u32, 1);
    let block = (BLOCK_SIZE.min(sort_size as u32).max(1), 1, 1);

    let cfg = launch_config(grid, block, shared_mem);

    let outer_u32 = outer_size as u32;
    let sort_u32 = sort_size as u32;
    let inner_u32 = inner_size as u32;
    let desc_u32 = descending as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&outer_u32);
    builder.arg(&sort_u32);
    builder.arg(&inner_u32);
    builder.arg(&desc_u32);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!(
            "CUDA sort_values_only kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

/// Launch argsort kernel (indices only, no values)
pub unsafe fn launch_argsort(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    indices_ptr: u64,
    outer_size: usize,
    sort_size: usize,
    inner_size: usize,
    descending: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = kernel_name("argsort", dtype);
    let func = get_kernel_function(&module, &kname)?;

    let elem_size = dtype.size_in_bytes();
    let shared_mem = sort_shared_mem_size(sort_size, elem_size);

    let grid = (outer_size as u32, inner_size as u32, 1);
    let block = (BLOCK_SIZE.min(sort_size as u32).max(1), 1, 1);

    let cfg = launch_config(grid, block, shared_mem);

    let outer_u32 = outer_size as u32;
    let sort_u32 = sort_size as u32;
    let inner_u32 = inner_size as u32;
    let desc_u32 = descending as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&indices_ptr);
    builder.arg(&outer_u32);
    builder.arg(&sort_u32);
    builder.arg(&inner_u32);
    builder.arg(&desc_u32);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("CUDA argsort kernel launch failed: {:?}", e)))?;

    Ok(())
}

/// Launch topk kernel
pub unsafe fn launch_topk(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    values_ptr: u64,
    indices_ptr: u64,
    outer_size: usize,
    sort_size: usize,
    inner_size: usize,
    k: usize,
    largest: bool,
    sorted: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = kernel_name("topk", dtype);
    let func = get_kernel_function(&module, &kname)?;

    let elem_size = dtype.size_in_bytes();
    let shared_mem = sort_shared_mem_size(sort_size, elem_size);

    let grid = (outer_size as u32, inner_size as u32, 1);
    let block = (BLOCK_SIZE.min(sort_size as u32).max(1), 1, 1);

    let cfg = launch_config(grid, block, shared_mem);

    let outer_u32 = outer_size as u32;
    let sort_u32 = sort_size as u32;
    let inner_u32 = inner_size as u32;
    let k_u32 = k as u32;
    let largest_u32 = largest as u32;
    let sorted_u32 = sorted as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&values_ptr);
    builder.arg(&indices_ptr);
    builder.arg(&outer_u32);
    builder.arg(&sort_u32);
    builder.arg(&inner_u32);
    builder.arg(&k_u32);
    builder.arg(&largest_u32);
    builder.arg(&sorted_u32);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("CUDA topk kernel launch failed: {:?}", e)))?;

    Ok(())
}

/// Launch count_nonzero kernel
pub unsafe fn launch_count_nonzero(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    count_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = kernel_name("count_nonzero", dtype);
    let func = get_kernel_function(&module, &kname)?;

    let (grid_size, _, _) = elementwise_launch_config(numel);
    let grid = (grid_size.min(256), 1, 1); // Limit grid size for atomic efficiency
    let block = (BLOCK_SIZE, 1, 1);

    let cfg = launch_config(grid, block, 0);
    let n = numel as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&count_ptr);
    builder.arg(&n);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!("CUDA count_nonzero kernel launch failed: {:?}", e))
    })?;

    Ok(())
}

/// Launch gather_nonzero kernel
pub unsafe fn launch_gather_nonzero(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    indices_ptr: u64,
    counter_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = kernel_name("gather_nonzero", dtype);
    let func = get_kernel_function(&module, &kname)?;

    let (grid_size, _, _) = elementwise_launch_config(numel);
    let grid = (grid_size.min(256), 1, 1);
    let block = (BLOCK_SIZE, 1, 1);

    let cfg = launch_config(grid, block, 0);
    let n = numel as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&indices_ptr);
    builder.arg(&counter_ptr);
    builder.arg(&n);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!("CUDA gather_nonzero kernel launch failed: {:?}", e))
    })?;

    Ok(())
}

/// Launch flat_to_multi_index kernel
pub unsafe fn launch_flat_to_multi_index(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    flat_indices_ptr: u64,
    multi_indices_ptr: u64,
    nnz: usize,
    ndim: usize,
    shape_ptr: u64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let func = get_kernel_function(&module, "flat_to_multi_index")?;

    let (grid_size, _, _) = elementwise_launch_config(nnz);
    let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let nnz_u32 = nnz as u32;
    let ndim_u32 = ndim as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&flat_indices_ptr);
    builder.arg(&multi_indices_ptr);
    builder.arg(&nnz_u32);
    builder.arg(&ndim_u32);
    builder.arg(&shape_ptr);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!(
            "CUDA flat_to_multi_index kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

/// Launch searchsorted kernel
pub unsafe fn launch_searchsorted(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    seq_ptr: u64,
    values_ptr: u64,
    output_ptr: u64,
    seq_len: usize,
    num_values: usize,
    right: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = kernel_name("searchsorted", dtype);
    let func = get_kernel_function(&module, &kname)?;

    let (grid_size, _, _) = elementwise_launch_config(num_values);
    let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let seq_len_u32 = seq_len as u32;
    let num_values_u32 = num_values as u32;
    let right_u32 = right as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&seq_ptr);
    builder.arg(&values_ptr);
    builder.arg(&output_ptr);
    builder.arg(&seq_len_u32);
    builder.arg(&num_values_u32);
    builder.arg(&right_u32);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("CUDA searchsorted kernel launch failed: {:?}", e)))?;

    Ok(())
}

/// Launch count_unique kernel
pub unsafe fn launch_count_unique(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    sorted_input_ptr: u64,
    count_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = kernel_name("count_unique", dtype);
    let func = get_kernel_function(&module, &kname)?;

    let (grid_size, _, _) = elementwise_launch_config(numel);
    let grid = (grid_size.min(256), 1, 1);
    let cfg = launch_config(grid, (BLOCK_SIZE, 1, 1), 0);
    let n = numel as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&sorted_input_ptr);
    builder.arg(&count_ptr);
    builder.arg(&n);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("CUDA count_unique kernel launch failed: {:?}", e)))?;

    Ok(())
}

/// Launch extract_unique kernel
pub unsafe fn launch_extract_unique(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    sorted_input_ptr: u64,
    output_ptr: u64,
    counter_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = kernel_name("extract_unique", dtype);
    let func = get_kernel_function(&module, &kname)?;

    let (grid_size, _, _) = elementwise_launch_config(numel);
    let grid = (grid_size.min(256), 1, 1);
    let cfg = launch_config(grid, (BLOCK_SIZE, 1, 1), 0);
    let n = numel as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&sorted_input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&counter_ptr);
    builder.arg(&n);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!("CUDA extract_unique kernel launch failed: {:?}", e))
    })?;

    Ok(())
}

/// Launch bincount kernel - counts occurrences of each index
pub unsafe fn launch_bincount(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    indices_ptr: u64,
    counts_ptr: u64,
    numel: usize,
    num_bins: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let func = get_kernel_function(&module, "bincount")?;

    let (grid_size, _, _) = elementwise_launch_config(numel);
    let grid = (grid_size.min(256), 1, 1);
    let cfg = launch_config(grid, (BLOCK_SIZE, 1, 1), 0);

    let n = numel as u32;
    let bins = num_bins as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&indices_ptr);
    builder.arg(&counts_ptr);
    builder.arg(&n);
    builder.arg(&bins);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("CUDA bincount kernel launch failed: {:?}", e)))?;

    Ok(())
}
