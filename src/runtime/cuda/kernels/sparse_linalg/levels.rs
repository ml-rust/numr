//! GPU-native level computation and structural analysis kernel launchers
//!
//! These kernels eliminate GPU↔CPU transfers for:
//! - i64 → i32 casting
//! - Level computation (iterative BFS)
//! - Max reduction
//! - Histogram and scatter operations
//! - LU/IC structure analysis

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::{
    BLOCK_SIZE, get_kernel_function, get_or_load_module, grid_size, launch_config, launch_error,
};

/// Module name for sparse level computation kernels
const SPARSE_LEVELS_MODULE: &str = "sparse_levels";
use crate::error::Result;

// ============================================================================
// Type Casting
// ============================================================================

/// Cast i64 GPU tensor to i32 GPU tensor (no CPU transfer)
///
/// # Safety
///
/// - `input` and `output` must be valid device memory pointers on the device associated with
///   `context`, each with at least `n` elements of their respective types.
/// - Values in `input` that exceed `i32::MAX` or are below `i32::MIN` will be truncated.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn launch_cast_i64_to_i32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    input: u64,
    output: u64,
    n: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LEVELS_MODULE)?;
    let func = get_kernel_function(&module, "cast_i64_to_i32")?;
    let cfg = launch_config((grid_size(n as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input);
    builder.arg(&output);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("cast_i64_to_i32", e))?;
    Ok(())
}

// ============================================================================
// Level Computation
// ============================================================================

/// Compute level schedule for lower triangular matrix via iterative BFS on GPU
///
/// # Safety
///
/// - `row_ptrs`, `col_indices`, `levels`, and `changed` must be valid device memory pointers on
///   the device associated with `context`.
/// - `row_ptrs` must have at least `n + 1` i32 elements; `col_indices` has `nnz` elements.
/// - `levels` must have at least `n` i32 elements (initialized by caller before first call).
/// - `changed` must point to a single i32 flag in device memory.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn launch_compute_levels_lower_iter(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs: u64,
    col_indices: u64,
    levels: u64,
    changed: u64,
    n: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LEVELS_MODULE)?;
    let func = get_kernel_function(&module, "compute_levels_lower_iter")?;
    let cfg = launch_config((grid_size(n as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&row_ptrs);
    builder.arg(&col_indices);
    builder.arg(&levels);
    builder.arg(&changed);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("compute_levels_lower_iter", e))?;
    Ok(())
}

/// Compute level schedule for upper triangular matrix via iterative BFS on GPU
///
/// # Safety
///
/// - `row_ptrs`, `col_indices`, `levels`, and `changed` must be valid device memory pointers on
///   the device associated with `context`.
/// - `row_ptrs` must have at least `n + 1` i32 elements; `col_indices` has `nnz` elements.
/// - `levels` must have at least `n` i32 elements (initialized by caller before first call).
/// - `changed` must point to a single i32 flag in device memory.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn launch_compute_levels_upper_iter(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs: u64,
    col_indices: u64,
    levels: u64,
    changed: u64,
    n: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LEVELS_MODULE)?;
    let func = get_kernel_function(&module, "compute_levels_upper_iter")?;
    let cfg = launch_config((grid_size(n as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&row_ptrs);
    builder.arg(&col_indices);
    builder.arg(&levels);
    builder.arg(&changed);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("compute_levels_upper_iter", e))?;
    Ok(())
}

// ============================================================================
// Reduction
// ============================================================================

/// Find maximum level value via single-block parallel reduction
///
/// # Safety
///
/// - `data` must be a valid device memory pointer with at least `n` i32 elements.
/// - `result` must point to a single i32 element in device memory where the result is written.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn launch_reduce_max_i32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    data: u64,
    result: u64,
    n: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LEVELS_MODULE)?;
    let func = get_kernel_function(&module, "reduce_max_i32")?;
    let shared_size = (BLOCK_SIZE as usize) * std::mem::size_of::<i32>();
    let cfg = launch_config((1, 1, 1), (BLOCK_SIZE, 1, 1), shared_size as u32);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&data);
    builder.arg(&result);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("reduce_max_i32", e))?;
    Ok(())
}

// ============================================================================
// Histogram and Scatter
// ============================================================================

/// Count occurrences of each level via atomic histogram
///
/// # Safety
///
/// - `levels` must be a valid device memory pointer with at least `n` i32 elements.
/// - `counts` must be a valid device memory pointer pre-allocated to hold the histogram
///   (size must be at least `max_level + 1` as determined by the caller).
/// - All values in `levels` must be non-negative and within bounds of the `counts` array.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn launch_histogram_levels(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    levels: u64,
    counts: u64,
    n: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LEVELS_MODULE)?;
    let func = get_kernel_function(&module, "histogram_levels")?;
    let cfg = launch_config((grid_size(n as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&levels);
    builder.arg(&counts);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("histogram_levels", e))?;
    Ok(())
}

/// Scatter rows by level into the `level_rows` array using atomic counters
///
/// # Safety
///
/// - `levels`, `level_ptrs`, `level_rows`, and `level_counters` must be valid device memory
///   pointers on the device associated with `context`.
/// - `levels` and `level_counters` must have at least `n` elements.
/// - `level_ptrs` must have at least `num_levels + 1` elements (prefix sums of level sizes).
/// - `level_rows` must have at least `n` elements.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn launch_scatter_by_level(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    levels: u64,
    level_ptrs: u64,
    level_rows: u64,
    level_counters: u64,
    n: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LEVELS_MODULE)?;
    let func = get_kernel_function(&module, "scatter_by_level")?;
    let cfg = launch_config((grid_size(n as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&levels);
    builder.arg(&level_ptrs);
    builder.arg(&level_rows);
    builder.arg(&level_counters);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("scatter_by_level", e))?;
    Ok(())
}
