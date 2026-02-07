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

/// Compute level schedule for lower triangular (iterative BFS on GPU)
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

/// Compute level schedule for upper triangular (iterative BFS on GPU)
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

/// Find maximum level value via reduction
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

/// Count occurrences of each level
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

/// Scatter rows by level into level_rows array
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
