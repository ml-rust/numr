//! ILU(0) and IC(0) factorization kernel launchers

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::{
    BLOCK_SIZE, SPARSE_LINALG_MODULE, get_kernel_function, get_or_load_module, grid_size,
    launch_config, launch_error,
};
use crate::error::Result;

// ============================================================================
// ILU(0) Level Kernel Launchers
// ============================================================================

/// Launch ILU(0) level kernel - f32
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_ilu0_level_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    level_rows: u64,
    level_size: i32,
    row_ptrs: u64,
    col_indices: u64,
    values: u64,
    diag_indices: u64,
    n: i32,
    diagonal_shift: f32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "ilu0_level_f32")?;
    let cfg = launch_config((grid_size(level_size as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&level_rows);
    builder.arg(&level_size);
    builder.arg(&row_ptrs);
    builder.arg(&col_indices);
    builder.arg(&values);
    builder.arg(&diag_indices);
    builder.arg(&n);
    builder.arg(&diagonal_shift);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("ilu0_level_f32", e))?;
    Ok(())
}

/// Launch ILU(0) level kernel - f64
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_ilu0_level_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    level_rows: u64,
    level_size: i32,
    row_ptrs: u64,
    col_indices: u64,
    values: u64,
    diag_indices: u64,
    n: i32,
    diagonal_shift: f64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "ilu0_level_f64")?;
    let cfg = launch_config((grid_size(level_size as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&level_rows);
    builder.arg(&level_size);
    builder.arg(&row_ptrs);
    builder.arg(&col_indices);
    builder.arg(&values);
    builder.arg(&diag_indices);
    builder.arg(&n);
    builder.arg(&diagonal_shift);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("ilu0_level_f64", e))?;
    Ok(())
}

// ============================================================================
// IC(0) Level Kernel Launchers
// ============================================================================

/// Launch IC(0) level kernel - f32
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_ic0_level_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    level_rows: u64,
    level_size: i32,
    row_ptrs: u64,
    col_indices: u64,
    values: u64,
    diag_indices: u64,
    n: i32,
    diagonal_shift: f32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "ic0_level_f32")?;
    let cfg = launch_config((grid_size(level_size as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&level_rows);
    builder.arg(&level_size);
    builder.arg(&row_ptrs);
    builder.arg(&col_indices);
    builder.arg(&values);
    builder.arg(&diag_indices);
    builder.arg(&n);
    builder.arg(&diagonal_shift);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("ic0_level_f32", e))?;
    Ok(())
}

/// Launch IC(0) level kernel - f64
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_ic0_level_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    level_rows: u64,
    level_size: i32,
    row_ptrs: u64,
    col_indices: u64,
    values: u64,
    diag_indices: u64,
    n: i32,
    diagonal_shift: f64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "ic0_level_f64")?;
    let cfg = launch_config((grid_size(level_size as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&level_rows);
    builder.arg(&level_size);
    builder.arg(&row_ptrs);
    builder.arg(&col_indices);
    builder.arg(&values);
    builder.arg(&diag_indices);
    builder.arg(&n);
    builder.arg(&diagonal_shift);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("ic0_level_f64", e))?;
    Ok(())
}
