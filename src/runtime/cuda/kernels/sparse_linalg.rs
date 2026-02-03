//! CUDA kernel launchers for level-scheduled sparse linear algebra
//!
//! This module provides Rust wrappers for launching level-scheduled
//! sparse triangular solve, ILU(0), and IC(0) kernels.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{get_kernel_function, get_or_load_module, launch_config};
use crate::error::{Error, Result};

/// Module name for sparse linear algebra kernels
pub const SPARSE_LINALG_MODULE: &str = "sparse_linalg";

const BLOCK_SIZE: u32 = 256;

// ============================================================================
// Level-Scheduled Triangular Solve Launchers
// ============================================================================

/// Launch level-scheduled lower triangular solve kernel (forward substitution)
///
/// # Safety
///
/// Caller must ensure all pointers are valid device pointers with correct sizes.
pub unsafe fn launch_sparse_trsv_lower_level_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    level_rows: u64,
    level_size: i32,
    row_ptrs: u64,
    col_indices: u64,
    values: u64,
    b: u64,
    x: u64,
    n: i32,
    unit_diagonal: bool,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "sparse_trsv_lower_level_f32")?;

        let grid_size = ((level_size as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);
        let unit_diag_i32: i32 = if unit_diagonal { 1 } else { 0 };

        let mut builder = stream.launch_builder(&func);
        builder.arg(&level_rows);
        builder.arg(&level_size);
        builder.arg(&row_ptrs);
        builder.arg(&col_indices);
        builder.arg(&values);
        builder.arg(&b);
        builder.arg(&x);
        builder.arg(&n);
        builder.arg(&unit_diag_i32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA sparse_trsv_lower_level_f32 launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch level-scheduled lower triangular solve kernel (forward substitution) - f64
pub unsafe fn launch_sparse_trsv_lower_level_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    level_rows: u64,
    level_size: i32,
    row_ptrs: u64,
    col_indices: u64,
    values: u64,
    b: u64,
    x: u64,
    n: i32,
    unit_diagonal: bool,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "sparse_trsv_lower_level_f64")?;

        let grid_size = ((level_size as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);
        let unit_diag_i32: i32 = if unit_diagonal { 1 } else { 0 };

        let mut builder = stream.launch_builder(&func);
        builder.arg(&level_rows);
        builder.arg(&level_size);
        builder.arg(&row_ptrs);
        builder.arg(&col_indices);
        builder.arg(&values);
        builder.arg(&b);
        builder.arg(&x);
        builder.arg(&n);
        builder.arg(&unit_diag_i32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA sparse_trsv_lower_level_f64 launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch level-scheduled upper triangular solve kernel (backward substitution)
pub unsafe fn launch_sparse_trsv_upper_level_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    level_rows: u64,
    level_size: i32,
    row_ptrs: u64,
    col_indices: u64,
    values: u64,
    b: u64,
    x: u64,
    n: i32,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "sparse_trsv_upper_level_f32")?;

        let grid_size = ((level_size as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&level_rows);
        builder.arg(&level_size);
        builder.arg(&row_ptrs);
        builder.arg(&col_indices);
        builder.arg(&values);
        builder.arg(&b);
        builder.arg(&x);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA sparse_trsv_upper_level_f32 launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch level-scheduled upper triangular solve kernel - f64
pub unsafe fn launch_sparse_trsv_upper_level_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    level_rows: u64,
    level_size: i32,
    row_ptrs: u64,
    col_indices: u64,
    values: u64,
    b: u64,
    x: u64,
    n: i32,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "sparse_trsv_upper_level_f64")?;

        let grid_size = ((level_size as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&level_rows);
        builder.arg(&level_size);
        builder.arg(&row_ptrs);
        builder.arg(&col_indices);
        builder.arg(&values);
        builder.arg(&b);
        builder.arg(&x);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA sparse_trsv_upper_level_f64 launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

// ============================================================================
// ILU(0) Level Kernel Launchers
// ============================================================================

/// Launch ILU(0) level kernel - f32
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
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "ilu0_level_f32")?;

        let grid_size = ((level_size as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&level_rows);
        builder.arg(&level_size);
        builder.arg(&row_ptrs);
        builder.arg(&col_indices);
        builder.arg(&values);
        builder.arg(&diag_indices);
        builder.arg(&n);
        builder.arg(&diagonal_shift);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA ilu0_level_f32 launch failed: {:?}", e)))?;

        Ok(())
    }
}

/// Launch ILU(0) level kernel - f64
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
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "ilu0_level_f64")?;

        let grid_size = ((level_size as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&level_rows);
        builder.arg(&level_size);
        builder.arg(&row_ptrs);
        builder.arg(&col_indices);
        builder.arg(&values);
        builder.arg(&diag_indices);
        builder.arg(&n);
        builder.arg(&diagonal_shift);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA ilu0_level_f64 launch failed: {:?}", e)))?;

        Ok(())
    }
}

// ============================================================================
// IC(0) Level Kernel Launchers
// ============================================================================

/// Launch IC(0) level kernel - f32
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
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "ic0_level_f32")?;

        let grid_size = ((level_size as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&level_rows);
        builder.arg(&level_size);
        builder.arg(&row_ptrs);
        builder.arg(&col_indices);
        builder.arg(&values);
        builder.arg(&diag_indices);
        builder.arg(&n);
        builder.arg(&diagonal_shift);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA ic0_level_f32 launch failed: {:?}", e)))?;

        Ok(())
    }
}

/// Launch IC(0) level kernel - f64
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
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "ic0_level_f64")?;

        let grid_size = ((level_size as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&level_rows);
        builder.arg(&level_size);
        builder.arg(&row_ptrs);
        builder.arg(&col_indices);
        builder.arg(&values);
        builder.arg(&diag_indices);
        builder.arg(&n);
        builder.arg(&diagonal_shift);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA ic0_level_f64 launch failed: {:?}", e)))?;

        Ok(())
    }
}

// ============================================================================
// Utility Kernel Launchers
// ============================================================================

/// Launch kernel to find diagonal indices in CSR matrix
pub unsafe fn launch_find_diag_indices(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs: u64,
    col_indices: u64,
    diag_indices: u64,
    n: i32,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "find_diag_indices")?;

        let grid_size = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&row_ptrs);
        builder.arg(&col_indices);
        builder.arg(&diag_indices);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA find_diag_indices launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch copy kernel - f32
#[allow(dead_code)]
pub unsafe fn launch_copy_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    src: u64,
    dst: u64,
    n: i32,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "copy_f32")?;

        let grid_size = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src);
        builder.arg(&dst);
        builder.arg(&n);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA copy_f32 launch failed: {:?}", e)))?;

        Ok(())
    }
}

/// Launch copy kernel - f64
#[allow(dead_code)]
pub unsafe fn launch_copy_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    src: u64,
    dst: u64,
    n: i32,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "copy_f64")?;

        let grid_size = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src);
        builder.arg(&dst);
        builder.arg(&n);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA copy_f64 launch failed: {:?}", e)))?;

        Ok(())
    }
}
