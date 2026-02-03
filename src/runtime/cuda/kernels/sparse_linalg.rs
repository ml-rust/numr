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
// Multi-RHS Level-Scheduled Triangular Solve Launchers
// These kernels handle multiple right-hand sides [n, nrhs] in parallel
// ============================================================================

/// Launch multi-RHS lower triangular solve kernel (forward substitution) - f32
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_sparse_trsv_lower_level_multi_rhs_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    level_rows: u64,
    level_size: i32,
    nrhs: i32,
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
        let func = get_kernel_function(&module, "sparse_trsv_lower_level_multi_rhs_f32")?;

        let total_work = (level_size * nrhs) as u32;
        let grid_size = (total_work + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);
        let unit_diag_i32: i32 = if unit_diagonal { 1 } else { 0 };

        let mut builder = stream.launch_builder(&func);
        builder.arg(&level_rows);
        builder.arg(&level_size);
        builder.arg(&nrhs);
        builder.arg(&row_ptrs);
        builder.arg(&col_indices);
        builder.arg(&values);
        builder.arg(&b);
        builder.arg(&x);
        builder.arg(&n);
        builder.arg(&unit_diag_i32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA sparse_trsv_lower_level_multi_rhs_f32 launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch multi-RHS lower triangular solve kernel - f64
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_sparse_trsv_lower_level_multi_rhs_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    level_rows: u64,
    level_size: i32,
    nrhs: i32,
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
        let func = get_kernel_function(&module, "sparse_trsv_lower_level_multi_rhs_f64")?;

        let total_work = (level_size * nrhs) as u32;
        let grid_size = (total_work + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);
        let unit_diag_i32: i32 = if unit_diagonal { 1 } else { 0 };

        let mut builder = stream.launch_builder(&func);
        builder.arg(&level_rows);
        builder.arg(&level_size);
        builder.arg(&nrhs);
        builder.arg(&row_ptrs);
        builder.arg(&col_indices);
        builder.arg(&values);
        builder.arg(&b);
        builder.arg(&x);
        builder.arg(&n);
        builder.arg(&unit_diag_i32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA sparse_trsv_lower_level_multi_rhs_f64 launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch multi-RHS upper triangular solve kernel (backward substitution) - f32
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_sparse_trsv_upper_level_multi_rhs_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    level_rows: u64,
    level_size: i32,
    nrhs: i32,
    row_ptrs: u64,
    col_indices: u64,
    values: u64,
    b: u64,
    x: u64,
    n: i32,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "sparse_trsv_upper_level_multi_rhs_f32")?;

        let total_work = (level_size * nrhs) as u32;
        let grid_size = (total_work + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&level_rows);
        builder.arg(&level_size);
        builder.arg(&nrhs);
        builder.arg(&row_ptrs);
        builder.arg(&col_indices);
        builder.arg(&values);
        builder.arg(&b);
        builder.arg(&x);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA sparse_trsv_upper_level_multi_rhs_f32 launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch multi-RHS upper triangular solve kernel - f64
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_sparse_trsv_upper_level_multi_rhs_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    level_rows: u64,
    level_size: i32,
    nrhs: i32,
    row_ptrs: u64,
    col_indices: u64,
    values: u64,
    b: u64,
    x: u64,
    n: i32,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "sparse_trsv_upper_level_multi_rhs_f64")?;

        let total_work = (level_size * nrhs) as u32;
        let grid_size = (total_work + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&level_rows);
        builder.arg(&level_size);
        builder.arg(&nrhs);
        builder.arg(&row_ptrs);
        builder.arg(&col_indices);
        builder.arg(&values);
        builder.arg(&b);
        builder.arg(&x);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA sparse_trsv_upper_level_multi_rhs_f64 launch failed: {:?}",
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

// ============================================================================
// LU Split Scatter Kernel Launchers
// ============================================================================

/// Launch kernel to scatter values from factored LU matrix to separate L and U arrays - f32
///
/// # Arguments
/// * `src_values` - Source values array from factored matrix
/// * `l_values` - Output L values array
/// * `u_values` - Output U values array
/// * `l_map` - Mapping: l_map[i] = destination index in l_values, or -1 if not in L
/// * `u_map` - Mapping: u_map[i] = destination index in u_values, or -1 if not in U
/// * `nnz` - Number of non-zero elements in source
pub unsafe fn launch_split_lu_scatter_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    src_values: u64,
    l_values: u64,
    u_values: u64,
    l_map: u64,
    u_map: u64,
    nnz: i32,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "split_lu_scatter_f32")?;

        let grid_size = ((nnz as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_values);
        builder.arg(&l_values);
        builder.arg(&u_values);
        builder.arg(&l_map);
        builder.arg(&u_map);
        builder.arg(&nnz);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA split_lu_scatter_f32 launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch kernel to scatter values from factored LU matrix to separate L and U arrays - f64
pub unsafe fn launch_split_lu_scatter_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    src_values: u64,
    l_values: u64,
    u_values: u64,
    l_map: u64,
    u_map: u64,
    nnz: i32,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "split_lu_scatter_f64")?;

        let grid_size = ((nnz as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_values);
        builder.arg(&l_values);
        builder.arg(&u_values);
        builder.arg(&l_map);
        builder.arg(&u_map);
        builder.arg(&nnz);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA split_lu_scatter_f64 launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

// ============================================================================
// Lower Triangle Extraction Scatter Kernel Launchers
// ============================================================================

/// Launch kernel to scatter values from source to lower triangular output - f32
///
/// # Arguments
/// * `src_values` - Source values array
/// * `dst_values` - Output values array (lower triangular)
/// * `lower_map` - Mapping: lower_map[i] = destination index, or -1 if not in lower
/// * `nnz` - Number of non-zero elements in source
pub unsafe fn launch_extract_lower_scatter_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    src_values: u64,
    dst_values: u64,
    lower_map: u64,
    nnz: i32,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "extract_lower_scatter_f32")?;

        let grid_size = ((nnz as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_values);
        builder.arg(&dst_values);
        builder.arg(&lower_map);
        builder.arg(&nnz);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA extract_lower_scatter_f32 launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch kernel to scatter values from source to lower triangular output - f64
pub unsafe fn launch_extract_lower_scatter_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    src_values: u64,
    dst_values: u64,
    lower_map: u64,
    nnz: i32,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
        let func = get_kernel_function(&module, "extract_lower_scatter_f64")?;

        let grid_size = ((nnz as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_values);
        builder.arg(&dst_values);
        builder.arg(&lower_map);
        builder.arg(&nnz);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA extract_lower_scatter_f64 launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}
