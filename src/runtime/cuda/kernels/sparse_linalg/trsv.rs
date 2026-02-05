//! Level-scheduled sparse triangular solve launchers (CSR and CSC formats)

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::{
    BLOCK_SIZE, SPARSE_LINALG_MODULE, get_kernel_function, get_or_load_module, grid_size,
    launch_config, launch_error,
};
use crate::error::Result;

// ============================================================================
// CSR Format - Single RHS
// ============================================================================

/// Launch level-scheduled lower triangular solve kernel (forward substitution) - f32
///
/// # Safety
///
/// Caller must ensure all pointers are valid device pointers with correct sizes.
#[allow(clippy::too_many_arguments)]
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
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_trsv_lower_level_f32")?;
    let cfg = launch_config((grid_size(level_size as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);
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
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_trsv_lower_level_f32", e))?;
    Ok(())
}

/// Launch level-scheduled lower triangular solve kernel - f64
#[allow(clippy::too_many_arguments)]
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
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_trsv_lower_level_f64")?;
    let cfg = launch_config((grid_size(level_size as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);
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
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_trsv_lower_level_f64", e))?;
    Ok(())
}

/// Launch level-scheduled upper triangular solve kernel (backward substitution) - f32
#[allow(clippy::too_many_arguments)]
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
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_trsv_upper_level_f32")?;
    let cfg = launch_config((grid_size(level_size as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&level_rows);
    builder.arg(&level_size);
    builder.arg(&row_ptrs);
    builder.arg(&col_indices);
    builder.arg(&values);
    builder.arg(&b);
    builder.arg(&x);
    builder.arg(&n);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_trsv_upper_level_f32", e))?;
    Ok(())
}

/// Launch level-scheduled upper triangular solve kernel - f64
#[allow(clippy::too_many_arguments)]
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
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_trsv_upper_level_f64")?;
    let cfg = launch_config((grid_size(level_size as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&level_rows);
    builder.arg(&level_size);
    builder.arg(&row_ptrs);
    builder.arg(&col_indices);
    builder.arg(&values);
    builder.arg(&b);
    builder.arg(&x);
    builder.arg(&n);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_trsv_upper_level_f64", e))?;
    Ok(())
}

// ============================================================================
// CSR Format - Multi-RHS
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
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_trsv_lower_level_multi_rhs_f32")?;
    let total_work = (level_size * nrhs) as u32;
    let cfg = launch_config((grid_size(total_work), 1, 1), (BLOCK_SIZE, 1, 1), 0);
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
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }
        .map_err(|e| launch_error("sparse_trsv_lower_level_multi_rhs_f32", e))?;
    Ok(())
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
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_trsv_lower_level_multi_rhs_f64")?;
    let total_work = (level_size * nrhs) as u32;
    let cfg = launch_config((grid_size(total_work), 1, 1), (BLOCK_SIZE, 1, 1), 0);
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
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }
        .map_err(|e| launch_error("sparse_trsv_lower_level_multi_rhs_f64", e))?;
    Ok(())
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
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_trsv_upper_level_multi_rhs_f32")?;
    let total_work = (level_size * nrhs) as u32;
    let cfg = launch_config((grid_size(total_work), 1, 1), (BLOCK_SIZE, 1, 1), 0);

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
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }
        .map_err(|e| launch_error("sparse_trsv_upper_level_multi_rhs_f32", e))?;
    Ok(())
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
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_trsv_upper_level_multi_rhs_f64")?;
    let total_work = (level_size * nrhs) as u32;
    let cfg = launch_config((grid_size(total_work), 1, 1), (BLOCK_SIZE, 1, 1), 0);

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
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }
        .map_err(|e| launch_error("sparse_trsv_upper_level_multi_rhs_f64", e))?;
    Ok(())
}

// ============================================================================
// CSC Format - Single RHS (for LU solve)
// ============================================================================

/// Launch CSC lower triangular solve kernel - f32
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_sparse_trsv_csc_lower_level_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    level_cols: u64,
    level_size: i32,
    col_ptrs: u64,
    row_indices: u64,
    values: u64,
    diag_ptr: u64,
    b: u64,
    n: i32,
    unit_diagonal: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_trsv_csc_lower_level_f32")?;
    let cfg = launch_config((grid_size(level_size as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);
    let unit_diag_i32: i32 = if unit_diagonal { 1 } else { 0 };

    let mut builder = stream.launch_builder(&func);
    builder.arg(&level_cols);
    builder.arg(&level_size);
    builder.arg(&col_ptrs);
    builder.arg(&row_indices);
    builder.arg(&values);
    builder.arg(&diag_ptr);
    builder.arg(&b);
    builder.arg(&n);
    builder.arg(&unit_diag_i32);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }
        .map_err(|e| launch_error("sparse_trsv_csc_lower_level_f32", e))?;
    Ok(())
}

/// Launch CSC lower triangular solve kernel - f64
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_sparse_trsv_csc_lower_level_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    level_cols: u64,
    level_size: i32,
    col_ptrs: u64,
    row_indices: u64,
    values: u64,
    diag_ptr: u64,
    b: u64,
    n: i32,
    unit_diagonal: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_trsv_csc_lower_level_f64")?;
    let cfg = launch_config((grid_size(level_size as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);
    let unit_diag_i32: i32 = if unit_diagonal { 1 } else { 0 };

    let mut builder = stream.launch_builder(&func);
    builder.arg(&level_cols);
    builder.arg(&level_size);
    builder.arg(&col_ptrs);
    builder.arg(&row_indices);
    builder.arg(&values);
    builder.arg(&diag_ptr);
    builder.arg(&b);
    builder.arg(&n);
    builder.arg(&unit_diag_i32);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }
        .map_err(|e| launch_error("sparse_trsv_csc_lower_level_f64", e))?;
    Ok(())
}

/// Launch CSC upper triangular solve kernel - f32
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_sparse_trsv_csc_upper_level_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    level_cols: u64,
    level_size: i32,
    col_ptrs: u64,
    row_indices: u64,
    values: u64,
    diag_ptr: u64,
    b: u64,
    n: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_trsv_csc_upper_level_f32")?;
    let cfg = launch_config((grid_size(level_size as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&level_cols);
    builder.arg(&level_size);
    builder.arg(&col_ptrs);
    builder.arg(&row_indices);
    builder.arg(&values);
    builder.arg(&diag_ptr);
    builder.arg(&b);
    builder.arg(&n);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }
        .map_err(|e| launch_error("sparse_trsv_csc_upper_level_f32", e))?;
    Ok(())
}

/// Launch CSC upper triangular solve kernel - f64
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_sparse_trsv_csc_upper_level_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    level_cols: u64,
    level_size: i32,
    col_ptrs: u64,
    row_indices: u64,
    values: u64,
    diag_ptr: u64,
    b: u64,
    n: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_trsv_csc_upper_level_f64")?;
    let cfg = launch_config((grid_size(level_size as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&level_cols);
    builder.arg(&level_size);
    builder.arg(&col_ptrs);
    builder.arg(&row_indices);
    builder.arg(&values);
    builder.arg(&diag_ptr);
    builder.arg(&b);
    builder.arg(&n);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }
        .map_err(|e| launch_error("sparse_trsv_csc_upper_level_f64", e))?;
    Ok(())
}
