//! Sparse primitive operation kernel launchers
//!
//! These are the building blocks for sparse factorization algorithms:
//! scatter, axpy, gather_clear, divide_pivot, clear, apply_row_perm

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::{
    BLOCK_SIZE, SPARSE_LINALG_MODULE, get_kernel_function, get_or_load_module, grid_size,
    launch_config, launch_error,
};
use crate::error::Result;

// ============================================================================
// Scatter Operations
// ============================================================================

/// Scatters values into work vector: work[row_indices[i]] = values[i] - f32
pub unsafe fn launch_sparse_scatter_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    values: u64,
    row_indices: u64,
    work: u64,
    nnz: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_scatter_f32")?;
    let cfg = launch_config((grid_size(nnz as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&values);
    builder.arg(&row_indices);
    builder.arg(&work);
    builder.arg(&nnz);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_scatter_f32", e))?;
    Ok(())
}

/// Scatters values into work vector - f64
pub unsafe fn launch_sparse_scatter_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    values: u64,
    row_indices: u64,
    work: u64,
    nnz: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_scatter_f64")?;
    let cfg = launch_config((grid_size(nnz as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&values);
    builder.arg(&row_indices);
    builder.arg(&work);
    builder.arg(&nnz);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_scatter_f64", e))?;
    Ok(())
}

// ============================================================================
// AXPY Operations
// ============================================================================

/// Computes: work[row_indices[i]] -= scale * values[i] - f32
pub unsafe fn launch_sparse_axpy_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    scale: f32,
    values: u64,
    row_indices: u64,
    work: u64,
    nnz: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_axpy_f32")?;
    let cfg = launch_config((grid_size(nnz as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&scale);
    builder.arg(&values);
    builder.arg(&row_indices);
    builder.arg(&work);
    builder.arg(&nnz);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_axpy_f32", e))?;
    Ok(())
}

/// Computes: work[row_indices[i]] -= scale * values[i] - f64
pub unsafe fn launch_sparse_axpy_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    scale: f64,
    values: u64,
    row_indices: u64,
    work: u64,
    nnz: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_axpy_f64")?;
    let cfg = launch_config((grid_size(nnz as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&scale);
    builder.arg(&values);
    builder.arg(&row_indices);
    builder.arg(&work);
    builder.arg(&nnz);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_axpy_f64", e))?;
    Ok(())
}

// ============================================================================
// Gather and Clear Operations
// ============================================================================

/// Gathers: output[i] = work[row_indices[i]], then clears work[row_indices[i]] = 0 - f32
pub unsafe fn launch_sparse_gather_clear_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    work: u64,
    row_indices: u64,
    output: u64,
    nnz: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_gather_clear_f32")?;
    let cfg = launch_config((grid_size(nnz as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&work);
    builder.arg(&row_indices);
    builder.arg(&output);
    builder.arg(&nnz);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_gather_clear_f32", e))?;
    Ok(())
}

/// Gathers and clears - f64
pub unsafe fn launch_sparse_gather_clear_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    work: u64,
    row_indices: u64,
    output: u64,
    nnz: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_gather_clear_f64")?;
    let cfg = launch_config((grid_size(nnz as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&work);
    builder.arg(&row_indices);
    builder.arg(&output);
    builder.arg(&nnz);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_gather_clear_f64", e))?;
    Ok(())
}

// ============================================================================
// Divide by Pivot Operations
// ============================================================================

/// Computes: work[row_indices[i]] *= inv_pivot - f32
pub unsafe fn launch_sparse_divide_pivot_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    work: u64,
    row_indices: u64,
    inv_pivot: f32,
    nnz: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_divide_pivot_f32")?;
    let cfg = launch_config((grid_size(nnz as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&work);
    builder.arg(&row_indices);
    builder.arg(&inv_pivot);
    builder.arg(&nnz);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_divide_pivot_f32", e))?;
    Ok(())
}

/// Divide by pivot - f64
pub unsafe fn launch_sparse_divide_pivot_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    work: u64,
    row_indices: u64,
    inv_pivot: f64,
    nnz: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_divide_pivot_f64")?;
    let cfg = launch_config((grid_size(nnz as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&work);
    builder.arg(&row_indices);
    builder.arg(&inv_pivot);
    builder.arg(&nnz);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_divide_pivot_f64", e))?;
    Ok(())
}

// ============================================================================
// Clear Operations
// ============================================================================

/// Clears: work[row_indices[i]] = 0 - f32
pub unsafe fn launch_sparse_clear_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    work: u64,
    row_indices: u64,
    nnz: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_clear_f32")?;
    let cfg = launch_config((grid_size(nnz as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&work);
    builder.arg(&row_indices);
    builder.arg(&nnz);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_clear_f32", e))?;
    Ok(())
}

/// Clears - f64
pub unsafe fn launch_sparse_clear_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    work: u64,
    row_indices: u64,
    nnz: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_clear_f64")?;
    let cfg = launch_config((grid_size(nnz as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&work);
    builder.arg(&row_indices);
    builder.arg(&nnz);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_clear_f64", e))?;
    Ok(())
}

// ============================================================================
// Row Permutation Operations
// ============================================================================

/// Applies row permutation: y[i] = b[perm[i]] - f32
pub unsafe fn launch_apply_row_perm_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    b: u64,
    perm: u64,
    y: u64,
    n: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "apply_row_perm_f32")?;
    let cfg = launch_config((grid_size(n as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&b);
    builder.arg(&perm);
    builder.arg(&y);
    builder.arg(&n);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("apply_row_perm_f32", e))?;
    Ok(())
}

/// Applies row permutation - f64
pub unsafe fn launch_apply_row_perm_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    b: u64,
    perm: u64,
    y: u64,
    n: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "apply_row_perm_f64")?;
    let cfg = launch_config((grid_size(n as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&b);
    builder.arg(&perm);
    builder.arg(&y);
    builder.arg(&n);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("apply_row_perm_f64", e))?;
    Ok(())
}
