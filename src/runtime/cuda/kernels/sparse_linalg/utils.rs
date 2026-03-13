//! Sparse utility kernel launchers
//!
//! Helper operations for sparse matrix algorithms:
//! - find_diag_indices (CSR and CSC)
//! - copy
//! - split_lu_scatter
//! - extract_lower_scatter

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::{
    BLOCK_SIZE, SPARSE_LINALG_MODULE, get_kernel_function, get_or_load_module, grid_size,
    launch_config, launch_error,
};
use crate::error::Result;

// ============================================================================
// Diagonal Index Finding
// ============================================================================

/// Find diagonal indices in CSR matrix
///
/// For each row i, finds the index within that row's entries where col == i (diagonal).
/// Stores -1 if no diagonal entry exists.
///
/// # Safety
///
/// - `row_ptrs`, `col_indices`, and `diag_indices` must be valid device memory pointers on the
///   device associated with `context`.
/// - `row_ptrs` must have at least `n + 1` elements; `diag_indices` must have at least `n`.
/// - `col_indices` must have at least `nnz` elements (as encoded in `row_ptrs`).
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn launch_find_diag_indices(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs: u64,
    col_indices: u64,
    diag_indices: u64,
    n: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "find_diag_indices")?;
    let cfg = launch_config((grid_size(n as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&row_ptrs);
    builder.arg(&col_indices);
    builder.arg(&diag_indices);
    builder.arg(&n);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("find_diag_indices", e))?;
    Ok(())
}

/// Find diagonal indices in CSC matrix
///
/// For each column j, finds the index within that column's entries where row == j (diagonal).
/// Stores -1 if no diagonal entry exists.
///
/// # Safety
///
/// - `col_ptrs`, `row_indices`, and `diag_ptr` must be valid device memory pointers on the
///   device associated with `context`.
/// - `col_ptrs` must have at least `n + 1` elements; `diag_ptr` must have at least `n`.
/// - `row_indices` must have at least `nnz` elements (as encoded in `col_ptrs`).
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn launch_find_diag_indices_csc(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    col_ptrs: u64,
    row_indices: u64,
    diag_ptr: u64,
    n: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "find_diag_indices_csc")?;
    let cfg = launch_config((grid_size(n as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&col_ptrs);
    builder.arg(&row_indices);
    builder.arg(&diag_ptr);
    builder.arg(&n);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("find_diag_indices_csc", e))?;
    Ok(())
}

// ============================================================================
// Copy Operations (may be unused but kept for potential future use)
// ============================================================================

/// Copy `n` f32 elements from `src` to `dst` on device (GPU kernel)
///
/// # Safety
///
/// - `src` and `dst` must be valid device memory pointers on the device associated with `context`.
/// - Both buffers must have at least `n` f32 elements.
/// - `src` and `dst` must not alias.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
#[allow(dead_code)]
pub unsafe fn launch_copy_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    src: u64,
    dst: u64,
    n: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "copy_f32")?;
    let cfg = launch_config((grid_size(n as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&src);
    builder.arg(&dst);
    builder.arg(&n);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("copy_f32", e))?;
    Ok(())
}

/// Copy `n` f64 elements from `src` to `dst` on device (GPU kernel)
///
/// # Safety
///
/// - `src` and `dst` must be valid device memory pointers on the device associated with `context`.
/// - Both buffers must have at least `n` f64 elements.
/// - `src` and `dst` must not alias.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
#[allow(dead_code)]
pub unsafe fn launch_copy_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    src: u64,
    dst: u64,
    n: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "copy_f64")?;
    let cfg = launch_config((grid_size(n as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&src);
    builder.arg(&dst);
    builder.arg(&n);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("copy_f64", e))?;
    Ok(())
}

// ============================================================================
// LU Split Scatter Operations
// ============================================================================

/// Scatter values from factored LU matrix to separate L and U arrays - f32
///
/// # Arguments
/// * `src_values` - Source values array from factored matrix
/// * `l_values` - Output L values array
/// * `u_values` - Output U values array
/// * `l_map` - Mapping: l_map[i] = destination index in l_values, or -1 if not in L
/// * `u_map` - Mapping: u_map[i] = destination index in u_values, or -1 if not in U
/// * `nnz` - Number of non-zero elements in source
///
/// # Safety
///
/// - `src_values`, `l_values`, `u_values`, `l_map`, and `u_map` must be valid device memory
///   pointers on the device associated with `context`, each with at least `nnz` elements.
/// - All mapped indices in `l_map` and `u_map` (excluding -1) must be valid indices into their
///   respective output arrays (no out-of-bounds access).
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
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
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "split_lu_scatter_f32")?;
    let cfg = launch_config((grid_size(nnz as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&src_values);
    builder.arg(&l_values);
    builder.arg(&u_values);
    builder.arg(&l_map);
    builder.arg(&u_map);
    builder.arg(&nnz);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("split_lu_scatter_f32", e))?;
    Ok(())
}

/// Scatter values from factored LU matrix to separate L and U arrays - f64
///
/// # Safety
///
/// - `src_values`, `l_values`, `u_values`, `l_map`, and `u_map` must be valid device memory
///   pointers on the device associated with `context`, each with at least `nnz` elements.
/// - All mapped indices in `l_map` and `u_map` (excluding -1) must be valid indices into their
///   respective output arrays (no out-of-bounds access).
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
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
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "split_lu_scatter_f64")?;
    let cfg = launch_config((grid_size(nnz as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&src_values);
    builder.arg(&l_values);
    builder.arg(&u_values);
    builder.arg(&l_map);
    builder.arg(&u_map);
    builder.arg(&nnz);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("split_lu_scatter_f64", e))?;
    Ok(())
}

// ============================================================================
// Lower Triangle Extraction
// ============================================================================

/// Scatter values from source to lower triangular output - f32
///
/// # Arguments
/// * `src_values` - Source values array
/// * `dst_values` - Output values array (lower triangular)
/// * `lower_map` - Mapping: lower_map[i] = destination index, or -1 if not in lower
/// * `nnz` - Number of non-zero elements in source
///
/// # Safety
///
/// - `src_values`, `dst_values`, and `lower_map` must be valid device memory pointers on the
///   device associated with `context`, each with at least `nnz` elements.
/// - All mapped indices in `lower_map` (excluding -1) must be valid indices into `dst_values`
///   (no out-of-bounds access).
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn launch_extract_lower_scatter_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    src_values: u64,
    dst_values: u64,
    lower_map: u64,
    nnz: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "extract_lower_scatter_f32")?;
    let cfg = launch_config((grid_size(nnz as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&src_values);
    builder.arg(&dst_values);
    builder.arg(&lower_map);
    builder.arg(&nnz);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("extract_lower_scatter_f32", e))?;
    Ok(())
}

/// Scatter values from source to lower triangular output - f64
///
/// # Safety
///
/// - `src_values`, `dst_values`, and `lower_map` must be valid device memory pointers on the
///   device associated with `context`, each with at least `nnz` elements.
/// - All mapped indices in `lower_map` (excluding -1) must be valid indices into `dst_values`
///   (no out-of-bounds access).
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn launch_extract_lower_scatter_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    src_values: u64,
    dst_values: u64,
    lower_map: u64,
    nnz: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "extract_lower_scatter_f64")?;
    let cfg = launch_config((grid_size(nnz as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&src_values);
    builder.arg(&dst_values);
    builder.arg(&lower_map);
    builder.arg(&nnz);
    // SAFETY: All pointers are valid device pointers with correct sizes (ensured by caller)
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("extract_lower_scatter_f64", e))?;
    Ok(())
}
