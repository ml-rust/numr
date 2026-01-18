//! COO Sparse Matrix CUDA Kernel Launchers
//!
//! Low-level kernel wrappers for COO sparse operations.
//! All functions are pub(crate) for use by high-level merge operations.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use cudarc::types::CudaTypeName;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::loader::{
    BLOCK_SIZE, get_kernel_function, get_or_load_module, kernel_names, launch_config,
};

/// Helper to compute launch config from element count
pub(crate) fn compute_launch_config(
    n: usize,
) -> crate::runtime::cuda::kernels::loader::LaunchConfig {
    let grid_size = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0)
}

// ============================================================================
// Key Computation Launchers
// ============================================================================

/// Launch COO key computation kernel
/// keys[i] = row_indices[i] * ncols + col_indices[i]
pub(crate) unsafe fn launch_coo_compute_keys(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_indices: u64,
    col_indices: u64,
    keys_out: u64,
    ncols: i64,
    nnz: usize,
) -> Result<()> {
    let kernel_name = "coo_compute_keys_i64";

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(nnz);

    let nnz_u32 = nnz as u32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&row_indices);
    builder.arg(&col_indices);
    builder.arg(&keys_out);
    builder.arg(&ncols);
    builder.arg(&nnz_u32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed (device={}, nnz={}, ncols={}): {:?}",
                kernel_name, device_index, nnz, ncols, e
            ))
        })?;
    }

    Ok(())
}

/// Launch COO index extraction kernel
/// row_indices[i] = keys[i] / ncols, col_indices[i] = keys[i] % ncols
pub(crate) unsafe fn launch_coo_extract_indices(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    keys: u64,
    row_indices_out: u64,
    col_indices_out: u64,
    ncols: i64,
    nnz: usize,
) -> Result<()> {
    let kernel_name = "coo_extract_indices_i64";

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(nnz);

    let nnz_u32 = nnz as u32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&keys);
    builder.arg(&row_indices_out);
    builder.arg(&col_indices_out);
    builder.arg(&ncols);
    builder.arg(&nnz_u32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed (device={}, nnz={}, ncols={}): {:?}",
                kernel_name, device_index, nnz, ncols, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// Concatenation Launchers
// ============================================================================

/// Launch key concatenation kernel
pub(crate) unsafe fn launch_coo_concat_keys(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    keys_a: u64,
    keys_b: u64,
    keys_out: u64,
    nnz_a: usize,
    nnz_b: usize,
) -> Result<()> {
    let kernel_name = "coo_concat_keys_i64";

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let total = nnz_a + nnz_b;
    let cfg = compute_launch_config(total);

    let nnz_a_u32 = nnz_a as u32;
    let nnz_b_u32 = nnz_b as u32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&keys_a);
    builder.arg(&keys_b);
    builder.arg(&keys_out);
    builder.arg(&nnz_a_u32);
    builder.arg(&nnz_b_u32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch value concatenation with source flags kernel
pub(crate) unsafe fn launch_coo_concat_values_with_source<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    values_a: u64,
    values_b: u64,
    values_out: u64,
    source_out: u64,
    nnz_a: usize,
    nnz_b: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "coo_concat_values_with_source_f32",
        "f64" => "coo_concat_values_with_source_f64",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for COO concat: {}",
                T::NAME
            )));
        }
    };

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let total = nnz_a + nnz_b;
    let cfg = compute_launch_config(total);

    let nnz_a_u32 = nnz_a as u32;
    let nnz_b_u32 = nnz_b as u32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&values_a);
    builder.arg(&values_b);
    builder.arg(&values_out);
    builder.arg(&source_out);
    builder.arg(&nnz_a_u32);
    builder.arg(&nnz_b_u32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// Unique Marking Launcher
// ============================================================================

/// Launch unique position marking kernel
pub(crate) unsafe fn launch_coo_mark_unique(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    keys: u64,
    unique_flags: u64,
    n: usize,
) -> Result<()> {
    let kernel_name = "coo_mark_unique_i64";

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(n);

    let n_u32 = n as u32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&keys);
    builder.arg(&unique_flags);
    builder.arg(&n_u32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// Merge Operation Launchers
// ============================================================================

/// Launch COO merge add kernel (union semantics)
pub(crate) unsafe fn launch_coo_merge_add<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    sorted_keys: u64,
    sorted_values: u64,
    source_flags: u64,
    unique_positions: u64,
    out_keys: u64,
    out_values: u64,
    n: usize,
    num_unique: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "coo_merge_add_f32",
        "f64" => "coo_merge_add_f64",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for COO merge add: {}",
                T::NAME
            )));
        }
    };

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(n);

    let n_u32 = n as u32;
    let num_unique_u32 = num_unique as u32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&sorted_keys);
    builder.arg(&sorted_values);
    builder.arg(&source_flags);
    builder.arg(&unique_positions);
    builder.arg(&out_keys);
    builder.arg(&out_values);
    builder.arg(&n_u32);
    builder.arg(&num_unique_u32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch COO merge sub kernel (union semantics)
pub(crate) unsafe fn launch_coo_merge_sub<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    sorted_keys: u64,
    sorted_values: u64,
    source_flags: u64,
    unique_positions: u64,
    out_keys: u64,
    out_values: u64,
    n: usize,
    num_unique: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "coo_merge_sub_f32",
        "f64" => "coo_merge_sub_f64",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for COO merge sub: {}",
                T::NAME
            )));
        }
    };

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(n);

    let n_u32 = n as u32;
    let num_unique_u32 = num_unique as u32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&sorted_keys);
    builder.arg(&sorted_values);
    builder.arg(&source_flags);
    builder.arg(&unique_positions);
    builder.arg(&out_keys);
    builder.arg(&out_values);
    builder.arg(&n_u32);
    builder.arg(&num_unique_u32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch intersection counting kernel
pub(crate) unsafe fn launch_coo_count_intersections(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    sorted_keys: u64,
    source_flags: u64,
    intersection_flags: u64,
    n: usize,
) -> Result<()> {
    let kernel_name = "coo_count_intersections_i64";

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(n);

    let n_u32 = n as u32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&sorted_keys);
    builder.arg(&source_flags);
    builder.arg(&intersection_flags);
    builder.arg(&n_u32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch COO merge mul kernel (intersection semantics)
pub(crate) unsafe fn launch_coo_merge_mul<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    sorted_keys: u64,
    sorted_values: u64,
    source_flags: u64,
    intersection_flags: u64,
    output_positions: u64,
    out_keys: u64,
    out_values: u64,
    n: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "coo_merge_mul_f32",
        "f64" => "coo_merge_mul_f64",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for COO merge mul: {}",
                T::NAME
            )));
        }
    };

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(n);

    let n_u32 = n as u32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&sorted_keys);
    builder.arg(&sorted_values);
    builder.arg(&source_flags);
    builder.arg(&intersection_flags);
    builder.arg(&output_positions);
    builder.arg(&out_keys);
    builder.arg(&out_values);
    builder.arg(&n_u32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch COO merge div kernel (intersection semantics)
pub(crate) unsafe fn launch_coo_merge_div<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    sorted_keys: u64,
    sorted_values: u64,
    source_flags: u64,
    intersection_flags: u64,
    output_positions: u64,
    out_keys: u64,
    out_values: u64,
    n: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "coo_merge_div_f32",
        "f64" => "coo_merge_div_f64",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for COO merge div: {}",
                T::NAME
            )));
        }
    };

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(n);

    let n_u32 = n as u32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&sorted_keys);
    builder.arg(&sorted_values);
    builder.arg(&source_flags);
    builder.arg(&intersection_flags);
    builder.arg(&output_positions);
    builder.arg(&out_keys);
    builder.arg(&out_values);
    builder.arg(&n_u32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// Zero Filtering Launchers
// ============================================================================

/// Launch non-zero marking kernel
pub(crate) unsafe fn launch_coo_mark_nonzero<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    values: u64,
    nonzero_flags: u64,
    threshold: f64,
    n: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "coo_mark_nonzero_f32",
        "f64" => "coo_mark_nonzero_f64",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for COO mark nonzero: {}",
                T::NAME
            )));
        }
    };

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(n);

    let n_u32 = n as u32;
    let threshold_f32 = threshold as f32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&values);
    builder.arg(&nonzero_flags);
    if T::NAME == "f32" {
        builder.arg(&threshold_f32);
    } else {
        builder.arg(&threshold);
    }
    builder.arg(&n_u32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch compaction kernel
pub(crate) unsafe fn launch_coo_compact<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    keys_in: u64,
    values_in: u64,
    flags: u64,
    positions: u64,
    keys_out: u64,
    values_out: u64,
    n: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "coo_compact_f32",
        "f64" => "coo_compact_f64",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for COO compact: {}",
                T::NAME
            )));
        }
    };

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(n);

    let n_u32 = n as u32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&keys_in);
    builder.arg(&values_in);
    builder.arg(&flags);
    builder.arg(&positions);
    builder.arg(&keys_out);
    builder.arg(&values_out);
    builder.arg(&n_u32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// GPU Sort using Thrust
// ============================================================================

/// Sort (i64 keys, i32 indices) using Thrust stable_sort_by_key - FULLY ON GPU
/// Sorts IN-PLACE, so keys and indices are both input and output
pub(crate) unsafe fn launch_thrust_sort_pairs_i64_i32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    keys: u64,
    indices: u64,
    n: u32,
) -> Result<()> {
    let kernel_name = "thrust_sort_pairs_i64_i32_kernel";

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    // Launch with single thread (thrust handles parallelism internally)
    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);

    let mut builder = stream.launch_builder(&function);
    builder.arg(&keys);
    builder.arg(&indices);
    builder.arg(&n);

    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed (n={}, keys=0x{:x}, indices=0x{:x}): {:?}",
                kernel_name, n, keys, indices, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// Index and Gather Kernel Launchers
// ============================================================================

/// Initialize indices array [0, 1, 2, ..., n-1]
pub(crate) unsafe fn launch_coo_init_indices(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    indices: u64,
    n: usize,
) -> Result<()> {
    let kernel_name = "coo_init_indices_i32";

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(n);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&function);
    builder.arg(&indices);
    builder.arg(&n_u32);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!(
            "CUDA {} kernel launch failed (device={}, n={}, indices=0x{:x}): {:?}",
            kernel_name, device_index, n, indices, e
        ))
    })?;

    Ok(())
}

/// Gather values using indices (permutation)
pub(crate) unsafe fn launch_coo_gather<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    values_in: u64,
    indices: u64,
    values_out: u64,
    n: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "coo_gather_f32",
        "f64" => "coo_gather_f64",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for coo_gather: {}",
                T::NAME
            )));
        }
    };

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(n);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&function);
    builder.arg(&values_in);
    builder.arg(&indices);
    builder.arg(&values_out);
    builder.arg(&n_u32);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!(
            "CUDA {} kernel launch failed: {:?}",
            kernel_name, e
        ))
    })?;

    Ok(())
}

/// Gather i32 values using indices
pub(crate) unsafe fn launch_coo_gather_i32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    values_in: u64,
    indices: u64,
    values_out: u64,
    n: usize,
) -> Result<()> {
    let kernel_name = "coo_gather_i32";

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(n);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&function);
    builder.arg(&values_in);
    builder.arg(&indices);
    builder.arg(&values_out);
    builder.arg(&n_u32);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!(
            "CUDA {} kernel launch failed (device={}, n={}, dtype=i32): {:?}",
            kernel_name, device_index, n, e
        ))
    })?;

    Ok(())
}

/// Gather i64 values using indices (for row/col indices)
pub(crate) unsafe fn launch_coo_gather_i64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    values_in: u64,
    indices: u64,
    values_out: u64,
    n: usize,
) -> Result<()> {
    let kernel_name = "coo_gather_i64";

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(n);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&function);
    builder.arg(&values_in);
    builder.arg(&indices);
    builder.arg(&values_out);
    builder.arg(&n_u32);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!(
            "CUDA {} kernel launch failed (device={}, n={}, dtype=i64): {:?}",
            kernel_name, device_index, n, e
        ))
    })?;

    Ok(())
}

/// Merge duplicates with add operation
pub(crate) unsafe fn launch_coo_merge_duplicates_add<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    sorted_keys: u64,
    sorted_values: u64,
    sorted_sources: u64,
    unique_flags: u64,
    output_positions: u64,
    out_keys: u64,
    out_values: u64,
    n: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "coo_merge_duplicates_add_f32",
        "f64" => "coo_merge_duplicates_add_f64",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for coo_merge_duplicates_add: {}",
                T::NAME
            )));
        }
    };

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(n);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&function);
    builder.arg(&sorted_keys);
    builder.arg(&sorted_values);
    builder.arg(&sorted_sources);
    builder.arg(&unique_flags);
    builder.arg(&output_positions);
    builder.arg(&out_keys);
    builder.arg(&out_values);
    builder.arg(&n_u32);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!(
            "CUDA {} kernel launch failed (device={}, n={}, dtype={}, op=add): {:?}",
            kernel_name,
            device_index,
            n,
            T::NAME,
            e
        ))
    })?;

    Ok(())
}
