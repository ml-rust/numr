//! CSC (Compressed Sparse Column) merge kernel launchers
//!
//! Low-level count and compute launchers plus high-level public merge operations
//! for CSC format sparse matrices.

#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use cudarc::types::CudaTypeName;
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::runtime::cuda::CudaRuntime;
use crate::tensor::Tensor;

use super::super::loader::{
    BLOCK_SIZE, get_kernel_function, get_or_load_module, kernel_names, launch_config,
};

// ============================================================================
// Count Kernels
// ============================================================================

/// Launch CSC intersect count kernel (for mul/div)
///
/// # Safety
///
/// - `col_ptrs_a`, `row_indices_a`, `col_ptrs_b`, `row_indices_b`, and `col_counts` must be
///   valid device memory pointers on the device associated with `context`.
/// - `ncols` must match the number of columns in both input CSC matrices.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub(super) unsafe fn launch_csc_intersect_count(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    col_ptrs_a: u64,
    row_indices_a: u64,
    col_ptrs_b: u64,
    row_indices_b: u64,
    col_counts: u64,
    ncols: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
        let func = get_kernel_function(&module, "csc_intersect_count")?;

        let block_size = BLOCK_SIZE;
        let grid_size = (ncols as u32 + block_size - 1) / block_size;
        let ncols_i32 = ncols as i32;

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&col_ptrs_a);
        builder.arg(&row_indices_a);
        builder.arg(&col_ptrs_b);
        builder.arg(&row_indices_b);
        builder.arg(&col_counts);
        builder.arg(&ncols_i32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA CSC intersect count kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

// ============================================================================
// Compute Kernels
// ============================================================================

/// Launch CSC add compute kernel
///
/// # Safety
///
/// - All pointer arguments must be valid device memory pointers on the device associated
///   with `context`. Output buffers must be pre-allocated to the correct sizes.
/// - `ncols` must match the number of columns in both input CSC matrices.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub(super) unsafe fn launch_csc_add_compute<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    col_ptrs_a: u64,
    row_indices_a: u64,
    values_a: u64,
    col_ptrs_b: u64,
    row_indices_b: u64,
    values_b: u64,
    out_col_ptrs: u64,
    out_row_indices: u64,
    out_values: u64,
    ncols: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "csc_add_compute_f32",
        "f64" => "csc_add_compute_f64",
        "__half" => "csc_add_compute_f16",
        "__nv_bfloat16" => "csc_add_compute_bf16",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for sparse CSC add: {}",
                T::NAME
            )));
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let block_size = BLOCK_SIZE;
        let grid_size = (ncols as u32 + block_size - 1) / block_size;
        let ncols_i32 = ncols as i32;

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&col_ptrs_a);
        builder.arg(&row_indices_a);
        builder.arg(&values_a);
        builder.arg(&col_ptrs_b);
        builder.arg(&row_indices_b);
        builder.arg(&values_b);
        builder.arg(&out_col_ptrs);
        builder.arg(&out_row_indices);
        builder.arg(&out_values);
        builder.arg(&ncols_i32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA CSC add compute kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch CSC sub compute kernel
///
/// # Safety
///
/// - All pointer arguments must be valid device memory pointers on the device associated
///   with `context`. Output buffers must be pre-allocated to the correct sizes.
/// - `ncols` must match the number of columns in both input CSC matrices.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub(super) unsafe fn launch_csc_sub_compute<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    col_ptrs_a: u64,
    row_indices_a: u64,
    values_a: u64,
    col_ptrs_b: u64,
    row_indices_b: u64,
    values_b: u64,
    out_col_ptrs: u64,
    out_row_indices: u64,
    out_values: u64,
    ncols: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "csc_sub_compute_f32",
        "f64" => "csc_sub_compute_f64",
        "__half" => "csc_sub_compute_f16",
        "__nv_bfloat16" => "csc_sub_compute_bf16",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for sparse CSC sub: {}",
                T::NAME
            )));
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let block_size = BLOCK_SIZE;
        let grid_size = (ncols as u32 + block_size - 1) / block_size;
        let ncols_i32 = ncols as i32;

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&col_ptrs_a);
        builder.arg(&row_indices_a);
        builder.arg(&values_a);
        builder.arg(&col_ptrs_b);
        builder.arg(&row_indices_b);
        builder.arg(&values_b);
        builder.arg(&out_col_ptrs);
        builder.arg(&out_row_indices);
        builder.arg(&out_values);
        builder.arg(&ncols_i32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA CSC sub compute kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch CSC mul compute kernel
///
/// # Safety
///
/// - All pointer arguments must be valid device memory pointers on the device associated
///   with `context`. Output buffers must be pre-allocated to the correct sizes.
/// - `ncols` must match the number of columns in both input CSC matrices.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub(super) unsafe fn launch_csc_mul_compute<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    col_ptrs_a: u64,
    row_indices_a: u64,
    values_a: u64,
    col_ptrs_b: u64,
    row_indices_b: u64,
    values_b: u64,
    out_col_ptrs: u64,
    out_row_indices: u64,
    out_values: u64,
    ncols: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "csc_mul_compute_f32",
        "f64" => "csc_mul_compute_f64",
        "__half" => "csc_mul_compute_f16",
        "__nv_bfloat16" => "csc_mul_compute_bf16",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for sparse CSC mul: {}",
                T::NAME
            )));
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let block_size = BLOCK_SIZE;
        let grid_size = (ncols as u32 + block_size - 1) / block_size;
        let ncols_i32 = ncols as i32;

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&col_ptrs_a);
        builder.arg(&row_indices_a);
        builder.arg(&values_a);
        builder.arg(&col_ptrs_b);
        builder.arg(&row_indices_b);
        builder.arg(&values_b);
        builder.arg(&out_col_ptrs);
        builder.arg(&out_row_indices);
        builder.arg(&out_values);
        builder.arg(&ncols_i32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA CSC mul compute kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch CSC div compute kernel
///
/// # Safety
///
/// - All pointer arguments must be valid device memory pointers on the device associated
///   with `context`. Output buffers must be pre-allocated to the correct sizes.
/// - `ncols` must match the number of columns in both input CSC matrices.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub(super) unsafe fn launch_csc_div_compute<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    col_ptrs_a: u64,
    row_indices_a: u64,
    values_a: u64,
    col_ptrs_b: u64,
    row_indices_b: u64,
    values_b: u64,
    out_col_ptrs: u64,
    out_row_indices: u64,
    out_values: u64,
    ncols: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "csc_div_compute_f32",
        "f64" => "csc_div_compute_f64",
        "__half" => "csc_div_compute_f16",
        "__nv_bfloat16" => "csc_div_compute_bf16",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for sparse CSC div: {}",
                T::NAME
            )));
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let block_size = BLOCK_SIZE;
        let grid_size = (ncols as u32 + block_size - 1) / block_size;
        let ncols_i32 = ncols as i32;

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&col_ptrs_a);
        builder.arg(&row_indices_a);
        builder.arg(&values_a);
        builder.arg(&col_ptrs_b);
        builder.arg(&row_indices_b);
        builder.arg(&values_b);
        builder.arg(&out_col_ptrs);
        builder.arg(&out_row_indices);
        builder.arg(&out_values);
        builder.arg(&ncols_i32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA CSC div compute kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

// ============================================================================
// High-level CSC Merge Operations
// ============================================================================

/// Two-pass CSC addition: C = A + B (union semantics)
///
/// # Safety
///
/// All tensor arguments must contain valid CUDA device pointers with correct sizes
/// for the given sparse CSC format. `ncols` must match the sparse matrix dimensions.
pub unsafe fn csc_add_merge<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    col_ptrs_a: &Tensor<CudaRuntime>,
    row_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    col_ptrs_b: &Tensor<CudaRuntime>,
    row_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    ncols: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    use super::super::sparse_strategy::AddMerge;
    super::generic::generic_csc_merge::<T, AddMerge>(
        context,
        stream,
        device_index,
        device,
        dtype,
        col_ptrs_a,
        row_indices_a,
        values_a,
        col_ptrs_b,
        row_indices_b,
        values_b,
        ncols,
    )
}

/// Two-pass CSC subtraction: C = A - B (union semantics)
///
/// # Safety
///
/// All tensor arguments must contain valid CUDA device pointers with correct sizes
/// for the given sparse CSC format. `ncols` must match the sparse matrix dimensions.
pub unsafe fn csc_sub_merge<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    col_ptrs_a: &Tensor<CudaRuntime>,
    row_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    col_ptrs_b: &Tensor<CudaRuntime>,
    row_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    ncols: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    use super::super::sparse_strategy::SubMerge;
    super::generic::generic_csc_merge::<T, SubMerge>(
        context,
        stream,
        device_index,
        device,
        dtype,
        col_ptrs_a,
        row_indices_a,
        values_a,
        col_ptrs_b,
        row_indices_b,
        values_b,
        ncols,
    )
}

/// Two-pass CSC element-wise multiplication: C = A .* B (intersection semantics)
///
/// # Safety
///
/// All tensor arguments must contain valid CUDA device pointers with correct sizes
/// for the given sparse CSC format. `ncols` must match the sparse matrix dimensions.
pub unsafe fn csc_mul_merge<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    col_ptrs_a: &Tensor<CudaRuntime>,
    row_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    col_ptrs_b: &Tensor<CudaRuntime>,
    row_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    ncols: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    use super::super::sparse_strategy::MulMerge;
    super::generic::generic_csc_merge::<T, MulMerge>(
        context,
        stream,
        device_index,
        device,
        dtype,
        col_ptrs_a,
        row_indices_a,
        values_a,
        col_ptrs_b,
        row_indices_b,
        values_b,
        ncols,
    )
}

/// Two-pass CSC element-wise division: C = A ./ B (intersection semantics)
///
/// # Safety
///
/// All tensor arguments must contain valid CUDA device pointers with correct sizes
/// for the given sparse CSC format. `ncols` must match the sparse matrix dimensions.
pub unsafe fn csc_div_merge<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    col_ptrs_a: &Tensor<CudaRuntime>,
    row_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    col_ptrs_b: &Tensor<CudaRuntime>,
    row_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    ncols: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    use super::super::sparse_strategy::DivMerge;
    super::generic::generic_csc_merge::<T, DivMerge>(
        context,
        stream,
        device_index,
        device,
        dtype,
        col_ptrs_a,
        row_indices_a,
        values_a,
        col_ptrs_b,
        row_indices_b,
        values_b,
        ncols,
    )
}
