//! Sparse matrix element-wise merge kernel launchers
//!
//! Two-pass algorithm for CSR element-wise operations:
//! 1. Count output size per row
//! 2. Exclusive scan to get row_ptrs
//! 3. Compute merged output

#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use cudarc::types::CudaTypeName;
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, get_kernel_function, get_or_load_module, kernel_names, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::runtime::cuda::CudaRuntime;
use crate::tensor::Tensor;

// ============================================================================
// Generic Kernel Launcher Helpers (DRY principle)
// ============================================================================

/// Get dtype-specific kernel name suffix
fn dtype_suffix<T: CudaTypeName>() -> Result<&'static str> {
    match T::NAME {
        "f32" => Ok("f32"),
        "f64" => Ok("f64"),
        "__half" => Ok("f16"),
        "__nv_bfloat16" => Ok("bf16"),
        _ => Err(Error::Internal(format!(
            "Unsupported dtype for sparse operation: {}",
            T::NAME
        ))),
    }
}

/// Generic launcher for kernels without dtype template (count kernels)
///
/// Eliminates duplication across count kernel launchers
unsafe fn launch_count_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    kernel_name: &str,
    row_ptrs_a: u64,
    col_indices_a: u64,
    row_ptrs_b: u64,
    col_indices_b: u64,
    row_counts: u64,
    nrows: usize,
    error_context: &str,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
    let func = get_kernel_function(&module, kernel_name)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (nrows as u32 + block_size - 1) / block_size;
    let nrows_i32 = nrows as i32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&row_ptrs_a);
    builder.arg(&col_indices_a);
    builder.arg(&row_ptrs_b);
    builder.arg(&col_indices_b);
    builder.arg(&row_counts);
    builder.arg(&nrows_i32);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("{} kernel launch failed: {:?}", error_context, e)))?;

    Ok(())
}

/// Generic launcher for dtype-templated compute kernels (CSR format)
///
/// Eliminates duplication across CSR add/sub/mul/div compute launchers
unsafe fn launch_csr_compute_kernel<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    kernel_base_name: &str,
    row_ptrs_a: u64,
    col_indices_a: u64,
    values_a: u64,
    row_ptrs_b: u64,
    col_indices_b: u64,
    values_b: u64,
    out_row_ptrs: u64,
    out_col_indices: u64,
    out_values: u64,
    nrows: usize,
    error_context: &str,
) -> Result<()> {
    let suffix = dtype_suffix::<T>()?;
    let kernel_name = format!("{}_{}", kernel_base_name, suffix);

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (nrows as u32 + block_size - 1) / block_size;
    let nrows_i32 = nrows as i32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&row_ptrs_a);
    builder.arg(&col_indices_a);
    builder.arg(&values_a);
    builder.arg(&row_ptrs_b);
    builder.arg(&col_indices_b);
    builder.arg(&values_b);
    builder.arg(&out_row_ptrs);
    builder.arg(&out_col_indices);
    builder.arg(&out_values);
    builder.arg(&nrows_i32);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("{} kernel launch failed: {:?}", error_context, e)))?;

    Ok(())
}

/// Generic launcher for dtype-templated compute kernels (CSC format)
///
/// Eliminates duplication across CSC add/sub/mul/div compute launchers
unsafe fn launch_csc_compute_kernel<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    kernel_base_name: &str,
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
    error_context: &str,
) -> Result<()> {
    let suffix = dtype_suffix::<T>()?;
    let kernel_name = format!("{}_{}", kernel_base_name, suffix);

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

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

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("{} kernel launch failed: {:?}", error_context, e)))?;

    Ok(())
}

// ============================================================================
// Exclusive Scan (Prefix Sum)
// ============================================================================

/// Compute exclusive scan (prefix sum) on GPU tensor
///
/// Input: [3, 1, 4, 2]
/// Output: [0, 3, 4, 8, 10] (n+1 elements, last is total sum)
///
/// Uses GPU-native parallel scan (no CPU transfer)
fn exclusive_scan_i32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    input: &Tensor<CudaRuntime>,
) -> Result<(Tensor<CudaRuntime>, usize)> {
    let device = input.device();

    // Use GPU scan (imported from scan module)
    unsafe { super::scan::exclusive_scan_i32_gpu(context, stream, device_index, device, input) }
}

// ============================================================================
// Count Kernels
// ============================================================================

/// Launch CSR merge count kernel (for add/sub operations)
///
/// Counts output size per row using union semantics
unsafe fn launch_csr_merge_count(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs_a: u64,
    col_indices_a: u64,
    row_ptrs_b: u64,
    col_indices_b: u64,
    row_counts: u64,
    nrows: usize,
) -> Result<()> {
    launch_count_kernel(
        context,
        stream,
        device_index,
        "csr_merge_count",
        row_ptrs_a,
        col_indices_a,
        row_ptrs_b,
        col_indices_b,
        row_counts,
        nrows,
        "CUDA sparse merge count",
    )
}

/// Launch CSR mul count kernel (intersection semantics)
unsafe fn launch_csr_mul_count(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs_a: u64,
    col_indices_a: u64,
    row_ptrs_b: u64,
    col_indices_b: u64,
    row_counts: u64,
    nrows: usize,
) -> Result<()> {
    launch_count_kernel(
        context,
        stream,
        device_index,
        "csr_mul_count",
        row_ptrs_a,
        col_indices_a,
        row_ptrs_b,
        col_indices_b,
        row_counts,
        nrows,
        "CUDA sparse mul count",
    )
}

// ============================================================================
// Compute Kernels
// ============================================================================

/// Launch CSR add compute kernel
unsafe fn launch_csr_add_compute<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs_a: u64,
    col_indices_a: u64,
    values_a: u64,
    row_ptrs_b: u64,
    col_indices_b: u64,
    values_b: u64,
    out_row_ptrs: u64,
    out_col_indices: u64,
    out_values: u64,
    nrows: usize,
) -> Result<()> {
    launch_csr_compute_kernel::<T>(
        context,
        stream,
        device_index,
        "csr_add_compute",
        row_ptrs_a,
        col_indices_a,
        values_a,
        row_ptrs_b,
        col_indices_b,
        values_b,
        out_row_ptrs,
        out_col_indices,
        out_values,
        nrows,
        "CUDA sparse add compute",
    )
}

/// Launch CSR sub compute kernel
unsafe fn launch_csr_sub_compute<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs_a: u64,
    col_indices_a: u64,
    values_a: u64,
    row_ptrs_b: u64,
    col_indices_b: u64,
    values_b: u64,
    out_row_ptrs: u64,
    out_col_indices: u64,
    out_values: u64,
    nrows: usize,
) -> Result<()> {
    launch_csr_compute_kernel::<T>(
        context,
        stream,
        device_index,
        "csr_sub_compute",
        row_ptrs_a,
        col_indices_a,
        values_a,
        row_ptrs_b,
        col_indices_b,
        values_b,
        out_row_ptrs,
        out_col_indices,
        out_values,
        nrows,
        "CUDA sparse sub compute",
    )
}

/// Launch CSR mul compute kernel
unsafe fn launch_csr_mul_compute<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs_a: u64,
    col_indices_a: u64,
    values_a: u64,
    row_ptrs_b: u64,
    col_indices_b: u64,
    values_b: u64,
    out_row_ptrs: u64,
    out_col_indices: u64,
    out_values: u64,
    nrows: usize,
) -> Result<()> {
    launch_csr_compute_kernel::<T>(
        context,
        stream,
        device_index,
        "csr_mul_compute",
        row_ptrs_a,
        col_indices_a,
        values_a,
        row_ptrs_b,
        col_indices_b,
        values_b,
        out_row_ptrs,
        out_col_indices,
        out_values,
        nrows,
        "CUDA sparse mul compute",
    )
}

/// Launch CSR div compute kernel
unsafe fn launch_csr_div_compute<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs_a: u64,
    col_indices_a: u64,
    values_a: u64,
    row_ptrs_b: u64,
    col_indices_b: u64,
    values_b: u64,
    out_row_ptrs: u64,
    out_col_indices: u64,
    out_values: u64,
    nrows: usize,
) -> Result<()> {
    launch_csr_compute_kernel::<T>(
        context,
        stream,
        device_index,
        "csr_div_compute",
        row_ptrs_a,
        col_indices_a,
        values_a,
        row_ptrs_b,
        col_indices_b,
        values_b,
        out_row_ptrs,
        out_col_indices,
        out_values,
        nrows,
        "CUDA sparse div compute",
    )
}

/// Launch CSC intersect count kernel (for mul/div)
unsafe fn launch_csc_intersect_count(
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

/// Launch CSC add compute kernel
unsafe fn launch_csc_add_compute<T: CudaTypeName>(
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
unsafe fn launch_csc_sub_compute<T: CudaTypeName>(
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
unsafe fn launch_csc_mul_compute<T: CudaTypeName>(
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
unsafe fn launch_csc_div_compute<T: CudaTypeName>(
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
// High-level Merge Operations
// ============================================================================

/// Two-pass CSR addition: C = A + B (union semantics)
///
/// Now uses generic_csr_merge with AddMerge strategy to eliminate duplication.
pub unsafe fn csr_add_merge<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_ptrs_a: &Tensor<CudaRuntime>,
    col_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    row_ptrs_b: &Tensor<CudaRuntime>,
    col_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    nrows: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    use super::sparse_strategy::AddMerge;
    generic_csr_merge::<T, AddMerge>(
        context,
        stream,
        device_index,
        device,
        dtype,
        row_ptrs_a,
        col_indices_a,
        values_a,
        row_ptrs_b,
        col_indices_b,
        values_b,
        nrows,
    )
}

/// Two-pass CSR subtraction: C = A - B (union semantics)
///
/// Now uses generic_csr_merge with SubMerge strategy to eliminate duplication.
pub unsafe fn csr_sub_merge<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_ptrs_a: &Tensor<CudaRuntime>,
    col_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    row_ptrs_b: &Tensor<CudaRuntime>,
    col_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    nrows: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    use super::sparse_strategy::SubMerge;
    generic_csr_merge::<T, SubMerge>(
        context,
        stream,
        device_index,
        device,
        dtype,
        row_ptrs_a,
        col_indices_a,
        values_a,
        row_ptrs_b,
        col_indices_b,
        values_b,
        nrows,
    )
}

/// Two-pass CSR element-wise multiplication: C = A .* B (intersection semantics)
///
/// Now uses generic_csr_merge with MulMerge strategy to eliminate duplication.
pub unsafe fn csr_mul_merge<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_ptrs_a: &Tensor<CudaRuntime>,
    col_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    row_ptrs_b: &Tensor<CudaRuntime>,
    col_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    nrows: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    use super::sparse_strategy::MulMerge;
    generic_csr_merge::<T, MulMerge>(
        context,
        stream,
        device_index,
        device,
        dtype,
        row_ptrs_a,
        col_indices_a,
        values_a,
        row_ptrs_b,
        col_indices_b,
        values_b,
        nrows,
    )
}

/// Two-pass CSR element-wise division: C = A ./ B (intersection semantics)
pub unsafe fn csr_div_merge<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_ptrs_a: &Tensor<CudaRuntime>,
    col_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    row_ptrs_b: &Tensor<CudaRuntime>,
    col_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    nrows: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    use super::sparse_strategy::DivMerge;
    generic_csr_merge::<T, DivMerge>(
        context,
        stream,
        device_index,
        device,
        dtype,
        row_ptrs_a,
        col_indices_a,
        values_a,
        row_ptrs_b,
        col_indices_b,
        values_b,
        nrows,
    )
}

// ============================================================================
// High-level CSC Merge Operations
// ============================================================================

/// Two-pass CSC addition: C = A + B (union semantics)
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
    use super::sparse_strategy::AddMerge;
    generic_csc_merge::<T, AddMerge>(
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
    use super::sparse_strategy::SubMerge;
    generic_csc_merge::<T, SubMerge>(
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
    use super::sparse_strategy::MulMerge;
    generic_csc_merge::<T, MulMerge>(
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
    use super::sparse_strategy::DivMerge;
    generic_csc_merge::<T, DivMerge>(
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

// ============================================================================
// Generic Merge Implementation (Zero Duplication)
// ============================================================================

use super::sparse_strategy::{MergeStrategy, SparseFormat};

/// Generic two-pass CSR merge using strategy pattern
///
/// Eliminates code duplication across add/sub/mul/div operations by abstracting
/// the merge semantics through the MergeStrategy trait.
///
/// # Type Parameters
///
/// * `T` - Element type (f32, f64, etc.)
/// * `S` - Merge strategy (AddMerge, SubMerge, MulMerge, DivMerge)
///
/// # Algorithm
///
/// 1. **Count**: Determine output size per row using strategy-specific semantics
/// 2. **Scan**: Compute row_ptrs via exclusive prefix sum
/// 3. **Compute**: Merge values using strategy-specific operation
pub unsafe fn generic_csr_merge<T: CudaTypeName, S: MergeStrategy>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_ptrs_a: &Tensor<CudaRuntime>,
    col_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    row_ptrs_b: &Tensor<CudaRuntime>,
    col_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    nrows: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    // Pass 1: Count output size per row
    let row_counts = Tensor::<CudaRuntime>::zeros(&[nrows], DType::I32, device);

    // Launch count kernel (union vs intersection semantics determined by strategy)
    let count_kernel_name = S::count_kernel_name(SparseFormat::Csr);
    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
    let function = get_kernel_function(&module, count_kernel_name)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (nrows as u32 + block_size - 1) / block_size;
    let nrows_i32 = nrows as i32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&function);

    // Store pointers to avoid temporary value issues
    let row_ptrs_a_ptr = row_ptrs_a.storage().ptr();
    let col_indices_a_ptr = col_indices_a.storage().ptr();
    let row_ptrs_b_ptr = row_ptrs_b.storage().ptr();
    let col_indices_b_ptr = col_indices_b.storage().ptr();
    let row_counts_ptr = row_counts.storage().ptr();

    builder.arg(&row_ptrs_a_ptr);
    builder.arg(&col_indices_a_ptr);
    builder.arg(&row_ptrs_b_ptr);
    builder.arg(&col_indices_b_ptr);
    builder.arg(&row_counts_ptr);
    builder.arg(&nrows_i32);

    // SAFETY: Kernel launch is unsafe because:
    // 1. Raw pointers are passed to CUDA kernel
    // 2. Kernel accesses GPU memory
    // Safety requirements satisfied:
    // - All pointers are valid GPU memory addresses from CudaRuntime tensors
    // - Tensor lifetimes ensure memory is valid during kernel execution
    // - nrows matches the actual tensor dimensions
    // - Stream synchronization ensures no data races
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed (nrows={}, strategy={:?}): {:?}",
                count_kernel_name,
                nrows,
                S::OP,
                e
            ))
        })?;
    }

    // Synchronize to ensure counts are ready
    stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("Stream synchronize failed: {:?}", e)))?;

    // Exclusive scan to get row_ptrs and total_nnz
    let (out_row_ptrs, total_nnz) = exclusive_scan_i32(context, stream, device_index, &row_counts)?;

    // Pass 2: Allocate output and compute merged result
    let out_col_indices = Tensor::<CudaRuntime>::zeros(&[total_nnz], DType::I32, device);
    let out_values = Tensor::<CudaRuntime>::zeros(&[total_nnz], dtype, device);

    // Launch compute kernel (operation-specific)
    let compute_kernel_name = S::compute_kernel_name(SparseFormat::Csr, T::NAME);
    let function = get_kernel_function(&module, &compute_kernel_name)?;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&function);

    // Store pointers to avoid temporary value issues
    let row_ptrs_a_ptr = row_ptrs_a.storage().ptr();
    let col_indices_a_ptr = col_indices_a.storage().ptr();
    let values_a_ptr = values_a.storage().ptr();
    let row_ptrs_b_ptr = row_ptrs_b.storage().ptr();
    let col_indices_b_ptr = col_indices_b.storage().ptr();
    let values_b_ptr = values_b.storage().ptr();
    let out_row_ptrs_ptr = out_row_ptrs.storage().ptr();
    let out_col_indices_ptr = out_col_indices.storage().ptr();
    let out_values_ptr = out_values.storage().ptr();

    builder.arg(&row_ptrs_a_ptr);
    builder.arg(&col_indices_a_ptr);
    builder.arg(&values_a_ptr);
    builder.arg(&row_ptrs_b_ptr);
    builder.arg(&col_indices_b_ptr);
    builder.arg(&values_b_ptr);
    builder.arg(&out_row_ptrs_ptr);
    builder.arg(&out_col_indices_ptr);
    builder.arg(&out_values_ptr);
    builder.arg(&nrows_i32);

    // SAFETY: Kernel launch is unsafe because:
    // 1. Raw pointers are passed to CUDA kernel
    // 2. Kernel writes to output tensors
    // Safety requirements satisfied:
    // - All input pointers are valid GPU memory from input tensors
    // - Output tensors allocated with correct size (total_nnz from scan)
    // - Tensor ownership prevents concurrent modification
    // - Stream ordering ensures count kernel completed before compute kernel
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed (nrows={}, total_nnz={}, strategy={:?}): {:?}",
                compute_kernel_name,
                nrows,
                total_nnz,
                S::OP,
                e
            ))
        })?;
    }

    Ok((out_row_ptrs, out_col_indices, out_values))
}

/// Generic two-pass CSC merge using strategy pattern
///
/// CSC variant of generic_csr_merge. See generic_csr_merge for details.
pub unsafe fn generic_csc_merge<T: CudaTypeName, S: MergeStrategy>(
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
    // Pass 1: Count output size per column
    let col_counts = Tensor::<CudaRuntime>::zeros(&[ncols], DType::I32, device);

    // Launch count kernel
    let count_kernel_name = S::count_kernel_name(SparseFormat::Csc);
    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
    let function = get_kernel_function(&module, count_kernel_name)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (ncols as u32 + block_size - 1) / block_size;
    let ncols_i32 = ncols as i32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&function);

    // Store pointers to avoid temporary value issues
    let col_ptrs_a_ptr = col_ptrs_a.storage().ptr();
    let row_indices_a_ptr = row_indices_a.storage().ptr();
    let col_ptrs_b_ptr = col_ptrs_b.storage().ptr();
    let row_indices_b_ptr = row_indices_b.storage().ptr();
    let col_counts_ptr = col_counts.storage().ptr();

    builder.arg(&col_ptrs_a_ptr);
    builder.arg(&row_indices_a_ptr);
    builder.arg(&col_ptrs_b_ptr);
    builder.arg(&row_indices_b_ptr);
    builder.arg(&col_counts_ptr);
    builder.arg(&ncols_i32);

    // SAFETY: Kernel launch is unsafe because:
    // 1. Raw pointers are passed to CUDA kernel
    // 2. Kernel accesses GPU memory
    // Safety requirements satisfied:
    // - All pointers are valid GPU memory addresses from CudaRuntime tensors
    // - Tensor lifetimes ensure memory is valid during kernel execution
    // - ncols matches the actual tensor dimensions
    // - Stream synchronization ensures no data races
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed (ncols={}, strategy={:?}): {:?}",
                count_kernel_name,
                ncols,
                S::OP,
                e
            ))
        })?;
    }

    // Synchronize to ensure counts are ready
    stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("Stream synchronize failed: {:?}", e)))?;

    // Exclusive scan to get col_ptrs and total_nnz
    let (out_col_ptrs, total_nnz) = exclusive_scan_i32(context, stream, device_index, &col_counts)?;

    // Pass 2: Allocate output and compute merged result
    let out_row_indices = Tensor::<CudaRuntime>::zeros(&[total_nnz], DType::I32, device);
    let out_values = Tensor::<CudaRuntime>::zeros(&[total_nnz], dtype, device);

    // Launch compute kernel
    let compute_kernel_name = S::compute_kernel_name(SparseFormat::Csc, T::NAME);
    let function = get_kernel_function(&module, &compute_kernel_name)?;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&function);

    // Store pointers to avoid temporary value issues
    let col_ptrs_a_ptr = col_ptrs_a.storage().ptr();
    let row_indices_a_ptr = row_indices_a.storage().ptr();
    let values_a_ptr = values_a.storage().ptr();
    let col_ptrs_b_ptr = col_ptrs_b.storage().ptr();
    let row_indices_b_ptr = row_indices_b.storage().ptr();
    let values_b_ptr = values_b.storage().ptr();
    let out_col_ptrs_ptr = out_col_ptrs.storage().ptr();
    let out_row_indices_ptr = out_row_indices.storage().ptr();
    let out_values_ptr = out_values.storage().ptr();

    builder.arg(&col_ptrs_a_ptr);
    builder.arg(&row_indices_a_ptr);
    builder.arg(&values_a_ptr);
    builder.arg(&col_ptrs_b_ptr);
    builder.arg(&row_indices_b_ptr);
    builder.arg(&values_b_ptr);
    builder.arg(&out_col_ptrs_ptr);
    builder.arg(&out_row_indices_ptr);
    builder.arg(&out_values_ptr);
    builder.arg(&ncols_i32);

    // SAFETY: Kernel launch is unsafe because:
    // 1. Raw pointers are passed to CUDA kernel
    // 2. Kernel writes to output tensors
    // Safety requirements satisfied:
    // - All input pointers are valid GPU memory from input tensors
    // - Output tensors allocated with correct size (total_nnz from scan)
    // - Tensor ownership prevents concurrent modification
    // - Stream ordering ensures count kernel completed before compute kernel
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed (ncols={}, total_nnz={}, strategy={:?}): {:?}",
                compute_kernel_name,
                ncols,
                total_nnz,
                S::OP,
                e
            ))
        })?;
    }

    Ok((out_col_ptrs, out_row_indices, out_values))
}
