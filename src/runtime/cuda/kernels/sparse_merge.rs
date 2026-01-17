//! Sparse matrix element-wise merge kernel launchers
//!
//! Two-pass algorithm for CSR element-wise operations:
//! 1. Count output size per row
//! 2. Exclusive scan to get row_ptrs
//! 3. Compute merged output

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
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
        let func = get_kernel_function(&module, "csr_merge_count")?;

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

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA sparse merge count kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
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
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
        let func = get_kernel_function(&module, "csr_mul_count")?;

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

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA sparse mul count kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
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
    let kernel_name = match T::NAME {
        "f32" => "csr_add_compute_f32",
        "f64" => "csr_add_compute_f64",
        "__half" => "csr_add_compute_f16",
        "__nv_bfloat16" => "csr_add_compute_bf16",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for sparse add: {}",
                T::NAME
            )));
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

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

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA sparse add compute kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
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
    let kernel_name = match T::NAME {
        "f32" => "csr_sub_compute_f32",
        "f64" => "csr_sub_compute_f64",
        "__half" => "csr_sub_compute_f16",
        "__nv_bfloat16" => "csr_sub_compute_bf16",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for sparse sub: {}",
                T::NAME
            )));
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

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

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA sparse sub compute kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
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
    let kernel_name = match T::NAME {
        "f32" => "csr_mul_compute_f32",
        "f64" => "csr_mul_compute_f64",
        "__half" => "csr_mul_compute_f16",
        "__nv_bfloat16" => "csr_mul_compute_bf16",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for sparse mul: {}",
                T::NAME
            )));
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

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

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA sparse mul compute kernel launch failed: {:?}",
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
    // Pass 1: Count output size per row
    let row_counts = Tensor::<CudaRuntime>::zeros(&[nrows], DType::I32, device);

    unsafe {
        launch_csr_merge_count(
            context,
            stream,
            device_index,
            row_ptrs_a.storage().ptr(),
            col_indices_a.storage().ptr(),
            row_ptrs_b.storage().ptr(),
            col_indices_b.storage().ptr(),
            row_counts.storage().ptr(),
            nrows,
        )?;
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

    unsafe {
        launch_csr_add_compute::<T>(
            context,
            stream,
            device_index,
            row_ptrs_a.storage().ptr(),
            col_indices_a.storage().ptr(),
            values_a.storage().ptr(),
            row_ptrs_b.storage().ptr(),
            col_indices_b.storage().ptr(),
            values_b.storage().ptr(),
            out_row_ptrs.storage().ptr(),
            out_col_indices.storage().ptr(),
            out_values.storage().ptr(),
            nrows,
        )?;
    }

    Ok((out_row_ptrs, out_col_indices, out_values))
}

/// Two-pass CSR subtraction: C = A - B (union semantics)
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
    // Pass 1: Count output size per row (same as add)
    let row_counts = Tensor::<CudaRuntime>::zeros(&[nrows], DType::I32, device);

    unsafe {
        launch_csr_merge_count(
            context,
            stream,
            device_index,
            row_ptrs_a.storage().ptr(),
            col_indices_a.storage().ptr(),
            row_ptrs_b.storage().ptr(),
            col_indices_b.storage().ptr(),
            row_counts.storage().ptr(),
            nrows,
        )?;
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

    unsafe {
        launch_csr_sub_compute::<T>(
            context,
            stream,
            device_index,
            row_ptrs_a.storage().ptr(),
            col_indices_a.storage().ptr(),
            values_a.storage().ptr(),
            row_ptrs_b.storage().ptr(),
            col_indices_b.storage().ptr(),
            values_b.storage().ptr(),
            out_row_ptrs.storage().ptr(),
            out_col_indices.storage().ptr(),
            out_values.storage().ptr(),
            nrows,
        )?;
    }

    Ok((out_row_ptrs, out_col_indices, out_values))
}

/// Two-pass CSR element-wise multiplication: C = A .* B (intersection semantics)
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
    // Pass 1: Count output size per row (intersection)
    let row_counts = Tensor::<CudaRuntime>::zeros(&[nrows], DType::I32, device);

    unsafe {
        launch_csr_mul_count(
            context,
            stream,
            device_index,
            row_ptrs_a.storage().ptr(),
            col_indices_a.storage().ptr(),
            row_ptrs_b.storage().ptr(),
            col_indices_b.storage().ptr(),
            row_counts.storage().ptr(),
            nrows,
        )?;
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

    unsafe {
        launch_csr_mul_compute::<T>(
            context,
            stream,
            device_index,
            row_ptrs_a.storage().ptr(),
            col_indices_a.storage().ptr(),
            values_a.storage().ptr(),
            row_ptrs_b.storage().ptr(),
            col_indices_b.storage().ptr(),
            values_b.storage().ptr(),
            out_row_ptrs.storage().ptr(),
            out_col_indices.storage().ptr(),
            out_values.storage().ptr(),
            nrows,
        )?;
    }

    Ok((out_row_ptrs, out_col_indices, out_values))
}
