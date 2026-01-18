//! Sparse utility kernel launchers
//!
//! GPU-native implementations to eliminate CPU transfers:
//! - CSR filtering (remove values below threshold)
//! - Row/column sums
//! - NNZ counting per row/column
//! - Sparse<->Dense conversion

#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use cudarc::types::CudaTypeName;
use std::sync::Arc;

use super::loader::{BLOCK_SIZE, get_kernel_function, get_or_load_module, launch_config};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::runtime::cuda::CudaRuntime;
use crate::tensor::Tensor;

// ============================================================================
// Module name
// ============================================================================

pub const SPARSE_UTILS_MODULE: &str = "sparse_utils";

// ============================================================================
// Helper: dtype suffix for kernel names
// ============================================================================

fn dtype_suffix<T: CudaTypeName>() -> Result<&'static str> {
    match T::NAME {
        "float" => Ok("f32"),
        "double" => Ok("f64"),
        "__half" => Ok("f16"),
        "__nv_bfloat16" => Ok("bf16"),
        _ => Err(Error::Internal(format!(
            "Unsupported dtype for sparse utils: {}",
            T::NAME
        ))),
    }
}

// ============================================================================
// Helper: Cast I32 to I64
// ============================================================================

/// Cast I32 tensor to I64 (for row_ptrs after scan)
unsafe fn cast_i32_to_i64_gpu(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    input: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let n = input.numel();
    let output = Tensor::<CudaRuntime>::zeros(&[n], DType::I64, device);

    // Use cast kernel from cast.rs
    use super::cast::launch_cast;

    let input_ptr = input.storage().ptr();
    let output_ptr = output.storage().ptr();

    unsafe {
        launch_cast(
            context,
            stream,
            device_index,
            DType::I32,
            DType::I64,
            input_ptr,
            output_ptr,
            n,
        )?;
    }

    Ok(output)
}

// ============================================================================
// CSR Filtering (two-pass)
// ============================================================================

/// Pass 1: Count values above threshold per row
unsafe fn launch_filter_csr_count<T: CudaTypeName + Copy + cudarc::driver::DeviceRepr>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs: u64,
    values: u64,
    row_counts: u64,
    nrows: usize,
    threshold: T,
) -> Result<()> {
    let suffix = dtype_suffix::<T>()?;
    let kernel_name = format!("filter_csr_count_{}", suffix);

    let module = get_or_load_module(context, device_index, SPARSE_UTILS_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (nrows as u32 + block_size - 1) / block_size;
    let nrows_u32 = nrows as u32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&row_ptrs);
    builder.arg(&values);
    builder.arg(&row_counts);
    builder.arg(&nrows_u32);
    builder.arg(&threshold);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!(
            "CUDA filter_csr_count kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

/// Pass 2: Copy filtered values and indices
unsafe fn launch_filter_csr_compute<T: CudaTypeName + Copy + cudarc::driver::DeviceRepr>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs: u64,
    col_indices: u64,
    values: u64,
    out_row_ptrs: u64,
    out_col_indices: u64,
    out_values: u64,
    nrows: usize,
    threshold: T,
) -> Result<()> {
    let suffix = dtype_suffix::<T>()?;
    let kernel_name = format!("filter_csr_compute_{}", suffix);

    let module = get_or_load_module(context, device_index, SPARSE_UTILS_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (nrows as u32 + block_size - 1) / block_size;
    let nrows_u32 = nrows as u32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&row_ptrs);
    builder.arg(&col_indices);
    builder.arg(&values);
    builder.arg(&out_row_ptrs);
    builder.arg(&out_col_indices);
    builder.arg(&out_values);
    builder.arg(&nrows_u32);
    builder.arg(&threshold);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!(
            "CUDA filter_csr_compute kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

/// High-level CSR filtering (GPU-only, no CPU transfer)
pub unsafe fn filter_csr_values_gpu<T: CudaTypeName + Copy + cudarc::driver::DeviceRepr>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_ptrs: &Tensor<CudaRuntime>,
    col_indices: &Tensor<CudaRuntime>,
    values: &Tensor<CudaRuntime>,
    shape: [usize; 2],
    threshold: T,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    let [nrows, _] = shape;

    // Handle empty matrices
    if values.shape()[0] == 0 {
        return Ok((row_ptrs.clone(), col_indices.clone(), values.clone()));
    }

    // Pass 1: Count values above threshold per row
    let row_counts = Tensor::<CudaRuntime>::zeros(&[nrows], DType::I32, device);

    unsafe {
        launch_filter_csr_count::<T>(
            context,
            stream,
            device_index,
            row_ptrs.storage().ptr(),
            values.storage().ptr(),
            row_counts.storage().ptr(),
            nrows,
            threshold,
        )?;
    }

    // Synchronize before scan
    stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("Stream synchronize failed: {:?}", e)))?;

    // Exclusive scan to get output row_ptrs (I32)
    let (out_row_ptrs_i32, total_nnz) =
        super::scan::exclusive_scan_i32_gpu(context, stream, device_index, device, &row_counts)?;

    // Convert I32 row_ptrs to I64 (CSR format uses I64)
    let out_row_ptrs =
        cast_i32_to_i64_gpu(context, stream, device_index, device, &out_row_ptrs_i32)?;

    // Pass 2: Copy filtered data
    let out_col_indices = Tensor::<CudaRuntime>::zeros(&[total_nnz], DType::I64, device);
    let out_values = Tensor::<CudaRuntime>::zeros(&[total_nnz], dtype, device);

    unsafe {
        launch_filter_csr_compute::<T>(
            context,
            stream,
            device_index,
            row_ptrs.storage().ptr(),
            col_indices.storage().ptr(),
            values.storage().ptr(),
            out_row_ptrs.storage().ptr(),
            out_col_indices.storage().ptr(),
            out_values.storage().ptr(),
            nrows,
            threshold,
        )?;
    }

    Ok((out_row_ptrs, out_col_indices, out_values))
}

// ============================================================================
// Row/Column Sums
// ============================================================================

/// CSR row-wise sum (GPU kernel)
pub unsafe fn csr_sum_rows_gpu<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_ptrs: &Tensor<CudaRuntime>,
    values: &Tensor<CudaRuntime>,
    nrows: usize,
) -> Result<Tensor<CudaRuntime>> {
    let suffix = dtype_suffix::<T>()?;
    let kernel_name = format!("csr_sum_rows_{}", suffix);

    let module = get_or_load_module(context, device_index, SPARSE_UTILS_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let out = Tensor::<CudaRuntime>::zeros(&[nrows], dtype, device);

    let block_size = BLOCK_SIZE;
    let grid_size = (nrows as u32 + block_size - 1) / block_size;
    let nrows_u32 = nrows as u32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);

    let row_ptrs_ptr = row_ptrs.storage().ptr();
    let values_ptr = values.storage().ptr();
    let out_ptr = out.storage().ptr();

    let mut builder = stream.launch_builder(&func);
    builder.arg(&row_ptrs_ptr);
    builder.arg(&values_ptr);
    builder.arg(&out_ptr);
    builder.arg(&nrows_u32);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("CUDA csr_sum_rows kernel launch failed: {:?}", e)))?;

    Ok(out)
}

/// CSC column-wise sum (GPU kernel)
pub unsafe fn csc_sum_cols_gpu<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    col_ptrs: &Tensor<CudaRuntime>,
    values: &Tensor<CudaRuntime>,
    ncols: usize,
) -> Result<Tensor<CudaRuntime>> {
    let suffix = dtype_suffix::<T>()?;
    let kernel_name = format!("csc_sum_cols_{}", suffix);

    let module = get_or_load_module(context, device_index, SPARSE_UTILS_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let out = Tensor::<CudaRuntime>::zeros(&[ncols], dtype, device);

    let block_size = BLOCK_SIZE;
    let grid_size = (ncols as u32 + block_size - 1) / block_size;
    let ncols_u32 = ncols as u32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);

    let col_ptrs_ptr = col_ptrs.storage().ptr();
    let values_ptr = values.storage().ptr();
    let out_ptr = out.storage().ptr();

    let mut builder = stream.launch_builder(&func);
    builder.arg(&col_ptrs_ptr);
    builder.arg(&values_ptr);
    builder.arg(&out_ptr);
    builder.arg(&ncols_u32);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("CUDA csc_sum_cols kernel launch failed: {:?}", e)))?;

    Ok(out)
}

// ============================================================================
// NNZ Counting
// ============================================================================

/// Count non-zeros per row (pointer difference)
pub unsafe fn csr_nnz_per_row_gpu(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    row_ptrs: &Tensor<CudaRuntime>,
    nrows: usize,
) -> Result<Tensor<CudaRuntime>> {
    let module = get_or_load_module(context, device_index, SPARSE_UTILS_MODULE)?;
    let func = get_kernel_function(&module, "csr_nnz_per_row_kernel")?;

    let out = Tensor::<CudaRuntime>::zeros(&[nrows], DType::I64, device);

    let block_size = BLOCK_SIZE;
    let grid_size = (nrows as u32 + block_size - 1) / block_size;
    let nrows_u32 = nrows as u32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);

    let row_ptrs_ptr = row_ptrs.storage().ptr();
    let out_ptr = out.storage().ptr();

    let mut builder = stream.launch_builder(&func);
    builder.arg(&row_ptrs_ptr);
    builder.arg(&out_ptr);
    builder.arg(&nrows_u32);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!(
            "CUDA csr_nnz_per_row kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(out)
}

/// Count non-zeros per column (pointer difference)
pub unsafe fn csc_nnz_per_col_gpu(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    col_ptrs: &Tensor<CudaRuntime>,
    ncols: usize,
) -> Result<Tensor<CudaRuntime>> {
    let module = get_or_load_module(context, device_index, SPARSE_UTILS_MODULE)?;
    let func = get_kernel_function(&module, "csc_nnz_per_col_kernel")?;

    let out = Tensor::<CudaRuntime>::zeros(&[ncols], DType::I64, device);

    let block_size = BLOCK_SIZE;
    let grid_size = (ncols as u32 + block_size - 1) / block_size;
    let ncols_u32 = ncols as u32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);

    let col_ptrs_ptr = col_ptrs.storage().ptr();
    let out_ptr = out.storage().ptr();

    let mut builder = stream.launch_builder(&func);
    builder.arg(&col_ptrs_ptr);
    builder.arg(&out_ptr);
    builder.arg(&ncols_u32);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!(
            "CUDA csc_nnz_per_col kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(out)
}

// ============================================================================
// Sparse to Dense Conversion
// ============================================================================

/// Expand CSR to dense matrix (GPU kernel)
pub unsafe fn csr_to_dense_gpu<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_ptrs: &Tensor<CudaRuntime>,
    col_indices: &Tensor<CudaRuntime>,
    values: &Tensor<CudaRuntime>,
    shape: [usize; 2],
) -> Result<Tensor<CudaRuntime>> {
    let [nrows, ncols] = shape;

    let suffix = dtype_suffix::<T>()?;
    let kernel_name = format!("csr_to_dense_{}", suffix);

    let module = get_or_load_module(context, device_index, SPARSE_UTILS_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let out = Tensor::<CudaRuntime>::zeros(&[nrows, ncols], dtype, device);

    let block_size = BLOCK_SIZE;
    let grid_size = (nrows as u32 + block_size - 1) / block_size;
    let nrows_u32 = nrows as u32;
    let ncols_u32 = ncols as u32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);

    let row_ptrs_ptr = row_ptrs.storage().ptr();
    let col_indices_ptr = col_indices.storage().ptr();
    let values_ptr = values.storage().ptr();
    let out_ptr = out.storage().ptr();

    let mut builder = stream.launch_builder(&func);
    builder.arg(&row_ptrs_ptr);
    builder.arg(&col_indices_ptr);
    builder.arg(&values_ptr);
    builder.arg(&out_ptr);
    builder.arg(&nrows_u32);
    builder.arg(&ncols_u32);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("CUDA csr_to_dense kernel launch failed: {:?}", e)))?;

    Ok(out)
}

// ============================================================================
// Dense to COO Conversion (two-pass)
// ============================================================================

/// Pass 1: Count non-zeros per row
unsafe fn launch_dense_to_coo_count<T: CudaTypeName + Copy + cudarc::driver::DeviceRepr>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    input: u64,
    row_counts: u64,
    nrows: usize,
    ncols: usize,
    threshold: T,
) -> Result<()> {
    let suffix = dtype_suffix::<T>()?;
    let kernel_name = format!("dense_to_coo_count_{}", suffix);

    let module = get_or_load_module(context, device_index, SPARSE_UTILS_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (nrows as u32 + block_size - 1) / block_size;
    let nrows_u32 = nrows as u32;
    let ncols_u32 = ncols as u32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&input);
    builder.arg(&row_counts);
    builder.arg(&nrows_u32);
    builder.arg(&ncols_u32);
    builder.arg(&threshold);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!(
            "CUDA dense_to_coo_count kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

/// Pass 2: Extract COO triplets
unsafe fn launch_dense_to_coo_extract<T: CudaTypeName + Copy + cudarc::driver::DeviceRepr>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    input: u64,
    offsets: u64,
    row_indices: u64,
    col_indices: u64,
    values: u64,
    nrows: usize,
    ncols: usize,
    threshold: T,
) -> Result<()> {
    let suffix = dtype_suffix::<T>()?;
    let kernel_name = format!("dense_to_coo_extract_{}", suffix);

    let module = get_or_load_module(context, device_index, SPARSE_UTILS_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (nrows as u32 + block_size - 1) / block_size;
    let nrows_u32 = nrows as u32;
    let ncols_u32 = ncols as u32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&input);
    builder.arg(&offsets);
    builder.arg(&row_indices);
    builder.arg(&col_indices);
    builder.arg(&values);
    builder.arg(&nrows_u32);
    builder.arg(&ncols_u32);
    builder.arg(&threshold);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!(
            "CUDA dense_to_coo_extract kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

/// High-level dense to COO conversion (GPU-only)
pub unsafe fn dense_to_coo_gpu<T: CudaTypeName + Copy + cudarc::driver::DeviceRepr>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    input: &Tensor<CudaRuntime>,
    threshold: T,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(Error::ShapeMismatch {
            expected: vec![0, 0], // placeholder
            got: shape.to_vec(),
        });
    }

    let nrows = shape[0];
    let ncols = shape[1];

    // Pass 1: Count non-zeros per row
    let row_counts = Tensor::<CudaRuntime>::zeros(&[nrows], DType::I32, device);

    unsafe {
        launch_dense_to_coo_count::<T>(
            context,
            stream,
            device_index,
            input.storage().ptr(),
            row_counts.storage().ptr(),
            nrows,
            ncols,
            threshold,
        )?;
    }

    // Synchronize before scan
    stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("Stream synchronize failed: {:?}", e)))?;

    // Exclusive scan to get offsets (I32)
    let (offsets_i32, total_nnz) =
        super::scan::exclusive_scan_i32_gpu(context, stream, device_index, device, &row_counts)?;

    // Convert I32 offsets to I64 (COO format uses I64)
    let offsets = cast_i32_to_i64_gpu(context, stream, device_index, device, &offsets_i32)?;

    // Pass 2: Extract COO data
    let row_indices = Tensor::<CudaRuntime>::zeros(&[total_nnz], DType::I64, device);
    let col_indices = Tensor::<CudaRuntime>::zeros(&[total_nnz], DType::I64, device);
    let values = Tensor::<CudaRuntime>::zeros(&[total_nnz], dtype, device);

    unsafe {
        launch_dense_to_coo_extract::<T>(
            context,
            stream,
            device_index,
            input.storage().ptr(),
            offsets.storage().ptr(),
            row_indices.storage().ptr(),
            col_indices.storage().ptr(),
            values.storage().ptr(),
            nrows,
            ncols,
            threshold,
        )?;
    }

    Ok((row_indices, col_indices, values))
}
