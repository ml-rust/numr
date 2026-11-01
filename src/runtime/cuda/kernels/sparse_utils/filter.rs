//! CSR filtering: drop stored values below a threshold and rebuild the CSR.

#![allow(unsafe_op_in_unsafe_fn)]

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::super::loader::{BLOCK_SIZE, get_kernel_function, get_or_load_module, launch_config};
use super::common::{SPARSE_UTILS_MODULE, cast_i32_to_i64_gpu, dtype_suffix};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::runtime::cuda::CudaRuntime;
use crate::runtime::cuda::GuardedStream;
use crate::tensor::Tensor;
// ============================================================================
// CSR Filtering (two-pass)
// ============================================================================

/// Pass 1: Count values above threshold per row
///
/// # Safety
///
/// - `row_ptrs`, `values`, and `row_counts` must be valid device memory pointers on the device
///   associated with `context`.
/// - `row_ptrs` must have at least `nrows + 1` elements; `row_counts` must have at least `nrows`.
/// - `values` must have at least as many elements as indicated by `row_ptrs[nrows]`.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
unsafe fn launch_filter_csr_count<T: Element + Copy + cudarc::driver::DeviceRepr>(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
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
///
/// # Safety
///
/// - All pointer arguments must be valid device memory pointers on the device associated with
///   `context`. Output buffers (`out_row_ptrs`, `out_col_indices`, `out_values`) must be
///   pre-allocated to the correct sizes (determined by the count pass and exclusive scan).
/// - `nrows` must match the number of rows in the input CSR matrix.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
unsafe fn launch_filter_csr_compute<T: Element + Copy + cudarc::driver::DeviceRepr>(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
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

/// Filter CSR values below a threshold (GPU-only, no CPU transfer)
///
/// Two-pass algorithm: count surviving values per row, exclusive scan to get output offsets,
/// then copy filtered data. All operations remain on the GPU.
///
/// # Safety
///
/// - `row_ptrs`, `col_indices`, and `values` must be valid `CudaRuntime` tensors on the device
///   associated with `context` with consistent CSR structure.
/// - `shape` must match the actual matrix dimensions.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn filter_csr_values_gpu<T: Element + Copy + cudarc::driver::DeviceRepr>(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
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
    let row_counts = Tensor::<CudaRuntime>::zeros(&[nrows], DType::I32, device)?;

    unsafe {
        launch_filter_csr_count::<T>(
            context,
            stream,
            device_index,
            row_ptrs.ptr(),
            values.ptr(),
            row_counts.ptr(),
            nrows,
            threshold,
        )?;
    }

    // Synchronize before scan
    stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("Stream synchronize failed: {:?}", e)))?;

    // Exclusive scan to get output row_ptrs (I32)
    let (out_row_ptrs_i32, total_nnz) = super::super::scan::exclusive_scan_i32_gpu(
        context,
        stream,
        device_index,
        device,
        &row_counts,
    )?;

    // Convert I32 row_ptrs to I64 (CSR format uses I64)
    let out_row_ptrs =
        cast_i32_to_i64_gpu(context, stream, device_index, device, &out_row_ptrs_i32)?;

    // Pass 2: Copy filtered data
    let out_col_indices = Tensor::<CudaRuntime>::zeros(&[total_nnz], DType::I64, device)?;
    let out_values = Tensor::<CudaRuntime>::zeros(&[total_nnz], dtype, device)?;

    unsafe {
        launch_filter_csr_compute::<T>(
            context,
            stream,
            device_index,
            row_ptrs.ptr(),
            col_indices.ptr(),
            values.ptr(),
            out_row_ptrs.ptr(),
            out_col_indices.ptr(),
            out_values.ptr(),
            nrows,
            threshold,
        )?;
    }

    Ok((out_row_ptrs, out_col_indices, out_values))
}
