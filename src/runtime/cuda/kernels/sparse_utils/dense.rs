//! CSR to dense, and dense to COO.

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
// Sparse to Dense Conversion
// ============================================================================

/// Expand CSR sparse matrix to a dense matrix (GPU kernel)
///
/// # Safety
///
/// - `row_ptrs`, `col_indices`, and `values` must be valid `CudaRuntime` tensors on the device
///   associated with `context` with a consistent CSR structure.
/// - `shape` must match the actual matrix dimensions: `row_ptrs` has `shape[0] + 1` elements,
///   `col_indices` and `values` have `nnz` elements, all column indices are in `[0, shape[1])`.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn csr_to_dense_gpu<T: Element>(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
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

    let out = Tensor::<CudaRuntime>::zeros(&[nrows, ncols], dtype, device)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (nrows as u32 + block_size - 1) / block_size;
    let nrows_u32 = nrows as u32;
    let ncols_u32 = ncols as u32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);

    let row_ptrs_ptr = row_ptrs.ptr();
    let col_indices_ptr = col_indices.ptr();
    let values_ptr = values.ptr();
    let out_ptr = out.ptr();

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

/// Pass 1: Count non-zeros per row for dense-to-COO conversion
///
/// # Safety
///
/// - `input` must be a valid device memory pointer for a 2D row-major array of at least
///   `nrows * ncols` elements of type `T`.
/// - `row_counts` must be a valid device memory pointer with at least `nrows` i32 elements.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
unsafe fn launch_dense_to_coo_count<T: Element + Copy + cudarc::driver::DeviceRepr>(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
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

/// Pass 2: Extract COO triplets from a dense matrix
///
/// # Safety
///
/// - `input` must be a valid device memory pointer for a 2D row-major array of at least
///   `nrows * ncols` elements of type `T`.
/// - `offsets`, `row_indices`, `col_indices`, and `values` must be valid device memory pointers
///   pre-allocated to hold at least `total_nnz` elements (as determined by the count pass).
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
unsafe fn launch_dense_to_coo_extract<T: Element + Copy + cudarc::driver::DeviceRepr>(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
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

/// Convert a dense matrix to COO sparse format (GPU-only, no CPU transfer)
///
/// Two-pass algorithm: count non-zeros per row, exclusive scan to get offsets, then extract
/// COO triplets. All operations remain on the GPU.
///
/// # Safety
///
/// - `input` must be a valid `CudaRuntime` tensor on the device associated with `context`.
/// - `input` must be a 2D tensor; calling with a tensor of any other rank returns an error.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn dense_to_coo_gpu<T: Element + Copy + cudarc::driver::DeviceRepr>(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
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
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: format!("dense_to_coo requires a 2D tensor, got {}D", shape.len()),
        });
    }

    let nrows = shape[0];
    let ncols = shape[1];

    // Pass 1: Count non-zeros per row
    let row_counts = Tensor::<CudaRuntime>::zeros(&[nrows], DType::I32, device)?;

    unsafe {
        launch_dense_to_coo_count::<T>(
            context,
            stream,
            device_index,
            input.ptr(),
            row_counts.ptr(),
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
    let (offsets_i32, total_nnz) = super::super::scan::exclusive_scan_i32_gpu(
        context,
        stream,
        device_index,
        device,
        &row_counts,
    )?;

    // Convert I32 offsets to I64 (COO format uses I64)
    let offsets = cast_i32_to_i64_gpu(context, stream, device_index, device, &offsets_i32)?;

    // Pass 2: Extract COO data
    let row_indices = Tensor::<CudaRuntime>::zeros(&[total_nnz], DType::I64, device)?;
    let col_indices = Tensor::<CudaRuntime>::zeros(&[total_nnz], DType::I64, device)?;
    let values = Tensor::<CudaRuntime>::zeros(&[total_nnz], dtype, device)?;

    unsafe {
        launch_dense_to_coo_extract::<T>(
            context,
            stream,
            device_index,
            input.ptr(),
            offsets.ptr(),
            row_indices.ptr(),
            col_indices.ptr(),
            values.ptr(),
            nrows,
            ncols,
            threshold,
        )?;
    }

    Ok((row_indices, col_indices, values))
}
