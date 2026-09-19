//! `gather` and `gather_2d` for the CUDA runtime.

use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::{launch_gather, launch_gather_2d};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{compute_contiguous_strides, ensure_contiguous};
use crate::tensor::Tensor;

use super::super::helpers::normalize_indices_to_i64;

/// Execute gather operation along dimension.
pub fn gather(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dim: usize,
    index: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let index_i64 = normalize_indices_to_i64(client, index)?;

    // Validate dimension
    let ndim = a.ndim();
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    // Validate index tensor has same number of dimensions
    if index_i64.ndim() != ndim {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: index_i64.shape().to_vec(),
        });
    }

    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let index_contig = ensure_contiguous(&index_i64)?;

    // Output has same shape as index
    let out_shape = index_i64.shape().to_vec();
    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

    // Prepare shape and stride arrays as host Vecs (passed as scalar args, no device alloc needed)
    let input_shape: Vec<u32> = a.shape().iter().map(|&s| s as u32).collect();
    let input_strides: Vec<u32> = compute_contiguous_strides(a.shape())
        .iter()
        .map(|&s| s as u32)
        .collect();
    let output_shape: Vec<u32> = out_shape.iter().map(|&s| s as u32).collect();
    let output_strides: Vec<u32> = compute_contiguous_strides(&out_shape)
        .iter()
        .map(|&s| s as u32)
        .collect();

    unsafe {
        launch_gather(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            index_contig.ptr(),
            out.ptr(),
            ndim,
            dim,
            &input_shape,
            &input_strides,
            &output_shape,
            &output_strides,
            out.numel(),
        )?;
    }

    Ok(out)
}

/// Execute gather_2d operation.
///
/// Gathers elements from a 2D matrix at specific (row, col) positions.
pub fn gather_2d(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    rows: &Tensor<CudaRuntime>,
    cols: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = input.dtype();
    let shape = input.shape();

    // Validate input is 2D
    if shape.len() != 2 {
        return Err(Error::ShapeMismatch {
            expected: vec![0, 0], // Indicates 2D expected
            got: shape.to_vec(),
        });
    }

    let nrows = shape[0];
    let ncols = shape[1];

    let rows_i64 = normalize_indices_to_i64(client, rows)?;
    let cols_i64 = normalize_indices_to_i64(client, cols)?;

    // Validate rows and cols are 1D and have same length
    if rows_i64.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![rows_i64.numel()],
            got: rows_i64.shape().to_vec(),
        });
    }

    if cols_i64.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![cols_i64.numel()],
            got: cols_i64.shape().to_vec(),
        });
    }

    let num_indices = rows_i64.numel();
    if cols_i64.numel() != num_indices {
        return Err(Error::ShapeMismatch {
            expected: vec![num_indices],
            got: cols_i64.shape().to_vec(),
        });
    }

    // Make all inputs contiguous
    let input_contig = ensure_contiguous(input)?;
    let rows_contig = ensure_contiguous(&rows_i64)?;
    let cols_contig = ensure_contiguous(&cols_i64)?;

    // Allocate output
    let out = Tensor::<CudaRuntime>::empty(&[num_indices], dtype, &client.device)?;

    unsafe {
        launch_gather_2d(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            input_contig.ptr(),
            rows_contig.ptr(),
            cols_contig.ptr(),
            out.ptr(),
            nrows,
            ncols,
            num_indices,
        )?;
    }

    Ok(out)
}
