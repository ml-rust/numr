//! `scatter` and `slice_assign` for the CUDA runtime.

use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::{launch_copy, launch_scatter, launch_slice_assign};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{compute_contiguous_strides, ensure_contiguous};
use crate::tensor::Tensor;

use super::super::helpers::normalize_indices_to_i64;

/// Execute scatter operation along dimension.
pub fn scatter(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dim: usize,
    index: &Tensor<CudaRuntime>,
    src: &Tensor<CudaRuntime>,
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

    // Validate src has same dtype as input
    let dtype = a.dtype();
    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    // Index and src must have same shape
    if index_i64.shape() != src.shape() {
        return Err(Error::ShapeMismatch {
            expected: index_i64.shape().to_vec(),
            got: src.shape().to_vec(),
        });
    }

    let a_contig = ensure_contiguous(a)?;
    let index_contig = ensure_contiguous(&index_i64)?;
    let src_contig = ensure_contiguous(src)?;

    // Output has same shape as input
    let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &client.device)?;

    // First, copy input to output (scatter modifies output in-place)
    unsafe {
        launch_copy(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            out.ptr(),
            a.numel(),
        )?;
    }

    // Prepare shape and stride arrays for GPU
    let output_shape: Vec<u32> = a.shape().iter().map(|&s| s as u32).collect();
    let output_strides: Vec<u32> = compute_contiguous_strides(a.shape())
        .iter()
        .map(|&s| s as u32)
        .collect();
    let src_shape: Vec<u32> = src.shape().iter().map(|&s| s as u32).collect();
    let src_strides: Vec<u32> = compute_contiguous_strides(src.shape())
        .iter()
        .map(|&s| s as u32)
        .collect();

    unsafe {
        launch_scatter(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            index_contig.ptr(),
            src_contig.ptr(),
            out.ptr(),
            ndim,
            dim,
            &output_shape,
            &output_strides,
            &src_shape,
            &src_strides,
            src.numel(),
        )?;
    }
    Ok(out)
}

/// Execute slice_assign operation: assign src into a slice of dst along dim.
pub fn slice_assign(
    client: &CudaClient,
    dst: &Tensor<CudaRuntime>,
    src: &Tensor<CudaRuntime>,
    dim: usize,
    start: usize,
) -> Result<Tensor<CudaRuntime>> {
    let ndim = dst.ndim();
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    if src.ndim() != ndim {
        return Err(Error::ShapeMismatch {
            expected: dst.shape().to_vec(),
            got: src.shape().to_vec(),
        });
    }
    for d in 0..ndim {
        if d != dim && src.shape()[d] != dst.shape()[d] {
            return Err(Error::ShapeMismatch {
                expected: dst.shape().to_vec(),
                got: src.shape().to_vec(),
            });
        }
    }

    let src_dim_size = src.shape()[dim];
    let dst_dim_size = dst.shape()[dim];
    if start + src_dim_size > dst_dim_size {
        return Err(Error::InvalidArgument {
            arg: "start",
            reason: format!(
                "start ({}) + src dim size ({}) exceeds dst dim size ({})",
                start, src_dim_size, dst_dim_size
            ),
        });
    }

    let dtype = dst.dtype();
    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    // Never clamp these with `.max(1)`: a clamp fires only on a genuinely zero
    // extent, and then defeats the launcher's own `total == 0` early return so the
    // kernel reads past an empty allocation. Unclamped, the slice-assign launch is
    // skipped and `out` keeps the copy of `dst` that CPU also produces.
    let outer_size: usize = dst.shape()[..dim].iter().product();
    let inner_size: usize = dst.shape()[dim + 1..].iter().product();

    let dst_contig = ensure_contiguous(dst)?;
    let src_contig = ensure_contiguous(src)?;

    let out = Tensor::<CudaRuntime>::empty(dst.shape(), dtype, &client.device)?;

    unsafe {
        // Copy dst → output
        launch_copy(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            dst_contig.ptr(),
            out.ptr(),
            dst_contig.numel(),
        )?;

        // Overwrite the slice with src
        launch_slice_assign(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            src_contig.ptr(),
            out.ptr(),
            outer_size,
            dst_dim_size,
            src_dim_size,
            inner_size,
            start,
        )?;
    }

    Ok(out)
}
