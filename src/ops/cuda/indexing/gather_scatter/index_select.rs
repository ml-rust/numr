//! `index_select` for the CUDA runtime.

use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::launch_index_select;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

use super::super::helpers::normalize_indices_to_i64;
use super::validate::validate_indices_or_capture;

/// Execute index_select operation.
pub fn index_select(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dim: usize,
    index: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let index_i64 = normalize_indices_to_i64(client, index)?;

    // Validate index is 1D
    if index_i64.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![index_i64.numel()],
            got: index_i64.shape().to_vec(),
        });
    }

    // Validate dimension
    let shape = a.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let index_contig = ensure_contiguous(&index_i64)?;

    // Compute output shape: same as input but dim[dim] = index.len()
    let index_len = index_i64.numel();
    let mut out_shape = shape.to_vec();
    out_shape[dim] = index_len;

    // A zero-element output has nothing to select. The kernel derives its grid
    // from the source's outer/inner extents, so launching it here reads off the
    // end of an empty allocation — an illegal access, not merely a wrong answer.
    if out_shape.iter().product::<usize>() == 0 {
        return Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);
    }

    // Compute dim_size for validation
    let dim_size = shape[dim];

    validate_indices_or_capture(client, &index_contig, index_len, dim_size)?;

    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

    // Compute outer/dim/inner sizes. Never clamp these with `.max(1)`: the
    // zero-element guard above already rules a zero out, and a clamp would make
    // the launcher's own `total == 0` check miss and read past the allocation.
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    unsafe {
        launch_index_select(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            index_contig.ptr(),
            out.ptr(),
            outer_size,
            dim_size,
            inner_size,
            index_len,
        )?;
    }

    Ok(out)
}
