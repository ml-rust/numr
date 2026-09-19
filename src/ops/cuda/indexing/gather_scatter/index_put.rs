//! `index_put` for the CUDA runtime.

use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::launch_index_put;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

use super::super::helpers::normalize_indices_to_i64;
use super::validate::validate_indices_or_capture;

/// Execute index_put operation.
pub fn index_put(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dim: usize,
    index: &Tensor<CudaRuntime>,
    src: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let index_i64 = normalize_indices_to_i64(client, index)?;

    // Validate index is 1D
    if index_i64.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![index_i64.numel()],
            got: index_i64.shape().to_vec(),
        });
    }

    // Validate src dtype matches
    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    let index_len = index_i64.numel();

    // Validate src shape: must match a's shape except at dim where it equals index_len
    let mut expected_src_shape = shape.to_vec();
    expected_src_shape[dim] = index_len;
    if src.shape() != expected_src_shape {
        return Err(Error::ShapeMismatch {
            expected: expected_src_shape,
            got: src.shape().to_vec(),
        });
    }

    let a_contig = ensure_contiguous(a)?;
    let index_contig = ensure_contiguous(&index_i64)?;
    let src_contig = ensure_contiguous(src)?;

    // Compute dim_size for validation
    let dim_size = shape[dim];

    validate_indices_or_capture(client, &index_contig, index_len, dim_size)?;

    // Clone a to output first
    let out = a_contig.clone();

    // Compute outer/dim/inner sizes. Never clamp these with `.max(1)`: a clamp
    // fires only on a genuinely zero extent, and then defeats the launcher's own
    // `total == 0` early return so the kernel reads past an empty allocation.
    // Unclamped, `total` is 0 and the launcher returns without dispatching, which
    // leaves `out` as the copy of `a` that CPU also produces.
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    unsafe {
        launch_index_put(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            index_contig.ptr(),
            src_contig.ptr(),
            out.ptr(),
            outer_size,
            dim_size,
            inner_size,
            index_len,
        )?;
    }

    Ok(out)
}
