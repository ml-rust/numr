//! `index_put` for the CUDA runtime.

use crate::error::Result;
use crate::runtime::cuda::kernels::launch_index_put;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

use super::super::super::tensor_inputs::{copy_admitted, materialize};
use super::super::helpers::normalize_indices_to_i64;
use super::index_put_admission::admit;
use super::validate::validate_indices_or_capture;

/// Return an independent input copy with indexed values replaced by `src`.
pub fn index_put(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dim: usize,
    index: &Tensor<CudaRuntime>,
    src: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let admitted = admit(client, a, dim, index, src)?;
    if admitted.index_len == 0 {
        return copy_admitted(client, a, admitted.input_count);
    }
    // Cast only a contiguous index, keeping every enqueue on the supplied stream.
    let index_materialized = materialize(client, index, admitted.index_len)?;
    let index_contig = index_materialized.as_ref().unwrap_or(index);
    let index_i64 = normalize_indices_to_i64(client, index_contig)?;
    validate_indices_or_capture(client, &index_i64, admitted.index_len, admitted.dim_size)?;

    let out = copy_admitted(client, a, admitted.input_count)?;
    if admitted.src_count == 0 {
        return Ok(out);
    }
    let src_materialized = materialize(client, src, admitted.src_count)?;
    let src_contig = src_materialized.as_ref().unwrap_or(src);

    // SAFETY: admission checks devices, layouts, allocation bytes, and kernel scalar ranges.
    // Caller storage is read-only. Every write targets independent output storage.
    unsafe {
        launch_index_put(
            &client.context,
            &client.stream,
            client.device.index,
            a.dtype(),
            index_i64.ptr(),
            src_contig.ptr(),
            out.ptr(),
            admitted.outer_size,
            admitted.dim_size,
            admitted.inner_size,
            admitted.index_len,
        )?;
    }

    Ok(out)
}
