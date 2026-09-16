//! WMMA pre-launch padding, shared by every CUDA matmul entry point that
//! pads a ragged `(N, K)` before a tensor-core launch: `matmul`,
//! `matmul_bias` (`ops/cuda/matmul.rs`), `matmul_bias_activation`,
//! `matmul_bias_residual` (`ops/cuda/gemm_epilogue.rs`), and `matmul_wide`
//! (`ops/cuda/matmul_wide.rs`).

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::ShapeOps;
use crate::runtime::cuda::kernels::{use_wmma_after_padding, wmma_padded_dims};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::traits::profile::DeviceCaps;
use crate::tensor::Tensor;

/// Pads `a` and `b` for the WMMA path when the pad pays, per
/// [`use_wmma_after_padding`]: K is the last dim of A and the second-last of
/// B, N the last dim of B; M and any leading batch dims are untouched.
/// Zero-padding is exact, and the extra N columns are the caller's to slice
/// off the result.
///
/// Returns `None` when the shape launches as it is: a rank-1 operand (the
/// pad spec addresses two dims of each, so it never applies to one) or a
/// shape `use_wmma_after_padding` rejects.
pub(crate) fn pad_ab_for_wmma(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    dtype: DType,
    caps: DeviceCaps,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Option<(Tensor<CudaRuntime>, Tensor<CudaRuntime>, usize, usize)>> {
    if !(a.rank() >= 2 && b.rank() >= 2 && use_wmma_after_padding(dtype, caps, m, n, k)) {
        return Ok(None);
    }
    let (n_pad, k_pad) = wmma_padded_dims(n, k);
    let a_pad = client.pad(a, &[0, k_pad - k, 0, 0], 0.0)?;
    let b_pad = client.pad(b, &[0, n_pad - n, 0, k_pad - k], 0.0)?;
    Ok(Some((a_pad, b_pad, n_pad, k_pad)))
}
