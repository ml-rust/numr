//! Higher-order moment statistics for CUDA runtime (skewness, kurtosis)
//!
//! Uses dtype promotion for reduced-precision types (F16, BF16, FP8) since
//! higher-order moments (x^3, x^4) overflow in low precision.

use crate::algorithm::linalg::helpers::{linalg_demote, linalg_promote};
use crate::error::Result;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::statistics_common;
use crate::tensor::Tensor;

/// Compute skewness (third standardized moment) using composition.
pub fn skew_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
) -> Result<Tensor<CudaRuntime>> {
    let (a_promoted, original_dtype) = linalg_promote(client, a)?;
    let result = statistics_common::skew_composite(client, &a_promoted, dims, keepdim, correction)?;
    linalg_demote(client, result, original_dtype)
}

/// Compute kurtosis (fourth standardized moment, excess) using composition.
pub fn kurtosis_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
) -> Result<Tensor<CudaRuntime>> {
    let (a_promoted, original_dtype) = linalg_promote(client, a)?;
    let result =
        statistics_common::kurtosis_composite(client, &a_promoted, dims, keepdim, correction)?;
    linalg_demote(client, result, original_dtype)
}
