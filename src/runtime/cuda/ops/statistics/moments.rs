//! Higher-order moment statistics for CUDA runtime (skewness, kurtosis)

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
    statistics_common::skew_composite(client, a, dims, keepdim, correction)
}

/// Compute kurtosis (fourth standardized moment, excess) using composition.
pub fn kurtosis_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
) -> Result<Tensor<CudaRuntime>> {
    statistics_common::kurtosis_composite(client, a, dims, keepdim, correction)
}
