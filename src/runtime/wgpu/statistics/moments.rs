//! Higher-order moment statistics for WebGPU runtime (skewness, kurtosis)

use crate::error::Result;
use crate::runtime::statistics_common;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Compute skewness (third standardized moment) using composition.
pub fn skew_impl(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
) -> Result<Tensor<WgpuRuntime>> {
    statistics_common::skew_composite(client, a, dims, keepdim, correction)
}

/// Compute kurtosis (fourth standardized moment, excess) using composition.
pub fn kurtosis_impl(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
) -> Result<Tensor<WgpuRuntime>> {
    statistics_common::kurtosis_composite(client, a, dims, keepdim, correction)
}
