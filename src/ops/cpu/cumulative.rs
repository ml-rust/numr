//! CPU implementation of cumulative operations.

use crate::error::Result;
use crate::ops::CumulativeOps;
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{cumprod_impl, cumsum_impl, logsumexp_impl},
};
use crate::tensor::Tensor;

/// CumulativeOps implementation for CPU runtime.
impl CumulativeOps<CpuRuntime> for CpuClient {
    fn cumsum(&self, a: &Tensor<CpuRuntime>, dim: isize) -> Result<Tensor<CpuRuntime>> {
        cumsum_impl(self, a, dim)
    }

    fn cumprod(&self, a: &Tensor<CpuRuntime>, dim: isize) -> Result<Tensor<CpuRuntime>> {
        cumprod_impl(self, a, dim)
    }

    fn logsumexp(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        logsumexp_impl(self, a, dims, keepdim)
    }
}
