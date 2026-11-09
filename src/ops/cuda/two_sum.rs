//! CUDA error-free addition through configured tensor primitives.

use crate::error::Result;
use crate::ops::TwoSumOps;
use crate::ops::impl_generic::two_sum::two_sum_impl;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

impl TwoSumOps<CudaRuntime> for CudaClient {
    fn two_sum(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        two_sum_impl(self, a, b)
    }
}
