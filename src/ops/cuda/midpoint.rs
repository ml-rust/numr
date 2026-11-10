//! CUDA midpoint through configured tensor primitives.

use crate::error::Result;
use crate::ops::MidpointOps;
use crate::ops::impl_generic::midpoint::midpoint_impl;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

impl MidpointOps<CudaRuntime> for CudaClient {
    fn midpoint(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        midpoint_impl(self, a, b)
    }
}
