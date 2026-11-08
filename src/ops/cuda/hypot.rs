//! Cuda scaled Euclidean magnitude.

use crate::error::Result;
use crate::ops::HypotOps;
use crate::ops::impl_generic::hypot::hypot_impl;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

impl HypotOps<CudaRuntime> for CudaClient {
    fn hypot(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        hypot_impl(self, a, b)
    }
}
