//! CUDA implementation of einsum operations.

use crate::error::Result;
use crate::ops::EinsumOps;
use crate::ops::impl_generic::einsum::einsum_impl;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

impl EinsumOps<CudaRuntime> for CudaClient {
    fn einsum(
        &self,
        notation: &str,
        inputs: &[&Tensor<CudaRuntime>],
    ) -> Result<Tensor<CudaRuntime>> {
        einsum_impl(self, notation, inputs)
    }
}
