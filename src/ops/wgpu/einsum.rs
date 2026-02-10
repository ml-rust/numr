//! WebGPU implementation of einsum operations.

use crate::error::Result;
use crate::ops::EinsumOps;
use crate::ops::impl_generic::einsum::einsum_impl;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

impl EinsumOps<WgpuRuntime> for WgpuClient {
    fn einsum(
        &self,
        notation: &str,
        inputs: &[&Tensor<WgpuRuntime>],
    ) -> Result<Tensor<WgpuRuntime>> {
        einsum_impl(self, notation, inputs)
    }
}
