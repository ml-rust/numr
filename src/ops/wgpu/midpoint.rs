//! WebGPU midpoint through configured tensor primitives.

use crate::error::Result;
use crate::ops::MidpointOps;
use crate::ops::impl_generic::midpoint::midpoint_impl;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

impl MidpointOps<WgpuRuntime> for WgpuClient {
    fn midpoint(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        midpoint_impl(self, a, b)
    }
}
