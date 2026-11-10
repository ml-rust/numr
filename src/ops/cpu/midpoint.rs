//! CPU midpoint through configured tensor primitives.

use crate::error::Result;
use crate::ops::MidpointOps;
use crate::ops::impl_generic::midpoint::midpoint_impl;
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

impl MidpointOps<CpuRuntime> for CpuClient {
    fn midpoint(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        midpoint_impl(self, a, b)
    }
}
