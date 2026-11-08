//! Cpu scaled Euclidean magnitude.

use crate::error::Result;
use crate::ops::HypotOps;
use crate::ops::impl_generic::hypot::hypot_impl;
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

impl HypotOps<CpuRuntime> for CpuClient {
    fn hypot(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        hypot_impl(self, a, b)
    }
}
