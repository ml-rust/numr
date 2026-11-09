//! CPU error-free addition through configured tensor primitives.

use crate::error::Result;
use crate::ops::TwoSumOps;
use crate::ops::impl_generic::two_sum::two_sum_impl;
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

impl TwoSumOps<CpuRuntime> for CpuClient {
    fn two_sum(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        two_sum_impl(self, a, b)
    }
}
