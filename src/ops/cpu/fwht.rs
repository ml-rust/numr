//! CPU implementation of the Walsh-Hadamard transform.

use crate::error::Result;
use crate::ops::FwhtOps;
use crate::runtime::cpu::{CpuClient, CpuRuntime, helpers::fwht_impl};
use crate::tensor::Tensor;

/// FwhtOps implementation for CPU runtime.
impl FwhtOps<CpuRuntime> for CpuClient {
    fn fwht(
        &self,
        x: &Tensor<CpuRuntime>,
        block_size: usize,
        signs: Option<&Tensor<CpuRuntime>>,
    ) -> Result<Tensor<CpuRuntime>> {
        fwht_impl(self, x, block_size, signs)
    }
}
