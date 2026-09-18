//! WebGPU implementation of the Walsh-Hadamard transform.

use crate::error::Result;
use crate::ops::FwhtOps;
use crate::runtime::wgpu::ops::native::native_fwht;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// FwhtOps implementation for WebGPU runtime. F32 only.
impl FwhtOps<WgpuRuntime> for WgpuClient {
    fn fwht(
        &self,
        x: &Tensor<WgpuRuntime>,
        block_size: usize,
        signs: Option<&Tensor<WgpuRuntime>>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_fwht(self, x, block_size, signs)
    }
}
