//! Cumulative operations for WebGPU runtime

use crate::error::Result;
use crate::ops::CumulativeOps;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::native::{native_cumprod, native_cumsum, native_logsumexp};
use crate::tensor::Tensor;

impl CumulativeOps<WgpuRuntime> for WgpuClient {
    fn cumsum(&self, a: &Tensor<WgpuRuntime>, dim: isize) -> Result<Tensor<WgpuRuntime>> {
        native_cumsum(self, a, dim)
    }

    fn cumprod(&self, a: &Tensor<WgpuRuntime>, dim: isize) -> Result<Tensor<WgpuRuntime>> {
        native_cumprod(self, a, dim)
    }

    fn logsumexp(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_logsumexp(self, a, dims, keepdim)
    }
}
