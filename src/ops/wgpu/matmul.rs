//! Matrix multiplication operations for WebGPU runtime

use crate::error::Result;
use crate::ops::MatmulOps;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::native::{native_matmul, native_matmul_bias};
use crate::tensor::Tensor;

impl MatmulOps<WgpuRuntime> for WgpuClient {
    fn matmul(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_matmul(self, a, b)
    }

    fn matmul_bias(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        bias: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_matmul_bias(self, a, b, bias)
    }
}
