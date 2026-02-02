//! Normalization operations for WebGPU runtime

use crate::error::Result;
use crate::ops::NormalizationOps;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::native::{native_layer_norm, native_rms_norm};
use crate::tensor::Tensor;

impl NormalizationOps<WgpuRuntime> for WgpuClient {
    fn rms_norm(
        &self,
        a: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        eps: f32,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_rms_norm(self, a, weight, eps)
    }

    fn layer_norm(
        &self,
        a: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        bias: &Tensor<WgpuRuntime>,
        eps: f32,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_layer_norm(self, a, weight, bias, eps)
    }
}
