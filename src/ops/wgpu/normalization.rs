//! Normalization operations for WebGPU runtime

use crate::error::Result;
use crate::ops::NormalizationOps;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::native::{
    native_fused_add_layer_norm, native_fused_add_layer_norm_bwd, native_fused_add_rms_norm,
    native_fused_add_rms_norm_bwd, native_group_norm, native_layer_norm, native_rms_norm,
};
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

    fn group_norm(
        &self,
        input: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        bias: &Tensor<WgpuRuntime>,
        num_groups: usize,
        eps: f32,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_group_norm(self, input, weight, bias, num_groups, eps)
    }

    fn fused_add_rms_norm(
        &self,
        x: &Tensor<WgpuRuntime>,
        residual: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        eps: f32,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        native_fused_add_rms_norm(self, x, residual, weight, eps)
    }

    fn fused_add_layer_norm(
        &self,
        x: &Tensor<WgpuRuntime>,
        residual: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        bias: &Tensor<WgpuRuntime>,
        eps: f32,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        native_fused_add_layer_norm(self, x, residual, weight, bias, eps)
    }

    fn fused_add_rms_norm_bwd(
        &self,
        grad: &Tensor<WgpuRuntime>,
        pre_norm: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        eps: f32,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        native_fused_add_rms_norm_bwd(self, grad, pre_norm, weight, eps)
    }

    fn fused_add_layer_norm_bwd(
        &self,
        grad: &Tensor<WgpuRuntime>,
        pre_norm: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        bias: &Tensor<WgpuRuntime>,
        eps: f32,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        native_fused_add_layer_norm_bwd(self, grad, pre_norm, weight, bias, eps)
    }
}
