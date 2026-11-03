//! WebGPU implementation of fused 2D convolution.
//!
//! Supports the dtypes WebGPU `conv2d` supports (F32). Any other float dtype
//! returns the error `conv2d` returns.

use crate::error::Result;
use crate::ops::impl_generic::{conv2d_fused_impl, conv2d_fused_into_impl};
use crate::ops::{Conv2dConfig, ConvActivation, ConvResidual, FusedConvOps};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

impl FusedConvOps<WgpuRuntime> for WgpuClient {
    fn conv2d_fused(
        &self,
        input: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        bias: Option<&Tensor<WgpuRuntime>>,
        config: &Conv2dConfig,
        activation: ConvActivation<'_, WgpuRuntime>,
        residual: Option<ConvResidual<'_, WgpuRuntime>>,
    ) -> Result<Tensor<WgpuRuntime>> {
        conv2d_fused_impl(self, input, weight, bias, config, activation, residual)
    }

    fn conv2d_fused_into(
        &self,
        out: &Tensor<WgpuRuntime>,
        input: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        bias: Option<&Tensor<WgpuRuntime>>,
        config: &Conv2dConfig,
        activation: ConvActivation<'_, WgpuRuntime>,
        residual: Option<ConvResidual<'_, WgpuRuntime>>,
    ) -> Result<()> {
        conv2d_fused_into_impl(self, out, input, weight, bias, config, activation, residual)
    }
}
