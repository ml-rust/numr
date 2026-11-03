//! CPU implementation of fused 2D convolution.

use crate::error::Result;
use crate::ops::impl_generic::{conv2d_fused_impl, conv2d_fused_into_impl};
use crate::ops::{Conv2dConfig, ConvActivation, ConvResidual, FusedConvOps};
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

impl FusedConvOps<CpuRuntime> for CpuClient {
    fn conv2d_fused(
        &self,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        bias: Option<&Tensor<CpuRuntime>>,
        config: &Conv2dConfig,
        activation: ConvActivation<'_, CpuRuntime>,
        residual: Option<ConvResidual<'_, CpuRuntime>>,
    ) -> Result<Tensor<CpuRuntime>> {
        conv2d_fused_impl(self, input, weight, bias, config, activation, residual)
    }

    fn conv2d_fused_into(
        &self,
        out: &Tensor<CpuRuntime>,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        bias: Option<&Tensor<CpuRuntime>>,
        config: &Conv2dConfig,
        activation: ConvActivation<'_, CpuRuntime>,
        residual: Option<ConvResidual<'_, CpuRuntime>>,
    ) -> Result<()> {
        conv2d_fused_into_impl(self, out, input, weight, bias, config, activation, residual)
    }
}
