//! CUDA implementation of fused 2D convolution.
//!
//! Every call takes the generic path.

use crate::error::Result;
use crate::ops::impl_generic::{conv2d_fused_impl, conv2d_fused_into_impl};
use crate::ops::{Conv2dConfig, ConvActivation, ConvResidual, FusedConvOps};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

impl FusedConvOps<CudaRuntime> for CudaClient {
    fn conv2d_fused(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: Option<&Tensor<CudaRuntime>>,
        config: &Conv2dConfig,
        activation: ConvActivation<'_, CudaRuntime>,
        residual: Option<ConvResidual<'_, CudaRuntime>>,
    ) -> Result<Tensor<CudaRuntime>> {
        conv2d_fused_impl(self, input, weight, bias, config, activation, residual)
    }

    fn conv2d_fused_into(
        &self,
        out: &Tensor<CudaRuntime>,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: Option<&Tensor<CudaRuntime>>,
        config: &Conv2dConfig,
        activation: ConvActivation<'_, CudaRuntime>,
        residual: Option<ConvResidual<'_, CudaRuntime>>,
    ) -> Result<()> {
        conv2d_fused_into_impl(self, out, input, weight, bias, config, activation, residual)
    }
}
