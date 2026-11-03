//! CUDA implementation of fused 2D convolution.
//!
//! F16 and BF16 NHWC calls with one group run the implicit-GEMM tensor-core
//! kernel when `conv_fused_native::native_args` accepts them. Every other
//! call takes the generic path.

use super::conv_fused_native::{launch_native, native_args};
use crate::error::Result;
use crate::ops::conv_fused_common::{validate_conv2d_fused, validate_conv2d_fused_out};
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
        let params = validate_conv2d_fused(input, weight, bias, config, activation, residual)?;
        let Some(mut args) = native_args(
            self, &params, config, input, weight, bias, activation, residual, None,
        ) else {
            return conv2d_fused_impl(self, input, weight, bias, config, activation, residual);
        };
        let out = Tensor::<CudaRuntime>::empty(&params.output_shape, input.dtype(), &self.device)?;
        args.out_ptr = out.ptr();
        launch_native(self, input.dtype(), &args)?;
        Ok(out)
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
        let params = validate_conv2d_fused(input, weight, bias, config, activation, residual)?;
        validate_conv2d_fused_out(out, &params, input, weight, bias, activation, residual)?;
        match native_args(
            self,
            &params,
            config,
            input,
            weight,
            bias,
            activation,
            residual,
            Some(out),
        ) {
            Some(args) => launch_native(self, input.dtype(), &args),
            None => {
                conv2d_fused_into_impl(self, out, input, weight, bias, config, activation, residual)
            }
        }
    }
}
