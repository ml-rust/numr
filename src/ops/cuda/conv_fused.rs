//! CUDA implementation of fused 2D convolution.
//!
//! F16 and BF16 calls with one group, NHWC or NCHW, run the implicit-GEMM
//! tensor-core kernel when `conv_fused_native::native_args` accepts them.
//! Every other call takes the generic path.

use super::conv_fused_native::{NativeCall, NativeOperands, launch_native, native_args};
use crate::error::Result;
use crate::ops::conv_fused_common::{
    FusedConv2dParams, validate_conv2d_fused, validate_conv2d_fused_out,
};
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
        let ops = NativeOperands {
            input,
            weight,
            bias,
            activation,
            residual,
            out: None,
        };
        let Some(mut args) = native_args(self, &native_call(&params, config), &ops) else {
            return conv2d_fused_impl(self, input, weight, bias, config, activation, residual);
        };
        let out = Tensor::<CudaRuntime>::empty(&params.output_shape, input.dtype(), &self.device)?;
        args.out_ptr = out.ptr();
        launch_native(self, input.dtype(), &args, weight)?;
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
        let ops = NativeOperands {
            input,
            weight,
            bias,
            activation,
            residual,
            out: Some(out),
        };
        match native_args(self, &native_call(&params, config), &ops) {
            Some(args) => launch_native(self, input.dtype(), &args, weight),
            None => {
                conv2d_fused_into_impl(self, out, input, weight, bias, config, activation, residual)
            }
        }
    }
}

/// Geometry, layout and accumulator of a validated fused call.
fn native_call<'a>(params: &'a FusedConv2dParams, config: &Conv2dConfig) -> NativeCall<'a> {
    NativeCall {
        conv: &params.conv,
        layout: params.layout,
        output_shape: params.output_shape,
        accumulation: config.accumulation,
    }
}
