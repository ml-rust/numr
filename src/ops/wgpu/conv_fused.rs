//! WebGPU implementation of fused 2D convolution.
//!
//! A call in F32 with one group runs one tile kernel, which fuses the K sum,
//! bias, activation and residual. Every other call takes the generic path.
//! That path supports the dtypes WebGPU `conv2d` supports (F32). Any other
//! float dtype returns the error `conv2d` returns.

use super::conv2d_tile::{TileCall, tile_grid_for};
use crate::error::Result;
use crate::ops::conv_fused_common::{
    FusedConv2dParams, validate_conv2d_fused, validate_conv2d_fused_out,
};
use crate::ops::impl_generic::{conv2d_fused_impl, conv2d_fused_into_impl};
use crate::ops::{BinaryOps, Conv2dConfig, ConvActivation, ConvResidual, FusedConvOps};
use crate::runtime::wgpu::ops::helpers::alloc_output;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// The tile call of validated operands, or `None` for the generic path.
fn tile_call<'a>(
    params: &FusedConv2dParams,
    input: &'a Tensor<WgpuRuntime>,
    weight: &'a Tensor<WgpuRuntime>,
    bias: Option<&'a Tensor<WgpuRuntime>>,
    activation: ConvActivation<'a, WgpuRuntime>,
    residual: Option<ConvResidual<'a, WgpuRuntime>>,
) -> Option<TileCall<'a>> {
    let grid = tile_grid_for(&params.conv, input.dtype())?;
    Some(TileCall {
        input,
        weight,
        bias,
        activation,
        residual,
        layout: params.layout,
        conv: params.conv,
        grid,
    })
}

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
        let params = validate_conv2d_fused(input, weight, bias, config, activation, residual)?;
        let Some(call) = tile_call(&params, input, weight, bias, activation, residual) else {
            return conv2d_fused_impl(self, input, weight, bias, config, activation, residual);
        };
        let buffers = call.buffers()?;
        let out = alloc_output(self, &params.output_shape, input.dtype())?;
        call.launch(self, &out, &buffers)?;
        Ok(out)
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
        let params = validate_conv2d_fused(input, weight, bias, config, activation, residual)?;
        validate_conv2d_fused_out(out, &params, input, weight, bias, activation, residual)?;
        let Some(call) = tile_call(&params, input, weight, bias, activation, residual) else {
            return conv2d_fused_into_impl(
                self, out, input, weight, bias, config, activation, residual,
            );
        };
        let buffers = call.buffers()?;
        if call.binds_output(out, &buffers)? {
            return call.launch(self, out, &buffers);
        }
        // A strided, offset or shared `out` takes the result through a copy.
        let result = alloc_output(self, &params.output_shape, input.dtype())?;
        call.launch(self, &result, &buffers)?;
        self.copy_into(out, &result)
    }
}
