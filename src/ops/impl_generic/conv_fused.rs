//! Generic fused 2D convolution built from `conv2d`, activations and scalar ops.
//!
//! This is the reference path for every backend. A native kernel replaces it
//! per backend and must match it within the parity-test tolerance.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::conv_fused_common::{
    FusedConv2dParams, NCHW_TO_NHWC, NHWC_TO_NCHW, validate_conv2d_fused, validate_conv2d_fused_out,
};
use crate::ops::traits::{
    ActivationOps, BinaryOps, Conv2dConfig, ConvActivation, ConvLayout, ConvOps, ConvResidual,
    ScalarOps,
};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Generic `conv2d_fused`: validate, then run conv, activation and residual.
///
/// NHWC runs as NCHW: the input and the OHWI weight are permuted views, and
/// the result is permuted back before the residual add.
pub fn conv2d_fused_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    weight: &Tensor<R>,
    bias: Option<&Tensor<R>>,
    config: &Conv2dConfig,
    activation: ConvActivation<'_, R>,
    residual: Option<ConvResidual<'_, R>>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ConvOps<R> + ActivationOps<R> + BinaryOps<R> + ScalarOps<R>,
{
    let params = validate_conv2d_fused(input, weight, bias, config, activation, residual)?;
    compute(
        client, &params, input, weight, bias, config, activation, residual,
    )
}

/// Generic `conv2d_fused_into`: validate `out`, compute, then `copy_into(out, ...)`.
pub fn conv2d_fused_into_impl<R, C>(
    client: &C,
    out: &Tensor<R>,
    input: &Tensor<R>,
    weight: &Tensor<R>,
    bias: Option<&Tensor<R>>,
    config: &Conv2dConfig,
    activation: ConvActivation<'_, R>,
    residual: Option<ConvResidual<'_, R>>,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: ConvOps<R> + ActivationOps<R> + BinaryOps<R> + ScalarOps<R>,
{
    let params = validate_conv2d_fused(input, weight, bias, config, activation, residual)?;
    validate_conv2d_fused_out(out, &params, input, weight, bias, activation, residual)?;
    let result = compute(
        client, &params, input, weight, bias, config, activation, residual,
    )?;
    client.copy_into(out, &result)
}

/// The math after validation: `residual + scale * act(conv(x, w) + bias)`.
fn compute<R, C>(
    client: &C,
    params: &FusedConv2dParams,
    input: &Tensor<R>,
    weight: &Tensor<R>,
    bias: Option<&Tensor<R>>,
    config: &Conv2dConfig,
    activation: ConvActivation<'_, R>,
    residual: Option<ConvResidual<'_, R>>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ConvOps<R> + ActivationOps<R> + BinaryOps<R> + ScalarOps<R>,
{
    let (input_nchw, weight_oihw) = match params.layout {
        ConvLayout::Nchw => (input.clone(), weight.clone()),
        ConvLayout::Nhwc => (
            input.permute(&NHWC_TO_NCHW)?,
            weight.permute(&NHWC_TO_NCHW)?,
        ),
    };
    let conv = client.conv2d(
        &input_nchw,
        &weight_oihw,
        bias,
        config.stride,
        config.padding,
        config.dilation,
        config.groups,
    )?;
    let activated = apply_activation(client, &conv, activation, params.conv.c_out)?;
    let y = match params.layout {
        ConvLayout::Nchw => activated,
        ConvLayout::Nhwc => activated.permute(&NCHW_TO_NHWC)?,
    };
    let y = match residual {
        None => y,
        // `x * 1.0` is exact, so a unit scale skips the extra pass.
        Some(ConvResidual { tensor, scale: 1.0 }) => client.add(tensor, &y)?,
        Some(ConvResidual { tensor, scale }) => {
            let scaled = client.mul_scalar(&y, scale)?;
            client.add(tensor, &scaled)?
        }
    };
    y.contiguous()
}

/// Apply the activation to an NCHW tensor.
///
/// A `[C_out]` `Prelu` weight is viewed as `[C_out, 1, 1]` so it broadcasts
/// over height and width. A `[1]` weight broadcasts as is.
fn apply_activation<R, C>(
    client: &C,
    y: &Tensor<R>,
    activation: ConvActivation<'_, R>,
    c_out: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ActivationOps<R>,
{
    match activation {
        ConvActivation::None => Ok(y.clone()),
        ConvActivation::Relu => client.relu(y),
        ConvActivation::LeakyRelu(slope) => client.leaky_relu(y, slope),
        ConvActivation::Prelu(slope) if slope.shape() == [c_out] && c_out != 1 => {
            let per_channel = slope.contiguous()?.reshape(&[c_out, 1, 1])?;
            client.prelu(y, &per_channel)
        }
        ConvActivation::Prelu(slope) => client.prelu(y, slope),
    }
}
