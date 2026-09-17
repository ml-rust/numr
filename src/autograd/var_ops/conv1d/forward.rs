//! Forward pass for differentiable 1D convolution.

use crate::autograd::Var;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, ConvOps, PaddingMode, ReduceOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

use super::backward::Conv1dBackward;

/// Differentiable 1D convolution.
///
/// Wraps the forward `conv1d` and builds autograd graph for backward.
///
/// # Arguments
/// * `input` - Input Var of shape `[batch, in_channels, length]`
/// * `weight` - Weight Var of shape `[out_channels, in_channels/groups, kernel_size]`
/// * `bias` - Optional bias Var of shape `[out_channels]`
/// * `stride` - Stride
/// * `padding` - Padding mode
/// * `dilation` - Dilation
/// * `groups` - Groups
/// * `client` - Runtime client
pub fn var_conv1d<R, C>(
    input: &Var<R>,
    weight: &Var<R>,
    bias: Option<&Var<R>>,
    stride: usize,
    padding: PaddingMode,
    dilation: usize,
    groups: usize,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ConvOps<R> + TensorOps<R> + ReduceOps<R> + BinaryOps<R> + ScalarOps<R>,
    R::Client: ConvOps<R> + TensorOps<R> + ReduceOps<R> + BinaryOps<R> + ScalarOps<R>,
{
    let output = client.conv1d(
        input.tensor(),
        weight.tensor(),
        bias.map(|b| b.tensor()),
        stride,
        padding,
        dilation,
        groups,
    )?;

    let needs_grad =
        input.requires_grad() || weight.requires_grad() || bias.is_some_and(|b| b.requires_grad());

    if needs_grad {
        let grad_fn = Conv1dBackward::<R>::new(
            input.id(),
            weight.id(),
            bias.map(|b| b.id()),
            input.tensor().clone(),
            weight.tensor().clone(),
            input.tensor().shape().to_vec(),
            stride,
            padding,
            dilation,
            groups,
            input.grad_fn().cloned(),
            weight.grad_fn().cloned(),
            bias.and_then(|b| b.grad_fn().cloned()),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}
