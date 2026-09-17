use super::super::ops::FusedAddLayerNormBackward;
use crate::autograd::Var;
use crate::error::Result;
use crate::ops::{NormalizationOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Fused Add + Layer Normalization: pre_norm = x + residual, output = layer_norm(pre_norm, weight, bias, eps)
///
/// Returns a single output variable. Both `x` and `residual` receive the same gradient.
///
/// # Arguments
///
/// * `x` - Input variable of shape `[..., hidden_size]`
/// * `residual` - Residual variable of same shape as `x`
/// * `weight` - Weight (gamma) variable of shape `[hidden_size]`
/// * `bias` - Bias (beta) variable of shape `[hidden_size]`
/// * `eps` - Small constant for numerical stability
/// * `client` - Runtime client
pub fn var_fused_add_layer_norm<R, C>(
    x: &Var<R>,
    residual: &Var<R>,
    weight: &Var<R>,
    bias: &Var<R>,
    eps: f32,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + NormalizationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let (output, pre_norm) = client.fused_add_layer_norm(
        x.tensor(),
        residual.tensor(),
        weight.tensor(),
        bias.tensor(),
        eps,
    )?;

    if x.requires_grad()
        || residual.requires_grad()
        || weight.requires_grad()
        || bias.requires_grad()
    {
        let grad_fn = FusedAddLayerNormBackward::<R>::new(
            x.id(),
            residual.id(),
            weight.id(),
            bias.id(),
            pre_norm,
            weight.tensor().clone(),
            bias.tensor().clone(),
            eps,
            x.grad_fn().cloned(),
            residual.grad_fn().cloned(),
            weight.grad_fn().cloned(),
            bias.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}
