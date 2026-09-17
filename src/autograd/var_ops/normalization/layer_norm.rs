use super::super::ops::LayerNormBackward;
use crate::autograd::Var;
use crate::error::Result;
use crate::ops::{NormalizationOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Layer Normalization: y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
///
/// Uses the fused `NormalizationOps::layer_norm` kernel for the forward pass
/// and tracks gradients for input, weight, and bias.
///
/// # Arguments
///
/// * `input` - Input variable of shape `[..., hidden_size]`
/// * `weight` - Weight (gamma) variable of shape `[hidden_size]`
/// * `bias` - Bias (beta) variable of shape `[hidden_size]`
/// * `eps` - Small constant for numerical stability
/// * `client` - Runtime client
pub fn var_layer_norm<R, C>(
    input: &Var<R>,
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
    let output = client.layer_norm(input.tensor(), weight.tensor(), bias.tensor(), eps)?;

    if input.requires_grad() || weight.requires_grad() || bias.requires_grad() {
        let grad_fn = LayerNormBackward::<R>::new(
            input.id(),
            weight.id(),
            bias.id(),
            input.tensor().clone(),
            weight.tensor().clone(),
            eps,
            input.grad_fn().cloned(),
            weight.grad_fn().cloned(),
            bias.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}
