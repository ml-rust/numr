use super::super::ops::RmsNormBackward;

use crate::autograd::Var;
use crate::error::Result;
use crate::ops::{NormalizationOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// RMS Normalization: y = x / rms(x) * weight
///
/// Uses the fused `NormalizationOps::rms_norm` kernel for the forward pass
/// and tracks gradients for both input and weight.
///
/// # Arguments
///
/// * `input` - Input variable of shape `[..., hidden_size]`
/// * `weight` - Weight variable of shape `[hidden_size]`
/// * `eps` - Small constant for numerical stability
/// * `client` - Runtime client
pub fn var_rms_norm<R, C>(input: &Var<R>, weight: &Var<R>, eps: f32, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + NormalizationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.rms_norm(input.tensor(), weight.tensor(), eps)?;

    if input.requires_grad() || weight.requires_grad() {
        let grad_fn = RmsNormBackward::<R>::new(
            input.id(),
            weight.id(),
            input.tensor().clone(),
            weight.tensor().clone(),
            eps,
            input.grad_fn().cloned(),
            weight.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}
