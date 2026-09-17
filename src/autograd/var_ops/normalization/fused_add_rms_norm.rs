use super::super::ops::FusedAddRmsNormBackward;
use crate::autograd::Var;
use crate::error::Result;
use crate::ops::{NormalizationOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Fused Add + RMS Normalization: pre_norm = x + residual, output = rms_norm(pre_norm, weight, eps)
///
/// Returns a single output variable. Both `x` and `residual` receive the same gradient.
///
/// # Arguments
///
/// * `x` - Input variable of shape `[..., hidden_size]`
/// * `residual` - Residual variable of same shape as `x`
/// * `weight` - Weight variable of shape `[hidden_size]`
/// * `eps` - Small constant for numerical stability
/// * `client` - Runtime client
pub fn var_fused_add_rms_norm<R, C>(
    x: &Var<R>,
    residual: &Var<R>,
    weight: &Var<R>,
    eps: f32,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + NormalizationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let (output, pre_norm) =
        client.fused_add_rms_norm(x.tensor(), residual.tensor(), weight.tensor(), eps)?;

    if x.requires_grad() || residual.requires_grad() || weight.requires_grad() {
        let grad_fn = FusedAddRmsNormBackward::<R>::new(
            x.id(),
            residual.id(),
            weight.id(),
            pre_norm,
            weight.tensor().clone(),
            eps,
            x.grad_fn().cloned(),
            residual.grad_fn().cloned(),
            weight.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}
