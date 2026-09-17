use super::super::ops::GroupNormBackward;
use crate::autograd::Var;
use crate::error::Result;
use crate::ops::{NormalizationOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Group Normalization with autograd support.
///
/// Input: `[batch, channels, *spatial]`
/// Normalizes over groups of channels independently.
///
/// # Arguments
/// * `input` - Input variable `[batch, channels, *spatial]`
/// * `weight` - Gamma variable `[channels]`
/// * `bias` - Beta variable `[channels]`
/// * `num_groups` - Number of groups (must divide channels)
/// * `eps` - Numerical stability constant
/// * `client` - Runtime client
pub fn var_group_norm<R, C>(
    input: &Var<R>,
    weight: &Var<R>,
    bias: &Var<R>,
    num_groups: usize,
    eps: f32,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + NormalizationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.group_norm(
        input.tensor(),
        weight.tensor(),
        bias.tensor(),
        num_groups,
        eps,
    )?;

    if input.requires_grad() || weight.requires_grad() || bias.requires_grad() {
        let grad_fn = GroupNormBackward::<R>::new(
            input.id(),
            weight.id(),
            bias.id(),
            input.tensor().clone(),
            weight.tensor().clone(),
            num_groups,
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
