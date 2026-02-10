//! Utility operations (clamp)

use super::ops::*;
use crate::autograd::Var;
use crate::error::Result;
use crate::ops::{CompareOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Clamp values to range: z = clamp(a, min, max)
///
/// Creates ClampBackward for gradient computation.
pub fn var_clamp<R, C>(a: &Var<R>, min_val: f64, max_val: f64, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
{
    let output = client.clamp(a.tensor(), min_val, max_val)?;

    if a.requires_grad() {
        let grad_fn = ClampBackward::<R>::new(
            a.id(),
            a.tensor().clone(),
            min_val,
            max_val,
            a.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}
