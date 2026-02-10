//! Statistical operations (var, std)

use super::ops::*;
use crate::autograd::Var;
use crate::error::Result;
use crate::ops::{ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Variance along dimensions: z = var(a, dims, correction)
///
/// Creates VarBackward for gradient computation.
pub fn var_var<R, C>(
    a: &Var<R>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.var(a.tensor(), dims, keepdim, correction)?;

    if a.requires_grad() {
        let grad_fn = VarBackward::<R>::new(
            a.id(),
            a.tensor().clone(),
            dims,
            keepdim,
            correction,
            a.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Standard deviation along dimensions: z = std(a, dims, correction)
///
/// Creates StdBackward for gradient computation.
pub fn var_std<R, C>(
    a: &Var<R>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.std(a.tensor(), dims, keepdim, correction)?;

    if a.requires_grad() {
        let grad_fn = StdBackward::<R>::new(
            a.id(),
            a.tensor().clone(),
            output.clone(),
            dims,
            keepdim,
            correction,
            a.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}
