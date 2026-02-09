//! Reduction operations (sum, mean, max, min)

use super::ops::*;
use crate::autograd::Var;
use crate::error::Result;
use crate::ops::{CompareOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Sum along dimensions: z = sum(a, dims)
pub fn var_sum<R, C>(a: &Var<R>, dims: &[usize], keepdim: bool, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
{
    let output = client.sum(a.tensor(), dims, keepdim)?;

    if a.requires_grad() {
        let grad_fn = SumBackward::<R>::new(a.id(), a.shape(), dims, keepdim, a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Mean along dimensions: z = mean(a, dims)
pub fn var_mean<R, C>(a: &Var<R>, dims: &[usize], keepdim: bool, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: ScalarOps<R>,
{
    let output = client.mean(a.tensor(), dims, keepdim)?;

    if a.requires_grad() {
        let grad_fn =
            MeanBackward::<R>::new(a.id(), a.shape(), dims, keepdim, a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Max along dimensions: z = max(a, dims)
pub fn var_max<R, C>(a: &Var<R>, dims: &[usize], keepdim: bool, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R> + CompareOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
{
    let output = client.max(a.tensor(), dims, keepdim)?;

    if a.requires_grad() {
        let grad_fn = MaxBackward::<R>::new(
            a.id(),
            a.tensor().clone(),
            dims,
            keepdim,
            a.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Min along dimensions: z = min(a, dims)
pub fn var_min<R, C>(a: &Var<R>, dims: &[usize], keepdim: bool, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R> + CompareOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
{
    let output = client.min(a.tensor(), dims, keepdim)?;

    if a.requires_grad() {
        let grad_fn = MinBackward::<R>::new(
            a.id(),
            a.tensor().clone(),
            dims,
            keepdim,
            a.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}
