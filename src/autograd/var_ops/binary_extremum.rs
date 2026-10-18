//! Shared backward for element-wise `maximum` and `minimum`.
//!
//! Each input receives the output gradient times a weight of 1 where it wins,
//! 0 where it loses and 0.5 on a tie, as PyTorch does. The gradient is then
//! summed back to the input's shape when the input was broadcast.

use crate::autograd::ops::{reduce_grad_for_broadcast, reduce_var_for_broadcast};
use crate::autograd::var_ops::var_mul;
use crate::autograd::{GradFn, Var};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, CompareOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Which element-wise extremum an op computes.
#[derive(Clone, Copy)]
pub(crate) enum Extremum {
    Maximum,
    Minimum,
}

/// Run the extremum on `a` and `b` and attach its backward.
pub(crate) fn var_extremum<R, C>(
    a: &Var<R>,
    b: &Var<R>,
    kind: Extremum,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + CompareOps<R> + ScalarOps<R>,
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    let output = match kind {
        Extremum::Maximum => client.maximum(a.tensor(), b.tensor())?,
        Extremum::Minimum => client.minimum(a.tensor(), b.tensor())?,
    };

    if !(a.requires_grad() || b.requires_grad()) {
        return Ok(Var::new(output, false));
    }

    let wins = |lhs: &Tensor<R>, rhs: &Tensor<R>, tie: &Tensor<R>| -> Result<Tensor<R>> {
        let strict = match kind {
            Extremum::Maximum => client.gt(lhs, rhs)?,
            Extremum::Minimum => client.lt(lhs, rhs)?,
        };
        client.add(&strict, &client.mul_scalar(tie, 0.5)?)
    };
    let tie = client.eq(a.tensor(), b.tensor())?;
    let weight_a = wins(a.tensor(), b.tensor(), &tie)?;
    let weight_b = wins(b.tensor(), a.tensor(), &tie)?;

    let grad_fn = ExtremumBackward::<R> {
        name: match kind {
            Extremum::Maximum => "MaximumBackward",
            Extremum::Minimum => "MinimumBackward",
        },
        input_ids: [a.id(), b.id()],
        weights: [weight_a, weight_b],
        shapes: [a.shape().to_vec(), b.shape().to_vec()],
        input_grad_fns: [a.grad_fn().cloned(), b.grad_fn().cloned()],
    };
    Ok(Var::from_op(output, Arc::new(grad_fn)))
}

/// Backward for `maximum` and `minimum`: inputs `[a, b]`.
pub(crate) struct ExtremumBackward<R: Runtime> {
    name: &'static str,
    input_ids: [TensorId; 2],
    /// Per-element weights of `[a, b]`, in the broadcast output shape.
    weights: [Tensor<R>; 2],
    /// Shapes of `[a, b]` before broadcasting.
    shapes: [Vec<usize>; 2],
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 2],
}

impl<R: Runtime<DType = DType>> GradFn<R> for ExtremumBackward<R>
where
    R::Client: TensorOps<R> + BinaryOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>, needed: &[bool]) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let mut grads = Vec::with_capacity(2);
        for slot in 0..2 {
            if needed.get(slot).copied().unwrap_or(true) {
                let scaled = client.mul(grad_output, &self.weights[slot])?;
                grads.push(Some(reduce_grad_for_broadcast(
                    &scaled,
                    &self.shapes[slot],
                )?));
            } else {
                grads.push(None);
            }
        }
        Ok(grads)
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        let mut grads = Vec::with_capacity(2);
        for slot in 0..2 {
            let weight = Var::new(self.weights[slot].clone(), false);
            let scaled = var_mul(grad_output, &weight, &client)?;
            grads.push(Some(reduce_var_for_broadcast(
                &scaled,
                &self.shapes[slot],
                &client,
            )?));
        }
        Ok(grads)
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        &self.weights
    }

    fn name(&self) -> &'static str {
        self.name
    }
}
