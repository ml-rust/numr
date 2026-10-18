//! Shared backward for element-wise ops: `dx = dy * f'(x)`.
//!
//! The op supplies a derivative function. The backward runs it on the saved
//! input and output when the gradient arrives, so the forward pass pays nothing.
//! Second-order backward treats `f'(x)` as a constant. The dependence on `dy`
//! stays on the graph.

use crate::autograd::var_ops::var_mul;
use crate::autograd::{GradFn, Var};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, TensorOps};
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Computes `f'(x)` from the saved input `x`, the saved output `y` and a scalar parameter.
pub(crate) type DerivativeFn<R> =
    fn(&<R as Runtime>::Client, &Tensor<R>, &Tensor<R>, f64) -> Result<Tensor<R>>;

/// Wrap `output` in a `Var` that scales its gradient by `derivative(x, y, param)`.
///
/// Returns a detached `Var` when `x` does not require grad.
pub(crate) fn unary_with_derivative<R: Runtime<DType = DType>>(
    x: &Var<R>,
    output: Tensor<R>,
    name: &'static str,
    param: f64,
    derivative: DerivativeFn<R>,
) -> Var<R>
where
    R::Client: TensorOps<R>,
{
    if !x.requires_grad() {
        return Var::new(output, false);
    }
    let grad_fn = DerivativeBackward::<R> {
        name,
        input_ids: [x.id()],
        saved: [x.tensor().clone(), output.clone()],
        param,
        derivative,
        input_grad_fn: x.grad_fn().cloned(),
    };
    Var::from_op(output, Arc::new(grad_fn))
}

/// Backward for an element-wise op: input `[x]`.
pub(crate) struct DerivativeBackward<R: Runtime> {
    name: &'static str,
    input_ids: [TensorId; 1],
    /// `[x, y]`.
    saved: [Tensor<R>; 2],
    param: f64,
    derivative: DerivativeFn<R>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime<DType = DType>> GradFn<R> for DerivativeBackward<R>
where
    R::Client: TensorOps<R> + BinaryOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let [x, y] = &self.saved;
        let d = (self.derivative)(&client, x, y, self.param)?;
        Ok(vec![Some(client.mul(grad_output, &d)?)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        let [x, y] = &self.saved;
        let d = (self.derivative)(&client, x, y, self.param)?;
        let dx = var_mul(grad_output, &Var::new(d, false), &client)?;
        Ok(vec![Some(dx)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        &self.saved
    }

    fn name(&self) -> &'static str {
        self.name
    }
}
