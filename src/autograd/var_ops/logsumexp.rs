//! Log-sum-exp with gradient support: `y = log(sum(exp(x)))` over `dims`.
//!
//! Backward is `dx = dy * exp(x - y)`, with `dy` and `y` reshaped to the input's
//! shape with the reduced dimensions set to 1 so they broadcast.

use crate::autograd::var_ops::var_mul;
use crate::autograd::{GradFn, Var};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, CumulativeOps, TensorOps, UnaryOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Log-sum-exp of `a` over `dims`.
///
/// `dims` must not be empty. `keepdim` keeps each reduced dimension with size 1.
pub fn var_logsumexp<R, C>(a: &Var<R>, dims: &[usize], keepdim: bool, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + CumulativeOps<R>,
    R::Client: TensorOps<R>,
{
    if dims.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "dims",
            reason: "logsumexp needs at least one dimension".to_string(),
        });
    }
    let output = client.logsumexp(a.tensor(), dims, keepdim)?;

    if a.requires_grad() {
        let mut kept_shape = a.shape().to_vec();
        for &d in dims {
            if let Some(size) = kept_shape.get_mut(d) {
                *size = 1;
            }
        }
        let grad_fn = LogSumExpBackward::<R> {
            input_ids: [a.id()],
            saved: [a.tensor().clone(), output.clone()],
            kept_shape,
            input_grad_fn: a.grad_fn().cloned(),
        };
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Backward for `var_logsumexp`: input `[a]`.
pub struct LogSumExpBackward<R: Runtime> {
    input_ids: [TensorId; 1],
    /// `[x, y]`.
    saved: [Tensor<R>; 2],
    /// The input's shape with each reduced dimension set to 1.
    kept_shape: Vec<usize>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime<DType = DType>> LogSumExpBackward<R> {
    /// `exp(x - y)`, the softmax weights, in the input's shape.
    fn weights(&self, client: &R::Client) -> Result<Tensor<R>>
    where
        R::Client: TensorOps<R> + BinaryOps<R> + UnaryOps<R>,
    {
        let [x, y] = &self.saved;
        let y_kept = y.contiguous()?.reshape(&self.kept_shape)?;
        client.exp(&client.sub(x, &y_kept)?)
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for LogSumExpBackward<R>
where
    R::Client: TensorOps<R> + BinaryOps<R> + UnaryOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let g_kept = grad_output.contiguous()?.reshape(&self.kept_shape)?;
        let dx = client.mul(&g_kept, &self.weights(&client)?)?;
        Ok(vec![Some(dx)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        let g_kept = crate::autograd::var_reshape(grad_output, &self.kept_shape)?;
        let weights = Var::new(self.weights(&client)?, false);
        Ok(vec![Some(var_mul(&g_kept, &weights, &client)?)])
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
        "LogSumExpBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::test_fd::{check_grad_f64, f64_tensor};
    use crate::autograd::{backward, var_sum};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    const X: [f64; 12] = [
        0.3, -1.2, 0.7, 2.1, -0.4, 1.5, 0.9, -2.2, 1.1, 0.0, -0.6, 0.8,
    ];

    #[test]
    fn fd_last_dim() {
        for keepdim in [false, true] {
            check_grad_f64(
                "var_logsumexp last dim",
                move |inputs, c| var_logsumexp(&inputs[0], &[1], keepdim, c),
                &[f64_tensor(&X, &[3, 4])],
            );
        }
    }

    #[test]
    fn fd_middle_dim() {
        for keepdim in [false, true] {
            check_grad_f64(
                "var_logsumexp middle dim",
                move |inputs, c| var_logsumexp(&inputs[0], &[1], keepdim, c),
                &[f64_tensor(&X, &[2, 3, 2])],
            );
        }
    }

    #[test]
    fn fd_two_dims() {
        check_grad_f64(
            "var_logsumexp two dims",
            |inputs, c| var_logsumexp(&inputs[0], &[0, 2], false, c),
            &[f64_tensor(&X, &[2, 3, 2])],
        );
    }

    #[test]
    fn gradient_is_softmax() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&[0.0, 0.0, 0.0, 0.0], &[4]), true);
        let y = var_logsumexp(&x, &[0], true, &client).unwrap();
        let loss = var_sum(&y, &[0], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        for v in grads.get(x.id()).unwrap().to_vec::<f64>() {
            assert!((v - 0.25).abs() < 1e-12);
        }
    }

    #[test]
    fn empty_dims_are_an_error() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&[1.0, 2.0], &[2]), true);
        assert!(var_logsumexp(&x, &[], false, &client).is_err());
    }

    #[test]
    fn no_grad_input_gives_detached_var() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&[1.0, 2.0], &[2]), false);
        let y = var_logsumexp(&x, &[0], false, &client).unwrap();
        assert!(!y.requires_grad());
        assert!(y.grad_fn().is_none());
    }
}
