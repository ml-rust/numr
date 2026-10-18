//! Constant padding with gradient support: `z = pad(a, padding, value)`.
//!
//! The gradient is the output gradient narrowed back to the input's extent on
//! each padded dimension. The padded border receives no gradient, and `value`
//! is a plain number with no gradient.

use crate::autograd::{GradFn, Var, var_narrow};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{ShapeOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Pad `a` with the constant `value`.
///
/// `padding` lists `(before, after)` pairs starting at the last dimension, in
/// the PyTorch order: `[last_before, last_after, prev_before, prev_after, ..]`.
/// It must have an even length and at most `a.ndim()` pairs. The sizes are
/// unsigned, so negative padding (cropping) is not expressible. Use
/// `var_narrow` to crop.
pub fn var_pad<R, C>(a: &Var<R>, padding: &[usize], value: f64, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ShapeOps<R>,
    R::Client: RuntimeClient<R> + TensorOps<R> + ShapeOps<R>,
{
    if !padding.len().is_multiple_of(2) || padding.len() / 2 > a.ndim() {
        return Err(Error::InvalidArgument {
            arg: "padding",
            reason: format!(
                "padding {padding:?} must hold (before, after) pairs for at most {} dimensions",
                a.ndim()
            ),
        });
    }
    let output = client.pad(a.tensor(), padding, value)?;

    if a.requires_grad() {
        let grad_fn = PadBackward::<R> {
            input_ids: [a.id()],
            regions: regions(a.shape(), padding),
            input_grad_fn: a.grad_fn().cloned(),
        };
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// `(dim, start, len)` of the input inside the output for each padded dimension.
fn regions(input_shape: &[usize], padding: &[usize]) -> Vec<(isize, usize, usize)> {
    let ndim = input_shape.len();
    padding
        .as_chunks::<2>()
        .0
        .iter()
        .enumerate()
        .filter(|(_, pair)| pair[0] != 0 || pair[1] != 0)
        .map(|(i, pair)| {
            let dim = ndim - 1 - i;
            (dim as isize, pair[0], input_shape[dim])
        })
        .collect()
}

/// Backward for `var_pad`: input `[a]`.
pub struct PadBackward<R: Runtime> {
    input_ids: [TensorId; 1],
    /// `(dim, start, len)` of the input inside the padded output.
    regions: Vec<(isize, usize, usize)>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime<DType = DType>> GradFn<R> for PadBackward<R>
where
    R::Client: RuntimeClient<R> + TensorOps<R> + ShapeOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        let mut grad = grad_output.clone();
        for &(dim, start, len) in &self.regions {
            grad = grad.narrow(dim, start, len)?;
        }
        Ok(vec![Some(grad.contiguous()?)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let mut grad = grad_output.alias();
        for &(dim, start, len) in &self.regions {
            grad = var_narrow(&grad, dim, start, len)?;
        }
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "PadBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::test_fd::{check_grad_f64, f64_tensor, grad_var};
    use crate::autograd::{backward, backward_with_graph, var_mul, var_sum};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    const X: [f64; 6] = [0.3, -1.2, 0.7, 2.1, -0.4, 1.5];

    #[test]
    fn forward_pads_last_dim_first() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let a = Var::new(f64_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
        let last = var_pad(&a, &[1, 1], 9.0, &client).unwrap();
        assert_eq!(last.shape(), &[2, 4]);
        assert_eq!(
            last.tensor().to_vec::<f64>(),
            vec![9.0, 1.0, 2.0, 9.0, 9.0, 3.0, 4.0, 9.0]
        );
        let both = var_pad(&a, &[0, 1, 2, 0], 0.0, &client).unwrap();
        assert_eq!(both.shape(), &[4, 3]);
    }

    #[test]
    fn fd_last_dim() {
        check_grad_f64(
            "var_pad last dim",
            |inputs, c| var_pad(&inputs[0], &[2, 1], 0.5, c),
            &[f64_tensor(&X, &[2, 3])],
        );
    }

    #[test]
    fn fd_first_dim_only() {
        check_grad_f64(
            "var_pad first dim",
            |inputs, c| var_pad(&inputs[0], &[0, 0, 1, 2], 0.0, c),
            &[f64_tensor(&X, &[2, 3])],
        );
    }

    #[test]
    fn fd_every_dim_asymmetric() {
        check_grad_f64(
            "var_pad all dims",
            |inputs, c| var_pad(&inputs[0], &[1, 0, 0, 2, 3, 1], -1.0, c),
            &[f64_tensor(&X, &[1, 2, 3])],
        );
    }

    #[test]
    fn border_gets_no_gradient_and_value_is_constant() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&[1.0, 2.0], &[2]), true);
        let w = Var::new(f64_tensor(&[5.0, 6.0, 7.0, 8.0], &[4]), false);
        let z = var_pad(&x, &[1, 1], 100.0, &client).unwrap();
        let loss = var_sum(&var_mul(&z, &w, &client).unwrap(), &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        assert_eq!(grads.get(x.id()).unwrap().to_vec::<f64>(), vec![6.0, 7.0]);
    }

    #[test]
    fn zero_padding_keeps_gradient_flowing() {
        check_grad_f64(
            "var_pad zero",
            |inputs, c| var_pad(&inputs[0], &[0, 0], 0.0, c),
            &[f64_tensor(&X, &[2, 3])],
        );
    }

    #[test]
    fn fd_second_order() {
        check_grad_f64(
            "var_pad second order",
            |inputs, c| {
                let z = var_pad(&inputs[0], &[1, 2], 0.0, c)?;
                let sq = var_mul(&z, &z, c)?;
                let cube = var_mul(&sq, &z, c)?;
                let loss = var_sum(&cube, &[], false, c)?;
                let grads = backward_with_graph(&loss, c)?;
                grad_var(&grads, inputs[0].id())
            },
            &[f64_tensor(&[0.5, -0.9, 1.3], &[3])],
        );
    }

    #[test]
    fn bad_padding_is_an_error() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let a = Var::new(f64_tensor(&X, &[2, 3]), true);
        assert!(var_pad(&a, &[1], 0.0, &client).is_err());
        assert!(var_pad(&a, &[1, 1, 1, 1, 1, 1], 0.0, &client).is_err());
    }

    #[test]
    fn no_grad_input_gives_detached_var() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let a = Var::new(f64_tensor(&X, &[6]), false);
        let z = var_pad(&a, &[1, 1], 0.0, &client).unwrap();
        assert!(!z.requires_grad());
        assert!(z.grad_fn().is_none());
    }
}
