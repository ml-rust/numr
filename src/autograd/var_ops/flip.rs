//! Flip with gradient support: `z = flip(a, dims)`.
//!
//! The backward of a flip is the same flip applied to the output gradient.

use crate::autograd::{GradFn, Var};
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Reverse `a` along each dimension in `dims`.
///
/// `dims` accepts negative indices and must not repeat a dimension.
/// The result is a contiguous copy. An empty `dims` returns a contiguous copy
/// of `a` without reordering.
pub fn var_flip<R: Runtime>(a: &Var<R>, dims: &[isize]) -> Result<Var<R>> {
    let dims = normalize_dims(dims, a.ndim())?;
    let output = flip_contiguous(a.tensor(), &dims)?;

    if a.requires_grad() {
        let grad_fn = FlipBackward::<R> {
            input_ids: [a.id()],
            dims,
            input_grad_fn: a.grad_fn().cloned(),
        };
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// `v` itself when its tensor is contiguous, else a contiguous copy on the graph.
pub(crate) fn var_contiguous<R: Runtime>(v: &Var<R>) -> Result<Var<R>> {
    if v.tensor().is_contiguous() {
        Ok(v.alias())
    } else {
        var_flip(v, &[])
    }
}

fn normalize_dims(dims: &[isize], ndim: usize) -> Result<Vec<isize>> {
    let mut seen = vec![false; ndim];
    let mut out = Vec::with_capacity(dims.len());
    for &dim in dims {
        let idx = if dim < 0 { dim + ndim as isize } else { dim };
        if idx < 0 || idx >= ndim as isize {
            return Err(Error::InvalidDimension { dim, ndim });
        }
        if std::mem::replace(&mut seen[idx as usize], true) {
            return Err(Error::InvalidArgument {
                arg: "dims",
                reason: format!("flip names dimension {idx} more than once"),
            });
        }
        out.push(idx);
    }
    Ok(out)
}

fn flip_contiguous<R: Runtime>(t: &Tensor<R>, dims: &[isize]) -> Result<Tensor<R>> {
    t.flip_dims(dims)?.contiguous()
}

/// Backward for `var_flip`: input `[a]`.
pub struct FlipBackward<R: Runtime> {
    input_ids: [TensorId; 1],
    /// Non-negative, distinct dimensions.
    dims: Vec<isize>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> GradFn<R> for FlipBackward<R> {
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        Ok(vec![Some(flip_contiguous(grad_output, &self.dims)?)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        Ok(vec![Some(var_flip(grad_output, &self.dims)?)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "FlipBackward"
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
    fn forward_reverses_each_named_dim() {
        let x = Var::new(f64_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]), false);
        let rows = var_flip(&x, &[0]).unwrap();
        assert_eq!(
            rows.tensor().to_vec::<f64>(),
            vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]
        );
        let both = var_flip(&x, &[0, -1]).unwrap();
        assert_eq!(
            both.tensor().to_vec::<f64>(),
            vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        );
        assert!(both.tensor().is_contiguous());
    }

    #[test]
    fn fd_each_dim_choice() {
        for dims in [vec![0isize], vec![1], vec![-1], vec![0, 1]] {
            check_grad_f64(
                "var_flip",
                move |inputs, _| var_flip(&inputs[0], &dims),
                &[f64_tensor(&X, &[2, 3])],
            );
        }
    }

    #[test]
    fn fd_three_dims() {
        check_grad_f64(
            "var_flip 3-D",
            |inputs, _| var_flip(&inputs[0], &[0, 2]),
            &[f64_tensor(
                &[
                    0.3, -1.2, 0.7, 2.1, -0.4, 1.5, 0.9, -2.2, 1.1, 0.0, -0.6, 0.8,
                ],
                &[2, 3, 2],
            )],
        );
    }

    #[test]
    fn gradient_is_flipped_output_gradient() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = Var::new(f64_tensor(&[1.0, 2.0, 3.0], &[3]), true);
        let w = Var::new(f64_tensor(&[10.0, 20.0, 30.0], &[3]), false);
        let z = var_mul(&var_flip(&x, &[0]).unwrap(), &w, &client).unwrap();
        let loss = var_sum(&z, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        assert_eq!(
            grads.get(x.id()).unwrap().to_vec::<f64>(),
            vec![30.0, 20.0, 10.0]
        );
    }

    #[test]
    fn fd_second_order() {
        check_grad_f64(
            "var_flip second order",
            |inputs, c| {
                let flipped = var_flip(&inputs[0], &[0])?;
                let z = var_mul(&flipped, &inputs[0], c)?;
                let sq = var_mul(&z, &z, c)?;
                let loss = var_sum(&sq, &[], false, c)?;
                let grads = backward_with_graph(&loss, c)?;
                grad_var(&grads, inputs[0].id())
            },
            &[f64_tensor(&[0.5, -0.9, 1.3, 0.2], &[4])],
        );
    }

    #[test]
    fn empty_dims_copy_a_strided_view() {
        let x = Var::new(f64_tensor(&X, &[2, 3]), false);
        let view = Var::new(x.tensor().permute(&[1, 0]).unwrap(), false);
        assert!(!view.tensor().is_contiguous());
        let copy = var_flip(&view, &[]).unwrap();
        assert!(copy.tensor().is_contiguous());
        assert_eq!(
            copy.tensor().to_vec::<f64>(),
            vec![0.3, 2.1, -1.2, -0.4, 0.7, 1.5]
        );
    }

    #[test]
    fn repeated_or_out_of_range_dims_are_errors() {
        let x = Var::new(f64_tensor(&X, &[2, 3]), true);
        assert!(var_flip(&x, &[0, -2]).is_err());
        assert!(var_flip(&x, &[2]).is_err());
        assert!(var_flip(&x, &[-3]).is_err());
    }

    #[test]
    fn no_grad_input_gives_detached_var() {
        let x = Var::new(f64_tensor(&X, &[6]), false);
        let z = var_flip(&x, &[0]).unwrap();
        assert!(!z.requires_grad());
        assert!(z.grad_fn().is_none());
    }
}
