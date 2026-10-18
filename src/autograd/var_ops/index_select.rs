//! Index select with gradient support: `z = a.index_select(dim, index)`.
//!
//! The gradient scatters the output gradient back into a zero tensor of the
//! input's shape. A repeated index sums every gradient row that names it.

use crate::autograd::var_ops::var_scatter_add;
use crate::autograd::{GradFn, Var};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{IndexingOps, ScatterReduceOp, TensorOps, TypeConversionOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Select entries of `a` along `dim` at the 1-D positions in `index`.
///
/// `index` is not differentiable. A repeated index returns the same entry more
/// than once, and the gradient sums over those copies.
pub fn var_index_select<R, C>(
    a: &Var<R>,
    dim: usize,
    index: &Tensor<R>,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + IndexingOps<R> + TypeConversionOps<R>,
    R::Client: TensorOps<R> + IndexingOps<R>,
{
    let output = client.index_select(a.tensor(), dim, index)?;

    if a.requires_grad() {
        let index = if index.dtype() == DType::I64 {
            index.clone()
        } else {
            client.cast(index, DType::I64)?
        };
        let grad_fn = IndexSelectBackward::<R> {
            input_ids: [a.id()],
            input_shape: a.shape().to_vec(),
            dim,
            index,
            input_grad_fn: a.grad_fn().cloned(),
        };
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Backward for `var_index_select`: input `[a]`.
pub struct IndexSelectBackward<R: Runtime> {
    input_ids: [TensorId; 1],
    input_shape: Vec<usize>,
    dim: usize,
    /// 1-D I64 positions.
    index: Tensor<R>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime<DType = DType>> IndexSelectBackward<R> {
    /// The index reshaped to `[1, .., m, .., 1]` and expanded to the gradient's shape.
    fn expanded_index(&self, grad_shape: &[usize]) -> Result<Tensor<R>> {
        if self.dim >= grad_shape.len() {
            return Err(Error::InvalidDimension {
                dim: self.dim as isize,
                ndim: grad_shape.len(),
            });
        }
        let mut shape = vec![1usize; grad_shape.len()];
        shape[self.dim] = self.index.numel();
        self.index
            .reshape(&shape)?
            .broadcast_to(grad_shape)?
            .contiguous()
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for IndexSelectBackward<R>
where
    R::Client: IndexingOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let index = self.expanded_index(grad_output.shape())?;
        let zeros =
            Tensor::<R>::zeros(&self.input_shape, grad_output.dtype(), grad_output.device())?;
        let grad = client.scatter_reduce(
            &zeros,
            self.dim,
            &index,
            &grad_output.contiguous()?,
            ScatterReduceOp::Sum,
            true,
        )?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        let g = grad_output.tensor();
        let index = self.expanded_index(g.shape())?;
        let zeros = Var::new(
            Tensor::<R>::zeros(&self.input_shape, g.dtype(), g.device())?,
            false,
        );
        let grad = var_scatter_add(&zeros, self.dim, &index, grad_output, &client)?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.index)
    }

    fn name(&self) -> &'static str {
        "IndexSelectBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::test_fd::{check_grad_f64, f64_tensor};
    use crate::autograd::{backward, var_sum};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    fn index(values: &[i64]) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(values, &[values.len()], &CpuDevice::new()).unwrap()
    }

    const A: [f64; 12] = [
        0.3, -1.2, 0.7, 2.1, -0.4, 1.5, 0.9, -2.2, 1.1, 0.0, -0.6, 0.8,
    ];

    #[test]
    fn fd_dim0_duplicate_indices() {
        let idx = index(&[2, 0, 2, 2]);
        check_grad_f64(
            "var_index_select dim 0",
            move |inputs, c| var_index_select(&inputs[0], 0, &idx, c),
            &[f64_tensor(&A, &[3, 4])],
        );
    }

    #[test]
    fn fd_dim1_duplicate_indices() {
        let idx = index(&[3, 1, 1, 0, 3]);
        check_grad_f64(
            "var_index_select dim 1",
            move |inputs, c| var_index_select(&inputs[0], 1, &idx, c),
            &[f64_tensor(&A, &[3, 4])],
        );
    }

    #[test]
    fn duplicates_accumulate_and_unselected_rows_get_zero() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let a = Var::new(f64_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]), true);
        let z = var_index_select(&a, 0, &index(&[0, 0, 2]), &client).unwrap();
        let loss = var_sum(&z, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        assert_eq!(
            grads.get(a.id()).unwrap().to_vec::<f64>(),
            vec![2.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn fd_second_order() {
        let idx = index(&[1, 1, 0]);
        check_grad_f64(
            "var_index_select second order",
            move |inputs, c| {
                let z = var_index_select(&inputs[0], 0, &idx, c)?;
                let cube = crate::autograd::var_mul(&crate::autograd::var_mul(&z, &z, c)?, &z, c)?;
                let loss = var_sum(&cube, &[], false, c)?;
                let grads = crate::autograd::backward_with_graph(&loss, c)?;
                crate::autograd::test_fd::grad_var(&grads, inputs[0].id())
            },
            &[f64_tensor(&[0.5, -0.9, 1.3], &[3])],
        );
    }

    #[test]
    fn no_grad_input_gives_detached_var() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let a = Var::new(f64_tensor(&[1.0, 2.0, 3.0], &[3]), false);
        let z = var_index_select(&a, 0, &index(&[2, 2]), &client).unwrap();
        assert!(!z.requires_grad());
        assert!(z.grad_fn().is_none());
        assert_eq!(z.tensor().to_vec::<f64>(), vec![3.0, 3.0]);
    }

    #[test]
    fn out_of_range_index_is_an_error() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let a = Var::new(f64_tensor(&[1.0, 2.0], &[2]), true);
        assert!(var_index_select(&a, 0, &index(&[2]), &client).is_err());
    }
}
