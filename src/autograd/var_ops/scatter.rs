//! Scatter with overwrite semantics and gradient support.
//!
//! `z = target` with `z[index[..]] = src[..]` along `dim`. For summing
//! semantics use `var_scatter_add`.

use crate::autograd::{GradFn, Var, var_gather};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::IndexingOps;
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Overwrite entries of `target` along `dim` with `src` at the positions in `index`.
///
/// For `dim == 1` on 2-D inputs, `z[i][index[i][j]] = src[i][j]`, and every
/// other entry equals `target`. `index` and `src` must have the same shape.
/// `index` holds integer positions and is not differentiable.
///
/// The gradient to `target` is the output gradient with the overwritten
/// positions set to zero. The gradient to `src` is `gather(grad, dim, index)`.
///
/// When `index` names a position more than once, the backend does not define
/// which `src` element wins. The gradient to `src` is correct only for unique
/// indices, because it hands the full output gradient to every duplicate. Use
/// `var_scatter_add` when duplicates must combine.
pub fn var_scatter<R, C>(
    target: &Var<R>,
    dim: usize,
    index: &Tensor<R>,
    src: &Var<R>,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + IndexingOps<R>,
    R::Client: IndexingOps<R>,
{
    let output = client.scatter(target.tensor(), dim, index, src.tensor())?;

    if target.requires_grad() || src.requires_grad() {
        let grad_fn = ScatterBackward::<R> {
            input_ids: [target.id(), src.id()],
            dim,
            index: index.clone(),
            src_shape: src.shape().to_vec(),
            input_grad_fns: [target.grad_fn().cloned(), src.grad_fn().cloned()],
        };
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Backward for `var_scatter`: inputs `[target, src]`.
pub struct ScatterBackward<R: Runtime> {
    input_ids: [TensorId; 2],
    dim: usize,
    index: Tensor<R>,
    src_shape: Vec<usize>,
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 2],
}

impl<R: Runtime<DType = DType>> GradFn<R> for ScatterBackward<R>
where
    R::Client: IndexingOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>, needed: &[bool]) -> Result<Vec<Option<Tensor<R>>>> {
        let want = |slot: usize| needed.get(slot).copied().unwrap_or(true);
        let client = R::default_client(grad_output.device());
        let grad = grad_output.contiguous()?;

        let grad_target = if want(0) {
            let zeros = Tensor::<R>::zeros(&self.src_shape, grad.dtype(), grad.device())?;
            Some(client.scatter(&grad, self.dim, &self.index, &zeros)?)
        } else {
            None
        };
        let grad_src = if want(1) {
            Some(client.gather(&grad, self.dim, &self.index)?)
        } else {
            None
        };
        Ok(vec![grad_target, grad_src])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let g = grad_output.tensor();
        let client = R::default_client(g.device());
        let zeros = Var::new(
            Tensor::<R>::zeros(&self.src_shape, g.dtype(), g.device())?,
            false,
        );
        let grad_target = var_scatter(grad_output, self.dim, &self.index, &zeros, &client)?;
        let grad_src = var_gather(grad_output, self.dim, &self.index, &client)?;
        Ok(vec![Some(grad_target), Some(grad_src)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.index)
    }

    fn name(&self) -> &'static str {
        "ScatterBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::test_fd::{check_grad_f64, f64_tensor, grad_var};
    use crate::autograd::{backward, backward_with_graph, var_mul, var_sum};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    fn index(values: &[i64], shape: &[usize]) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(values, shape, &CpuDevice::new()).unwrap()
    }

    const TARGET: [f64; 12] = [
        0.3, -1.2, 0.7, 2.1, -0.4, 1.5, 0.9, -2.2, 1.1, 0.0, -0.6, 0.8,
    ];

    #[test]
    fn forward_overwrites_named_positions() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let target = Var::new(f64_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]), false);
        let src = Var::new(f64_tensor(&[10.0, 20.0], &[2]), false);
        let z = var_scatter(&target, 0, &index(&[3, 1], &[2]), &src, &client).unwrap();
        assert_eq!(z.tensor().to_vec::<f64>(), vec![1.0, 20.0, 3.0, 10.0]);
    }

    #[test]
    fn overwritten_target_entries_get_zero_gradient() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let target = Var::new(f64_tensor(&[0.0; 4], &[4]), true);
        let src = Var::new(f64_tensor(&[1.0, 1.0], &[2]), true);
        let weight = Var::new(f64_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]), false);
        let z = var_scatter(&target, 0, &index(&[3, 1], &[2]), &src, &client).unwrap();
        let loss = var_sum(&var_mul(&z, &weight, &client).unwrap(), &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        assert_eq!(
            grads.get(target.id()).unwrap().to_vec::<f64>(),
            vec![1.0, 0.0, 3.0, 0.0]
        );
        assert_eq!(grads.get(src.id()).unwrap().to_vec::<f64>(), vec![4.0, 2.0]);
    }

    #[test]
    fn fd_dim0_unique_indices() {
        let idx = index(&[0, 1, 2, 0, 2, 0, 1, 1], &[2, 4]);
        check_grad_f64(
            "var_scatter dim 0",
            move |inputs, c| var_scatter(&inputs[0], 0, &idx, &inputs[1], c),
            &[
                f64_tensor(&TARGET, &[3, 4]),
                f64_tensor(&[1.0, -0.5, 0.25, 2.0, -1.5, 0.75, 1.25, -2.5], &[2, 4]),
            ],
        );
    }

    #[test]
    fn fd_dim1_unique_indices() {
        let idx = index(&[2, 0, 1, 2], &[2, 2]);
        check_grad_f64(
            "var_scatter dim 1",
            move |inputs, c| var_scatter(&inputs[0], 1, &idx, &inputs[1], c),
            &[
                f64_tensor(&[0.5, -0.25, 1.75, -1.0, 2.5, 0.125], &[2, 3]),
                f64_tensor(&[0.9, -1.1, 0.4, 1.3], &[2, 2]),
            ],
        );
    }

    #[test]
    fn fd_second_order() {
        let idx = index(&[1, 3], &[2]);
        check_grad_f64(
            "var_scatter second order",
            move |inputs, c| {
                let z = var_scatter(&inputs[0], 0, &idx, &inputs[1], c)?;
                let cube = var_mul(&var_mul(&z, &z, c)?, &z, c)?;
                let loss = var_sum(&cube, &[], false, c)?;
                let grads = backward_with_graph(&loss, c)?;
                grad_var(&grads, inputs[0].id())
            },
            &[
                f64_tensor(&[0.5, -1.0, 0.25, 0.8], &[4]),
                f64_tensor(&[0.3, -0.8], &[2]),
            ],
        );
    }

    #[test]
    fn one_grad_free_operand_still_flows_to_the_other() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let target = Var::new(f64_tensor(&[1.0, 2.0, 3.0], &[3]), false);
        let src = Var::new(f64_tensor(&[5.0], &[1]), true);
        let z = var_scatter(&target, 0, &index(&[1], &[1]), &src, &client).unwrap();
        assert!(z.requires_grad());
        let loss = var_sum(&z, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        assert_eq!(grads.get(src.id()).unwrap().to_vec::<f64>(), vec![1.0]);
    }

    #[test]
    fn no_grad_inputs_give_detached_var() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let target = Var::new(f64_tensor(&[1.0, 2.0], &[2]), false);
        let src = Var::new(f64_tensor(&[7.0], &[1]), false);
        let z = var_scatter(&target, 0, &index(&[0], &[1]), &src, &client).unwrap();
        assert!(!z.requires_grad());
        assert!(z.grad_fn().is_none());
        assert_eq!(z.tensor().to_vec::<f64>(), vec![7.0, 2.0]);
    }
}
