//! Conditional select with gradient support: `z = cond ? x : y`.
//!
//! `cond` holds no gradient. Each branch receives the output gradient where it
//! was selected and zero elsewhere, summed back to the branch's own shape when
//! `x`, `y` or `cond` was broadcast.

use crate::autograd::ops::{reduce_grad_for_broadcast, reduce_var_for_broadcast};
use crate::autograd::var_ops::var_mul;
use crate::autograd::{GradFn, Var};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{ConditionalOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Select `x` where `cond` is non-zero and `y` elsewhere, with broadcasting.
///
/// `x` and `y` must share a dtype. `cond` can have any numeric dtype.
pub fn var_where<R, C>(cond: &Tensor<R>, x: &Var<R>, y: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ConditionalOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.where_cond(cond, x.tensor(), y.tensor())?;

    if x.requires_grad() || y.requires_grad() {
        let grad_fn = WhereBackward::<R> {
            input_ids: [x.id(), y.id()],
            cond: cond.clone(),
            shapes: [x.shape().to_vec(), y.shape().to_vec()],
            input_grad_fns: [x.grad_fn().cloned(), y.grad_fn().cloned()],
        };
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Backward for `var_where`: inputs `[x, y]`.
pub struct WhereBackward<R: Runtime> {
    input_ids: [TensorId; 2],
    cond: Tensor<R>,
    /// Shapes of `[x, y]` before broadcasting.
    shapes: [Vec<usize>; 2],
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 2],
}

impl<R: Runtime<DType = DType>> GradFn<R> for WhereBackward<R>
where
    R::Client: TensorOps<R> + ConditionalOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>, needed: &[bool]) -> Result<Vec<Option<Tensor<R>>>> {
        let want = |slot: usize| needed.get(slot).copied().unwrap_or(true);
        let client = R::default_client(grad_output.device());
        let zeros = Tensor::<R>::zeros(
            grad_output.shape(),
            grad_output.dtype(),
            grad_output.device(),
        )?;

        let grad_x = if want(0) {
            let g = client.where_cond(&self.cond, grad_output, &zeros)?;
            Some(reduce_grad_for_broadcast(&g, &self.shapes[0])?)
        } else {
            None
        };
        let grad_y = if want(1) {
            let g = client.where_cond(&self.cond, &zeros, grad_output)?;
            Some(reduce_grad_for_broadcast(&g, &self.shapes[1])?)
        } else {
            None
        };
        Ok(vec![grad_x, grad_y])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        let g = grad_output.tensor();
        let zeros = Tensor::<R>::zeros(g.shape(), g.dtype(), g.device())?;
        let ones = Tensor::<R>::ones(g.shape(), g.dtype(), g.device())?;

        // 0/1 masks of the output shape keep the dependence on `grad_output` on the graph.
        let mask_x = Var::new(client.where_cond(&self.cond, &ones, &zeros)?, false);
        let mask_y = Var::new(client.where_cond(&self.cond, &zeros, &ones)?, false);
        let grad_x = var_mul(grad_output, &mask_x, &client)?;
        let grad_y = var_mul(grad_output, &mask_y, &client)?;
        Ok(vec![
            Some(reduce_var_for_broadcast(&grad_x, &self.shapes[0], &client)?),
            Some(reduce_var_for_broadcast(&grad_y, &self.shapes[1], &client)?),
        ])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.cond)
    }

    fn name(&self) -> &'static str {
        "WhereBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::test_fd::{check_grad_f64, f64_tensor};
    use crate::autograd::{backward, var_sum};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn fd_same_shape() {
        let cond = f64_tensor(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0], &[2, 3]);
        check_grad_f64(
            "var_where same shape",
            move |inputs, c| var_where(&cond, &inputs[0], &inputs[1], c),
            &[
                f64_tensor(&[0.5, -1.0, 2.0, 0.3, -0.7, 1.1], &[2, 3]),
                f64_tensor(&[1.5, 0.25, -2.0, 0.8, 0.9, -1.1], &[2, 3]),
            ],
        );
    }

    #[test]
    fn fd_broadcast_all_operands() {
        // cond [2, 1], x [3], y [2, 1] broadcast to [2, 3].
        let cond = f64_tensor(&[1.0, 0.0], &[2, 1]);
        check_grad_f64(
            "var_where broadcast",
            move |inputs, c| var_where(&cond, &inputs[0], &inputs[1], c),
            &[
                f64_tensor(&[0.5, -1.0, 2.0], &[3]),
                f64_tensor(&[1.5, -0.25], &[2, 1]),
            ],
        );
    }

    #[test]
    fn broadcast_gradients_have_operand_shapes() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let cond = f64_tensor(&[1.0, 0.0, 1.0, 1.0, 0.0, 0.0], &[2, 3]);
        let x = Var::new(f64_tensor(&[1.0, 2.0, 3.0], &[3]), true);
        let y = Var::new(f64_tensor(&[4.0], &[1]), true);
        let z = var_where(&cond, &x, &y, &client).unwrap();
        let loss = var_sum(&z, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        assert_eq!(
            grads.get(x.id()).unwrap().to_vec::<f64>(),
            vec![2.0, 0.0, 1.0]
        );
        assert_eq!(grads.get(y.id()).unwrap().to_vec::<f64>(), vec![3.0]);
    }

    #[test]
    fn one_grad_free_branch_still_flows_to_the_other() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let cond = f64_tensor(&[1.0, 0.0], &[2]);
        let x = Var::new(f64_tensor(&[1.0, 2.0], &[2]), false);
        let y = Var::new(f64_tensor(&[3.0, 4.0], &[2]), true);
        let z = var_where(&cond, &x, &y, &client).unwrap();
        assert!(z.requires_grad());
        let loss = var_sum(&z, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        assert_eq!(grads.get(y.id()).unwrap().to_vec::<f64>(), vec![0.0, 1.0]);
    }

    #[test]
    fn no_grad_inputs_give_detached_var() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let cond = f64_tensor(&[1.0, 0.0], &[2]);
        let x = Var::new(f64_tensor(&[1.0, 2.0], &[2]), false);
        let y = Var::new(f64_tensor(&[3.0, 4.0], &[2]), false);
        let z = var_where(&cond, &x, &y, &client).unwrap();
        assert!(!z.requires_grad());
        assert!(z.grad_fn().is_none());
        assert_eq!(z.tensor().to_vec::<f64>(), vec![1.0, 4.0]);
    }
}
