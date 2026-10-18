//! Backward implementation for scatter-add

use crate::autograd::{GradFn, Var, var_gather};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::IndexingOps;
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Backward for scatter-add: z = scatter_reduce(target, dim, index, src, Sum, include_self)
///
/// Gradients:
/// - dL/dtarget = dL/dz, since every target element passes through once.
/// - dL/dsrc = gather(dL/dz, dim, index), since each src element adds to one slot.
pub struct ScatterAddBackward<R: Runtime> {
    input_ids: [TensorId; 2],
    dim: usize,
    index: Tensor<R>,
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 2],
}

impl<R: Runtime> ScatterAddBackward<R> {
    /// Constructs a new backward function for scatter-add.
    pub fn new(
        target_id: TensorId,
        src_id: TensorId,
        dim: usize,
        index: Tensor<R>,
        target_grad_fn: Option<Arc<dyn GradFn<R>>>,
        src_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [target_id, src_id],
            dim,
            index,
            input_grad_fns: [target_grad_fn, src_grad_fn],
        }
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for ScatterAddBackward<R>
where
    R::Client: IndexingOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>, needed: &[bool]) -> Result<Vec<Option<Tensor<R>>>> {
        // A short mask reads as "needed", which matches `backward_all`.
        let want = |slot: usize| needed.get(slot).copied().unwrap_or(true);
        let grad_target = want(0).then(|| grad_output.clone());
        let grad_src = if want(1) {
            let client = R::default_client(grad_output.device());
            Some(client.gather(grad_output, self.dim, &self.index)?)
        } else {
            None
        };
        Ok(vec![grad_target, grad_src])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        let grad_target = grad_output.clone();
        let grad_src = var_gather(grad_output, self.dim, &self.index, &client)?;
        Ok(vec![Some(grad_target), Some(grad_src)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn name(&self) -> &'static str {
        "ScatterAddBackward"
    }
}
