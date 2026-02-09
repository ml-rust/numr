//! Backward implementations for indexing operations (gather)

use crate::autograd::GradFn;
use crate::autograd::var::Var;
use crate::error::Result;
use crate::ops::IndexingOps;
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

// ============================================================================
// GatherBackward
// ============================================================================

/// Backward for gather: z = gather(a, dim, index)
///
/// dL/da = zeros_like(a); scatter(zeros, dim, index, grad_output)
pub struct GatherBackward<R: Runtime> {
    input_id: TensorId,
    input_shape: Vec<usize>,
    dim: usize,
    index: Tensor<R>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> GatherBackward<R> {
    /// Constructs a new backward function for gather.
    pub fn new(
        input_id: TensorId,
        input_shape: &[usize],
        dim: usize,
        index: Tensor<R>,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            input_shape: input_shape.to_vec(),
            dim,
            index,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for GatherBackward<R>
where
    R::Client: IndexingOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let zeros =
            Tensor::<R>::zeros(&self.input_shape, grad_output.dtype(), grad_output.device());
        let grad_input = client.scatter(&zeros, self.dim, &self.index, grad_output)?;
        Ok(vec![Some(grad_input)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        let zeros = Tensor::<R>::zeros(
            &self.input_shape,
            grad_output.tensor().dtype(),
            grad_output.tensor().device(),
        );
        let grad_input = client.scatter(&zeros, self.dim, &self.index, grad_output.tensor())?;
        Ok(vec![Some(Var::new(grad_input, true))])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "GatherBackward"
    }
}
