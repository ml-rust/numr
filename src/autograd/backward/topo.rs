use crate::autograd::{GradFn, Var};
use crate::runtime::Runtime;
use crate::tensor::TensorId;
use std::collections::HashSet;
use std::sync::Arc;

/// Entry for topological sort: (tensor_id, grad_fn, input_ids)
pub(super) type TopoEntry<R> = (TensorId, Option<Arc<dyn GradFn<R>>>, Vec<TensorId>);

/// Build topological sort of computation graph using DFS post-order traversal
///
/// Returns nodes in topological order (inputs before outputs).
pub(super) fn topological_sort<R: Runtime>(loss: &Var<R>) -> Vec<TopoEntry<R>> {
    let mut result = Vec::new();
    let mut visited = HashSet::new();

    fn dfs<R: Runtime>(
        id: TensorId,
        grad_fn: Option<Arc<dyn GradFn<R>>>,
        visited: &mut HashSet<TensorId>,
        result: &mut Vec<TopoEntry<R>>,
    ) {
        if !visited.insert(id) {
            return;
        }

        let input_ids: Vec<TensorId> = grad_fn
            .as_ref()
            .map(|gf| gf.inputs().to_vec())
            .unwrap_or_default();

        // Get input grad_fns and visit inputs first (dependencies)
        if let Some(gf) = &grad_fn {
            for (input_id, input_grad_fn) in input_ids.iter().zip(gf.input_grad_fns()) {
                dfs(*input_id, input_grad_fn, visited, result);
            }
        }

        // Add this node after its inputs (post-order)
        result.push((id, grad_fn, input_ids));
    }

    dfs(
        loss.id(),
        loss.grad_fn().cloned(),
        &mut visited,
        &mut result,
    );
    result
}
