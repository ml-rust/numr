//! Backward pass implementation
//!
//! Implements reverse-mode automatic differentiation using topological sort
//! to traverse the computation graph and accumulate gradients.

use super::{GradFn, GradStore, Var};
use crate::error::{Error, Result};
use crate::ops::TensorOps;
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::collections::HashSet;
use std::sync::Arc;

/// Compute gradients via reverse-mode automatic differentiation
///
/// Starting from a scalar loss, traverses the computation graph in
/// reverse topological order, computing gradients for all tensors
/// that require them.
///
/// # Arguments
///
/// * `loss` - The scalar loss tensor to differentiate
/// * `client` - The runtime client for tensor operations
///
/// # Returns
///
/// A `GradStore` containing gradients for all tensors in the graph.
///
/// # Example
///
/// ```ignore
/// use numr::prelude::*;
/// use numr::autograd::{Var, backward};
///
/// let device = CpuDevice::new();
/// let client = CpuRuntime::default_client(&device);
///
/// // Create variables
/// let x = Var::new(Tensor::from_slice(&[2.0f32], &[1], &device), true);
/// let y = Var::new(Tensor::from_slice(&[3.0f32], &[1], &device), true);
///
/// // Forward: z = x * y
/// let z = x.mul(&y, &client)?;
///
/// // Backward
/// let grads = backward(&z, &client)?;
///
/// // dx = y = 3.0
/// let grad_x = grads.get(x.id()).unwrap();
/// ```
pub fn backward<R, C>(loss: &Var<R>, client: &C) -> Result<GradStore<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
{
    // Ensure loss is a scalar
    if loss.numel() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![1],
            got: loss.shape().to_vec(),
        });
    }

    // Ensure loss requires gradients
    if !loss.requires_grad() {
        return Err(Error::Internal(
            "backward() called on tensor that doesn't require grad".into(),
        ));
    }

    // Initialize gradient store
    let mut grad_store = GradStore::new();

    // Initialize gradient of loss with respect to itself: dL/dL = 1
    let one = Tensor::<R>::ones(loss.shape(), loss.tensor().dtype(), loss.tensor().device());
    grad_store.insert(loss.id(), one);

    // Build the computation graph and get topological order
    let topo_order = topological_sort(loss);

    // Traverse in reverse topological order (from output to inputs)
    for var_entry in topo_order.into_iter().rev() {
        let (var_id, grad_fn_opt, input_ids) = var_entry;

        // Get gradient for this node
        let grad_output = match grad_store.get(var_id) {
            Some(g) => g.clone(),
            None => continue, // No gradient flowing to this node
        };

        // If this node has a grad_fn, compute gradients for its inputs
        if let Some(grad_fn) = grad_fn_opt {
            // Compute gradients for inputs
            let input_grads = grad_fn.backward(&grad_output)?;

            // Accumulate gradients for each input
            for (input_id, input_grad_opt) in input_ids.iter().zip(input_grads.into_iter()) {
                if let Some(input_grad) = input_grad_opt {
                    // Accumulate gradient using tensor addition
                    grad_store.try_accumulate(*input_id, input_grad, |existing, new| {
                        client.add(&existing, &new)
                    })?;
                }
            }
        }
    }

    Ok(grad_store)
}

/// Entry for topological sort: (tensor_id, grad_fn, input_ids)
type TopoEntry<R> = (TensorId, Option<Arc<dyn GradFn<R>>>, Vec<TensorId>);

/// Build topological sort of computation graph using DFS post-order traversal
///
/// Returns nodes in topological order (inputs before outputs).
fn topological_sort<R: Runtime>(loss: &Var<R>) -> Vec<TopoEntry<R>> {
    let mut result = Vec::new();
    let mut visited = HashSet::new();

    fn dfs<R: Runtime>(
        id: TensorId,
        grad_fn: Option<Arc<dyn GradFn<R>>>,
        visited: &mut HashSet<TensorId>,
        result: &mut Vec<TopoEntry<R>>,
    ) {
        if visited.contains(&id) {
            return;
        }
        visited.insert(id);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_backward_requires_scalar() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Non-scalar tensor should fail
        let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let var = Var::new(tensor, true);

        let result = backward(&var, &client);
        assert!(result.is_err());
    }

    #[test]
    fn test_backward_leaf_variable() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Simple scalar leaf variable
        let tensor = Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[1], &device);
        let var = Var::new(tensor, true);

        let grads = backward(&var, &client).unwrap();

        // Gradient of loss w.r.t. itself should be 1
        let grad = grads.get(var.id()).unwrap();
        let grad_data: Vec<f32> = grad.to_vec();
        assert_eq!(grad_data, vec![1.0f32]);
    }
}
