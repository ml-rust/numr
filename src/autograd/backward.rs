//! Backward pass implementation

use super::{GradStore, Var};
use crate::error::Result;
use crate::runtime::Runtime;

/// Compute gradients via reverse-mode automatic differentiation
///
/// Starting from a scalar loss, traverses the computation graph in
/// reverse topological order, computing gradients for all tensors
/// that require them.
///
/// # Arguments
///
/// * `loss` - The scalar loss tensor to differentiate
///
/// # Returns
///
/// A `GradStore` containing gradients for all tensors in the graph.
pub fn backward<R: Runtime>(loss: &Var<R>) -> Result<GradStore<R>> {
    // TODO: Implement backward pass
    // 1. Initialize grads[loss.id] = ones_like(loss)
    // 2. Topological sort of computation graph
    // 3. For each node in reverse order:
    //    - Get grad_output from grads
    //    - Call grad_fn.backward(grad_output)
    //    - Accumulate results into grads for each input

    let grads = GradStore::new();

    // Placeholder implementation
    let _ = loss;

    Ok(grads)
}

// TODO: Implement topological sort helper
// fn topological_sort<R: Runtime>(root: &Var<R>) -> Vec<...> { ... }
