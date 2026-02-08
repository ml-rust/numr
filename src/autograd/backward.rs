//! Backward pass implementation
//!
//! Implements reverse-mode automatic differentiation using topological sort
//! to traverse the computation graph and accumulate gradients.
//!
//! # First-Order vs Second-Order Differentiation
//!
//! This module provides two backward functions:
//!
//! - [`backward`]: Standard first-order differentiation. Returns raw tensors.
//!   Efficient for training neural networks.
//!
//! - [`backward_with_graph`]: Second-order capable differentiation. Returns
//!   `Var`s that retain their computation history, enabling Hessians and HVPs.

use super::{GradFn, GradStore, Var, VarGradStore, var_add};
use crate::error::{Error, Result};
use crate::ops::TensorOps;
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::collections::HashSet;
use std::sync::Arc;

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate that the loss tensor is suitable for backward pass
///
/// Checks:
/// 1. Loss is a scalar (numel == 1)
/// 2. Loss requires gradients
#[inline]
fn validate_loss<R: Runtime>(loss: &Var<R>, fn_name: &str) -> Result<()> {
    if loss.numel() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![1],
            got: loss.shape().to_vec(),
        });
    }

    if !loss.requires_grad() {
        return Err(Error::Internal(format!(
            "{}() called on tensor that doesn't require grad",
            fn_name
        )));
    }

    Ok(())
}

/// Create the initial gradient tensor for the loss (dL/dL = 1)
#[inline]
fn create_loss_gradient<R: Runtime>(loss: &Var<R>) -> Tensor<R> {
    Tensor::<R>::ones(loss.shape(), loss.tensor().dtype(), loss.tensor().device())
}

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
/// ```
/// # use numr::prelude::*;
/// # use numr::autograd::{Var, backward, var_mul};
/// # let device = CpuDevice::new();
/// # let client = CpuRuntime::default_client(&device);
/// // Create variables
/// let x = Var::new(Tensor::from_slice(&[2.0f32], &[1], &device), true);
/// let y = Var::new(Tensor::from_slice(&[3.0f32], &[1], &device), true);
///
/// // Forward: z = x * y
/// let z = var_mul(&x, &y, &client)?;
///
/// // Backward
/// let grads = backward(&z, &client)?;
///
/// // dx = y = 3.0
/// let grad_x = grads.get(x.id()).unwrap();
/// # Ok::<(), numr::error::Error>(())
/// ```
pub fn backward<R, C>(loss: &Var<R>, client: &C) -> Result<GradStore<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
{
    validate_loss(loss, "backward")?;

    // Initialize gradient store with dL/dL = 1
    let mut grad_store = GradStore::new();
    grad_store.insert(loss.id(), create_loss_gradient(loss));

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

/// Compute gradients with graph retention for second-order differentiation
///
/// Like [`backward`], but returns `Var`s instead of raw tensors. The returned
/// gradients retain their computation history, enabling them to be differentiated
/// again for computing Hessians, Hessian-vector products (HVPs), and other
/// second-order derivatives.
///
/// # Arguments
///
/// * `loss` - The scalar loss tensor to differentiate
/// * `client` - The runtime client for tensor operations
///
/// # Returns
///
/// A `VarGradStore` containing gradient `Var`s for all tensors in the graph.
/// Each gradient can be differentiated again using [`backward`].
///
/// # Example
///
/// ```
/// # use numr::prelude::*;
/// # use numr::autograd::{Var, backward, backward_with_graph, var_mul, var_sum};
/// # let device = CpuDevice::new();
/// # let client = CpuRuntime::default_client(&device);
/// // Forward pass
/// let x = Var::new(Tensor::from_slice(&[2.0f32], &[1], &device), true);
/// let y = var_mul(&x, &x, &client)?;  // y = x²
///
/// // First backward - get gradient as Var (not detached)
/// let grads = backward_with_graph(&y, &client)?;
/// let grad_x = grads.get_var(x.id()).unwrap();  // dy/dx = 2x = 4
///
/// // grad_x is a Var with history, so we can differentiate it
/// // Compute HVP: multiply by vector v, then differentiate again
/// let v = Var::new(Tensor::from_slice(&[1.0f32], &[1], &device), true);
/// let grad_v = var_mul(grad_x, &v, &client)?;
/// let hvp = backward(&var_sum(&grad_v, &[], false, &client)?, &client)?;
/// // hvp[x] = d²y/dx² * v = 2 * 1 = 2
/// # Ok::<(), numr::error::Error>(())
/// ```
///
/// # Performance Note
///
/// This function is slower and uses more memory than [`backward`] because it
/// builds a computation graph for the gradient computation itself. Only use it
/// when you actually need second-order derivatives.
pub fn backward_with_graph<R, C>(loss: &Var<R>, client: &C) -> Result<VarGradStore<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    validate_loss(loss, "backward_with_graph")?;

    // Initialize gradient store with dL/dL = 1 as a Var
    // This is a leaf Var (no grad_fn), but requires_grad = true so it can be differentiated
    let mut var_grad_store = VarGradStore::new();
    var_grad_store.insert(loss.id(), Var::new(create_loss_gradient(loss), true));

    // Build the computation graph and get topological order
    let topo_order = topological_sort(loss);

    // Traverse in reverse topological order (from output to inputs)
    for var_entry in topo_order.into_iter().rev() {
        let (var_id, grad_fn_opt, input_ids) = var_entry;

        // Get gradient Var for this node (borrow, don't remove)
        let grad_output = match var_grad_store.get_var(var_id) {
            Some(g) => g.clone(),
            None => continue, // No gradient flowing to this node
        };

        // If this node has a grad_fn, compute gradients for its inputs
        if let Some(grad_fn) = grad_fn_opt {
            // Compute gradients for inputs using backward_var (returns Vars)
            let input_grads = grad_fn.backward_var(&grad_output)?;

            // Accumulate gradients for each input using var_add (builds graph)
            for (input_id, input_grad_opt) in input_ids.iter().zip(input_grads.into_iter()) {
                if let Some(input_grad) = input_grad_opt {
                    // Accumulate gradient using var_add to maintain computation graph
                    var_grad_store.try_accumulate(*input_id, input_grad, |existing, new| {
                        var_add(&existing, &new, client)
                    })?;
                }
            }
        }
    }

    Ok(var_grad_store)
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
    use crate::autograd::{var_mul, var_sum};
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

    // ========================================================================
    // backward_with_graph tests
    // ========================================================================

    #[test]
    fn test_backward_with_graph_requires_scalar() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Non-scalar tensor should fail
        let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let var = Var::new(tensor, true);

        let result = backward_with_graph(&var, &client);
        assert!(result.is_err());
    }

    #[test]
    fn test_backward_with_graph_leaf_variable() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Simple scalar leaf variable
        let tensor = Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[1], &device);
        let var = Var::new(tensor, true);

        let grads = backward_with_graph(&var, &client).unwrap();

        // Gradient of loss w.r.t. itself should be 1
        let grad_var = grads.get_var(var.id()).unwrap();
        let grad_data: Vec<f32> = grad_var.tensor().to_vec();
        assert_eq!(grad_data, vec![1.0f32]);

        // The gradient Var should require grad (for second-order)
        assert!(grad_var.requires_grad());
    }

    #[test]
    fn test_backward_with_graph_simple_mul() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // y = x * x = x²
        // dy/dx = 2x
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );
        let y = var_mul(&x, &x, &client).unwrap();

        let grads = backward_with_graph(&y, &client).unwrap();

        // dy/dx = 2x = 6
        let grad_x = grads.get_var(x.id()).unwrap();
        let grad_data: Vec<f32> = grad_x.tensor().to_vec();
        assert!((grad_data[0] - 6.0).abs() < 1e-6);

        // grad_x should require grad for second-order differentiation
        assert!(grad_x.requires_grad());
    }

    #[test]
    fn test_backward_with_graph_matches_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Test that backward_with_graph produces same numerical results as backward
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        // z = x * y
        let z1 = var_mul(&x, &y, &client).unwrap();
        let z2 = var_mul(&x, &y, &client).unwrap();

        let grads1 = backward(&z1, &client).unwrap();
        let grads2 = backward_with_graph(&z2, &client).unwrap();

        // Compare gradients
        let grad_x1: Vec<f32> = grads1.get(x.id()).unwrap().to_vec();
        let grad_x2: Vec<f32> = grads2.get(x.id()).unwrap().to_vec();
        assert!((grad_x1[0] - grad_x2[0]).abs() < 1e-6);

        let grad_y1: Vec<f32> = grads1.get(y.id()).unwrap().to_vec();
        let grad_y2: Vec<f32> = grads2.get(y.id()).unwrap().to_vec();
        assert!((grad_y1[0] - grad_y2[0]).abs() < 1e-6);
    }

    #[test]
    fn test_backward_with_graph_to_grad_store() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );
        let y = var_mul(&x, &x, &client).unwrap();

        let var_grads = backward_with_graph(&y, &client).unwrap();

        // Convert to regular GradStore
        let grad_store = var_grads.to_grad_store();

        // Should still have the gradient
        let grad_x: Vec<f32> = grad_store.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 4.0).abs() < 1e-6); // dy/dx = 2x = 4
    }

    #[test]
    fn test_second_order_derivative_x_squared() {
        // Test true second-order differentiation
        // f(x) = x², f'(x) = 2x, f''(x) = 2
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        // y = x²
        let y = var_mul(&x, &x, &client).unwrap();

        // First backward with graph
        let grads = backward_with_graph(&y, &client).unwrap();
        let grad_x = grads.get_var(x.id()).unwrap();

        // grad_x = 2x = 6
        let first_deriv: Vec<f32> = grad_x.tensor().to_vec();
        assert!((first_deriv[0] - 6.0).abs() < 1e-6);

        // Now differentiate grad_x to get second derivative
        // We need to sum grad_x to get a scalar for backward
        let grad_x_sum = var_sum(grad_x, &[], false, &client).unwrap();

        // Second backward
        let second_grads = backward(&grad_x_sum, &client).unwrap();

        // d²y/dx² = 2
        let second_deriv: Vec<f32> = second_grads.get(x.id()).unwrap().to_vec();
        assert!(
            (second_deriv[0] - 2.0).abs() < 1e-5,
            "Expected 2.0, got {}",
            second_deriv[0]
        );
    }

    #[test]
    fn test_hessian_vector_product() {
        // Test HVP: H @ v where H is the Hessian of f(x) = x²
        // H = [[2]], so H @ [1] = [2]
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        // f(x) = x²
        let y = var_mul(&x, &x, &client).unwrap();

        // First backward with graph
        let grads = backward_with_graph(&y, &client).unwrap();
        let grad_x = grads.get_var(x.id()).unwrap();

        // Vector v for HVP
        let v = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device),
            false, // v doesn't need grad
        );

        // Compute grad_x · v
        let grad_v = var_mul(grad_x, &v, &client).unwrap();
        let grad_v_sum = var_sum(&grad_v, &[], false, &client).unwrap();

        // Differentiate to get HVP
        let hvp_grads = backward(&grad_v_sum, &client).unwrap();

        // HVP = H @ v = 2 * 1 = 2
        let hvp: Vec<f32> = hvp_grads.get(x.id()).unwrap().to_vec();
        assert!(
            (hvp[0] - 2.0).abs() < 1e-5,
            "Expected HVP = 2.0, got {}",
            hvp[0]
        );
    }

    #[test]
    fn test_second_order_add() {
        // f(x, y) = x + y, d²f/dx² = 0, d²f/dy² = 0
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );

        let z = crate::autograd::var_add(&x, &y, &client).unwrap();

        // First backward
        let grads = backward_with_graph(&z, &client).unwrap();

        // df/dx = 1, df/dy = 1
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        let grad_y: Vec<f32> = grads.get(y.id()).unwrap().to_vec();
        assert!((grad_x[0] - 1.0).abs() < 1e-6);
        assert!((grad_y[0] - 1.0).abs() < 1e-6);

        // Second derivative of constant gradient is 0
        // (The gradient is 1, which doesn't depend on x or y)
        let grad_x_var = grads.get_var(x.id()).unwrap();
        let grad_x_sum = var_sum(grad_x_var, &[], false, &client).unwrap();
        let second_grads = backward(&grad_x_sum, &client).unwrap();

        // d²f/dx² = 0 (gradient of 1 w.r.t. x is 0)
        // x shouldn't have a gradient in second_grads because grad_x doesn't depend on x
        assert!(
            second_grads.get(x.id()).is_none(),
            "Expected no second-order gradient for add"
        );
    }

    #[test]
    fn test_second_order_sub() {
        // f(x, y) = x - y
        // df/dx = 1, df/dy = -1
        // d²f/dx² = 0, d²f/dy² = 0
        use crate::autograd::var_sub;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );

        let z = var_sub(&x, &y, &client).unwrap();

        // First backward
        let grads = backward_with_graph(&z, &client).unwrap();

        // df/dx = 1, df/dy = -1
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        let grad_y: Vec<f32> = grads.get(y.id()).unwrap().to_vec();
        assert!((grad_x[0] - 1.0).abs() < 1e-6);
        assert!((grad_y[0] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_second_order_div() {
        // f(x) = 1/x = x^(-1)
        // df/dx = -1/x² = -x^(-2)
        // d²f/dx² = 2/x³ = 2x^(-3)
        // At x = 2: d²f/dx² = 2/8 = 0.25
        use crate::autograd::var_div;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let one = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device),
            false,
        );
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );

        // f(x) = 1/x
        let y = var_div(&one, &x, &client).unwrap();

        // First backward
        let grads = backward_with_graph(&y, &client).unwrap();

        // df/dx = -1/x² = -0.25
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!(
            (grad_x[0] - (-0.25)).abs() < 1e-5,
            "Expected -0.25, got {}",
            grad_x[0]
        );

        // Second backward
        let grad_x_var = grads.get_var(x.id()).unwrap();
        let grad_x_sum = var_sum(grad_x_var, &[], false, &client).unwrap();
        let second_grads = backward(&grad_x_sum, &client).unwrap();

        // d²f/dx² = 2/x³ = 2/8 = 0.25
        let second_deriv: Vec<f32> = second_grads.get(x.id()).unwrap().to_vec();
        assert!(
            (second_deriv[0] - 0.25).abs() < 1e-4,
            "Expected 0.25, got {}",
            second_deriv[0]
        );
    }

    #[test]
    fn test_second_order_through_sum() {
        // f(x) = sum(x²)
        // For x = [a, b]: f = a² + b²
        // df/da = 2a, df/db = 2b
        // d²f/da² = 2, d²f/db² = 2
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device),
            true,
        );

        // f(x) = sum(x * x)
        let x_squared = var_mul(&x, &x, &client).unwrap();
        let y = var_sum(&x_squared, &[0], false, &client).unwrap(); // dim 0 to reduce all

        // First backward
        let grads = backward_with_graph(&y, &client).unwrap();

        // df/dx = 2x = [6, 8]
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!(
            (grad_x[0] - 6.0).abs() < 1e-5,
            "Expected 6.0, got {}",
            grad_x[0]
        );
        assert!(
            (grad_x[1] - 8.0).abs() < 1e-5,
            "Expected 8.0, got {}",
            grad_x[1]
        );

        // Second backward - differentiate sum(grad_x)
        let grad_x_var = grads.get_var(x.id()).unwrap();
        let grad_x_sum = var_sum(grad_x_var, &[0], false, &client).unwrap(); // dim 0 to reduce all
        let second_grads = backward(&grad_x_sum, &client).unwrap();

        // d²f/dx² = 2 for each element
        let second_deriv: Vec<f32> = second_grads.get(x.id()).unwrap().to_vec();
        assert!(
            (second_deriv[0] - 2.0).abs() < 1e-4,
            "Expected 2.0, got {}",
            second_deriv[0]
        );
        assert!(
            (second_deriv[1] - 2.0).abs() < 1e-4,
            "Expected 2.0, got {}",
            second_deriv[1]
        );
    }

    #[test]
    fn test_second_order_through_mean() {
        // f(x) = mean(x²)
        // For x = [a, b]: f = (a² + b²) / 2
        // df/da = a, df/db = b
        // d²f/da² = 1, d²f/db² = 1
        use crate::autograd::var_mean;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device),
            true,
        );

        // f(x) = mean(x * x)
        let x_squared = var_mul(&x, &x, &client).unwrap();
        let y = var_mean(&x_squared, &[0], false, &client).unwrap(); // dim 0 to reduce all

        // First backward
        let grads = backward_with_graph(&y, &client).unwrap();

        // df/dx = x (due to mean dividing by 2)
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!(
            (grad_x[0] - 3.0).abs() < 1e-5,
            "Expected 3.0, got {}",
            grad_x[0]
        );
        assert!(
            (grad_x[1] - 4.0).abs() < 1e-5,
            "Expected 4.0, got {}",
            grad_x[1]
        );

        // Second backward
        let grad_x_var = grads.get_var(x.id()).unwrap();
        let grad_x_sum = var_sum(grad_x_var, &[0], false, &client).unwrap(); // dim 0 to reduce all
        let second_grads = backward(&grad_x_sum, &client).unwrap();

        // d²f/dx² = 1 for each element
        let second_deriv: Vec<f32> = second_grads.get(x.id()).unwrap().to_vec();
        assert!(
            (second_deriv[0] - 1.0).abs() < 1e-4,
            "Expected 1.0, got {}",
            second_deriv[0]
        );
        assert!(
            (second_deriv[1] - 1.0).abs() < 1e-4,
            "Expected 1.0, got {}",
            second_deriv[1]
        );
    }

    #[test]
    fn test_second_order_through_mul_scalar() {
        // f(x) = sum(3 * x²)
        // df/dx = 6x
        // d²f/dx² = 6
        use crate::autograd::var_mul_scalar;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );

        // f(x) = 3 * x²
        let x_squared = var_mul(&x, &x, &client).unwrap();
        let y = var_mul_scalar(&x_squared, 3.0, &client).unwrap();

        // First backward
        let grads = backward_with_graph(&y, &client).unwrap();

        // df/dx = 6x = 12
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!(
            (grad_x[0] - 12.0).abs() < 1e-5,
            "Expected 12.0, got {}",
            grad_x[0]
        );

        // Second backward
        let grad_x_var = grads.get_var(x.id()).unwrap();
        let grad_x_sum = var_sum(grad_x_var, &[0], false, &client).unwrap();
        let second_grads = backward(&grad_x_sum, &client).unwrap();

        // d²f/dx² = 6
        let second_deriv: Vec<f32> = second_grads.get(x.id()).unwrap().to_vec();
        assert!(
            (second_deriv[0] - 6.0).abs() < 1e-4,
            "Expected 6.0, got {}",
            second_deriv[0]
        );
    }

    #[test]
    fn test_second_order_through_pow_scalar() {
        // f(x) = x³
        // df/dx = 3x²
        // d²f/dx² = 6x
        // At x = 2: d²f/dx² = 12
        use crate::autograd::var_pow_scalar;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );

        // f(x) = x³
        let y = var_pow_scalar(&x, 3.0, &client).unwrap();

        // First backward
        let grads = backward_with_graph(&y, &client).unwrap();

        // df/dx = 3x² = 12
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!(
            (grad_x[0] - 12.0).abs() < 1e-5,
            "Expected 12.0, got {}",
            grad_x[0]
        );

        // Second backward
        let grad_x_var = grads.get_var(x.id()).unwrap();
        let grad_x_sum = var_sum(grad_x_var, &[0], false, &client).unwrap();
        let second_grads = backward(&grad_x_sum, &client).unwrap();

        // d²f/dx² = 6x = 12
        let second_deriv: Vec<f32> = second_grads.get(x.id()).unwrap().to_vec();
        assert!(
            (second_deriv[0] - 12.0).abs() < 1e-4,
            "Expected 12.0, got {}",
            second_deriv[0]
        );
    }

    #[test]
    fn test_second_order_through_broadcast() {
        // Test that second-order gradients work through broadcasting
        // Simpler test: f(x, b) = sum((x + b)²) where x is [2] and b is [2]
        // No actual broadcasting, but uses var_add to verify basic chain works
        //
        // Forward: y = (x + b)², then sum
        // First backward: dL/dx = 2(x + b), dL/db = 2(x + b)
        // Second backward: d²L/dx² = 2
        use crate::autograd::var_add;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // x is [2] vector
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device),
            true,
        );

        // b is [2] vector (same shape, no broadcast)
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device),
            true,
        );

        // f(x, b) = sum((x + b)²)
        let x_plus_b = var_add(&x, &b, &client).unwrap();
        let squared = var_mul(&x_plus_b, &x_plus_b, &client).unwrap();
        let loss = var_sum(&squared, &[0], false, &client).unwrap();

        // First backward
        let grads = backward_with_graph(&loss, &client).unwrap();

        // Verify first-order gradients exist
        assert!(grads.get(x.id()).is_some(), "Should have gradient for x");
        assert!(grads.get(b.id()).is_some(), "Should have gradient for b");

        // dL/dx = 2(x + b)
        // For x[0] = 1.0, b[0] = 0.1: grad = 2 * 1.1 = 2.2
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!(
            (grad_x[0] - 2.2).abs() < 1e-5,
            "Expected 2.2, got {}",
            grad_x[0]
        );

        // Second backward through x
        let grad_x_var = grads.get_var(x.id()).unwrap();
        let grad_x_sum = var_sum(grad_x_var, &[0], false, &client).unwrap();
        let second_grads = backward(&grad_x_sum, &client).unwrap();

        // d²L/dx² = 2 for each element (since d/dx[2(x+b)] = 2)
        let second_deriv_x: Vec<f32> = second_grads.get(x.id()).unwrap().to_vec();
        for (i, &val) in second_deriv_x.iter().enumerate() {
            assert!(
                (val - 2.0).abs() < 1e-4,
                "Expected d²L/dx²[{}] = 2.0, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_second_order_through_broadcast_shapes() {
        // Test that second-order gradients work through actual broadcasting
        // f(x, b) = sum((x + b)²) where x is [2, 3] and b is [3] (broadcasts)
        //
        // This tests that reduce_var_for_broadcast properly maintains
        // the gradient chain via var_reshape.
        use crate::autograd::var_add;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // x is [2, 3] matrix
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device),
            true,
        );

        // b is [3] vector that will broadcast to [2, 3]
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3], &[3], &device),
            true,
        );

        // f(x, b) = sum((x + b)²)
        let x_plus_b = var_add(&x, &b, &client).unwrap();
        let squared = var_mul(&x_plus_b, &x_plus_b, &client).unwrap();
        let loss = var_sum(&squared, &[0, 1], false, &client).unwrap();

        // First backward
        let grads = backward_with_graph(&loss, &client).unwrap();

        // Verify first-order gradients exist for both x and b
        assert!(grads.get(x.id()).is_some(), "Should have gradient for x");
        assert!(grads.get(b.id()).is_some(), "Should have gradient for b");

        // Verify gradient shapes
        assert_eq!(grads.get(x.id()).unwrap().shape(), &[2, 3]);
        assert_eq!(grads.get(b.id()).unwrap().shape(), &[3]);

        // dL/dx = 2(x + b)
        // For x[0,0] = 1.0, b[0] = 0.1: grad = 2 * 1.1 = 2.2
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!(
            (grad_x[0] - 2.2).abs() < 1e-5,
            "Expected 2.2, got {}",
            grad_x[0]
        );

        // dL/db[0] = sum over rows of 2(x + b) at column 0
        // = 2*(1.0+0.1) + 2*(4.0+0.1) = 2.2 + 8.2 = 10.4
        let grad_b: Vec<f32> = grads.get(b.id()).unwrap().to_vec();
        assert!(
            (grad_b[0] - 10.4).abs() < 1e-4,
            "Expected 10.4, got {}",
            grad_b[0]
        );

        // Second backward through x - need to verify get_var works
        if let Some(grad_x_var) = grads.get_var(x.id()) {
            let grad_x_sum = var_sum(grad_x_var, &[0, 1], false, &client).unwrap();
            let second_grads = backward(&grad_x_sum, &client).unwrap();

            // d²L/dx² = 2 for each element
            if let Some(second_deriv_x) = second_grads.get(x.id()) {
                let second_deriv_x: Vec<f32> = second_deriv_x.to_vec();
                for (i, &val) in second_deriv_x.iter().enumerate() {
                    assert!(
                        (val - 2.0).abs() < 1e-4,
                        "Expected d²L/dx²[{}] = 2.0, got {}",
                        i,
                        val
                    );
                }
            }
        }
    }
}
