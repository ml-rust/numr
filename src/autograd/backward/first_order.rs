use super::driver::backward_driver;
use super::hook::{BackwardHook, NoOpHook};
use super::needed::Wanted;
use crate::autograd::{GradStore, Var};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::TensorOps;
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::TensorId;

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
/// let x = Var::new(Tensor::from_slice(&[2.0f32], &[1], &device)?, true);
/// let y = Var::new(Tensor::from_slice(&[3.0f32], &[1], &device)?, true);
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
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    backward_with_hooks(loss, client, &mut NoOpHook)
}

/// Compute gradients only for the tensors in `wanted`, and for the nodes on a
/// path to them.
///
/// Identical to [`backward`] on every id in `wanted`, bit for bit: the reverse
/// topological order and therefore the float accumulation order of each retained
/// node is unchanged. The difference is what is NOT done — a node whose input
/// cone contains no wanted id has its `GradFn::backward()` skipped entirely, so
/// neither the intermediate tensor nor the work that produces it is ever
/// materialized.
///
/// Pruning also reaches inside a node that IS needed. Each `GradFn::backward()`
/// call receives a per-input `needed` mask, so an op with independently priced
/// gradients skips the ones nothing reads. A frozen `Linear` is the case that
/// matters: its matmul node is needed for the activation gradient, yet the
/// weight gradient `A^T @ dL/dC` is dead, and `MatmulBackward` skips
/// that second matmul.
///
/// This matters for partial-finetuning graphs (LoRA, frozen backbones). There,
/// most graph nodes are only reachable from frozen weights, yet a plain
/// [`backward`] still computes and stores a full-size gradient for each of them
/// under an id nothing can read back.
///
/// # Arguments
///
/// * `loss` - The scalar loss tensor to differentiate
/// * `wanted` - Tensor ids whose gradients are required (typically the trainable
///   parameter ids)
/// * `client` - The runtime client for tensor operations
///
/// # Returns
///
/// A `GradStore<R>` holding a gradient for every wanted id that receives one,
/// plus the intermediate nodes on a path to them.
pub fn backward_wrt<R, C>(loss: &Var<R>, wanted: &[TensorId], client: &C) -> Result<GradStore<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    backward_wrt_with_hooks(loss, wanted, client, &mut NoOpHook)
}

/// Compute gradients only for `wanted`, with leaf-ready hooks.
///
/// Combines [`backward_wrt`] and [`backward_with_hooks`]. The hook fires for a
/// leaf only when that leaf is in `wanted` — a pruned leaf produces no gradient,
/// so there is nothing to report.
///
/// # Arguments
///
/// * `loss` - The scalar loss tensor to differentiate
/// * `wanted` - Tensor ids whose gradients are required
/// * `client` - The runtime client for tensor operations
/// * `hooks` - Hook implementation called when each wanted leaf gradient is ready
pub fn backward_wrt_with_hooks<R, C, H>(
    loss: &Var<R>,
    wanted: &[TensorId],
    client: &C,
    hooks: &mut H,
) -> Result<GradStore<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    H: BackwardHook<R>,
{
    let wanted = Wanted::Set(
        wanted
            .iter()
            .copied()
            .filter(|id| id.is_tracked())
            .collect(),
    );
    backward_driver(loss, &wanted, client, hooks, "backward_wrt_with_hooks")
}

/// Compute gradients with hooks that fire when leaf gradients are ready.
///
/// Identical to [`backward`], but calls `hooks.on_leaf_grad_ready(id, grad)`
/// after a leaf variable's gradient is fully accumulated. This enables
/// overlapping gradient communication with backward computation (e.g.,
/// bucketed allreduce in distributed training).
///
/// A leaf variable is one with no `grad_fn` (i.e., a model parameter or
/// input created with `requires_grad = true`). By the time the hook fires,
/// all upstream contributions to that leaf's gradient have been accumulated.
///
/// # Arguments
///
/// * `loss` - The scalar loss tensor to differentiate
/// * `client` - The runtime client for tensor operations
/// * `hooks` - Hook implementation called when each leaf gradient is ready
///
/// # Returns
///
/// A `GradStore` containing gradients for all tensors in the graph.
pub fn backward_with_hooks<R, C, H>(
    loss: &Var<R>,
    client: &C,
    hooks: &mut H,
) -> Result<GradStore<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    H: BackwardHook<R>,
{
    // `backward()` delegates here, so the validation error text has always named
    // `backward_with_hooks`. Keep it that way.
    backward_driver(loss, &Wanted::All, client, hooks, "backward_with_hooks")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{var_mul, var_sum};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    use std::cell::RefCell;
    use std::rc::Rc;

    /// Test hook that records leaf gradient notifications
    struct RecordingHook {
        leaf_ids: Rc<RefCell<Vec<TensorId>>>,
    }

    impl RecordingHook {
        fn new() -> (Self, Rc<RefCell<Vec<TensorId>>>) {
            let ids = Rc::new(RefCell::new(Vec::new()));
            (
                Self {
                    leaf_ids: ids.clone(),
                },
                ids,
            )
        }
    }

    // RecordingHook is not Send (due to Rc), so we wrap for single-threaded tests
    unsafe impl Send for RecordingHook {}

    impl BackwardHook<CpuRuntime> for RecordingHook {
        fn on_leaf_grad_ready(&mut self, id: TensorId, _grad: &Tensor<CpuRuntime>) {
            self.leaf_ids.borrow_mut().push(id);
        }
    }

    #[test]
    fn test_backward_with_hooks_matches_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap(),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device).unwrap(),
            true,
        );

        // z = x * y
        let z1 = var_mul(&x, &y, &client).unwrap();
        let z2 = var_mul(&x, &y, &client).unwrap();

        let grads1 = backward(&z1, &client).unwrap();

        let (mut hook, leaf_ids) = RecordingHook::new();
        let grads2 = backward_with_hooks(&z2, &client, &mut hook).unwrap();

        // Gradients match
        let gx1: Vec<f32> = grads1.get(x.id()).unwrap().to_vec();
        let gx2: Vec<f32> = grads2.get(x.id()).unwrap().to_vec();
        assert!((gx1[0] - gx2[0]).abs() < 1e-6);

        let gy1: Vec<f32> = grads1.get(y.id()).unwrap().to_vec();
        let gy2: Vec<f32> = grads2.get(y.id()).unwrap().to_vec();
        assert!((gy1[0] - gy2[0]).abs() < 1e-6);

        // The hook runs for both leaf variables
        let ids = leaf_ids.borrow();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&x.id()));
        assert!(ids.contains(&y.id()));
    }

    #[test]
    fn test_backward_with_hooks_no_hook_for_non_leaf() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device).unwrap(),
            true,
        );

        // y = sum(x * x) — intermediate x*x is NOT a leaf
        let x_sq = var_mul(&x, &x, &client).unwrap();
        let loss = var_sum(&x_sq, &[0], false, &client).unwrap();

        let (mut hook, leaf_ids) = RecordingHook::new();
        let _grads = backward_with_hooks(&loss, &client, &mut hook).unwrap();

        // Only x is a leaf, not x_sq or loss
        let ids = leaf_ids.borrow();
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&x.id()));
    }

    #[test]
    fn test_backward_still_returns_non_parameter_gradients() {
        // backward() keeps its old meaning: a gradient for every graph node,
        // including intermediates nothing will ever step.
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device).unwrap(),
            true,
        );
        let frozen = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0], &[2], &device).unwrap(),
            false,
        );

        let h = var_mul(&x, &frozen, &client).unwrap();
        let loss = var_sum(&h, &[0], false, &client).unwrap();

        let grads = backward(&loss, &client).unwrap();

        assert!(grads.contains(loss.id()), "loss seed dL/dL");
        assert!(grads.contains(h.id()), "intermediate node");
        assert!(!grads.contains(frozen.id()), "frozen operand gets no entry");
        assert!(grads.contains(x.id()), "parameter");
    }
    #[test]
    fn test_backward_requires_scalar() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // A non-scalar tensor is an error
        let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let var = Var::new(tensor, true);

        let result = backward(&var, &client);
        assert!(result.is_err());
    }

    #[test]
    fn test_backward_leaf_variable() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Simple scalar leaf variable
        let tensor = Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[1], &device).unwrap();
        let var = Var::new(tensor, true);

        let grads = backward(&var, &client).unwrap();

        // The gradient of the loss w.r.t. itself is 1
        let grad = grads.get(var.id()).unwrap();
        let grad_data: Vec<f32> = grad.to_vec();
        assert_eq!(grad_data, vec![1.0f32]);
    }
}
