use super::hook::BackwardHook;
use super::loss::{create_loss_gradient, validate_loss};
use super::needed::{Wanted, compute_needed};
use super::topo::topological_sort;
use crate::autograd::grad_mode::EnableGradGuard;
use crate::autograd::{GradStore, Var};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::TensorOps;
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::TensorId;

/// Single reverse-mode traversal shared by every first-order backward entry point.
///
/// Pruning is a reachability pre-pass from `wanted`, not a `requires_grad` edge
/// check: a node is skipped only when no wanted id lies anywhere below it. That
/// is closed under contribution — if node `n` is needed and `n` is an input of
/// consumer `c`, then `needed[c]` holds by construction — so no accumulation term
/// that reaches a wanted id is ever dropped, and the accumulation order of every
/// retained node is untouched.
pub(super) fn backward_driver<R, C, H>(
    loss: &Var<R>,
    wanted: &Wanted,
    client: &C,
    hooks: &mut H,
    fn_name: &str,
) -> Result<GradStore<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    H: BackwardHook<R>,
{
    validate_loss(loss, fn_name)?;

    // A `GradFn` can build `Var`s, as the checkpoint recompute does. Those must
    // record even when the caller runs backward inside `no_grad`.
    let _grad_mode = EnableGradGuard::new();

    // Initialize gradient store with dL/dL = 1
    let mut grad_store = GradStore::new();
    grad_store.insert(loss.id(), create_loss_gradient(loss)?);

    // Build the computation graph and get topological order
    let topo_order = topological_sort(loss);

    // Reachability pre-pass: which nodes lie on a path to a wanted gradient
    let needed = compute_needed(&topo_order, wanted);
    let is_needed = |id: TensorId| match &needed {
        None => id.is_tracked(),
        Some(set) => set.contains(&id),
    };

    // Traverse in reverse topological order (from output to inputs)
    for var_entry in topo_order.into_iter().rev() {
        let (var_id, grad_fn_opt, input_ids) = var_entry;

        // Nothing wanted lies below this node — skip before any tensor work runs
        if !is_needed(var_id) {
            continue;
        }

        // Get gradient for this node
        let grad_output = match grad_store.get(var_id) {
            Some(g) => g.clone(),
            None => continue, // No gradient flowing to this node
        };

        // If this node has a grad_fn, compute gradients for its inputs
        if let Some(grad_fn) = grad_fn_opt {
            // Per-slot mask, one entry per input, in `inputs()` order. This is
            // the same predicate the accumulation loop below applies, handed to
            // the op up front so it can skip producing a gradient nobody reads.
            // Under `Wanted::All` every entry is true and the op takes exactly
            // the branches it took before the mask existed.
            let input_needed: Vec<bool> = input_ids.iter().map(|id| is_needed(*id)).collect();

            // Compute gradients for inputs
            let input_grads = grad_fn.backward(&grad_output, &input_needed)?;

            // Accumulate gradients for each input
            for (input_id, input_grad_opt) in input_ids.iter().zip(input_grads) {
                if !is_needed(*input_id) {
                    continue;
                }
                if let Some(input_grad) = input_grad_opt {
                    // Accumulate gradient using tensor addition
                    grad_store.try_accumulate(*input_id, input_grad, |existing, new| {
                        client.add(&existing, &new)
                    })?;
                }
            }
        } else {
            // Leaf node (no grad_fn) with a gradient — notify hook
            hooks.on_leaf_grad_ready(var_id, &grad_output);
        }
    }

    Ok(grad_store)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::bits;
    use super::*;
    use crate::autograd::{
        GradFn, backward, backward_wrt, var_add, var_matmul, var_mul, var_sum, var_transpose,
    };
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// GradFn that records how many times it is actually executed.
    struct CountingBackward<R: Runtime> {
        input_ids: Vec<TensorId>,
        input_grad_fns: Vec<Option<Arc<dyn GradFn<R>>>>,
        calls: Arc<AtomicUsize>,
    }

    impl<R: Runtime> GradFn<R> for CountingBackward<R> {
        fn backward(
            &self,
            grad_output: &Tensor<R>,
            needed: &[bool],
        ) -> Result<Vec<Option<Tensor<R>>>> {
            assert_eq!(
                needed.len(),
                self.input_ids.len(),
                "driver must pass one mask entry per input"
            );
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![Some(grad_output.clone()); self.input_ids.len()])
        }

        fn inputs(&self) -> &[TensorId] {
            &self.input_ids
        }

        fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
            self.input_grad_fns.clone()
        }

        fn name(&self) -> &'static str {
            "CountingBackward"
        }
    }

    /// GradFn that panics if the driver asks it to compute a slot the test
    /// knows nothing wants.
    ///
    /// This is what turns "the mask reached the op" into an observable failure:
    /// if the driver ever handed a true entry for `forbidden`, the op would be
    /// obliged to produce that gradient, and the panic fires instead.
    struct PanicIfNeeded<R: Runtime> {
        input_ids: Vec<TensorId>,
        input_grad_fns: Vec<Option<Arc<dyn GradFn<R>>>>,
        forbidden: usize,
    }

    impl<R: Runtime> GradFn<R> for PanicIfNeeded<R> {
        fn backward(
            &self,
            grad_output: &Tensor<R>,
            needed: &[bool],
        ) -> Result<Vec<Option<Tensor<R>>>> {
            assert_eq!(
                needed.len(),
                self.input_ids.len(),
                "driver must pass one mask entry per input"
            );
            assert!(
                !needed[self.forbidden],
                "driver asked for slot {} that no wanted id depends on",
                self.forbidden
            );
            Ok(needed
                .iter()
                .map(|&want| want.then(|| grad_output.clone()))
                .collect())
        }

        fn inputs(&self) -> &[TensorId] {
            &self.input_ids
        }

        fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
            self.input_grad_fns.clone()
        }

        fn name(&self) -> &'static str {
            "PanicIfNeeded"
        }
    }

    /// GradFn that delegates to `inner` and records every mask it is handed.
    struct MaskSpy<R: Runtime> {
        inner: Arc<dyn GradFn<R>>,
        seen: Arc<std::sync::Mutex<Vec<Vec<bool>>>>,
    }

    impl<R: Runtime> GradFn<R> for MaskSpy<R> {
        fn backward(
            &self,
            grad_output: &Tensor<R>,
            needed: &[bool],
        ) -> Result<Vec<Option<Tensor<R>>>> {
            self.seen
                .lock()
                .expect("mask spy lock")
                .push(needed.to_vec());
            self.inner.backward(grad_output, needed)
        }

        fn inputs(&self) -> &[TensorId] {
            self.inner.inputs()
        }

        fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
            self.inner.input_grad_fns()
        }

        fn name(&self) -> &'static str {
            "MaskSpy"
        }
    }

    /// The driver hands every node a mask exactly as long as its input list.
    #[test]
    fn test_driver_mask_length_matches_inputs() {
        // CountingBackward and PanicIfNeeded both assert the length on entry;
        // this drives a graph mixing unary, binary and reduce nodes through the
        // real driver so those assertions actually run.
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.5f32, -2.25], &[2], &device).unwrap(),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.5f32, 4.0], &[2], &device).unwrap(),
            true,
        );

        let calls = Arc::new(AtomicUsize::new(0));
        let counted = Var::from_op(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device).unwrap(),
            Arc::new(CountingBackward::<CpuRuntime> {
                input_ids: vec![x.id(), y.id()],
                input_grad_fns: vec![None, None],
                calls: calls.clone(),
            }),
        );

        let h = var_mul(&counted, &y, &client).unwrap();
        let loss = var_sum(&h, &[0], false, &client).unwrap();

        let _ = backward(&loss, &client).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        let _ = backward_wrt(&loss, &[x.id()], &client).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// The driver delivers `false` for a slot nothing wants.
    ///
    /// The node itself IS needed — `x` lies below it — so it is not skipped
    /// wholesale. Only its second slot is dead, which is exactly the
    /// frozen-weight shape this mask exists for.
    #[test]
    fn test_driver_masks_dead_slot_of_a_needed_node() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap(),
            true,
        );
        let frozen = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[7.0f32], &[1], &device).unwrap(),
            false,
        );

        let node = Var::from_op(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap(),
            Arc::new(PanicIfNeeded::<CpuRuntime> {
                input_ids: vec![x.id(), frozen.id()],
                input_grad_fns: vec![None, None],
                forbidden: 1,
            }),
        );
        let loss = var_sum(&node, &[0], false, &client).unwrap();

        let grads = backward_wrt(&loss, &[x.id()], &client).unwrap();

        assert!(grads.contains(x.id()), "wanted gradient still produced");
        assert!(
            !grads.contains(frozen.id()),
            "dead slot produced no gradient"
        );
    }

    /// Frozen `Linear`, end to end through the real driver.
    ///
    /// `var_transpose(&w)` mints a fresh `TensorId` every step, so the matmul's
    /// second operand is an id nothing can ever look up. The spy shows the
    /// driver marks that slot false, and `MatmulBackward` returns `None` there
    /// — the `A^T @ dL/dC` GEMM never runs. `x` still gets the right gradient.
    #[test]
    fn test_frozen_linear_skips_weight_gradient_end_to_end() {
        use crate::autograd::ops::MatmulBackward;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Trainable activation
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device).unwrap(),
            true,
        );
        // Frozen weight
        let w = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0, 5.0, 6.0], &[2, 2], &device).unwrap(),
            false,
        );
        let w_t = var_transpose(&w).unwrap();

        // Same node `var_matmul` would build, wrapped so the mask is visible.
        let inner: Arc<dyn GradFn<CpuRuntime>> = Arc::new(MatmulBackward::<CpuRuntime>::new(
            x.id(),
            w_t.id(),
            x.tensor().clone(),
            w_t.tensor().clone(),
            x.grad_fn().cloned(),
            w_t.grad_fn().cloned(),
        ));
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        // Reuse `var_matmul` purely for the forward value; the graph edge comes
        // from the spy-wrapped node below.
        let y_tensor = var_matmul(&x, &w_t, &client).unwrap().tensor().clone();
        let y = Var::from_op(
            y_tensor,
            Arc::new(MaskSpy::<CpuRuntime> {
                inner,
                seen: seen.clone(),
            }),
        );
        let loss = var_sum(&y, &[0, 1], false, &client).unwrap();

        let grads = backward_wrt(&loss, &[x.id()], &client).unwrap();

        let masks = seen.lock().expect("mask spy lock");
        assert_eq!(masks.len(), 1, "the matmul node runs exactly once");
        assert_eq!(
            masks[0],
            vec![true, false],
            "activation wanted, transposed frozen weight dead"
        );

        assert!(
            !grads.contains(w_t.id()),
            "no gradient stored for the throwaway transposed id"
        );
        assert!(!grads.contains(w.id()), "frozen weight gets no gradient");

        // w = [[3,4],[5,6]], w_t = [[3,5],[4,6]], dL/dy = ones[1,2].
        // dL/dx = dL/dy @ w_t^T = ones[1,2] @ w = column sums of w = [8, 10].
        let grad_x: Vec<f32> = grads
            .get(x.id())
            .expect("gradient for x")
            .contiguous()
            .unwrap()
            .to_vec();
        assert_eq!(grad_x, vec![8.0f32, 10.0]);
    }

    /// The all-true path is untouched: a branching multi-input graph containing
    /// a guarded matmul gives bit-identical gradients whether the driver builds
    /// an all-true mask (`backward`) or derives one from a full wanted set.
    #[test]
    fn test_all_true_mask_is_bit_identical_through_guarded_ops() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.5f32, -2.25, 0.75, 3.0], &[2, 2], &device)
                .unwrap(),
            true,
        );
        let w = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.5f32, 1.25, -2.0, 3.5], &[2, 2], &device).unwrap(),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.1f32, -0.4, 2.25, 0.125], &[2, 2], &device)
                .unwrap(),
            true,
        );

        // `h` branches into two consumers, so its gradient accumulates two terms.
        let build = || {
            let h = var_matmul(&x, &w, &client).unwrap();
            let p = var_mul(&h, &b, &client).unwrap();
            let q = var_matmul(&h, &b, &client).unwrap();
            let s = var_add(&p, &q, &client).unwrap();
            var_sum(&s, &[0, 1], false, &client).unwrap()
        };

        let full = backward(&build(), &client).unwrap();
        let all_ids = [x.id(), w.id(), b.id()];
        let pruned = backward_wrt(&build(), &all_ids, &client).unwrap();

        for id in all_ids {
            let want = full.get(id).expect("baseline grad");
            let got = pruned.get(id).expect("pruned grad");
            assert_eq!(bits(want), bits(got), "gradient differs for {id:?}");
        }
    }
    #[test]
    fn test_backward_wrt_skips_execution_not_just_storage() {
        // A pruned node's GradFn must never run. Storing-then-discarding would
        // still pay for the tensor work; this asserts the work is not done.
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let build = |calls: &Arc<AtomicUsize>| {
            let trainable = Var::new(
                Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap(),
                true,
            );
            let frozen = Var::new(
                Tensor::<CpuRuntime>::from_slice(&[7.0f32], &[1], &device).unwrap(),
                false,
            );
            let counted = Var::from_op(
                Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap(),
                Arc::new(CountingBackward::<CpuRuntime> {
                    input_ids: vec![frozen.id()],
                    input_grad_fns: vec![None],
                    calls: calls.clone(),
                }),
            );
            let loss = var_add(&trainable, &counted, &client).unwrap();
            (trainable.id(), loss)
        };

        let full_calls = Arc::new(AtomicUsize::new(0));
        let (_, loss_full) = build(&full_calls);
        let _ = backward(&loss_full, &client).unwrap();
        assert_eq!(
            full_calls.load(Ordering::SeqCst),
            1,
            "backward executes the frozen-only node"
        );

        let pruned_calls = Arc::new(AtomicUsize::new(0));
        let (trainable_id, loss_pruned) = build(&pruned_calls);
        let _ = backward_wrt(&loss_pruned, &[trainable_id], &client).unwrap();
        assert_eq!(
            pruned_calls.load(Ordering::SeqCst),
            0,
            "backward_wrt must not execute a node with no wanted id below it"
        );
    }
}
