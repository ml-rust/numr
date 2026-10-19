//! Variable: tensor with gradient tracking

use super::GradFn;
use super::grad_mode::is_grad_enabled;
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// A tensor that tracks gradients for automatic differentiation
///
/// `Var` wraps a `Tensor` and optionally records how it was created
/// (via `grad_fn`), enabling reverse-mode autodiff.
///
/// # Lazy Autograd
///
/// During inference, `grad_fn` is `None` and there's zero overhead.
/// Gradient tracking only occurs during training when needed.
pub struct Var<R: Runtime> {
    /// The underlying tensor data
    tensor: Tensor<R>,

    /// Unique identifier for graph tracking
    id: TensorId,

    /// Whether this variable requires gradient computation
    requires_grad: bool,

    /// Function to compute gradients (None for leaf tensors)
    grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> Var<R> {
    /// Create a leaf variable (no gradient function)
    pub fn new(tensor: Tensor<R>, requires_grad: bool) -> Self {
        Self {
            id: tensor.id(),
            tensor,
            requires_grad,
            grad_fn: None,
        }
    }

    /// Create a leaf variable with a specific ID
    ///
    /// This is used for second-order differentiation where we need to
    /// preserve the original input IDs so gradients can be accumulated
    /// correctly across multiple backward passes.
    pub fn with_id(tensor: Tensor<R>, id: TensorId, requires_grad: bool) -> Self {
        Self {
            id,
            tensor,
            requires_grad,
            grad_fn: None,
        }
    }

    /// Create a variable with a specific ID and gradient function
    ///
    /// This is essential for second-order differentiation. When computing
    /// gradients of gradients, we need to preserve both:
    /// 1. The original input ID (so second-order gradients accumulate correctly)
    /// 2. The grad_fn chain (so the backward traversal can continue through
    ///    the original computation graph)
    ///
    /// Without the grad_fn, second-order backward would stop at saved tensors
    /// instead of continuing to the original inputs.
    ///
    /// While grad mode is disabled (see [`no_grad`](super::no_grad)), the
    /// result has `requires_grad == false` and no `grad_fn`.
    pub fn with_id_and_grad_fn(
        tensor: Tensor<R>,
        id: TensorId,
        grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        if !is_grad_enabled() {
            return Self {
                id,
                tensor,
                requires_grad: false,
                grad_fn: None,
            };
        }
        Self {
            id,
            tensor,
            requires_grad: true,
            grad_fn,
        }
    }

    /// Create from an operation result with a gradient function
    ///
    /// While grad mode is disabled (see [`no_grad`](super::no_grad)), the
    /// result has `requires_grad == false` and no `grad_fn`. The dropped
    /// `grad_fn` frees the tensors it saved.
    pub fn from_op(tensor: Tensor<R>, grad_fn: Arc<dyn GradFn<R>>) -> Self {
        if !is_grad_enabled() {
            return Self {
                id: TensorId::new(),
                tensor,
                requires_grad: false,
                grad_fn: None,
            };
        }
        Self {
            id: TensorId::new(),
            tensor,
            requires_grad: true,
            grad_fn: Some(grad_fn),
        }
    }

    /// Get the tensor ID
    #[inline]
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// The id an op records for this Var as one of its inputs.
    ///
    /// A Var with `requires_grad == false` is always a leaf. It records
    /// [`TensorId::UNTRACKED`], so no backward pass gives it a gradient entry.
    /// Gradients still flow to every other input of the same op.
    #[inline]
    pub fn graph_id(&self) -> TensorId {
        if self.requires_grad {
            self.id
        } else {
            TensorId::UNTRACKED
        }
    }

    /// Access the underlying tensor
    #[inline]
    pub fn tensor(&self) -> &Tensor<R> {
        &self.tensor
    }

    /// Check if this variable requires gradients
    #[inline]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Get the gradient function (if any)
    #[inline]
    pub fn grad_fn(&self) -> Option<&Arc<dyn GradFn<R>>> {
        self.grad_fn.as_ref()
    }

    /// Identity alias: same tensor, same [`TensorId`], same `requires_grad`,
    /// same `grad_fn`.
    ///
    /// Use this — NOT [`Clone::clone`] — whenever a layer returns its input
    /// unchanged (e.g. `Dropout` in eval mode, a disabled residual scale).
    /// `clone` mints a FRESH `TensorId`, which silently breaks two things:
    ///
    /// 1. If the input is a leaf parameter, `backward` records its gradient
    ///    under the clone's new id, so `GradStore::get(param.id())` returns
    ///    `None` and the parameter never updates.
    /// 2. Optimizer state (AdamW) is keyed by `TensorId`, so a drifting id
    ///    orphans the moment estimates.
    pub fn alias(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            id: self.id,
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn.clone(),
        }
    }

    /// Detach from the computation graph
    ///
    /// Returns a new variable that doesn't track gradients.
    pub fn detach(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            id: TensorId::new(),
            requires_grad: false,
            grad_fn: None,
        }
    }

    /// Set requires_grad flag
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
        if !requires_grad {
            self.grad_fn = None;
        }
    }

    // Delegate common methods to tensor

    /// Get the shape
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    /// Get the number of elements
    #[inline]
    pub fn numel(&self) -> usize {
        self.tensor.numel()
    }

    /// Get the number of dimensions
    #[inline]
    pub fn ndim(&self) -> usize {
        self.tensor.ndim()
    }
}

impl<R: Runtime> Clone for Var<R> {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            id: TensorId::new(),
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn.clone(),
        }
    }
}

impl<R: Runtime> std::fmt::Debug for Var<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Var")
            .field("id", &self.id)
            .field("shape", &self.tensor.shape())
            .field("requires_grad", &self.requires_grad)
            .field("has_grad_fn", &self.grad_fn.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::CpuRuntime;

    /// `alias` preserves the id (so gradients and TensorId-keyed optimizer
    /// state still find the variable); `clone` deliberately does not.
    #[test]
    fn alias_preserves_id_but_clone_does_not() {
        let device = <CpuRuntime as Runtime>::default_device();
        let t = Tensor::<CpuRuntime>::zeros(&[2, 3], DType::F32, &device).unwrap();
        let v = Var::new(t, true);

        let aliased = v.alias();
        assert_eq!(aliased.id(), v.id(), "alias must keep the same TensorId");
        assert_eq!(aliased.requires_grad(), v.requires_grad());
        assert_eq!(aliased.shape(), v.shape());

        let cloned = v.clone();
        assert_ne!(
            cloned.id(),
            v.id(),
            "clone is documented to mint a fresh TensorId — if this ever \
             changes, the `alias` call sites can be simplified"
        );
    }

    /// A no-op layer path must not strip `requires_grad == false`.
    #[test]
    fn alias_preserves_requires_grad_false() {
        let device = <CpuRuntime as Runtime>::default_device();
        let t = Tensor::<CpuRuntime>::zeros(&[4], DType::F32, &device).unwrap();
        let v = Var::new(t, false);
        let aliased = v.alias();
        assert!(!aliased.requires_grad());
        assert_eq!(aliased.id(), v.id());
    }

    #[test]
    fn graph_id_is_untracked_only_for_frozen_vars() {
        let device = <CpuRuntime as Runtime>::default_device();
        let t = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device).unwrap();
        let trainable = Var::new(t.clone(), true);
        let frozen = Var::new(t, false);
        assert_eq!(trainable.graph_id(), trainable.id());
        assert_eq!(frozen.graph_id(), TensorId::UNTRACKED);
        assert!(!TensorId::UNTRACKED.is_tracked());
        assert!(frozen.id().is_tracked());
    }

    mod frozen_leaves {
        use super::super::Var;
        use crate::autograd::{
            backward, backward_with_graph, backward_wrt, var_add, var_matmul, var_mul, var_sum,
        };
        use crate::runtime::Runtime;
        use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
        use crate::tensor::{Tensor, TensorId};

        const X: [f32; 6] = [0.5, -1.25, 2.0, 1.5, -0.75, 3.0];
        const BIAS: [f32; 3] = [0.25, -0.5, 1.0];
        const SCALE: [f32; 6] = [1.5, 0.75, -2.0, 0.5, 1.25, -1.0];
        const W: [f32; 6] = [2.0, -1.0, 0.5, 1.5, -0.25, 3.0];

        fn leaf(device: &CpuDevice, data: &[f32], shape: &[usize], rg: bool) -> Var<CpuRuntime> {
            Var::new(
                Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap(),
                rg,
            )
        }

        /// `loss = sum(((x + bias) * scale) @ w)`, with `bias` [3] broadcast over [2, 3].
        ///
        /// `x`, `bias` and `scale` meet the graph through add and mul, as in a GDN
        /// block. `w` meets it through matmul directly, with no transpose between.
        /// Bit `i` of `trainable` sets `requires_grad` on leaf `i`.
        fn build(client: &CpuClient, trainable: u32) -> (Vec<Var<CpuRuntime>>, Var<CpuRuntime>) {
            let device = CpuDevice::new();
            let rg = |i: u32| trainable & (1 << i) != 0;
            let leaves = vec![
                leaf(&device, &X, &[2, 3], rg(0)),
                leaf(&device, &BIAS, &[3], rg(1)),
                leaf(&device, &SCALE, &[2, 3], rg(2)),
                leaf(&device, &W, &[3, 2], rg(3)),
            ];
            let h = var_add(&leaves[0], &leaves[1], client).unwrap();
            let h = var_mul(&h, &leaves[2], client).unwrap();
            let y = var_matmul(&h, &leaves[3], client).unwrap();
            let loss = var_sum(&y, &[0, 1], false, client).unwrap();
            (leaves, loss)
        }

        fn values(t: &Tensor<CpuRuntime>) -> Vec<f32> {
            t.contiguous().unwrap().to_vec::<f32>()
        }

        /// Frozen leaves get no entry. Trainable leaves match the all-trainable run.
        fn check(
            leaves: &[Var<CpuRuntime>],
            reference: &[Var<CpuRuntime>],
            get: impl Fn(TensorId) -> Option<Vec<f32>>,
            ref_get: impl Fn(TensorId) -> Option<Vec<f32>>,
            mode: &str,
        ) {
            assert!(
                get(TensorId::UNTRACKED).is_none(),
                "{mode}: untracked entry"
            );
            for (i, (v, r)) in leaves.iter().zip(reference).enumerate() {
                if v.requires_grad() {
                    let want = ref_get(r.id()).expect("reference gradient");
                    assert_eq!(get(v.id()), Some(want), "{mode}: leaf {i} gradient");
                } else {
                    assert!(
                        get(v.id()).is_none(),
                        "{mode}: frozen leaf {i} has an entry"
                    );
                }
            }
        }

        #[test]
        fn frozen_leaves_get_no_gradient_in_any_mode() {
            let device = CpuDevice::new();
            let client = CpuRuntime::default_client(&device);

            let (reference, ref_loss) = build(&client, 0b1111);
            let ref_grads = backward(&ref_loss, &client).unwrap();
            let ref_get = |id| ref_grads.get(id).map(values);
            // The graph-keeping pass runs var ops, so it has its own reference.
            let ref_graph = backward_with_graph(&ref_loss, &client).unwrap();
            let ref_graph_get = |id| ref_graph.get(id).map(values);

            // Every mix with at least one trainable leaf.
            for trainable in 1..0b1111 {
                let (leaves, loss) = build(&client, trainable);
                let ids: Vec<TensorId> = leaves.iter().map(|v| v.id()).collect();

                let grads = backward(&loss, &client).unwrap();
                check(
                    &leaves,
                    &reference,
                    |id| grads.get(id).map(values),
                    ref_get,
                    "backward",
                );

                // Asking for a frozen id gives it nothing either.
                let wrt = backward_wrt(&loss, &ids, &client).unwrap();
                check(
                    &leaves,
                    &reference,
                    |id| wrt.get(id).map(values),
                    ref_get,
                    "wrt",
                );

                let graph = backward_with_graph(&loss, &client).unwrap();
                let graph_get = |id| graph.get(id).map(values);
                check(&leaves, &reference, graph_get, ref_graph_get, "with_graph");
            }
        }

        #[test]
        fn frozen_leaf_gets_no_second_order_gradient() {
            let device = CpuDevice::new();
            let client = CpuRuntime::default_client(&device);

            // f = sum(x * x * c), c frozen. df/dx = 2xc, d(sum(df/dx))/dx = 2c.
            let x = leaf(&device, &[1.0, 2.0], &[2], true);
            let c = leaf(&device, &[3.0, 4.0], &[2], false);
            let xx = var_mul(&x, &x, &client).unwrap();
            let f = var_sum(&var_mul(&xx, &c, &client).unwrap(), &[0], false, &client).unwrap();

            let first = backward_with_graph(&f, &client).unwrap();
            assert!(
                !first.contains(c.id()),
                "first order: frozen leaf has an entry"
            );
            assert!(
                !first.contains(TensorId::UNTRACKED),
                "first order: untracked entry"
            );
            let grad_x = first.get_var(x.id()).expect("first-order gradient for x");
            assert_eq!(values(grad_x.tensor()), vec![6.0, 16.0]);

            let g = var_sum(grad_x, &[0], false, &client).unwrap();
            let second = backward(&g, &client).unwrap();
            assert!(
                !second.contains(c.id()),
                "second order: frozen leaf has an entry"
            );
            assert!(
                !second.contains(TensorId::UNTRACKED),
                "second order: untracked entry"
            );
            let hx = second.get(x.id()).expect("second-order gradient for x");
            assert_eq!(values(hx), vec![6.0, 8.0]);
        }
    }
}
