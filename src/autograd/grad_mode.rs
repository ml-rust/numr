//! Thread-local grad mode: controls whether ops record a graph.
//!
//! Grad mode is enabled by default on every thread. While it is disabled,
//! [`Var::from_op`](super::Var::from_op) and
//! [`Var::with_id_and_grad_fn`](super::Var::with_id_and_grad_fn) return a
//! `Var` with `requires_grad == false` and no `grad_fn`. The `GradFn` is
//! dropped, so the tensors it saved are freed at once.
//!
//! Leaf constructors ignore grad mode. A parameter created inside
//! [`no_grad`] keeps the `requires_grad` flag its caller passed.
//!
//! The mode is per thread. A guard on one thread never changes another thread.
//!
//! # Example
//!
//! ```
//! # use numr::prelude::*;
//! # use numr::autograd::{Var, no_grad, var_mul};
//! # let device = CpuDevice::new();
//! # let client = CpuRuntime::default_client(&device);
//! let x = Var::new(Tensor::from_slice(&[2.0f32], &[1], &device)?, true);
//! let y = no_grad(|| var_mul(&x, &x, &client))?;
//! assert!(!y.requires_grad());
//! assert!(y.grad_fn().is_none());
//! # Ok::<(), numr::error::Error>(())
//! ```

use std::cell::Cell;
use std::marker::PhantomData;

thread_local! {
    /// Count of active [`NoGradGuard`]s since the last [`EnableGradGuard`].
    /// Zero means grad mode is enabled.
    static NO_GRAD_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Read the current depth. A thread in teardown reads as depth zero.
fn depth() -> usize {
    NO_GRAD_DEPTH.try_with(Cell::get).unwrap_or(0)
}

/// Write the depth. A thread in teardown ignores the write.
fn set_depth(value: usize) {
    let _ = NO_GRAD_DEPTH.try_with(|d| d.set(value));
}

/// Return `true` when ops on this thread record a graph.
#[inline]
pub fn is_grad_enabled() -> bool {
    depth() == 0
}

/// RAII guard that disables graph recording on this thread.
///
/// Drop restores the mode that was active at creation. Guards must drop in
/// reverse creation order, which scoped bindings give.
///
/// The guard is `!Send`. Grad mode is thread-local, so the guard must drop on
/// the thread that created it.
#[must_use = "grad mode is enabled again as soon as the guard drops"]
pub struct NoGradGuard {
    prev: usize,
    _not_send: PhantomData<*const ()>,
}

impl NoGradGuard {
    /// Disable graph recording until the guard drops.
    pub fn new() -> Self {
        let prev = depth();
        set_depth(prev.saturating_add(1));
        Self {
            prev,
            _not_send: PhantomData,
        }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_depth(self.prev);
    }
}

/// RAII guard that enables graph recording on this thread.
///
/// It overrides every enclosing [`NoGradGuard`] until it drops. Drop restores
/// the mode that was active at creation. The guard is `!Send`.
#[must_use = "the enclosing grad mode returns as soon as the guard drops"]
pub struct EnableGradGuard {
    prev: usize,
    _not_send: PhantomData<*const ()>,
}

impl EnableGradGuard {
    /// Enable graph recording until the guard drops.
    pub fn new() -> Self {
        let prev = depth();
        set_depth(0);
        Self {
            prev,
            _not_send: PhantomData,
        }
    }
}

impl Default for EnableGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for EnableGradGuard {
    fn drop(&mut self) {
        set_depth(self.prev);
    }
}

/// Run `f` with graph recording disabled on this thread.
pub fn no_grad<T>(f: impl FnOnce() -> T) -> T {
    let _guard = NoGradGuard::new();
    f()
}

/// Run `f` with graph recording enabled on this thread.
pub fn enable_grad<T>(f: impl FnOnce() -> T) -> T {
    let _guard = EnableGradGuard::new();
    f()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{Var, backward, backward_with_graph, var_mul, var_sum};
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    fn leaf(values: &[f64], requires_grad: bool) -> Var<CpuRuntime> {
        let device = CpuDevice::new();
        let t = Tensor::<CpuRuntime>::from_slice(values, &[values.len()], &device).unwrap();
        Var::new(t, requires_grad)
    }

    #[test]
    fn ops_inside_no_grad_have_no_grad_fn() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = leaf(&[2.0, 3.0], true);

        let y = no_grad(|| var_mul(&x, &x, &client)).unwrap();
        assert!(!y.requires_grad());
        assert!(y.grad_fn().is_none());

        let z = var_mul(&x, &x, &client).unwrap();
        assert!(z.requires_grad());
        assert!(z.grad_fn().is_some());
    }

    #[test]
    fn leaf_requires_grad_is_unchanged() {
        let x = leaf(&[1.0], true);
        let (inside, frozen) = no_grad(|| (leaf(&[1.0], true), leaf(&[1.0], false)));
        assert!(x.requires_grad());
        assert!(
            inside.requires_grad(),
            "a leaf keeps the flag its caller passed"
        );
        assert!(!frozen.requires_grad());
    }

    #[test]
    fn guards_nest_and_restore() {
        assert!(is_grad_enabled());
        {
            let _outer = NoGradGuard::new();
            assert!(!is_grad_enabled());
            {
                let _inner = NoGradGuard::new();
                assert!(!is_grad_enabled());
            }
            assert!(!is_grad_enabled(), "inner drop must keep the outer guard");
        }
        assert!(is_grad_enabled());
    }

    #[test]
    fn enable_grad_inside_no_grad_records() {
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = leaf(&[2.0], true);

        no_grad(|| {
            let recorded = enable_grad(|| {
                assert!(is_grad_enabled());
                let inner = no_grad(|| var_mul(&x, &x, &client)).unwrap();
                assert!(inner.grad_fn().is_none());
                var_mul(&x, &x, &client)
            })
            .unwrap();
            assert!(recorded.grad_fn().is_some());
            assert!(!is_grad_enabled(), "enable_grad exit restores no_grad");

            let skipped = var_mul(&x, &x, &client).unwrap();
            assert!(skipped.grad_fn().is_none());
        });
        assert!(is_grad_enabled());
    }

    #[test]
    fn guard_on_one_thread_does_not_affect_another() {
        let _guard = NoGradGuard::new();
        assert!(!is_grad_enabled());

        let other = std::thread::spawn(|| {
            let client = CpuRuntime::default_client(&CpuDevice::new());
            let x = leaf(&[2.0], true);
            let y = var_mul(&x, &x, &client).unwrap();
            (is_grad_enabled(), y.grad_fn().is_some())
        })
        .join()
        .unwrap();

        assert_eq!(other, (true, true));
        assert!(!is_grad_enabled());
    }

    #[test]
    fn backward_with_graph_records_inside_no_grad() {
        // y = x^3: dy/dx = 3x^2 = 12 and d2y/dx2 = 6x = 12 at x = 2.
        let client = CpuRuntime::default_client(&CpuDevice::new());
        let x = leaf(&[2.0], true);
        let sq = var_mul(&x, &x, &client).unwrap();
        let y = var_mul(&sq, &x, &client).unwrap();

        let gx = no_grad(|| {
            let grads = backward_with_graph(&y, &client).unwrap();
            grads.get_var(x.id()).unwrap().alias()
        });
        assert!(
            gx.requires_grad(),
            "first-order grad must stay differentiable"
        );
        assert!(gx.grad_fn().is_some());

        let s = var_sum(&gx, &[], false, &client).unwrap();
        let second = backward(&s, &client).unwrap();
        let h: Vec<f64> = second.get(x.id()).unwrap().to_vec();
        assert!((h[0] - 12.0).abs() < 1e-12, "expected 12, got {}", h[0]);
    }
}
