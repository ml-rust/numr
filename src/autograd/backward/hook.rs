use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};

/// Hook called during backward when a leaf variable's gradient is fully accumulated.
///
/// This enables overlapping gradient communication with backward computation
/// in distributed training scenarios (e.g., bucketed allreduce).
pub trait BackwardHook<R: Runtime>: Send {
    /// Called when a leaf variable's gradient is fully accumulated.
    ///
    /// At the point this is called, the gradient for `id` in the grad store
    /// is complete — all upstream contributions have been accumulated.
    fn on_leaf_grad_ready(&mut self, id: TensorId, grad: &Tensor<R>);
}

/// No-op backward hook for use when no hook behavior is needed.
pub struct NoOpHook;

impl<R: Runtime> BackwardHook<R> for NoOpHook {
    fn on_leaf_grad_ready(&mut self, _id: TensorId, _grad: &Tensor<R>) {}
}
