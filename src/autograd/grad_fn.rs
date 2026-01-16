//! Gradient function trait

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};

/// Trait for computing gradients during backward pass
///
/// Each operation that participates in autograd has an associated
/// `GradFn` that knows how to compute gradients for its inputs.
pub trait GradFn<R: Runtime>: Send + Sync {
    /// Compute gradients for input tensors given the gradient of the output
    ///
    /// Returns a vector of optional gradients - one per input.
    /// `None` indicates that input doesn't need a gradient.
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>>;

    /// Get the IDs of input tensors
    ///
    /// Used for topological sorting during backward pass.
    fn inputs(&self) -> &[TensorId];

    /// Get tensors saved during forward pass
    ///
    /// Some operations (like softmax) need forward outputs for backward.
    fn saved_tensors(&self) -> &[Tensor<R>] {
        &[]
    }

    /// Human-readable name for debugging
    fn name(&self) -> &'static str;
}
