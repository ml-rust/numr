//! Variable: tensor with gradient tracking

use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};
use super::GradFn;
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

    /// Create from an operation result with a gradient function
    pub fn from_op(tensor: Tensor<R>, grad_fn: Arc<dyn GradFn<R>>) -> Self {
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
