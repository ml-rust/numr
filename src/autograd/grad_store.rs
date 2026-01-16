//! Gradient storage and accumulation

use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};
use std::collections::HashMap;

/// Storage for gradients computed during backward pass
///
/// Gradients are stored by tensor ID and accumulated when a tensor
/// is used multiple times in the computation graph.
pub struct GradStore<R: Runtime> {
    grads: HashMap<TensorId, Tensor<R>>,
}

impl<R: Runtime> GradStore<R> {
    /// Create a new empty gradient store
    pub fn new() -> Self {
        Self {
            grads: HashMap::new(),
        }
    }

    /// Get the gradient for a tensor
    pub fn get(&self, id: TensorId) -> Option<&Tensor<R>> {
        self.grads.get(&id)
    }

    /// Insert a gradient (overwrites if exists)
    pub fn insert(&mut self, id: TensorId, grad: Tensor<R>) {
        self.grads.insert(id, grad);
    }

    /// Check if a gradient exists
    pub fn contains(&self, id: TensorId) -> bool {
        self.grads.contains_key(&id)
    }

    /// Remove and return a gradient
    pub fn remove(&mut self, id: TensorId) -> Option<Tensor<R>> {
        self.grads.remove(&id)
    }

    /// Get all tensor IDs with gradients
    pub fn keys(&self) -> impl Iterator<Item = &TensorId> {
        self.grads.keys()
    }

    /// Number of stored gradients
    pub fn len(&self) -> usize {
        self.grads.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }

    /// Clear all gradients
    pub fn clear(&mut self) {
        self.grads.clear();
    }

    /// Accumulate a gradient for a tensor
    ///
    /// If no gradient exists for this tensor, stores the gradient.
    /// If a gradient already exists, adds the new gradient to the existing one.
    ///
    /// # Arguments
    /// * `id` - The tensor ID to accumulate gradient for
    /// * `grad` - The gradient tensor to accumulate
    /// * `add_fn` - Function to add two tensors: `fn(existing, new) -> sum`
    ///
    /// This is used when a tensor is used multiple times in the computation graph,
    /// requiring its gradients to be summed according to the chain rule.
    pub fn accumulate<F>(&mut self, id: TensorId, grad: Tensor<R>, add_fn: F)
    where
        F: FnOnce(Tensor<R>, Tensor<R>) -> Tensor<R>,
    {
        if let Some(existing) = self.grads.remove(&id) {
            // Accumulate: existing + grad
            let accumulated = add_fn(existing, grad);
            self.grads.insert(id, accumulated);
        } else {
            // First gradient for this tensor
            self.grads.insert(id, grad);
        }
    }

    /// Accumulate a gradient, storing if none exists
    ///
    /// This is a simpler version that just overwrites if addition isn't available.
    /// Use `accumulate` with an add function for proper gradient accumulation.
    pub fn accumulate_or_insert(&mut self, id: TensorId, grad: Tensor<R>) {
        // TODO: Once tensor addition is implemented in TensorOps,
        // this should call accumulate() with the addition operation.
        // For now, we just insert (which overwrites).
        self.grads.insert(id, grad);
    }
}

impl<R: Runtime> Default for GradStore<R> {
    fn default() -> Self {
        Self::new()
    }
}
