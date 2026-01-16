//! Backward pass implementation
//!
//! **STATUS: NOT IMPLEMENTED**
//!
//! This module is a placeholder for the autograd backward pass.
//! The `backward()` function is exposed to allow API design validation,
//! but calling it will return an error until Phase 4 implementation.

use super::{GradStore, Var};
use crate::error::{Error, Result};
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
///
/// # Errors
///
/// Currently returns `Error::NotImplemented` as autograd is not yet implemented.
/// This is planned for Phase 4 of the numr roadmap.
///
/// # Example
///
/// ```ignore
/// // Will return Error::NotImplemented until Phase 4
/// let grads = backward(&loss)?;
/// ```
pub fn backward<R: Runtime>(loss: &Var<R>) -> Result<GradStore<R>> {
    // Autograd backward pass is not yet implemented (Phase 4)
    // Return an explicit error rather than silently returning empty gradients
    let _ = loss;
    Err(Error::NotImplemented {
        feature: "autograd backward pass",
    })
}

// TODO: Implement topological sort helper
// fn topological_sort<R: Runtime>(root: &Var<R>) -> Vec<...> { ... }
