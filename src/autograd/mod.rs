//! Automatic differentiation (autograd)
//!
//! This module provides reverse-mode automatic differentiation for
//! computing gradients of tensor computations.

mod var;
mod grad_fn;
mod backward;
mod grad_store;

pub mod ops;

pub use var::Var;
pub use grad_fn::GradFn;
pub use backward::backward;
pub use grad_store::GradStore;
