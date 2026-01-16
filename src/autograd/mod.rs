//! Automatic differentiation (autograd)
//!
//! This module provides reverse-mode automatic differentiation for
//! computing gradients of tensor computations.
//!
//! # Status: NOT IMPLEMENTED
//!
//! **WARNING:** This module exposes the autograd API for design validation,
//! but the `backward()` function is not yet implemented and will return
//! `Error::NotImplemented`. Autograd implementation is planned for Phase 4.
//!
//! The types (`Var`, `GradFn`, `GradStore`) are usable for API exploration,
//! but gradient computation will not work until Phase 4.

mod backward;
mod grad_fn;
mod grad_store;
mod var;

pub mod ops;

pub use backward::backward;
pub use grad_fn::GradFn;
pub use grad_store::GradStore;
pub use var::Var;
