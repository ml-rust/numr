//! Backward pass implementation
//!
//! Implements reverse-mode automatic differentiation using topological sort
//! to traverse the computation graph and accumulate gradients.
//!
//! # First-Order vs Second-Order Differentiation
//!
//! This module provides two backward functions:
//!
//! - [`backward`]: Standard first-order differentiation. Returns raw tensors.
//!   Efficient for training neural networks.
//!
//! - [`backward_with_graph`]: Second-order capable differentiation. Returns
//!   `Var`s that retain their computation history, enabling Hessians and HVPs.

mod driver;
mod first_order;
mod graph;
mod hook;
mod loss;
mod needed;
#[cfg(test)]
mod test_support;
mod topo;

pub use first_order::{backward, backward_with_hooks, backward_wrt, backward_wrt_with_hooks};
pub use graph::backward_with_graph;
pub use hook::{BackwardHook, NoOpHook};
