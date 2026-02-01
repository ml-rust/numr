//! Operation traits for tensor operations.
//!
//! This module contains trait definitions for various tensor operations.
//! Implementations are in the backend-specific modules (cpu/, cuda/, wgpu/).

mod compare;
mod kernel;
mod logical;
mod scalar;

pub use compare::CompareOps;
pub use kernel::Kernel;
pub use logical::LogicalOps;
pub use scalar::ScalarOps;

// Re-export TensorOps from parent (still in mod.rs for now)
pub use super::TensorOps;
