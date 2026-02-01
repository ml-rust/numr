//! Operation traits for tensor operations.
//!
//! This module contains trait definitions for various tensor operations.
//! Implementations are in the backend-specific modules (cpu/, cuda/, wgpu/).

mod compare;
mod complex;
mod conditional;
mod kernel;
mod logical;
mod matmul;
mod normalization;
mod scalar;
mod type_conversion;

pub use compare::CompareOps;
pub use complex::ComplexOps;
pub use conditional::ConditionalOps;
pub use kernel::Kernel;
pub use logical::LogicalOps;
pub use matmul::MatmulOps;
pub use normalization::NormalizationOps;
pub use scalar::ScalarOps;
pub use type_conversion::TypeConversionOps;

// Re-export TensorOps from parent (still in mod.rs for now)
pub use super::TensorOps;
