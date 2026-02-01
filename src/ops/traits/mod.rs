//! Operation traits for tensor operations.
//!
//! This module contains trait definitions for various tensor operations.
//! Implementations are in the backend-specific modules (cpu/, cuda/, wgpu/).

mod activation;
mod compare;
mod complex;
mod conditional;
mod cumulative;
mod indexing;
mod kernel;
mod logical;
mod matmul;
mod normalization;
mod random;
mod reduce;
mod scalar;
mod sorting;
mod statistics;
mod type_conversion;
mod utility;

pub use activation::ActivationOps;
pub use compare::CompareOps;
pub use complex::ComplexOps;
pub use conditional::ConditionalOps;
pub use cumulative::CumulativeOps;
pub use indexing::IndexingOps;
pub use kernel::Kernel;
pub use logical::LogicalOps;
pub use matmul::MatmulOps;
pub use normalization::NormalizationOps;
pub use random::RandomOps;
pub use reduce::ReduceOps;
pub use scalar::ScalarOps;
pub use sorting::SortingOps;
pub use statistics::StatisticalOps;
pub use type_conversion::TypeConversionOps;
pub use utility::UtilityOps;

// Re-export TensorOps from parent (still in mod.rs for now)
pub use super::TensorOps;
