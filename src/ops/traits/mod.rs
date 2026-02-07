//! Operation traits for tensor operations.
//!
//! This module contains trait definitions for various tensor operations.
//! Implementations are in the backend-specific modules (cpu/, cuda/, wgpu/).

mod activation;
mod advanced_random;
mod binary;
mod compare;
mod complex;
mod conditional;
mod conv;
mod cumulative;
mod distance;
mod indexing;
mod kernel;
mod linalg;
mod logical;
mod matmul;
pub mod multivariate;
mod normalization;
mod quasirandom;
mod random;
mod reduce;
mod scalar;
mod shape;
mod sorting;
mod statistics;
mod tensor_ops;
mod type_conversion;
mod unary;
mod utility;

pub use activation::ActivationOps;
pub use advanced_random::AdvancedRandomOps;
pub use binary::BinaryOps;
pub use compare::CompareOps;
pub use complex::ComplexOps;
pub use conditional::ConditionalOps;
pub use conv::{ConvOps, PaddingMode};
pub use cumulative::CumulativeOps;
pub use distance::{DistanceMetric, DistanceOps};
pub use indexing::{IndexingOps, ScatterReduceOp};
pub use kernel::Kernel;
pub use linalg::LinalgOps;
pub use logical::LogicalOps;
pub use matmul::MatmulOps;
pub use multivariate::MultivariateRandomOps;
pub use normalization::NormalizationOps;
pub use quasirandom::QuasiRandomOps;
pub use random::RandomOps;
pub use reduce::ReduceOps;
pub use scalar::ScalarOps;
pub use shape::ShapeOps;
pub use sorting::SortingOps;
pub use statistics::StatisticalOps;
pub use tensor_ops::TensorOps;
pub use type_conversion::TypeConversionOps;
pub use unary::UnaryOps;
pub use utility::{MeshgridIndexing, UtilityOps};
