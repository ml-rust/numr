//! CUDA kernel implementations for tensor operations
//!
//! This module provides native CUDA kernels for all tensor operations,
//! eliminating the need for CPU fallback in most cases.
//!
//! # Architecture
//!
//! Kernels are written in CUDA C++ (.cu files) and compiled to PTX by build.rs.
//! The PTX is loaded at runtime and cached per-device for efficient reuse.
//!
//! # Module Organization
//!
//! - `loader` - Kernel loading, caching, and generic launch infrastructure
//! - `binary` - Binary element-wise operations (add, sub, mul, div, pow, max, min)
//! - `unary` - Unary element-wise operations (neg, abs, sqrt, exp, log, etc.)
//! - `scalar` - Tensor-scalar operations (add_scalar, mul_scalar, etc.)
//! - `reduce` - Reduction operations (sum, max, min)
//! - `compare` - Comparison operations (eq, ne, lt, le, gt, ge)
//! - `activation` - Activation functions (relu, sigmoid, softmax, silu, gelu)
//! - `norm` - Normalization operations (rms_norm, layer_norm)
//!
//! # Kernel Files
//!
//! - `binary.cu` - Binary element-wise operations
//! - `unary.cu` - Unary element-wise operations
//! - `scalar.cu` - Tensor-scalar operations
//! - `reduce.cu` - Reduction operations
//! - `compare.cu` - Comparison operations
//! - `activation.cu` - Activation functions
//! - `norm.cu` - Normalization operations

mod activation;
mod binary;
mod compare;
mod loader;
mod norm;
mod reduce;
mod scalar;
mod unary;

pub use activation::*;
pub use binary::*;
pub use compare::*;
pub use norm::*;
pub use reduce::*;
pub use scalar::*;
pub use unary::*;

// Re-export commonly used items from loader for advanced users
#[allow(unused_imports)]
pub use loader::{BLOCK_SIZE, LaunchConfig, kernel_names};
