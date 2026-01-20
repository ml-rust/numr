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
//! - `binary` - Binary element-wise operations (add, sub, mul, div, pow, max, min, logical_and/or/xor)
//! - `unary` - Unary element-wise operations (neg, abs, sqrt, exp, log, sign, isnan, isinf, logical_not, etc.)
//! - `scalar` - Tensor-scalar operations (add_scalar, mul_scalar, etc.)
//! - `reduce` - Reduction operations (sum, max, min)
//! - `compare` - Comparison operations (eq, ne, lt, le, gt, ge)
//! - `activation` - Activation functions (relu, sigmoid, softmax, silu, gelu)
//! - `norm` - Normalization operations (rms_norm, layer_norm)
//! - `cast` - Type casting operations
//! - `utility` - Utility operations (fill)
//! - `ternary` - Ternary operations (where)
//! - `sparse_spmv` - Sparse matrix operations (SpMV, SpMM)
//! - `sparse_merge` - Sparse matrix merge operations
//! - `sparse_convert` - Sparse format conversions (COO/CSR/CSC)
//! - `sparse_coo` - COO sparse element-wise operations with GPU sorting
//! - `scan` - Prefix sum operations
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
//! - `cast.cu` - Type casting operations
//! - `utility.cu` - Utility operations
//! - `ternary.cu` - Ternary operations
//! - `sparse_spmv.cu` - Sparse matrix operations
//! - `sparse_merge.cu` - Sparse matrix merge operations
//! - `sparse_convert.cu` - Sparse format conversions
//! - `sparse_coo.cu` - COO sparse element-wise operations with GPU sorting
//! - `scan.cu` - Prefix sum operations

mod activation;
mod binary;
mod cast;
mod compare;
mod cumulative;
mod fft;
mod index;
mod linalg;
mod loader;
mod norm;
mod reduce;
mod scalar;
#[cfg(feature = "sparse")]
mod scan;
mod shape;
#[cfg(feature = "sparse")]
mod sparse_convert;
#[cfg(feature = "sparse")]
mod sparse_coo;
#[cfg(feature = "sparse")]
mod sparse_merge;
#[cfg(feature = "sparse")]
mod sparse_spmv;
#[cfg(feature = "sparse")]
mod sparse_strategy;
#[cfg(feature = "sparse")]
mod sparse_utils;
#[cfg(feature = "sparse")]
mod spgemm;
mod strided_copy;
mod ternary;
mod unary;
mod utility;

pub use activation::*;
pub use binary::*;
pub use cast::*;
pub use compare::*;
pub use cumulative::*;
pub use fft::*;
pub use index::*;
pub use linalg::*;
pub use norm::*;
pub use reduce::*;
pub use scalar::*;
#[cfg(feature = "sparse")]
#[allow(unused_imports)]
pub use scan::*;
pub use shape::*;
#[cfg(feature = "sparse")]
pub use sparse_convert::*;
#[cfg(feature = "sparse")]
pub use sparse_coo::*;
#[cfg(feature = "sparse")]
pub use sparse_merge::*;
#[cfg(feature = "sparse")]
pub use sparse_spmv::*;
#[cfg(feature = "sparse")]
#[allow(unused_imports)]
// Sparse strategy types (AddMerge, SubMerge, etc.) used internally in sparse_merge
pub use sparse_strategy::*;
#[cfg(feature = "sparse")]
pub use sparse_utils::*;
#[cfg(feature = "sparse")]
pub use spgemm::*;
pub use strided_copy::*;
pub use ternary::*;
pub use unary::*;
#[allow(unused_imports)] // Prepared for future tensor creation optimization
pub use utility::*;

// Re-export commonly used items from loader for advanced users
#[allow(unused_imports)]
pub use loader::{
    BLOCK_SIZE, LaunchConfig, kernel_names, launch_matmul_batched_kernel,
    launch_matmul_bias_batched_kernel, launch_matmul_bias_kernel, launch_matmul_kernel,
};
