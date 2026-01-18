//! WGSL compute shader infrastructure for WebGPU operations
//!
//! This module provides native WGSL compute shaders for tensor operations.
//! All operations run entirely on the GPU without CPU fallback.
//!
//! # Module Structure
//!
//! - `pipeline` - Pipeline caching and dispatch utilities
//! - `elementwise` - Element-wise operation launchers (binary, unary, scalar, compare)
//! - `reduce` - Reduction operation launchers (sum, mean, max, min, softmax, argmax)
//! - `matmul` - Matrix multiplication launchers
//! - `norm` - Normalization operation launchers (rms_norm, layer_norm)
//! - `linalg` - Linear algebra kernel launchers
//! - `copy` - Copy operation shaders (strided to contiguous)

pub mod copy;
pub mod elementwise;
pub mod linalg;
pub mod matmul;
pub mod norm;
pub mod reduce;

mod elementwise_wgsl;
mod linalg_wgsl;
mod matmul_wgsl;
mod norm_wgsl;
mod pipeline;
mod reduce_wgsl;

pub use pipeline::{LayoutKey, PipelineCache, WORKGROUP_SIZE, workgroup_count};
