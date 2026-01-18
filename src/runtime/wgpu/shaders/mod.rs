//! WGSL compute shader infrastructure for WebGPU operations
//!
//! This module provides native WGSL compute shaders for tensor operations.
//! All operations run entirely on the GPU without CPU fallback.
//!
//! # Module Structure
//!
//! - `pipeline` - Pipeline caching and dispatch utilities
//! - `linalg` - Linear algebra kernel launchers
//! - `linalg_wgsl` - WGSL shader source code
//! - `copy` - Copy operation shaders (strided to contiguous)

pub mod copy;
pub mod linalg;
mod linalg_wgsl;
mod pipeline;

pub use pipeline::{LayoutKey, PipelineCache, WORKGROUP_SIZE, workgroup_count};
