//! WGSL compute shader infrastructure for WebGPU operations
//!
//! This module provides native WGSL compute shaders for tensor operations.
//! All operations run entirely on the GPU without CPU fallback.
//!
//! # Multi-DType Support
//!
//! Shaders are generated per-dtype using the `generator` module:
//! - F32, I32, U32 are always supported
//! - F16 requires WebGPU f16 extension
//!
//! # Module Structure
//!
//! - `generator` - WGSL shader source generation per dtype
//! - `pipeline` - Pipeline caching and dispatch utilities
//! - `typed_kernels` - TypedKernel<T> implementations for compile-time dtype enforcement
//! - `elementwise` - Legacy element-wise operation launchers (deprecated)
//! - `reduce` - Legacy reduction operation launchers (deprecated)
//! - `matmul` - Legacy matrix multiplication launchers (deprecated)
//! - `norm` - Legacy normalization operation launchers (deprecated)
//! - `linalg` - Linear algebra kernel launchers
//! - `copy` - Copy operation shaders (strided to contiguous)

pub mod complex;
pub mod copy;
pub mod cumulative;
pub mod dtype_support;
pub mod fft;
pub mod generator;
pub mod index;
pub mod linalg;
pub mod shape;
pub mod sort;
pub mod special;
pub mod typed_kernels;

// Legacy modules (to be replaced by typed_kernels)
pub mod elementwise;
pub mod matmul;
pub mod norm;
pub mod reduce;

mod elementwise_wgsl;
mod linalg_wgsl;
mod matmul_wgsl;
mod norm_wgsl;
mod pipeline;
mod reduce_wgsl;

pub use complex::{launch_angle_real, launch_complex_op};
pub use cumulative::{
    launch_cumprod, launch_cumprod_strided, launch_cumsum, launch_cumsum_strided, launch_logsumexp,
    launch_logsumexp_strided,
};
pub use generator::{
    dtype_suffix, generate_all_casts_from, generate_arange_shader, generate_binary_shader,
    generate_cast_shader, generate_cat_shader, generate_compare_shader, generate_cumprod_shader,
    generate_cumprod_strided_shader, generate_cumsum_shader, generate_cumsum_strided_shader,
    generate_eye_shader, generate_fill_shader, generate_gather_shader,
    generate_index_select_shader, generate_linspace_shader, generate_logsumexp_shader,
    generate_logsumexp_strided_shader, generate_masked_fill_shader, generate_masked_select_shader,
    generate_matmul_shader, generate_norm_shader, generate_reduce_shader, generate_scalar_shader,
    generate_scatter_shader, generate_unary_shader, is_wgpu_supported, is_wgsl_float, is_wgsl_int,
    wgsl_type,
};
pub use pipeline::{LayoutKey, PipelineCache, WORKGROUP_SIZE, workgroup_count};
pub use special::{launch_special_binary, launch_special_ternary, launch_special_unary};
