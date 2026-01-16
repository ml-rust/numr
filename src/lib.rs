//! # numr
//!
//! A high-performance numerical computing library for Rust.
//!
//! `numr` provides n-dimensional arrays (tensors) with:
//! - **Multiple backends**: CPU (with SIMD), CUDA, WebGPU
//! - **Automatic differentiation**: Reverse-mode autodiff for optimization
//! - **Zero-copy views**: Efficient slicing, transposing, reshaping
//! - **Type-safe operations**: Compile-time backend selection, runtime dtype
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use numr::prelude::*;
//!
//! // Create a tensor
//! let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
//! let b = Tensor::<CpuRuntime>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
//!
//! // Operations
//! let c = &a + &b;
//! let d = a.matmul(&b);
//! ```
//!
//! ## Feature Flags
//!
//! - `cpu` (default): CPU backend with SIMD optimizations
//! - `cuda`: NVIDIA CUDA backend
//! - `wgpu`: Cross-platform GPU via WebGPU
//! - `rayon` (default): Parallel CPU operations
//! - `f16`: Half-precision float support

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_inception)]

pub mod dtype;
pub mod error;
pub mod tensor;
pub mod runtime;
pub mod ops;
pub mod autograd;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::dtype::DType;
    pub use crate::tensor::{Tensor, Layout};
    pub use crate::runtime::{Runtime, Device, RuntimeClient};
    pub use crate::error::{Error, Result};

    #[cfg(feature = "cpu")]
    pub use crate::runtime::cpu::CpuRuntime;

    #[cfg(feature = "cuda")]
    pub use crate::runtime::cuda::CudaRuntime;

    #[cfg(feature = "wgpu")]
    pub use crate::runtime::wgpu::WgpuRuntime;
}

/// Default runtime based on enabled features
///
/// - With `cuda` feature: `CudaRuntime`
/// - With `wgpu` feature (no cuda): `WgpuRuntime`
/// - Otherwise: `CpuRuntime`
#[cfg(feature = "cuda")]
pub type DefaultRuntime = runtime::cuda::CudaRuntime;

/// Default runtime based on enabled features
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type DefaultRuntime = runtime::wgpu::WgpuRuntime;

/// Default runtime based on enabled features
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
pub type DefaultRuntime = runtime::cpu::CpuRuntime;
