//! # numr
//!
//! **High-performance numerical computing for Rust with multi-backend GPU acceleration.**
//!
//! numr provides n-dimensional arrays (tensors), linear algebra, FFT, and automatic
//! differentiation - with the same API across CPU, CUDA, and WebGPU backends.
//!
//! ## Why numr?
//!
//! - **Multi-backend**: Same code runs on CPU, CUDA, and WebGPU
//! - **No vendor lock-in**: Native kernels, not cuBLAS/MKL wrappers
//! - **Pure Rust**: No Python runtime, no FFI overhead, single binary deployment
//! - **Autograd included**: Reverse-mode automatic differentiation built-in
//! - **Sparse tensors**: CSR, CSC, COO formats with GPU support
//!
//! ## Features
//!
//! - **Tensors**: N-dimensional arrays with broadcasting, slicing, views
//! - **Linear algebra**: Matmul, LU, QR, SVD, Cholesky, eigendecomposition
//! - **FFT**: Fast Fourier transforms (1D, 2D, ND)
//! - **Element-wise ops**: Full set of math functions
//! - **Reductions**: Sum, mean, max, min, argmax, argmin along axes
//! - **Multiple dtypes**: f64, f32, f16, bf16, fp8, integers, bool
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use numr::prelude::*;
//!
//! let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
//! let b = Tensor::<CpuRuntime>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2])?;
//!
//! let c = &a + &b;
//! let d = a.matmul(&b)?;
//! ```
//!
//! ## Feature Flags
//!
//! - `cpu` (default): CPU backend
//! - `cuda`: NVIDIA CUDA backend
//! - `wgpu`: Cross-platform GPU via WebGPU
//! - `rayon` (default): Multi-threaded CPU operations
//! - `f16`: Half-precision floats (F16, BF16)
//! - `fp8`: 8-bit floats (FP8E4M3, FP8E5M2)
//! - `sparse`: Sparse tensor formats (CSR, CSC, COO)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_inception)]

pub mod algorithm;
pub mod autograd;
pub mod dtype;
pub mod error;
pub mod ops;
pub mod runtime;
#[cfg(feature = "sparse")]
pub mod sparse;
pub mod tensor;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::dtype::DType;
    pub use crate::error::{Error, Result};
    pub use crate::runtime::{Device, Runtime, RuntimeClient};
    pub use crate::tensor::{Layout, Tensor};

    #[cfg(feature = "cpu")]
    pub use crate::runtime::cpu::CpuRuntime;

    #[cfg(feature = "cuda")]
    pub use crate::runtime::cuda::CudaRuntime;

    #[cfg(feature = "wgpu")]
    pub use crate::runtime::wgpu::WgpuRuntime;

    #[cfg(feature = "sparse")]
    pub use crate::sparse::{SparseFormat, SparseOps, SparseTensor};
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
