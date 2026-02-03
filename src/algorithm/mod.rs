//! Algorithm contracts for runtime backends
//!
//! This module defines trait-based contracts that ensure all backends implement
//! the same mathematical algorithms, guaranteeing numerical parity across
//! CPU, CUDA, WebGPU, ROCm, and other compute devices.
//!
//! # Trait-First Architecture
//!
//! Every algorithm is defined as a trait FIRST, then implemented per backend:
//!
//! 1. **Define trait here** - Specifies exact algorithm with pseudocode/description
//! 2. **Implement for each backend** - CPU, CUDA, WebGPU follow same algorithm
//! 3. **Compile-time enforcement** - Missing implementations cause errors
//!
//! # Available Algorithm Contracts
//!
//! - [`MatmulAlgorithm`] - Tiled GEMM with register blocking for dense matmul
//! - [`LinearAlgebraAlgorithms`] - LU, Cholesky, QR, solve, inverse, det, trace
//! - [`FftAlgorithms`] - FFT, IFFT, RFFT using Stockham autosort algorithm
//! - [`SparseAlgorithms`] - SpGEMM, SpMV, DSMM (requires `sparse` feature)
//!
//! # Signal Processing
//!
//! Signal processing operations (convolution, STFT, window functions) are provided
//! by the **solvr** library, which builds on numr's FFT primitives.
//!
//! # Universal Algorithm Design
//!
//! Algorithms are designed to be portable across all backends:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                  Algorithm Trait                         │
//! │  (defines: tile sizes, register blocking, pseudocode)   │
//! └──────────────────────┬──────────────────────────────────┘
//!                        │
//!     ┌──────────────────┼──────────────────┐
//!     ▼                  ▼                  ▼
//! ┌─────────┐      ┌──────────┐      ┌───────────┐
//! │   CPU   │      │   CUDA   │      │   WebGPU  │
//! │ (SIMD)  │      │ (shared  │      │ (workgrp  │
//! │         │      │  memory) │      │  memory)  │
//! └─────────┘      └──────────┘      └───────────┘
//! ```

pub mod fft;
pub mod linalg;
pub mod matmul;
pub mod special;

#[cfg(feature = "sparse")]
pub mod sparse;

#[cfg(feature = "sparse")]
pub mod sparse_linalg;

#[cfg(feature = "sparse")]
pub mod sparse_linalg_cpu;

#[cfg(feature = "sparse")]
pub mod sparse_linalg_levels;

pub use linalg::{
    CholeskyDecomposition, EigenDecomposition, GeneralEigenDecomposition, LinearAlgebraAlgorithms,
    LuDecomposition, MatrixFunctionsAlgorithms, MatrixNormOrder, QrDecomposition,
    SchurDecomposition, SvdDecomposition, machine_epsilon, validate_linalg_dtype,
    validate_matrix_2d, validate_square_matrix,
};

pub use matmul::{MatmulAlgorithm, TileConfig};

pub use special::{
    EULER_MASCHERONI, LANCZOS_COEFFICIENTS, LANCZOS_G, LN_SQRT_2PI, SQRT_PI, SpecialFunctions,
    TWO_OVER_SQRT_PI, validate_special_dtype,
};

#[cfg(feature = "sparse")]
pub use sparse::{
    SparseAlgorithms, validate_dsmm_shapes, validate_dtype_match, validate_spgemm_shapes,
    zero_tolerance,
};

#[cfg(feature = "sparse")]
pub use sparse_linalg::{
    IcDecomposition, IcOptions, IluDecomposition, IluOptions, SparseLinAlgAlgorithms,
    validate_square_sparse, validate_triangular_solve_dims,
};

#[cfg(feature = "sparse")]
pub use sparse_linalg_cpu::{ic0_cpu, ilu0_cpu, sparse_solve_triangular_cpu};

#[cfg(feature = "sparse")]
pub use sparse_linalg_levels::{
    LevelSchedule, compute_levels_ilu, compute_levels_lower, compute_levels_upper, flatten_levels,
};
