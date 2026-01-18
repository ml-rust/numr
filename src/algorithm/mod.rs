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
//! - [`SparseAlgorithms`] - SpGEMM, SpMV, DSMM (requires `sparse` feature)
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

pub mod linalg;
pub mod matmul;

#[cfg(feature = "sparse")]
pub mod sparse;

pub use linalg::{
    CholeskyDecomposition, LinearAlgebraAlgorithms, LuDecomposition, QrDecomposition,
    SvdDecomposition, machine_epsilon, validate_linalg_dtype, validate_matrix_2d,
    validate_square_matrix,
};

pub use matmul::{MatmulAlgorithm, TileConfig};

#[cfg(feature = "sparse")]
pub use sparse::{
    SparseAlgorithms, validate_dsmm_shapes, validate_dtype_match, validate_spgemm_shapes,
    zero_tolerance,
};
