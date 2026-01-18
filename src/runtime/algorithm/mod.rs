//! Algorithm contracts for runtime backends
//!
//! This module defines trait-based contracts that ensure all backends implement
//! the same mathematical algorithms, guaranteeing numerical parity across
//! CPU, CUDA, WebGPU, and other compute devices.
//!
//! # Trait-First Architecture
//!
//! Every algorithm is defined as a trait FIRST, then implemented per backend:
//!
//! 1. **Define trait here** - Specifies exact algorithm with pseudocode
//! 2. **Implement for each backend** - CPU, CUDA, WebGPU follow same algorithm
//! 3. **Compile-time enforcement** - Missing implementations cause errors
//!
//! # Available Algorithm Contracts
//!
//! - [`LinearAlgebraAlgorithms`] - LU, Cholesky, QR, solve, inverse, det, trace
//! - [`SparseAlgorithms`] - SpGEMM, SpMV, DSMM (requires `sparse` feature)

pub mod linalg;

#[cfg(feature = "sparse")]
pub mod sparse;

pub use linalg::{
    CholeskyDecomposition, LinearAlgebraAlgorithms, LuDecomposition, QrDecomposition,
    SvdDecomposition, machine_epsilon, validate_linalg_dtype, validate_matrix_2d,
    validate_square_matrix,
};

#[cfg(feature = "sparse")]
pub use sparse::{
    SparseAlgorithms, validate_dsmm_shapes, validate_dtype_match, validate_spgemm_shapes,
    zero_tolerance,
};
