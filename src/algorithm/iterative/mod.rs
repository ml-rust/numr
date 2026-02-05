//! Iterative solvers for sparse linear systems
//!
//! This module provides Krylov subspace methods for solving Ax = b where A is
//! a sparse matrix. All methods support optional preconditioning to improve
//! convergence.
//!
//! # Available Solvers
//!
//! - **GMRES** - Generalized Minimal Residual method for non-symmetric systems
//! - **BiCGSTAB** - Bi-Conjugate Gradient Stabilized for non-symmetric systems
//! - **Adaptive GMRES** - GMRES with automatic preconditioner upgrading
//!
//! # Architecture
//!
//! Iterative solvers are **composite operations** that compose primitive ops:
//! - Vector norms, dot products (ReduceOps)
//! - Vector addition, subtraction, scaling (BinaryOps, ScalarOps)
//! - Sparse matrix-vector multiply (SparseOps)
//! - Triangular solves for preconditioning (SparseLinAlgAlgorithms)
//!
//! The generic implementations in `impl_generic/` define THE algorithm,
//! ensuring identical behavior across all backends.

mod helpers;
mod traits;
mod types;

pub mod impl_generic;

// Re-export types
pub use types::{
    AdaptiveGmresResult, AdaptivePreconditionerOptions, BiCgStabOptions, BiCgStabResult,
    ConvergenceReason, GmresDiagnostics, GmresOptions, GmresResult, PreconditionerType,
    StagnationParams,
};

// Re-export traits
pub use traits::{IterativeSolvers, validate_iterative_inputs};

// Re-export implementations
pub use impl_generic::{adaptive_gmres_impl, bicgstab_impl, gmres_impl};
