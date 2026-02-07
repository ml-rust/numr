//! Iterative solvers for sparse linear systems and eigenvalue problems
//!
//! # Available Linear Solvers
//!
//! - **CG** - Conjugate Gradient for symmetric positive definite systems
//! - **MINRES** - Minimum Residual for symmetric indefinite systems
//! - **CGS** - Conjugate Gradient Squared for non-symmetric systems
//! - **BiCGSTAB** - Bi-Conjugate Gradient Stabilized for non-symmetric systems
//! - **GMRES** - Generalized Minimal Residual for non-symmetric systems
//! - **Adaptive GMRES** - GMRES with automatic preconditioner upgrading
//!
//! # Available Eigensolvers
//!
//! - **Lanczos** - For large sparse symmetric matrices
//! - **Arnoldi (IRAM)** - For large sparse non-symmetric matrices
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
    AdaptiveGmresResult, AdaptivePreconditionerOptions, BiCgStabOptions, BiCgStabResult, CgOptions,
    CgResult, CgsOptions, CgsResult, ConvergenceReason, GmresDiagnostics, GmresOptions,
    GmresResult, MinresOptions, MinresResult, PreconditionerType, SparseEigComplexResult,
    SparseEigOptions, SparseEigResult, StagnationParams, WhichEigenvalues,
};

// Re-export traits
pub use traits::{IterativeSolvers, validate_iterative_inputs};

// Re-export implementations
pub use impl_generic::{
    adaptive_gmres_impl, arnoldi_eig_impl, bicgstab_impl, cg_impl, cgs_impl, gmres_impl,
    lanczos_eig_impl, minres_impl,
};
