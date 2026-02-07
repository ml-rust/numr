//! Iterative solvers for sparse linear systems and eigenvalue problems
//!
//! # Available Linear Solvers
//!
//! - **CG** - Conjugate Gradient for symmetric positive definite systems
//! - **MINRES** - Minimum Residual for symmetric indefinite systems
//! - **CGS** - Conjugate Gradient Squared for non-symmetric systems
//! - **BiCGSTAB** - Bi-Conjugate Gradient Stabilized for non-symmetric systems
//! - **GMRES** - Generalized Minimal Residual for non-symmetric systems
//! - **LGMRES** - Loose GMRES with augmented restarts
//! - **QMR** - Quasi-Minimal Residual for non-symmetric systems
//! - **Jacobi** - Weighted Jacobi iteration for diagonally dominant systems
//! - **SOR** - Successive Over-Relaxation
//! - **Adaptive GMRES** - GMRES with automatic preconditioner upgrading
//!
//! # Available Eigensolvers / SVD
//!
//! - **Lanczos** - For large sparse symmetric matrices
//! - **Arnoldi (IRAM)** - For large sparse non-symmetric matrices
//! - **Sparse SVD** - Lanczos bidiagonalization for singular values
//!
//! # Preconditioners
//!
//! - **ILU(0)** - Incomplete LU with zero fill-in
//! - **IC(0)** - Incomplete Cholesky with zero fill-in
//! - **AMG** - Algebraic Multigrid (V-cycle with Jacobi smoothing)
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
    AdaptiveGmresResult, AdaptivePreconditionerOptions, AmgCycleType, AmgHierarchy, AmgOptions,
    BiCgStabOptions, BiCgStabResult, CgOptions, CgResult, CgsOptions, CgsResult, ConvergenceReason,
    GmresDiagnostics, GmresOptions, GmresResult, JacobiOptions, JacobiResult, LgmresOptions,
    LgmresResult, MinresOptions, MinresResult, PreconditionerType, QmrOptions, QmrResult,
    SorOptions, SorResult, SparseEigComplexResult, SparseEigOptions, SparseEigResult,
    SparseSvdResult, StagnationParams, SvdsOptions, WhichEigenvalues, WhichSingularValues,
};

// Re-export traits
pub use traits::{IterativeSolvers, validate_iterative_inputs};

// Re-export implementations
pub use impl_generic::{
    adaptive_gmres_impl, amg_preconditioned_cg, amg_setup, amg_vcycle, arnoldi_eig_impl,
    bicgstab_impl, cg_impl, cgs_impl, gmres_impl, jacobi_impl, lanczos_eig_impl, lgmres_impl,
    minres_impl, qmr_impl, sor_impl, svds_impl,
};
