//! Sparse Linear Algebra Algorithm Contracts
//!
//! This module defines traits for sparse linear algebra algorithms that are
//! essential for iterative solvers. All backends must implement the same
//! algorithms to ensure numerical parity.
//!
//! # Algorithms
//!
//! - **ILU(0)**: Incomplete LU factorization with zero fill-in
//! - **ILU(k)**: Incomplete LU factorization with level-k fill-in
//! - **IC(0)**: Incomplete Cholesky factorization with zero fill-in
//! - **Sparse Triangular Solve**: Forward/backward substitution for sparse triangular matrices
//!
//! # Use Cases
//!
//! These algorithms are primarily used as **preconditioners** for iterative solvers:
//!
//! - **ILU(0)** → Preconditioner for GMRES, BiCGSTAB (non-symmetric systems)
//! - **ILU(k)** → Stronger preconditioner with more fill-in for difficult systems
//! - **IC(0)** → Preconditioner for CG (symmetric positive definite systems)
//! - **Sparse Triangular Solve** → Application of the preconditioner (L⁻¹ or U⁻¹)
//!
//! # Symbolic/Numeric Split
//!
//! For repeated solves with the same sparsity pattern (e.g., Newton iterations
//! in ODE solvers), the factorization can be split into:
//!
//! 1. **Symbolic phase**: Compute fill pattern based on sparsity structure
//! 2. **Numeric phase**: Compute actual values using precomputed pattern
//!
//! This gives ~10-50x speedup for repeated solves since the expensive symbolic
//! analysis is only done once.
//!
//! # Algorithm Guarantees
//!
//! All backends produce identical results (within floating-point tolerance):
//! - Same sparsity pattern in output
//! - Same numerical values (order of operations matches)
//! - Same handling of near-zero pivots

pub mod cpu;
pub mod levels;
pub mod traits;
pub mod types;

// Re-export types
pub use types::{
    IcDecomposition, IcOptions, IluDecomposition, IluFillLevel, IluMetrics, IluOptions,
    IlukDecomposition, IlukOptions, IlukSymbolic, SymbolicIlu0,
};

// Re-export trait and validation helpers
pub use traits::{SparseLinAlgAlgorithms, validate_square_sparse, validate_triangular_solve_dims};

// Re-export level scheduling
pub use levels::{
    LevelSchedule, compute_levels_ilu, compute_levels_lower, compute_levels_upper, flatten_levels,
};

// Re-export CPU implementations
pub use cpu::{
    ic0_cpu, ilu0_cpu, ilu0_numeric_cpu, ilu0_symbolic_cpu, iluk_cpu, iluk_numeric_cpu,
    iluk_symbolic_cpu, sparse_solve_triangular_cpu,
};
