//! Sparse Linear Algebra Algorithm Contracts
//!
//! This module defines traits for sparse linear algebra algorithms that are
//! essential for iterative solvers. All backends must implement the same
//! algorithms to ensure numerical parity.
//!
//! # Algorithms
//!
//! ## Preprocessing (CPU-only, structure-based)
//!
//! - **COLAMD**: Column Approximate Minimum Degree ordering to minimize fill-in
//! - **Maximum Transversal**: Hopcroft-Karp algorithm for row permutation
//!
//! ## Factorization (multi-backend, value-based)
//!
//! - **Sparse LU**: Direct LU factorization with partial pivoting (Gilbert-Peierls)
//! - **ILU(0)**: Incomplete LU factorization with zero fill-in
//! - **ILU(k)**: Incomplete LU factorization with level-k fill-in
//! - **IC(0)**: Incomplete Cholesky factorization with zero fill-in
//! - **Sparse Triangular Solve**: Forward/backward substitution for sparse triangular matrices
//!
//! # Use Cases
//!
//! - **COLAMD** → Reduce fill-in before LU/QR factorization
//! - **Maximum Transversal** → Row permutation for diagonal nonzeros, BTF preprocessing
//! - **Sparse LU** → Direct solver for sparse linear systems
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
//! 1. **Ordering phase**: COLAMD/maximum transversal (once per pattern)
//! 2. **Symbolic phase**: Compute fill pattern based on sparsity structure
//! 3. **Numeric phase**: Compute actual values using precomputed pattern
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
pub mod lu;
pub mod matching;
pub mod ordering;
pub mod symbolic;
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
    LevelSchedule, compute_levels_csc_lower, compute_levels_csc_upper, compute_levels_ilu,
    compute_levels_lower, compute_levels_upper, flatten_levels,
};

// Re-export shared symbolic implementations
pub use symbolic::{ilu0_symbolic_impl, iluk_symbolic_impl};

// Re-export CPU implementations
pub use cpu::{
    ic0_cpu, ilu0_cpu, ilu0_numeric_cpu, ilu0_symbolic_cpu, iluk_cpu, iluk_numeric_cpu,
    iluk_symbolic_cpu, sparse_solve_triangular_cpu,
};

// Re-export sparse LU types and functions
pub use lu::{
    LuFactors, LuMetrics, LuOptions, LuSymbolic, LuSymbolicSimple, LuWorkspace, SparseLuKernels,
    SparseLuOps, sparse_lu_cpu, sparse_lu_cpu_with_metrics, sparse_lu_cpu_with_workspace,
    sparse_lu_cpu_with_workspace_and_metrics, sparse_lu_simple_cpu, sparse_lu_solve_cpu,
};

// Re-export ordering algorithms
pub use ordering::{ColamdOptions, ColamdStats, SparseOrdering, colamd};

// Re-export matching algorithms
pub use matching::{BipartiteMatching, MatchingResult, hopcroft_karp, maximum_transversal};
