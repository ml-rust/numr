//! Linear algebra CUDA kernel launchers
//!
//! Split into categorical submodules for maintainability:
//! - `basic` - trace, diag, diagflat, identity, copy, transpose
//! - `solvers` - forward/backward substitution, determinant, permutation
//! - `decompositions` - LU, Cholesky, QR
//! - `svd` - SVD Jacobi algorithm
//! - `eigen` - eigendecomposition (symmetric, general, Schur)
//! - `advanced` - rsf2csf, QZ decomposition

mod advanced;
mod basic;
mod decompositions;
mod eigen;
mod solvers;
mod svd;

// Re-export all launcher functions
pub use advanced::*;
pub use basic::*;
pub use decompositions::*;
pub use eigen::*;
pub use solvers::*;
pub use svd::*;
