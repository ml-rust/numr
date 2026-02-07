//! Linear algebra CUDA kernel launchers
//!
//! Split into categorical submodules for maintainability:
//! - `basic` - trace, diag, diagflat, identity, copy, transpose
//! - `solvers` - forward/backward substitution, determinant, permutation
//! - `decompositions` - LU, Cholesky, QR
//! - `svd` - SVD Jacobi algorithm
//! - `eigen` - eigendecomposition (symmetric, general, Schur)
//! - `advanced` - rsf2csf, QZ decomposition
//! - `matrix_funcs` - matrix functions (exp, log, sqrt) on quasi-triangular matrices

mod advanced;
mod banded;
mod basic;
mod decompositions;
mod eigen;
mod matrix_funcs;
mod solvers;
mod svd;

// Re-export all launcher functions
pub use advanced::*;
pub use banded::*;
pub use basic::*;
pub use decompositions::*;
pub use eigen::*;
pub use matrix_funcs::*;
pub use solvers::*;
pub use svd::*;
