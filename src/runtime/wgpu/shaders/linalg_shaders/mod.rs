//! Linear algebra WGSL shader modules
//!
//! Split into focused submodules to maintain file size limits.
//! Each submodule contains shader source code for related operations.
//!
//! # Module Organization
//!
//! - `basic_ops` - Trace, diagonal, identity operations
//! - `solvers` - Forward/backward substitution
//! - `decompositions` - LU, Cholesky, QR decompositions
//! - `utilities` - Determinant, permutation, column ops, rank helpers
//! - `svd` - Singular value decomposition (Jacobi)
//! - `eig_symmetric` - Symmetric eigendecomposition (Jacobi)
//! - `schur` - Schur decomposition (Hessenberg + QR iteration)
//! - `eig_general` - General eigendecomposition
//! - `matrix_functions` - Matrix exponential, square root, logarithm
//!
//! # Lazy Shader Compilation
//!
//! Each shader module is compiled on first use via the `PipelineCache`. This allows
//! applications that only use a subset of linalg operations to avoid the compilation
//! overhead of unused shaders.

pub mod basic_ops;
pub mod decompositions;
pub mod eig_general;
pub mod eig_symmetric;
pub mod matrix_functions;
pub mod schur;
pub mod solvers;
pub mod svd;
pub mod utilities;
