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
//!
//! For backward compatibility, the combined shader is available via
//! [`super::linalg_wgsl::LINALG_SHADER`].

pub mod basic_ops;
pub mod decompositions;
pub mod eig_general;
pub mod eig_symmetric;
pub mod matrix_functions;
pub mod schur;
pub mod solvers;
pub mod svd;
pub mod utilities;

// Individual shader constants for lazy per-category compilation.
// Each launcher uses only the shader it needs, which is compiled on first use.
pub use basic_ops::BASIC_OPS_SHADER;
pub use decompositions::DECOMPOSITIONS_SHADER;
pub use eig_general::EIG_GENERAL_SHADER;
pub use eig_symmetric::EIG_SYMMETRIC_SHADER;
pub use matrix_functions::MATRIX_FUNCTIONS_SHADER;
pub use schur::SCHUR_SHADER;
pub use solvers::SOLVERS_SHADER;
pub use svd::SVD_SHADER;
pub use utilities::UTILITIES_SHADER;
