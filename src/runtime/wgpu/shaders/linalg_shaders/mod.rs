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
//! # Individual Shader Constants
//!
//! Individual shader constants are exported for fine-grained pipeline management.
//! This allows users to compile only the shaders they need, reducing compilation
//! time and memory usage for specialized applications.
//!
//! For backward compatibility, the combined shader is available via
//! [`super::linalg_wgsl::LINALG_SHADER`] or [`super::linalg_wgsl::get_combined_linalg_shader()`].

pub mod basic_ops;
pub mod decompositions;
pub mod eig_general;
pub mod eig_symmetric;
pub mod matrix_functions;
pub mod schur;
pub mod solvers;
pub mod svd;
pub mod utilities;

// Re-export all shader constants for convenient access.
// These are marked allow(dead_code) because the current launcher implementation
// uses the combined LINALG_SHADER for simplicity. Individual constants are exported
// for future fine-grained shader compilation and for external crate consumers who
// may want to compile only specific operations.
#[allow(dead_code)]
pub use basic_ops::BASIC_OPS_SHADER;
#[allow(dead_code)]
pub use decompositions::DECOMPOSITIONS_SHADER;
#[allow(dead_code)]
pub use eig_general::EIG_GENERAL_SHADER;
#[allow(dead_code)]
pub use eig_symmetric::EIG_SYMMETRIC_SHADER;
#[allow(dead_code)]
pub use matrix_functions::MATRIX_FUNCTIONS_SHADER;
#[allow(dead_code)]
pub use schur::SCHUR_SHADER;
#[allow(dead_code)]
pub use solvers::SOLVERS_SHADER;
#[allow(dead_code)]
pub use svd::SVD_SHADER;
#[allow(dead_code)]
pub use utilities::UTILITIES_SHADER;
