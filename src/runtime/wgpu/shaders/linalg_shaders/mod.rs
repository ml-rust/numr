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

// Individual shader constants for potential future fine-grained compilation.
// Currently unused - all launchers use the combined LINALG_SHADER for simplicity.
// Exported as public API for potential future optimization where applications
// could compile only the shaders they need, reducing compilation time and memory.
//
// TODO: Consider implementing fine-grained shader compilation to use these
// individual modules instead of the monolithic combined shader.
#[allow(unused_imports)]
pub use basic_ops::BASIC_OPS_SHADER;
#[allow(unused_imports)]
pub use decompositions::DECOMPOSITIONS_SHADER;
#[allow(unused_imports)]
pub use eig_general::EIG_GENERAL_SHADER;
#[allow(unused_imports)]
pub use eig_symmetric::EIG_SYMMETRIC_SHADER;
#[allow(unused_imports)]
pub use matrix_functions::MATRIX_FUNCTIONS_SHADER;
#[allow(unused_imports)]
pub use schur::SCHUR_SHADER;
#[allow(unused_imports)]
pub use solvers::SOLVERS_SHADER;
#[allow(unused_imports)]
pub use svd::SVD_SHADER;
#[allow(unused_imports)]
pub use utilities::UTILITIES_SHADER;
