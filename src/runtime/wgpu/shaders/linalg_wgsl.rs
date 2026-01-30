//! WGSL shader source code for linear algebra operations
//!
//! This module provides the combined linear algebra shader used by all linalg operations.
//! The shader source is maintained in `linalg_combined.wgsl` which contains all operations:
//!
//! - Basic ops: trace, diagonal, identity
//! - Solvers: forward/backward substitution
//! - Decompositions: LU, Cholesky, QR
//! - Utilities: determinant, permutation, column operations
//! - SVD: Singular value decomposition (Jacobi)
//! - Eigendecomposition: Symmetric and general cases
//! - Schur decomposition
//! - Matrix functions: expm, sqrtm, logm
//!
//! # Future Work
//!
//! Individual shader modules exist in `linalg_shaders/` for potential fine-grained
//! compilation, but are not currently used. This could reduce shader compilation time
//! for specialized applications that only need specific operations.

/// Combined linear algebra shader containing all operations.
///
/// This shader is used by all linear algebra launchers and includes all operations
/// from basic matrix ops to advanced decompositions.
pub const LINALG_SHADER: &str = include_str!("linalg_combined.wgsl");
