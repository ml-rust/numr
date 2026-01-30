//! WGSL shader source code for linear algebra operations
//!
//! This module re-exports shader constants from the split `linalg/` submodule.
//! The actual shader source is split across multiple files for maintainability
//! while staying within the 500-line limit per file.
//!
//! # Architecture
//!
//! Each operation category has its own shader module with a `*_SHADER` constant.
//! These can be used independently or combined via `LINALG_SHADER` for backward
//! compatibility with existing launcher code.
//!
//! # Module Organization
//!
//! - `linalg/basic_ops.rs` - Trace, diagonal, identity (BASIC_OPS_SHADER)
//! - `linalg/solvers.rs` - Forward/backward substitution (SOLVERS_SHADER)
//! - `linalg/decompositions.rs` - LU, Cholesky, QR (DECOMPOSITIONS_SHADER)
//! - `linalg/utilities.rs` - Det, permutation, column ops (UTILITIES_SHADER)
//! - `linalg/svd.rs` - SVD Jacobi algorithm (SVD_SHADER)
//! - `linalg/eig_symmetric.rs` - Symmetric eigendecomposition (EIG_SYMMETRIC_SHADER)
//! - `linalg/schur.rs` - Schur decomposition (SCHUR_SHADER)
//! - `linalg/eig_general.rs` - General eigendecomposition (EIG_GENERAL_SHADER)
//! - `linalg/matrix_functions.rs` - expm, sqrtm, logm (MATRIX_FUNCTIONS_SHADER)

// Re-export individual shader modules for fine-grained use
pub use super::linalg_shaders::{
    BASIC_OPS_SHADER, DECOMPOSITIONS_SHADER, EIG_GENERAL_SHADER, EIG_SYMMETRIC_SHADER,
    MATRIX_FUNCTIONS_SHADER, SCHUR_SHADER, SOLVERS_SHADER, SVD_SHADER, UTILITIES_SHADER,
};

/// Combined linear algebra shader for backward compatibility.
///
/// This is a runtime-concatenated version of all shader modules.
/// For new code, consider using individual shader modules for
/// finer-grained pipeline management and smaller shader compilation.
///
/// Note: This constant is lazily initialized on first use.
pub fn get_combined_linalg_shader() -> String {
    format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
        BASIC_OPS_SHADER,
        SOLVERS_SHADER,
        DECOMPOSITIONS_SHADER,
        UTILITIES_SHADER,
        SVD_SHADER,
        EIG_SYMMETRIC_SHADER,
        SCHUR_SHADER,
        EIG_GENERAL_SHADER,
        MATRIX_FUNCTIONS_SHADER,
    )
}

// For backward compatibility with existing code that expects LINALG_SHADER as &str,
// we provide it via lazy_static or use the original combined shader.
// Since we can't use lazy_static without adding dependencies, the launcher
// should be updated to call get_combined_linalg_shader() instead.

/// Legacy combined shader constant.
///
/// **Deprecated**: Use `get_combined_linalg_shader()` or individual shader modules.
/// This constant is kept for backward compatibility but will be removed in v1.0.
pub const LINALG_SHADER: &str = include_str!("linalg_combined.wgsl");
