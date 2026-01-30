//! Linear algebra WGSL kernel launchers
//!
//! This module re-exports all linalg launcher functions from the split submodules
//! in `linalg_launchers/`. The actual implementations are organized by category:
//!
//! - `basic_ops` - trace, diag, diagflat, create_identity
//! - `solvers` - forward/backward substitution
//! - `decompositions` - LU, Cholesky, QR
//! - `utilities` - determinant, permutation, column ops, rank helpers
//! - `svd` - SVD Jacobi algorithm
//! - `eig` - eigendecomposition (symmetric and general)
//! - `matrix_functions` - exp, sqrt, log of matrices

// Re-export all launcher functions for backward compatibility
pub use super::linalg_launchers::{
    // Utilities
    launch_apply_lu_permutation,
    // Solvers
    launch_backward_sub,
    // Decompositions
    launch_cholesky_decompose,
    launch_count_above_threshold,
    // Basic operations
    launch_create_identity,
    launch_det_from_lu,
    launch_diag,
    launch_diagflat,
    // Eigendecomposition
    launch_eig_general,
    launch_eig_jacobi_symmetric,
    // Matrix functions
    launch_exp_quasi_triangular,
    launch_extract_column,
    launch_forward_sub,
    launch_log_quasi_triangular,
    launch_lu_decompose,
    launch_matrix_copy,
    launch_max_abs,
    launch_qr_decompose,
    launch_scatter_column,
    launch_schur_decompose,
    launch_sqrt_quasi_triangular,
    // SVD
    launch_svd_jacobi,
    launch_trace,
};
