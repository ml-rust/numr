//! Linear algebra WGSL kernel launchers
//!
//! Split into focused submodules for maintainability:
//! - `basic_ops` - trace, diag, diagflat, create_identity
//! - `solvers` - forward/backward substitution
//! - `decompositions` - LU, Cholesky, QR
//! - `utilities` - determinant, permutation, column ops, rank helpers
//! - `svd` - SVD Jacobi algorithm
//! - `eig` - eigendecomposition (symmetric and general)
//! - `matrix_functions` - exp, sqrt, log of matrices

mod basic_ops;
mod decompositions;
mod eig;
mod matrix_functions;
mod solvers;
mod svd;
mod utilities;

// Re-export all launcher functions
pub use basic_ops::{
    launch_create_identity, launch_diag, launch_diagflat, launch_khatri_rao, launch_kron,
    launch_trace,
};
pub use decompositions::{launch_cholesky_decompose, launch_lu_decompose, launch_qr_decompose};
pub use eig::{
    launch_eig_general, launch_eig_jacobi_symmetric, launch_qz_decompose, launch_rsf2csf,
    launch_schur_decompose,
};
pub use matrix_functions::{
    launch_exp_quasi_triangular, launch_log_quasi_triangular, launch_sqrt_quasi_triangular,
};
pub use solvers::{launch_backward_sub, launch_forward_sub};
pub use svd::launch_svd_jacobi;
pub use utilities::{
    launch_apply_lu_permutation, launch_count_above_threshold, launch_det_from_lu,
    launch_extract_column, launch_matrix_copy, launch_max_abs, launch_scatter_column,
};

/// Helper macro to check dtype is F32 (only supported type for linalg)
macro_rules! check_dtype_f32 {
    ($dtype:expr, $op:expr) => {
        if $dtype != crate::dtype::DType::F32 {
            return Err(crate::error::Error::UnsupportedDType {
                dtype: $dtype,
                op: $op,
            });
        }
    };
}

pub(crate) use check_dtype_f32;
