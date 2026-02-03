//! WGSL shader generation for sparse linear algebra operations.
//!
//! This module re-exports from the split submodules for backward compatibility.
//! The actual implementations are in:
//! - `sparse_trsv.rs` - Sparse triangular solve shaders
//! - `sparse_factorize.rs` - ILU(0) and IC(0) factorization shaders
//! - `sparse_utils.rs` - Utility shaders (find_diag, copy)
//! - `sparse_split.rs` - Split LU and extract lower triangle shaders

// Re-export from split modules for backward compatibility
pub use super::sparse_factorize::{generate_ic0_level_shader, generate_ilu0_level_shader};
pub use super::sparse_split::{
    generate_extract_lower_count_shader, generate_extract_lower_scatter_shader,
    generate_split_lu_count_shader, generate_split_lu_scatter_l_shader,
    generate_split_lu_scatter_shader, generate_split_lu_scatter_u_shader,
};
pub use super::sparse_trsv::{
    generate_sparse_trsv_lower_multi_rhs_shader, generate_sparse_trsv_lower_shader,
    generate_sparse_trsv_upper_multi_rhs_shader, generate_sparse_trsv_upper_shader,
};
pub use super::sparse_utils::{generate_copy_shader, generate_find_diag_indices_shader};
