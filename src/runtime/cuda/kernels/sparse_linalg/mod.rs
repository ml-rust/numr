//! CUDA kernel launchers for level-scheduled sparse linear algebra
//!
//! This module provides Rust wrappers for launching level-scheduled
//! sparse triangular solve, ILU(0), IC(0), and LU factorization kernels.
//!
//! # Module Organization
//!
//! - `trsv` - Triangular solve launchers (CSR and CSC formats)
//! - `ilu_ic` - ILU(0) and IC(0) factorization launchers
//! - `primitives` - Sparse primitive operations (scatter, axpy, gather, etc.)
//! - `utils` - Utility launchers (find_diag, copy, split_lu, etc.)

mod ilu_ic;
mod primitives;
mod trsv;
mod utils;

pub use ilu_ic::*;
pub use primitives::*;
pub use trsv::*;
pub use utils::*;

// Re-export loader items for submodules
pub(crate) use super::loader::{get_kernel_function, get_or_load_module, launch_config};
use crate::error::Error;

/// Module name for sparse linear algebra kernels
pub const SPARSE_LINALG_MODULE: &str = "sparse_linalg";

/// Default block size for sparse kernels
pub(crate) const BLOCK_SIZE: u32 = 256;

/// Compute grid size for a given number of elements
#[inline]
pub(crate) fn grid_size(n: u32) -> u32 {
    (n + BLOCK_SIZE - 1) / BLOCK_SIZE
}

/// Create a launch error message
#[inline]
pub(crate) fn launch_error(kernel_name: &str, e: impl std::fmt::Debug) -> Error {
    Error::Internal(format!("CUDA {} launch failed: {:?}", kernel_name, e))
}
