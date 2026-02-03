//! CPU-specific sparse linear algebra algorithms
//!
//! This module provides CPU implementations of sparse linear algebra algorithms.
//! These are inherently sequential algorithms that work directly on CPU memory.
//!
//! # Note
//!
//! These implementations use `to_vec()` to extract data, which is efficient
//! for CPU tensors (just reading RAM) but would be slow for GPU tensors.
//! GPU backends should use level-scheduled native kernels instead.

mod ic0;
mod ilu0;
mod triangular_solve;

pub use ic0::ic0_cpu;
pub use ilu0::ilu0_cpu;
pub use triangular_solve::sparse_solve_triangular_cpu;

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Validate dtype for sparse linear algebra (CPU supports F32 and F64)
pub fn validate_cpu_dtype(dtype: DType) -> Result<()> {
    if dtype != DType::F32 && dtype != DType::F64 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_linalg_cpu",
        });
    }
    Ok(())
}
