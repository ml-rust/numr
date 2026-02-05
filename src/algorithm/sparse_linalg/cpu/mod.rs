//! CPU implementations of sparse linear algebra algorithms
//!
//! These implementations use sequential row-by-row algorithms optimized for CPU.
//! GPU backends would use level-scheduled parallel kernels instead.

mod ic0;
mod ilu0;
mod iluk;
mod triangular_solve;

pub use ic0::ic0_cpu;
pub use ilu0::{ilu0_cpu, ilu0_numeric_cpu, ilu0_symbolic_cpu};
pub use iluk::{iluk_cpu, iluk_numeric_cpu, iluk_symbolic_cpu};
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
