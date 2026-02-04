//! Core polynomial algorithms shared across all backends
//!
//! This module provides the unified implementation for polynomial operations.
//! All backends (CPU, CUDA, WebGPU) call these functions to ensure numerical parity.
//!
//! # Design
//!
//! These functions operate through the trait interface (LinearAlgebraAlgorithms,
//! BinaryOps, IndexingOps, etc.) using ONLY tensor operations. This ensures all
//! operations stay on the original device without GPU↔CPU transfers.
//!
//! # No GPU↔CPU Transfers
//!
//! All algorithms are implemented using tensor operations only:
//! - `index_select` for accessing individual coefficients
//! - `scatter_reduce` for convolution operations
//! - `arange` and `eye` for tensor construction
//! - Broadcasting for element-wise operations

mod polyfromroots;
mod polymul;
mod polyroots;
mod polyval;

pub use polyfromroots::polyfromroots_impl;
pub use polymul::polymul_impl;
pub use polyroots::polyroots_impl;
pub use polyval::polyval_impl;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// DType support flags for backend validation
///
/// Used to specify which floating-point dtypes a backend supports.
///
/// # Examples
///
/// ```ignore
/// // CPU and CUDA support both F32 and F64
/// let support = DTypeSupport::FULL;
/// assert!(support.check(DType::F64, "polyroots").is_ok());
///
/// // WebGPU only supports F32 (WGSL has no f64)
/// let support = DTypeSupport::F32_ONLY;
/// assert!(support.check(DType::F64, "polyroots").is_err());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DTypeSupport {
    /// Whether F32 dtype is supported
    pub f32: bool,
    /// Whether F64 dtype is supported
    pub f64: bool,
}

impl DTypeSupport {
    /// Full floating-point support (F32 and F64)
    ///
    /// Used for CPU and CUDA backends which support both single and double precision.
    pub const FULL: Self = Self {
        f32: true,
        f64: true,
    };

    /// F32 only support
    ///
    /// Used for WebGPU backend since WGSL does not support 64-bit floats.
    pub const F32_ONLY: Self = Self {
        f32: true,
        f64: false,
    };

    /// Check if dtype is supported for the given operation
    ///
    /// Returns `Ok(())` if supported, `Err(UnsupportedDType)` otherwise.
    pub fn check(&self, dtype: DType, op: &'static str) -> Result<()> {
        match dtype {
            DType::F32 if self.f32 => Ok(()),
            DType::F64 if self.f64 => Ok(()),
            DType::F32 | DType::F64 => Err(Error::UnsupportedDType { dtype, op }),
            _ => Err(Error::UnsupportedDType { dtype, op }),
        }
    }
}

// ============================================================================
// Helper Functions (shared across algorithm implementations)
// ============================================================================

/// Create a single-element I64 index tensor
pub(crate) fn create_index_tensor<R: Runtime>(index: usize, device: &R::Device) -> Tensor<R> {
    Tensor::<R>::from_slice(&[index as i64], &[1], device)
}

/// Create an arange-like I64 tensor [start, start+1, ..., end-1]
pub(crate) fn create_arange_tensor<R: Runtime>(
    start: usize,
    end: usize,
    device: &R::Device,
) -> Tensor<R> {
    let indices: Vec<i64> = (start..end).map(|i| i as i64).collect();
    Tensor::<R>::from_slice(&indices, &[indices.len()], device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_support() {
        assert!(DTypeSupport::FULL.check(DType::F32, "test").is_ok());
        assert!(DTypeSupport::FULL.check(DType::F64, "test").is_ok());
        assert!(DTypeSupport::FULL.check(DType::I32, "test").is_err());

        assert!(DTypeSupport::F32_ONLY.check(DType::F32, "test").is_ok());
        assert!(DTypeSupport::F32_ONLY.check(DType::F64, "test").is_err());
    }
}
