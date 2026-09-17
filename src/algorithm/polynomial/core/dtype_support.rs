//! DType support flags for backend validation.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// DType support flags for backend validation
///
/// Used to specify which floating-point dtypes a backend supports,
/// and which index dtype to use for indexing operations.
///
/// # Examples
///
/// ```
/// use numr::algorithm::polynomial::core::DTypeSupport;
/// use numr::dtype::DType;
///
/// // CPU and CUDA support both F32 and F64, use I64 indices
/// let support = DTypeSupport::FULL;
/// assert!(support.check(DType::F64, "polyroots").is_ok());
/// assert_eq!(support.index_dtype, DType::I64);
///
/// // WebGPU only supports F32, uses I32 indices (WGSL has no i64)
/// let support = DTypeSupport::F32_ONLY;
/// assert!(support.check(DType::F64, "polyroots").is_err());
/// assert_eq!(support.index_dtype, DType::I32);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DTypeSupport {
    /// Whether F32 dtype is supported
    pub f32: bool,
    /// Whether F64 dtype is supported
    pub f64: bool,
    /// Index dtype to use for indexing operations
    /// CPU/CUDA use I64, WebGPU uses I32 (WGSL has no i64 type)
    pub index_dtype: DType,
}

impl DTypeSupport {
    /// Full floating-point support (F32 and F64) with I64 indices
    ///
    /// Used for CPU and CUDA backends which support both single and double precision.
    pub const FULL: Self = Self {
        f32: true,
        f64: true,
        index_dtype: DType::I64,
    };

    /// F32 only support with I32 indices
    ///
    /// Used for WebGPU backend since WGSL does not support 64-bit floats or integers.
    pub const F32_ONLY: Self = Self {
        f32: true,
        f64: false,
        index_dtype: DType::I32,
    };

    /// Check if dtype is supported for the given operation
    ///
    /// Returns `Ok(())` if supported, `Err(UnsupportedDType)` otherwise.
    pub fn check(&self, dtype: DType, op: &'static str) -> Result<()> {
        match dtype {
            DType::F32 if self.f32 => Ok(()),
            DType::F64 if self.f64 => Ok(()),
            // F16, BF16, FP8 supported if F32 is supported (they convert to/from F32)
            DType::F16 | DType::BF16 | DType::FP8E4M3 | DType::FP8E5M2 if self.f32 => Ok(()),
            DType::F32 | DType::F64 => Err(Error::UnsupportedDType { dtype, op }),
            _ => Err(Error::UnsupportedDType { dtype, op }),
        }
    }
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
