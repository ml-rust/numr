//! DType support for WebGPU operations.
//!
//! WebGPU is a 32-bit compute backend. All element-wise, scalar, comparison,
//! and activation operations are F32 only. Cast supports F32 ↔ I32 ↔ U32
//! because type conversions are necessary for indexing interop.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Returns true only for F32 (all WebGPU compute ops are F32-only).
pub fn is_wgpu_compute_supported(dtype: DType) -> bool {
    dtype == DType::F32
}

/// Validate F32 for unary operations.
pub fn check_unary_dtype_support(op: &'static str, dtype: DType) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

/// Validate F32 for binary operations.
pub fn check_binary_dtype_support(op: &'static str, dtype: DType) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

/// Validate F32 for scalar operations.
pub fn check_scalar_dtype_support(op: &'static str, dtype: DType) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

/// Validate F32 for comparison operations.
pub fn check_compare_dtype_support(op: &'static str, dtype: DType) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_supported() {
        assert!(check_unary_dtype_support("neg", DType::F32).is_ok());
        assert!(check_binary_dtype_support("add", DType::F32).is_ok());
        assert!(check_scalar_dtype_support("add_scalar", DType::F32).is_ok());
        assert!(check_compare_dtype_support("eq", DType::F32).is_ok());
    }

    #[test]
    fn test_non_f32_rejected() {
        assert!(check_unary_dtype_support("abs", DType::I32).is_err());
        assert!(check_binary_dtype_support("add", DType::U32).is_err());
        assert!(check_scalar_dtype_support("mul_scalar", DType::I32).is_err());
        assert!(check_compare_dtype_support("lt", DType::U32).is_err());
    }
}
