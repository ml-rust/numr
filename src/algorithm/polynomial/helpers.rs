//! Helper functions for polynomial operations

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Validate polynomial coefficient tensor
///
/// Requirements:
/// - Must be 1D
/// - Must have at least one element (degree >= 0)
pub fn validate_polynomial_coeffs(shape: &[usize]) -> Result<usize> {
    if shape.len() != 1 {
        return Err(Error::Internal(format!(
            "Expected 1D coefficient tensor, got {}D tensor with shape {:?}",
            shape.len(),
            shape
        )));
    }

    let n = shape[0];
    if n == 0 {
        return Err(Error::Internal(
            "Polynomial coefficient tensor cannot be empty".to_string(),
        ));
    }

    Ok(n)
}

/// Validate polynomial roots tensor
///
/// Requirements:
/// - Must be 1D (or empty)
/// - Empty is allowed (0 roots = constant polynomial)
pub fn validate_polynomial_roots(shape: &[usize]) -> Result<usize> {
    if shape.len() != 1 {
        return Err(Error::Internal(format!(
            "Expected 1D roots tensor, got {}D tensor with shape {:?}",
            shape.len(),
            shape
        )));
    }

    Ok(shape[0])
}

/// Validate dtype for polynomial operations
pub fn validate_polynomial_dtype(dtype: DType) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 => Ok(()),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "polynomial",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_polynomial_coeffs() {
        assert!(validate_polynomial_coeffs(&[3]).is_ok());
        assert!(validate_polynomial_coeffs(&[1]).is_ok());
        assert!(validate_polynomial_coeffs(&[0]).is_err());
        assert!(validate_polynomial_coeffs(&[2, 3]).is_err());
        assert!(validate_polynomial_coeffs(&[]).is_err());
    }

    #[test]
    fn test_validate_polynomial_dtype() {
        assert!(validate_polynomial_dtype(DType::F32).is_ok());
        assert!(validate_polynomial_dtype(DType::F64).is_ok());
        assert!(validate_polynomial_dtype(DType::I32).is_err());
    }

    #[test]
    fn test_validate_polynomial_roots() {
        assert!(validate_polynomial_roots(&[3]).is_ok());
        assert!(validate_polynomial_roots(&[0]).is_ok()); // Empty is OK for roots
        assert!(validate_polynomial_roots(&[2, 3]).is_err());
    }
}
