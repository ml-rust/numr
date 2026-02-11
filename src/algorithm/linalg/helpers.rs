//! Helper functions for linear algebra operations
//!
//! Validation utilities and common helper functions used across backends.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::TypeConversionOps;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Validate matrix is 2D
pub fn validate_matrix_2d(shape: &[usize]) -> Result<(usize, usize)> {
    if shape.len() != 2 {
        return Err(Error::Internal(format!(
            "Expected 2D matrix, got {}D tensor with shape {:?}",
            shape.len(),
            shape
        )));
    }
    Ok((shape[0], shape[1]))
}

/// Validate matrix is square
pub fn validate_square_matrix(shape: &[usize]) -> Result<usize> {
    let (m, n) = validate_matrix_2d(shape)?;
    if m != n {
        return Err(Error::ShapeMismatch {
            expected: vec![m, m],
            got: vec![m, n],
        });
    }
    Ok(n)
}

/// Validate dtypes match for linear algebra operations.
///
/// Accepts all floating-point types. Reduced-precision types (F16, BF16, FP8)
/// are accepted but callers should promote to F32 before computation.
pub fn validate_linalg_dtype(dtype: DType) -> Result<()> {
    if dtype.is_float() {
        Ok(())
    } else {
        Err(Error::UnsupportedDType {
            dtype,
            op: "linear algebra",
        })
    }
}

/// Returns the working dtype for linalg computation.
/// F32/F64 are used directly; all other float types are promoted to F32.
pub fn linalg_working_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F32 | DType::F64 => dtype,
        _ => DType::F32,
    }
}

/// Promote a tensor to its linalg working dtype (F32 for reduced-precision types).
///
/// Returns the promoted tensor and the original dtype. If the tensor is already
/// F32/F64, returns it by reference (no allocation). Use [`linalg_demote`] to
/// cast results back to the original dtype.
pub fn linalg_promote<'a, R, C>(
    client: &C,
    tensor: &'a Tensor<R>,
) -> Result<(std::borrow::Cow<'a, Tensor<R>>, DType)>
where
    R: Runtime,
    C: TypeConversionOps<R>,
{
    let original_dtype = tensor.dtype();
    let working = linalg_working_dtype(original_dtype);
    if working != original_dtype {
        Ok((
            std::borrow::Cow::Owned(client.cast(tensor, working)?),
            original_dtype,
        ))
    } else {
        Ok((std::borrow::Cow::Borrowed(tensor), original_dtype))
    }
}

/// Cast a result tensor back to the original dtype after linalg computation.
///
/// No-op if `original_dtype` matches the tensor's current dtype.
pub fn linalg_demote<R, C>(
    client: &C,
    result: Tensor<R>,
    original_dtype: DType,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TypeConversionOps<R>,
{
    if result.dtype() != original_dtype {
        client.cast(&result, original_dtype)
    } else {
        Ok(result)
    }
}

/// Machine epsilon for floating point comparison
pub fn machine_epsilon(dtype: DType) -> f64 {
    match dtype {
        DType::F32 => f32::EPSILON as f64,
        DType::F64 => f64::EPSILON,
        _ => f32::EPSILON as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_square_matrix() {
        assert!(validate_square_matrix(&[3, 3]).is_ok());
        assert!(validate_square_matrix(&[2, 3]).is_err());
        assert!(validate_square_matrix(&[3, 2, 1]).is_err());
    }

    #[test]
    fn test_validate_linalg_dtype() {
        assert!(validate_linalg_dtype(DType::F32).is_ok());
        assert!(validate_linalg_dtype(DType::F64).is_ok());
        assert!(validate_linalg_dtype(DType::I32).is_err());
    }
}
