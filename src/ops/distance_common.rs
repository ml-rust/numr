//! Shared validation and utility functions for distance operations.
//!
//! This module contains common validation logic used across all backend implementations
//! (CPU, CUDA, WebGPU) to ensure consistency and eliminate code duplication.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Validates that a tensor is 2-dimensional.
///
/// # Arguments
/// * `shape` - The tensor shape to validate
/// * `arg_name` - Name of the argument for error messages
/// * `op` - Name of the operation for error messages
///
/// # Errors
/// Returns `InvalidArgument` if the tensor is not 2D.
#[inline]
pub fn validate_2d_tensor(shape: &[usize], arg_name: &'static str, op: &'static str) -> Result<()> {
    if shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: arg_name,
            reason: format!("{} expects 2D tensor, got {}D", op, shape.len()),
        });
    }
    Ok(())
}

/// Validates that a tensor is 1-dimensional.
///
/// # Arguments
/// * `shape` - The tensor shape to validate
/// * `arg_name` - Name of the argument for error messages
/// * `op` - Name of the operation for error messages
///
/// # Errors
/// Returns `InvalidArgument` if the tensor is not 1D.
#[inline]
pub fn validate_1d_tensor(shape: &[usize], arg_name: &'static str, op: &'static str) -> Result<()> {
    if shape.len() != 1 {
        return Err(Error::InvalidArgument {
            arg: arg_name,
            reason: format!("{} expects 1D tensor, got {}D", op, shape.len()),
        });
    }
    Ok(())
}

/// Validates that two tensors have matching feature dimensions (last axis).
///
/// # Arguments
/// * `x_shape` - Shape of the first tensor
/// * `y_shape` - Shape of the second tensor
/// * `op` - Name of the operation for error messages
///
/// # Errors
/// Returns `InvalidArgument` if the last dimensions don't match.
#[inline]
pub fn validate_same_dimension(
    x_shape: &[usize],
    y_shape: &[usize],
    op: &'static str,
) -> Result<()> {
    let d_x = x_shape[x_shape.len() - 1];
    let d_y = y_shape[y_shape.len() - 1];
    if d_x != d_y {
        return Err(Error::InvalidArgument {
            arg: "y",
            reason: format!(
                "{} requires same dimensionality, got x.shape[1]={}, y.shape[1]={}",
                op, d_x, d_y
            ),
        });
    }
    Ok(())
}

/// Validates that a dtype is a floating-point type.
///
/// # Arguments
/// * `dtype` - The dtype to validate
/// * `op` - Name of the operation for error messages
///
/// # Errors
/// Returns `UnsupportedDType` if the dtype is not a float type.
#[inline]
pub fn validate_float_dtype(dtype: DType, op: &'static str) -> Result<()> {
    if !dtype.is_float() {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

/// Validates that two tensors have the same dtype.
///
/// # Arguments
/// * `x_dtype` - Dtype of the first tensor
/// * `y_dtype` - Dtype of the second tensor
/// * `op` - Name of the operation for error messages
///
/// # Errors
/// Returns `InvalidArgument` if the dtypes don't match.
#[inline]
pub fn validate_same_dtype(x_dtype: DType, y_dtype: DType, op: &'static str) -> Result<()> {
    if y_dtype != x_dtype {
        return Err(Error::InvalidArgument {
            arg: "y",
            reason: format!(
                "{} requires same dtype, got x.dtype={}, y.dtype={}",
                op, x_dtype, y_dtype
            ),
        });
    }
    Ok(())
}

/// Validates that a tensor has at least a minimum number of elements in a dimension.
///
/// # Arguments
/// * `n` - Actual size of the dimension
/// * `min` - Minimum required size
/// * `arg_name` - Name of the argument for error messages
/// * `op` - Name of the operation for error messages
///
/// # Errors
/// Returns `InvalidArgument` if n < min.
#[inline]
pub fn validate_min_points(
    n: usize,
    min: usize,
    arg_name: &'static str,
    op: &'static str,
) -> Result<()> {
    if n < min {
        return Err(Error::InvalidArgument {
            arg: arg_name,
            reason: format!("{} requires at least {} points, got {}", op, min, n),
        });
    }
    Ok(())
}

/// Validates that a condensed distance vector has the correct length for n points.
///
/// Condensed form has length n*(n-1)/2 for n points.
///
/// # Arguments
/// * `actual` - Actual length of the condensed vector
/// * `n` - Number of points
/// * `arg_name` - Name of the argument for error messages
/// * `op` - Name of the operation for error messages
///
/// # Errors
/// Returns `InvalidArgument` if the length doesn't match the expected value.
#[inline]
pub fn validate_condensed_length(
    actual: usize,
    n: usize,
    arg_name: &'static str,
    op: &'static str,
) -> Result<()> {
    let expected = n * (n - 1) / 2;
    if actual != expected {
        return Err(Error::InvalidArgument {
            arg: arg_name,
            reason: format!(
                "{} with n={} expects condensed length {}, got {}",
                op, n, expected, actual
            ),
        });
    }
    Ok(())
}

/// Validates that a matrix is square (rows == cols).
///
/// # Arguments
/// * `shape` - The tensor shape to validate
/// * `arg_name` - Name of the argument for error messages
/// * `op` - Name of the operation for error messages
///
/// # Errors
/// Returns `InvalidArgument` if the matrix is not square.
#[inline]
pub fn validate_square_matrix(
    shape: &[usize],
    arg_name: &'static str,
    op: &'static str,
) -> Result<()> {
    if shape[0] != shape[1] {
        return Err(Error::InvalidArgument {
            arg: arg_name,
            reason: format!("{} expects square matrix, got shape {:?}", op, shape),
        });
    }
    Ok(())
}

/// Validates that a dtype is supported by the backend.
///
/// For WebGPU, only F32 is supported. For CPU/CUDA, all float types are supported.
///
/// # Arguments
/// * `dtype` - The dtype to validate
/// * `supported_dtypes` - List of supported dtypes for this backend
/// * `op` - Name of the operation for error messages
///
/// # Errors
/// Returns `UnsupportedDType` if the dtype is not in the supported list.
#[inline]
pub fn validate_dtype_supported(
    dtype: DType,
    supported_dtypes: &[DType],
    op: &'static str,
) -> Result<()> {
    if !supported_dtypes.contains(&dtype) {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_2d_tensor() {
        assert!(validate_2d_tensor(&[3, 4], "x", "test").is_ok());
        assert!(validate_2d_tensor(&[3], "x", "test").is_err());
        assert!(validate_2d_tensor(&[3, 4, 5], "x", "test").is_err());
    }

    #[test]
    fn test_validate_1d_tensor() {
        assert!(validate_1d_tensor(&[10], "x", "test").is_ok());
        assert!(validate_1d_tensor(&[3, 4], "x", "test").is_err());
    }

    #[test]
    fn test_validate_same_dimension() {
        assert!(validate_same_dimension(&[3, 4], &[5, 4], "test").is_ok());
        assert!(validate_same_dimension(&[3, 4], &[5, 5], "test").is_err());
    }

    #[test]
    fn test_validate_float_dtype() {
        assert!(validate_float_dtype(DType::F32, "test").is_ok());
        assert!(validate_float_dtype(DType::F64, "test").is_ok());
        assert!(validate_float_dtype(DType::I32, "test").is_err());
    }

    #[test]
    fn test_validate_same_dtype() {
        assert!(validate_same_dtype(DType::F32, DType::F32, "test").is_ok());
        assert!(validate_same_dtype(DType::F32, DType::F64, "test").is_err());
    }

    #[test]
    fn test_validate_min_points() {
        assert!(validate_min_points(5, 2, "x", "test").is_ok());
        assert!(validate_min_points(1, 2, "x", "test").is_err());
    }

    #[test]
    fn test_validate_condensed_length() {
        // n=3 -> 3*2/2 = 3
        assert!(validate_condensed_length(3, 3, "x", "test").is_ok());
        assert!(validate_condensed_length(4, 3, "x", "test").is_err());
        // n=5 -> 5*4/2 = 10
        assert!(validate_condensed_length(10, 5, "x", "test").is_ok());
    }

    #[test]
    fn test_validate_square_matrix() {
        assert!(validate_square_matrix(&[5, 5], "x", "test").is_ok());
        assert!(validate_square_matrix(&[5, 4], "x", "test").is_err());
    }

    #[test]
    fn test_validate_dtype_supported() {
        let supported = vec![DType::F32, DType::F64];
        assert!(validate_dtype_supported(DType::F32, &supported, "test").is_ok());
        assert!(validate_dtype_supported(DType::F16, &supported, "test").is_err());
    }
}
