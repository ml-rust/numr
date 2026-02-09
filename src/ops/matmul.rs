//! Matrix multiplication helpers
//!
//! This module contains helper types and functions for matrix multiplication.
//! The actual operations are defined in the `TensorOps` trait.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Matrix multiplication parameters
#[derive(Copy, Clone, Debug)]
pub struct MatmulParams {
    /// Number of rows in A (M)
    pub m: usize,
    /// Number of columns in A / rows in B (K)
    pub k: usize,
    /// Number of columns in B (N)
    pub n: usize,
    /// Whether A is transposed
    pub trans_a: bool,
    /// Whether B is transposed
    pub trans_b: bool,
    /// Batch size (1 for regular matmul)
    pub batch: usize,
}

impl MatmulParams {
    /// Create params for standard matmul: C`[M,N]` = A`[M,K]` @ B`[K,N]`
    pub fn new(m: usize, k: usize, n: usize) -> Self {
        Self {
            m,
            k,
            n,
            trans_a: false,
            trans_b: false,
            batch: 1,
        }
    }

    /// Create params for batched matmul
    pub fn batched(batch: usize, m: usize, k: usize, n: usize) -> Self {
        Self {
            m,
            k,
            n,
            trans_a: false,
            trans_b: false,
            batch,
        }
    }

    /// Set A transposition
    pub fn with_trans_a(mut self, trans: bool) -> Self {
        self.trans_a = trans;
        self
    }

    /// Set B transposition
    pub fn with_trans_b(mut self, trans: bool) -> Self {
        self.trans_b = trans;
        self
    }

    /// Output shape
    pub fn output_shape(&self) -> Vec<usize> {
        if self.batch > 1 {
            vec![self.batch, self.m, self.n]
        } else {
            vec![self.m, self.n]
        }
    }
}

/// Validate matmul shapes and return dimensions (m, k, n)
///
/// Returns None if shapes are incompatible.
pub fn validate_matmul_shapes(
    a_shape: &[usize],
    b_shape: &[usize],
) -> Option<(usize, usize, usize)> {
    // Handle 1D vectors
    let (a_rows, a_cols) = match a_shape.len() {
        0 => return None,
        1 => (1, a_shape[0]),
        _ => {
            let ndim = a_shape.len();
            (a_shape[ndim - 2], a_shape[ndim - 1])
        }
    };

    let (b_rows, b_cols) = match b_shape.len() {
        0 => return None,
        1 => (b_shape[0], 1),
        _ => {
            let ndim = b_shape.len();
            (b_shape[ndim - 2], b_shape[ndim - 1])
        }
    };

    // Inner dimensions must match
    if a_cols != b_rows {
        return None;
    }

    Some((a_rows, a_cols, b_cols))
}

/// Compute output shape for matmul
pub fn matmul_output_shape(a_shape: &[usize], b_shape: &[usize]) -> Option<Vec<usize>> {
    let (m, _k, n) = validate_matmul_shapes(a_shape, b_shape)?;

    // Handle batched matmul
    let a_batch: Vec<_> = a_shape
        .iter()
        .take(a_shape.len().saturating_sub(2))
        .copied()
        .collect();
    let b_batch: Vec<_> = b_shape
        .iter()
        .take(b_shape.len().saturating_sub(2))
        .copied()
        .collect();

    // Broadcast batch dimensions
    let batch = super::broadcast_shape(&a_batch, &b_batch)?;

    let mut result = batch;
    result.push(m);
    result.push(n);
    Some(result)
}

/// Validate matmul_bias shapes and return dimensions (m, k, n)
///
/// Checks that:
/// - A and B are compatible for matmul (inner dimensions match)
/// - bias is 1D
/// - bias length matches output columns (N)
///
/// Returns None if shapes are incompatible.
pub fn validate_matmul_bias_shapes(
    a_shape: &[usize],
    b_shape: &[usize],
    bias_shape: &[usize],
) -> Option<(usize, usize, usize)> {
    // First validate matmul shapes
    let (m, k, n) = validate_matmul_shapes(a_shape, b_shape)?;

    // Bias must be 1D
    if bias_shape.len() != 1 {
        return None;
    }

    // Bias length must match output columns (N)
    if bias_shape[0] != n {
        return None;
    }

    Some((m, k, n))
}

/// Compute output shape for matmul_bias
///
/// Same as matmul_output_shape - bias doesn't change output dimensions.
pub fn matmul_bias_output_shape(
    a_shape: &[usize],
    b_shape: &[usize],
    bias_shape: &[usize],
) -> Option<Vec<usize>> {
    // Validate bias shape
    validate_matmul_bias_shapes(a_shape, b_shape, bias_shape)?;

    // Output shape is same as matmul
    matmul_output_shape(a_shape, b_shape)
}

/// Validate that all three tensors (A, B, bias) have the same dtype for matmul_bias.
///
/// This is the **canonical** dtype validation for matmul_bias - use this function
/// in all backend implementations to ensure consistent error handling.
///
/// # Returns
/// - `Ok(dtype)` if all dtypes match
/// - `Err(DTypeMismatch)` if any dtypes differ
pub fn validate_matmul_bias_dtypes(
    a_dtype: DType,
    b_dtype: DType,
    bias_dtype: DType,
) -> Result<DType> {
    if a_dtype != b_dtype {
        return Err(Error::DTypeMismatch {
            lhs: a_dtype,
            rhs: b_dtype,
        });
    }
    if a_dtype != bias_dtype {
        return Err(Error::DTypeMismatch {
            lhs: a_dtype,
            rhs: bias_dtype,
        });
    }
    Ok(a_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_matmul_shapes() {
        // Valid 2D matmul
        assert_eq!(validate_matmul_shapes(&[2, 3], &[3, 4]), Some((2, 3, 4)));

        // Invalid: inner dimensions don't match
        assert_eq!(validate_matmul_shapes(&[2, 3], &[4, 5]), None);

        // 1D vectors
        assert_eq!(validate_matmul_shapes(&[3], &[3, 4]), Some((1, 3, 4)));
        assert_eq!(validate_matmul_shapes(&[2, 3], &[3]), Some((2, 3, 1)));
    }

    #[test]
    fn test_matmul_output_shape() {
        // Basic 2D matmul
        assert_eq!(matmul_output_shape(&[2, 3], &[3, 4]), Some(vec![2, 4]));

        // Batched matmul
        assert_eq!(
            matmul_output_shape(&[5, 2, 3], &[5, 3, 4]),
            Some(vec![5, 2, 4])
        );

        // Broadcast batches
        assert_eq!(
            matmul_output_shape(&[5, 2, 3], &[3, 4]),
            Some(vec![5, 2, 4])
        );
    }

    #[test]
    fn test_validate_matmul_bias_shapes() {
        // Valid: 2D matmul with 1D bias
        assert_eq!(
            validate_matmul_bias_shapes(&[2, 3], &[3, 4], &[4]),
            Some((2, 3, 4))
        );

        // Invalid: inner dimensions don't match
        assert_eq!(validate_matmul_bias_shapes(&[2, 3], &[4, 5], &[5]), None);

        // Invalid: bias is 2D
        assert_eq!(validate_matmul_bias_shapes(&[2, 3], &[3, 4], &[2, 4]), None);

        // Invalid: bias length doesn't match N
        assert_eq!(validate_matmul_bias_shapes(&[2, 3], &[3, 4], &[3]), None);

        // Valid: batched matmul with 1D bias
        assert_eq!(
            validate_matmul_bias_shapes(&[5, 2, 3], &[5, 3, 4], &[4]),
            Some((2, 3, 4))
        );
    }

    #[test]
    fn test_matmul_bias_output_shape() {
        // Basic 2D matmul_bias
        assert_eq!(
            matmul_bias_output_shape(&[2, 3], &[3, 4], &[4]),
            Some(vec![2, 4])
        );

        // Batched matmul_bias
        assert_eq!(
            matmul_bias_output_shape(&[5, 2, 3], &[5, 3, 4], &[4]),
            Some(vec![5, 2, 4])
        );

        // Invalid bias shape returns None
        assert_eq!(matmul_bias_output_shape(&[2, 3], &[3, 4], &[3]), None);
    }

    #[test]
    fn test_validate_matmul_bias_dtypes() {
        // All same dtype - should succeed
        assert!(validate_matmul_bias_dtypes(DType::F32, DType::F32, DType::F32).is_ok());
        assert_eq!(
            validate_matmul_bias_dtypes(DType::F32, DType::F32, DType::F32).unwrap(),
            DType::F32
        );
        assert_eq!(
            validate_matmul_bias_dtypes(DType::F64, DType::F64, DType::F64).unwrap(),
            DType::F64
        );

        // A and B mismatch
        let result = validate_matmul_bias_dtypes(DType::F32, DType::F64, DType::F32);
        assert!(result.is_err());
        match result {
            Err(Error::DTypeMismatch { lhs, rhs }) => {
                assert_eq!(lhs, DType::F32);
                assert_eq!(rhs, DType::F64);
            }
            _ => panic!("Expected DTypeMismatch error"),
        }

        // A and bias mismatch
        let result = validate_matmul_bias_dtypes(DType::F32, DType::F32, DType::I32);
        assert!(result.is_err());
        match result {
            Err(Error::DTypeMismatch { lhs, rhs }) => {
                assert_eq!(lhs, DType::F32);
                assert_eq!(rhs, DType::I32);
            }
            _ => panic!("Expected DTypeMismatch error"),
        }
    }
}
