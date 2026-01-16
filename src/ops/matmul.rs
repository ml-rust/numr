//! Matrix multiplication helpers
//!
//! This module contains helper types and functions for matrix multiplication.
//! The actual operations are defined in the `TensorOps` trait.

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
    /// Create params for standard matmul: C[M,N] = A[M,K] @ B[K,N]
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
}
