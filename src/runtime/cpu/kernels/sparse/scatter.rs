//! Scatter kernels for sparse operations
//!
//! Scatter a sparse column into a dense work vector.

/// Scatter sparse column into dense work vector
///
/// work[row_indices[i]] = values[i] for i in 0..nnz
///
/// # Arguments
///
/// * `values` - Sparse column values
/// * `row_indices` - Row indices for each value (i64)
/// * `work` - Dense work vector, modified in place
#[inline]
pub fn scatter_column(values: &[f64], row_indices: &[i64], work: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                scatter_column_avx2(values, row_indices, work);
            }
            return;
        }
    }
    scatter_column_scalar(values, row_indices, work);
}

/// Scatter with i32 indices (for GPU compatibility)
#[inline]
pub fn scatter_column_i32(values: &[f64], row_indices: &[i32], work: &mut [f64]) {
    for (i, &val) in values.iter().enumerate() {
        let row = row_indices[i] as usize;
        work[row] = val;
    }
}

/// Scalar scatter implementation
#[inline]
fn scatter_column_scalar(values: &[f64], row_indices: &[i64], work: &mut [f64]) {
    for (i, &val) in values.iter().enumerate() {
        let row = row_indices[i] as usize;
        work[row] = val;
    }
}

/// SIMD-accelerated scatter (AVX2)
///
/// Note: True scatter requires AVX-512. This unrolls for better ILP.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scatter_column_avx2(values: &[f64], row_indices: &[i64], work: &mut [f64]) {
    let n = values.len();
    let mut i = 0;

    // Process 4 elements at a time (unrolled for ILP)
    while i + 4 <= n {
        let row0 = row_indices[i] as usize;
        let row1 = row_indices[i + 1] as usize;
        let row2 = row_indices[i + 2] as usize;
        let row3 = row_indices[i + 3] as usize;

        work[row0] = values[i];
        work[row1] = values[i + 1];
        work[row2] = values[i + 2];
        work[row3] = values[i + 3];

        i += 4;
    }

    // Handle remainder
    while i < n {
        let row = row_indices[i] as usize;
        work[row] = values[i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scatter_column() {
        let values = vec![1.0, 2.0, 3.0];
        let row_indices = vec![0i64, 2, 4];
        let mut work = vec![0.0; 6];

        scatter_column(&values, &row_indices, &mut work);

        assert_eq!(work[0], 1.0);
        assert_eq!(work[1], 0.0);
        assert_eq!(work[2], 2.0);
        assert_eq!(work[3], 0.0);
        assert_eq!(work[4], 3.0);
        assert_eq!(work[5], 0.0);
    }

    #[test]
    fn test_scatter_large() {
        let n = 100;
        let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let row_indices: Vec<i64> = (0..n).map(|i| (i * 2) as i64).collect();
        let mut work = vec![0.0; n * 2];

        scatter_column(&values, &row_indices, &mut work);

        for i in 0..n {
            assert_eq!(work[i * 2], i as f64);
            assert_eq!(work[i * 2 + 1], 0.0);
        }
    }
}
