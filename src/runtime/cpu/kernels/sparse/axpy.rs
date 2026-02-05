//! Sparse AXPY kernels
//!
//! Sparse AXPY: work[indices] -= scale * values
//! This is the hot loop in Gilbert-Peierls sparse LU.

/// Sparse AXPY: work[row_indices[i]] -= scale * values[i]
///
/// # Arguments
///
/// * `scale` - Scalar multiplier
/// * `values` - Sparse vector values
/// * `row_indices` - Row indices for each value (i64)
/// * `work` - Dense work vector, modified in place
#[inline]
pub fn sparse_axpy(scale: f64, values: &[f64], row_indices: &[i64], work: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                sparse_axpy_avx2(scale, values, row_indices, work);
            }
            return;
        }
    }
    sparse_axpy_scalar(scale, values, row_indices, work);
}

/// Sparse AXPY with i32 indices (for GPU compatibility)
#[inline]
pub fn sparse_axpy_i32(scale: f64, values: &[f64], row_indices: &[i32], work: &mut [f64]) {
    for (i, &val) in values.iter().enumerate() {
        let row = row_indices[i] as usize;
        work[row] -= scale * val;
    }
}

/// Scalar sparse AXPY implementation
#[inline]
fn sparse_axpy_scalar(scale: f64, values: &[f64], row_indices: &[i64], work: &mut [f64]) {
    for (i, &val) in values.iter().enumerate() {
        let row = row_indices[i] as usize;
        work[row] -= scale * val;
    }
}

/// SIMD-accelerated sparse AXPY (AVX2 + FMA)
///
/// Unrolled for better instruction-level parallelism.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sparse_axpy_avx2(scale: f64, values: &[f64], row_indices: &[i64], work: &mut [f64]) {
    let n = values.len();
    let mut i = 0;

    // Process 4 elements at a time with unrolling
    while i + 4 <= n {
        let row0 = row_indices[i] as usize;
        let row1 = row_indices[i + 1] as usize;
        let row2 = row_indices[i + 2] as usize;
        let row3 = row_indices[i + 3] as usize;

        // Load, compute, store (unrolled for better ILP)
        let v0 = values[i];
        let v1 = values[i + 1];
        let v2 = values[i + 2];
        let v3 = values[i + 3];

        work[row0] -= scale * v0;
        work[row1] -= scale * v1;
        work[row2] -= scale * v2;
        work[row3] -= scale * v3;

        i += 4;
    }

    // Handle remainder
    while i < n {
        let row = row_indices[i] as usize;
        work[row] -= scale * values[i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_axpy() {
        let values = vec![1.0, 2.0, 3.0];
        let row_indices = vec![0i64, 2, 4];
        let mut work = vec![10.0, 0.0, 20.0, 0.0, 30.0, 0.0];

        sparse_axpy(2.0, &values, &row_indices, &mut work);

        assert_eq!(work[0], 10.0 - 2.0 * 1.0); // 8.0
        assert_eq!(work[2], 20.0 - 2.0 * 2.0); // 16.0
        assert_eq!(work[4], 30.0 - 2.0 * 3.0); // 24.0
    }

    #[test]
    fn test_sparse_axpy_large() {
        let n = 100;
        let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let row_indices: Vec<i64> = (0..n).map(|i| i as i64).collect();
        let mut work: Vec<f64> = (0..n).map(|i| (i * 10) as f64).collect();
        let scale = 0.5;

        sparse_axpy(scale, &values, &row_indices, &mut work);

        for i in 0..n {
            let expected = (i * 10) as f64 - scale * (i as f64);
            assert!((work[i] - expected).abs() < 1e-10);
        }
    }
}
