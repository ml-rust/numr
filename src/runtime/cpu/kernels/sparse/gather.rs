//! Gather kernels for sparse operations
//!
//! Gather values from dense work vector to sparse output, with clearing.

/// Gather nonzeros from work vector and clear
///
/// `output[i]` = `work[row_indices[i]]`, then `work[row_indices[i]]` = 0.0
///
/// # Arguments
///
/// * `work` - Dense work vector, cleared after gather
/// * `row_indices` - Row indices to gather (i64)
/// * `output` - Output sparse values
#[inline]
pub fn gather_and_clear(work: &mut [f64], row_indices: &[i64], output: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                gather_and_clear_avx2(work, row_indices, output);
            }
            return;
        }
    }
    gather_and_clear_scalar(work, row_indices, output);
}

/// Gather with i32 indices (for GPU compatibility)
#[inline]
pub fn gather_and_clear_i32(work: &mut [f64], row_indices: &[i32], output: &mut [f64]) {
    for (i, &row_idx) in row_indices.iter().enumerate() {
        let row = row_idx as usize;
        output[i] = work[row];
        work[row] = 0.0;
    }
}

/// Scalar gather and clear implementation
#[inline]
fn gather_and_clear_scalar(work: &mut [f64], row_indices: &[i64], output: &mut [f64]) {
    for (i, &row_idx) in row_indices.iter().enumerate() {
        let row = row_idx as usize;
        output[i] = work[row];
        work[row] = 0.0;
    }
}

/// SIMD-accelerated gather and clear (AVX2)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gather_and_clear_avx2(work: &mut [f64], row_indices: &[i64], output: &mut [f64]) {
    let n = row_indices.len();
    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= n {
        let row0 = row_indices[i] as usize;
        let row1 = row_indices[i + 1] as usize;
        let row2 = row_indices[i + 2] as usize;
        let row3 = row_indices[i + 3] as usize;

        // Gather values
        output[i] = work[row0];
        output[i + 1] = work[row1];
        output[i + 2] = work[row2];
        output[i + 3] = work[row3];

        // Clear work vector
        work[row0] = 0.0;
        work[row1] = 0.0;
        work[row2] = 0.0;
        work[row3] = 0.0;

        i += 4;
    }

    // Handle remainder
    while i < n {
        let row = row_indices[i] as usize;
        output[i] = work[row];
        work[row] = 0.0;
        i += 1;
    }
}

/// Divide elements by pivot value
///
/// `work[row_indices[i]]` /= `pivot` for i in 0..nnz
#[inline]
pub fn divide_by_pivot(work: &mut [f64], row_indices: &[i64], pivot: f64) {
    let inv_pivot = 1.0 / pivot;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                divide_by_pivot_avx2(work, row_indices, inv_pivot);
            }
            return;
        }
    }
    divide_by_pivot_scalar(work, row_indices, inv_pivot);
}

/// Scalar divide by pivot
#[inline]
fn divide_by_pivot_scalar(work: &mut [f64], row_indices: &[i64], inv_pivot: f64) {
    for &row_idx in row_indices {
        let row = row_idx as usize;
        work[row] *= inv_pivot;
    }
}

/// SIMD-accelerated divide by pivot
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn divide_by_pivot_avx2(work: &mut [f64], row_indices: &[i64], inv_pivot: f64) {
    let n = row_indices.len();
    let mut i = 0;

    while i + 4 <= n {
        let row0 = row_indices[i] as usize;
        let row1 = row_indices[i + 1] as usize;
        let row2 = row_indices[i + 2] as usize;
        let row3 = row_indices[i + 3] as usize;

        work[row0] *= inv_pivot;
        work[row1] *= inv_pivot;
        work[row2] *= inv_pivot;
        work[row3] *= inv_pivot;

        i += 4;
    }

    while i < n {
        let row = row_indices[i] as usize;
        work[row] *= inv_pivot;
        i += 1;
    }
}

/// Swap two rows in work vector and permutation
#[inline]
pub fn swap_rows(work: &mut [f64], perm: &mut [usize], row_a: usize, row_b: usize) {
    if row_a != row_b {
        work.swap(row_a, row_b);
        perm.swap(row_a, row_b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_and_clear() {
        let mut work = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let row_indices = vec![0i64, 2, 4];
        let mut output = vec![0.0; 3];

        gather_and_clear(&mut work, &row_indices, &mut output);

        assert_eq!(output, vec![1.0, 2.0, 3.0]);
        assert_eq!(work, vec![0.0; 6]);
    }

    #[test]
    fn test_divide_by_pivot() {
        let mut work = vec![2.0, 0.0, 4.0, 0.0, 6.0, 0.0];
        let row_indices = vec![0i64, 2, 4];

        divide_by_pivot(&mut work, &row_indices, 2.0);

        assert_eq!(work[0], 1.0);
        assert_eq!(work[2], 2.0);
        assert_eq!(work[4], 3.0);
    }

    #[test]
    fn test_swap_rows() {
        let mut work = vec![1.0, 2.0, 3.0, 4.0];
        let mut perm = vec![0, 1, 2, 3];

        swap_rows(&mut work, &mut perm, 1, 3);

        assert_eq!(work, vec![1.0, 4.0, 3.0, 2.0]);
        assert_eq!(perm, vec![0, 3, 2, 1]);
    }

    #[test]
    fn test_gather_large() {
        let n = 100;
        let mut work: Vec<f64> = (0..n * 2)
            .map(|i| if i % 2 == 0 { (i / 2) as f64 } else { 0.0 })
            .collect();
        let row_indices: Vec<i64> = (0..n).map(|i| (i * 2) as i64).collect();
        let mut output = vec![0.0; n];

        gather_and_clear(&mut work, &row_indices, &mut output);

        for i in 0..n {
            assert_eq!(output[i], i as f64);
        }
        for &v in &work {
            assert_eq!(v, 0.0);
        }
    }
}
