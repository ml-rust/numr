//! Pivot finding kernels for sparse LU
//!
//! Find the index of maximum absolute value for partial pivoting.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Find index of maximum absolute value in work[start..end]
///
/// Returns (index, max_abs_value) where index is in [start, end)
///
/// # Arguments
///
/// * `work` - Dense work vector
/// * `start` - Start index (inclusive)
/// * `end` - End index (exclusive)
#[inline]
pub fn find_pivot(work: &[f64], start: usize, end: usize) -> (usize, f64) {
    if start >= end {
        return (start, 0.0);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return find_pivot_avx2(work, start, end);
            }
        }
    }
    find_pivot_scalar(work, start, end)
}

/// Find pivot in a range specified by indices
///
/// Returns (index, max_abs_value) where index is the actual row index
#[inline]
pub fn find_pivot_range(work: &[f64], indices: &[i64]) -> (usize, f64) {
    if indices.is_empty() {
        return (0, 0.0);
    }

    let mut max_idx = indices[0] as usize;
    let mut max_val = work[max_idx].abs();

    for &idx in indices.iter().skip(1) {
        let i = idx as usize;
        let abs_val = work[i].abs();
        if abs_val > max_val {
            max_val = abs_val;
            max_idx = i;
        }
    }

    (max_idx, max_val)
}

/// Scalar pivot search implementation
#[inline]
fn find_pivot_scalar(work: &[f64], start: usize, end: usize) -> (usize, f64) {
    let mut max_idx = start;
    let mut max_val = work[start].abs();

    for i in (start + 1)..end {
        let abs_val = work[i].abs();
        if abs_val > max_val {
            max_val = abs_val;
            max_idx = i;
        }
    }

    (max_idx, max_val)
}

/// SIMD-accelerated pivot search (AVX2)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_pivot_avx2(work: &[f64], start: usize, end: usize) -> (usize, f64) {
    let n = end - start;
    if n < 8 {
        return find_pivot_scalar(work, start, end);
    }

    // Mask for absolute value (clear sign bit)
    let abs_mask = _mm256_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));

    let mut i = start;
    let mut max_idx_scalar = start;
    let mut max_val_scalar = 0.0f64;

    // Process 4 doubles at a time
    while i + 4 <= end {
        // SAFETY: We checked i + 4 <= end and work has length >= end
        let vals = unsafe { _mm256_loadu_pd(work.as_ptr().add(i)) };
        let abs_vals = _mm256_and_pd(vals, abs_mask);

        // Horizontal max within vector
        let max_in_vec = {
            let temp = _mm256_max_pd(abs_vals, _mm256_permute4x64_pd(abs_vals, 0b01_00_11_10));
            let temp2 = _mm256_max_pd(temp, _mm256_permute_pd(temp, 0b0101));
            _mm256_cvtsd_f64(temp2)
        };

        if max_in_vec > max_val_scalar {
            // Find which index has the max
            let mut arr = [0.0f64; 4];
            // SAFETY: arr is a properly aligned f64 array
            unsafe { _mm256_storeu_pd(arr.as_mut_ptr(), abs_vals) };
            for (j, &v) in arr.iter().enumerate() {
                if v > max_val_scalar {
                    max_val_scalar = v;
                    max_idx_scalar = i + j;
                }
            }
        }

        i += 4;
    }

    // Handle remainder
    while i < end {
        let abs_val = work[i].abs();
        if abs_val > max_val_scalar {
            max_val_scalar = abs_val;
            max_idx_scalar = i;
        }
        i += 1;
    }

    (max_idx_scalar, max_val_scalar)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_pivot() {
        let work = vec![1.0, -5.0, 3.0, -2.0, 4.0];

        let (idx, val) = find_pivot(&work, 0, 5);
        assert_eq!(idx, 1);
        assert_eq!(val, 5.0);

        let (idx, val) = find_pivot(&work, 2, 5);
        assert_eq!(idx, 4);
        assert_eq!(val, 4.0);
    }

    #[test]
    fn test_find_pivot_large() {
        let mut work: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1).collect();
        work[73] = 999.0;

        let (idx, val) = find_pivot(&work, 0, 100);
        assert_eq!(idx, 73);
        assert!((val - 999.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_pivot_negative() {
        let work = vec![1.0, -10.0, 5.0, -3.0];

        let (idx, val) = find_pivot(&work, 0, 4);
        assert_eq!(idx, 1);
        assert_eq!(val, 10.0);
    }

    #[test]
    fn test_find_pivot_range() {
        let work = vec![1.0, -5.0, 3.0, -2.0, 4.0, 10.0];
        let indices = vec![1i64, 3, 5];

        let (idx, val) = find_pivot_range(&work, &indices);
        assert_eq!(idx, 5);
        assert_eq!(val, 10.0);
    }
}
