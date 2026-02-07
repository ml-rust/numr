//! Semiring matrix multiplication kernels
//!
//! Fused triple-loop kernel parameterized by semiring operations.
//! O(n²) memory vs O(n³) for decomposed unsqueeze+broadcast+reduce.

use crate::dtype::Element;
use crate::ops::semiring::SemiringOp;

/// Semiring matrix multiplication kernel: C[i,j] = ⊕_k (A[i,k] ⊗ B[k,j])
///
/// # Arguments
/// * `a` - Pointer to matrix A (m × k), row-major with leading dimension lda
/// * `b` - Pointer to matrix B (k × n), row-major with leading dimension ldb
/// * `out` - Pointer to output matrix C (m × n), row-major with leading dimension ldc
/// * `m`, `n`, `k` - Matrix dimensions
/// * `lda`, `ldb`, `ldc` - Leading dimensions (row stride in elements)
/// * `op` - The semiring operation
///
/// # Safety
/// - All pointers must be valid for the specified dimensions and strides
/// - `out` must not alias with `a` or `b`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn semiring_matmul_kernel<T: Element>(
    a: *const T,
    b: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    op: SemiringOp,
) {
    let identity = op.reduce_identity::<T>();

    // Initialize output with reduce identity
    for i in 0..m {
        for j in 0..n {
            *out.add(i * ldc + j) = identity;
        }
    }

    // Fused semiring matmul
    // Using ijk order for simplicity; the reduce is accumulated per (i,j).
    for i in 0..m {
        for j in 0..n {
            let mut acc = identity;
            for kk in 0..k {
                let a_val = *a.add(i * lda + kk);
                let b_val = *b.add(kk * ldb + j);
                let combined = op.combine(a_val, b_val);
                acc = op.reduce(acc, combined);
            }
            *out.add(i * ldc + j) = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_plus_2x2() {
        // Shortest path: A = [[0, 3], [7, 1]], B = [[0, 2], [5, 0]]
        // C[0,0] = min(0+0, 3+5) = min(0, 8) = 0
        // C[0,1] = min(0+2, 3+0) = min(2, 3) = 2
        // C[1,0] = min(7+0, 1+5) = min(7, 6) = 6
        // C[1,1] = min(7+2, 1+0) = min(9, 1) = 1
        let a = [0.0f32, 3.0, 7.0, 1.0];
        let b = [0.0f32, 2.0, 5.0, 0.0];
        let mut c = [0.0f32; 4];

        unsafe {
            semiring_matmul_kernel(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                2,
                2,
                2,
                2,
                2,
                2,
                SemiringOp::MinPlus,
            );
        }

        assert_eq!(c, [0.0, 2.0, 6.0, 1.0]);
    }

    #[test]
    fn test_max_plus_2x2() {
        // Longest path: A = [[0, 3], [7, 1]], B = [[0, 2], [5, 0]]
        // C[0,0] = max(0+0, 3+5) = max(0, 8) = 8
        // C[0,1] = max(0+2, 3+0) = max(2, 3) = 3
        // C[1,0] = max(7+0, 1+5) = max(7, 6) = 7
        // C[1,1] = max(7+2, 1+0) = max(9, 1) = 9
        let a = [0.0f32, 3.0, 7.0, 1.0];
        let b = [0.0f32, 2.0, 5.0, 0.0];
        let mut c = [0.0f32; 4];

        unsafe {
            semiring_matmul_kernel(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                2,
                2,
                2,
                2,
                2,
                2,
                SemiringOp::MaxPlus,
            );
        }

        assert_eq!(c, [8.0, 3.0, 7.0, 9.0]);
    }

    #[test]
    fn test_max_min_2x2() {
        // Bottleneck: A = [[5, 3], [2, 8]], B = [[4, 1], [6, 7]]
        // C[0,0] = max(min(5,4), min(3,6)) = max(4, 3) = 4
        // C[0,1] = max(min(5,1), min(3,7)) = max(1, 3) = 3
        // C[1,0] = max(min(2,4), min(8,6)) = max(2, 6) = 6
        // C[1,1] = max(min(2,1), min(8,7)) = max(1, 7) = 7
        let a = [5.0f32, 3.0, 2.0, 8.0];
        let b = [4.0f32, 1.0, 6.0, 7.0];
        let mut c = [0.0f32; 4];

        unsafe {
            semiring_matmul_kernel(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                2,
                2,
                2,
                2,
                2,
                2,
                SemiringOp::MaxMin,
            );
        }

        assert_eq!(c, [4.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn test_min_max_2x2() {
        // Fuzzy: A = [[5, 3], [2, 8]], B = [[4, 1], [6, 7]]
        // C[0,0] = min(max(5,4), max(3,6)) = min(5, 6) = 5
        // C[0,1] = min(max(5,1), max(3,7)) = min(5, 7) = 5
        // C[1,0] = min(max(2,4), max(8,6)) = min(4, 8) = 4
        // C[1,1] = min(max(2,1), max(8,7)) = min(2, 8) = 2
        let a = [5.0f32, 3.0, 2.0, 8.0];
        let b = [4.0f32, 1.0, 6.0, 7.0];
        let mut c = [0.0f32; 4];

        unsafe {
            semiring_matmul_kernel(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                2,
                2,
                2,
                2,
                2,
                2,
                SemiringOp::MinMax,
            );
        }

        assert_eq!(c, [5.0, 5.0, 4.0, 2.0]);
    }

    #[test]
    fn test_non_square() {
        // MinPlus: A (2x3) @ B (3x2)
        // A = [[1, 2, 3], [4, 5, 6]]
        // B = [[7, 8], [9, 10], [11, 12]]
        // C[0,0] = min(1+7, 2+9, 3+11) = min(8, 11, 14) = 8
        // C[0,1] = min(1+8, 2+10, 3+12) = min(9, 12, 15) = 9
        // C[1,0] = min(4+7, 5+9, 6+11) = min(11, 14, 17) = 11
        // C[1,1] = min(4+8, 5+10, 6+12) = min(12, 15, 18) = 12
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = [0.0f32; 4];

        unsafe {
            semiring_matmul_kernel(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                2,
                2,
                3,
                3,
                2,
                2,
                SemiringOp::MinPlus,
            );
        }

        assert_eq!(c, [8.0, 9.0, 11.0, 12.0]);
    }

    #[test]
    fn test_i32_min_plus() {
        let a = [0i32, 3, 7, 1];
        let b = [0i32, 2, 5, 0];
        let mut c = [0i32; 4];

        unsafe {
            semiring_matmul_kernel(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                2,
                2,
                2,
                2,
                2,
                2,
                SemiringOp::MinPlus,
            );
        }

        assert_eq!(c, [0, 2, 6, 1]);
    }
}
