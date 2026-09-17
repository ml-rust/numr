//! Fused matmul + bias kernel: `matmul_bias_kernel`.

use super::super::wide_acc::WideAcc;
use crate::dtype::Element;

/// Fused matrix multiplication with bias addition: C = A @ B + bias
///
/// Single-pass implementation that initializes C with bias, then accumulates
/// the matmul result. This is more cache-efficient than separate matmul + bias
/// because it avoids an extra memory round-trip through the output matrix.
///
/// # Arguments
/// * `a` - Pointer to matrix A (m × k), row-major with leading dimension lda
/// * `b` - Pointer to matrix B (k × n), row-major with leading dimension ldb
/// * `bias` - Pointer to bias vector (n elements, broadcast across rows)
/// * `out` - Pointer to output matrix C (m × n), row-major with leading dimension ldc
/// * `m`, `n`, `k` - Matrix dimensions
/// * `lda`, `ldb`, `ldc` - Leading dimensions (row stride in elements)
///
/// # Safety
/// - All pointers must be valid for the specified dimensions and strides
/// - `out` must not alias with `a`, `b`, or `bias`
/// - `bias` must have at least `n` elements
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bias_kernel<T: Element>(
    a: *const T,
    b: *const T,
    bias: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    // Same accumulator-width rule as `matmul_kernel`: the bias is only the
    // starting value of a dot product that still has to be accumulated wide.
    if T::DTYPE.is_int() {
        matmul_bias_scalar_acc::<T, i128>(a, b, bias, out, m, n, k, lda, ldb, ldc);
        return;
    }

    // Dispatch to fused SIMD for f32/f64, f16/bf16 via f32 conversion.
    // Gated on both architectures for the reason in `matmul_kernel`.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::super::simd::matmul;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                matmul::matmul_bias_f32(
                    a as *const f32,
                    b as *const f32,
                    bias as *const f32,
                    out as *mut f32,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                );
                return;
            }
            DType::F64 => {
                matmul::matmul_bias_f64(
                    a as *const f64,
                    b as *const f64,
                    bias as *const f64,
                    out as *mut f64,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => {
                matmul::half_convert::matmul_bias_via_f32(a, b, bias, out, m, n, k, lda, ldb, ldc);
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    if T::DTYPE.is_narrow_float() {
        matmul_bias_scalar_acc::<T, f32>(a, b, bias, out, m, n, k, lda, ldb, ldc);
        return;
    }

    // Scalar fallback with fused bias
    matmul_bias_scalar(a, b, bias, out, m, n, k, lda, ldb, ldc);
}

/// Fused matmul + bias with a wide accumulator.
///
/// # Safety
/// Same as [`matmul_bias_kernel`].
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_bias_scalar_acc<T: Element, A: WideAcc>(
    a: *const T,
    b: *const T,
    bias: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    let mut row_acc = vec![A::ZERO; n];

    for i in 0..m {
        for (j, slot) in row_acc.iter_mut().enumerate() {
            *slot = A::from_elem(*bias.add(j));
        }

        for kk in 0..k {
            let a_val = A::from_elem(*a.add(i * lda + kk));
            for (j, slot) in row_acc.iter_mut().enumerate() {
                let prod = a_val.wide_mul(A::from_elem(*b.add(kk * ldb + j)));
                *slot = slot.wide_add(prod);
            }
        }

        for (j, slot) in row_acc.iter().enumerate() {
            *out.add(i * ldc + j) = slot.to_elem::<T>();
        }
    }
}

/// Scalar matmul with fused bias for all Element types
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_bias_scalar<T: Element>(
    a: *const T,
    b: *const T,
    bias: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    // Initialize output with bias (single write pass)
    for i in 0..m {
        for j in 0..n {
            *out.add(i * ldc + j) = *bias.add(j);
        }
    }

    // Accumulate matmul result (ikj order for cache locality)
    for i in 0..m {
        for kk in 0..k {
            let a_val = *a.add(i * lda + kk);
            for j in 0..n {
                let b_val = *b.add(kk * ldb + j);
                let out_ptr = out.add(i * ldc + j);
                *out_ptr = *out_ptr + a_val * b_val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same accumulator defect through the fused bias kernel.
    ///
    /// The bias is only the starting value of a dot product that still has to
    /// be accumulated wide, so an i32 accumulator fails here for exactly the
    /// reason it fails in `matmul_kernel`.
    #[test]
    fn test_matmul_bias_i32_saturates_instead_of_wrapping() {
        let a = [2_000_000_000i32, 2_000_000_000];
        let b = [1i32, 1, 1, -1];
        let bias = [7i32, 7];
        let mut c = [0i32; 2];

        unsafe {
            matmul_bias_kernel(
                a.as_ptr(),
                b.as_ptr(),
                bias.as_ptr(),
                c.as_mut_ptr(),
                1,
                2,
                2,
                2,
                2,
                2,
            )
        };
        assert_eq!(c, [i32::MAX, 7]);
    }
}
