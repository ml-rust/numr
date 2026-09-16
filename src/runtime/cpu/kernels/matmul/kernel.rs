//! The general matmul kernels: `matmul_kernel` and `matmul_bias_kernel`.

use super::super::wide_acc::WideAcc;
use crate::dtype::Element;

/// Matrix multiplication with automatic SIMD dispatch: C = A @ B
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64:
/// - AVX-512: 6×16 f32 microkernel, 6×8 f64 microkernel
/// - AVX2+FMA: 6×8 f32 microkernel, 6×4 f64 microkernel
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Arguments
/// * `a` - Pointer to matrix A (m × k), row-major with leading dimension lda
/// * `b` - Pointer to matrix B (k × n), row-major with leading dimension ldb
/// * `out` - Pointer to output matrix C (m × n), row-major with leading dimension ldc
/// * `m`, `n`, `k` - Matrix dimensions
/// * `lda`, `ldb`, `ldc` - Leading dimensions (row stride in elements)
///
/// # Safety
/// - All pointers must be valid for the specified dimensions and strides
/// - `out` must not alias with `a` or `b`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_kernel<T: Element>(
    a: *const T,
    b: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    // Integers accumulate exactly, never in the element type. An older AVX2 i32
    // kernel used `_mm256_add_epi32` and wrapped mid-dot-product, reporting the
    // wrong sign even when the final result was representable.
    //
    // i32 on AVX2 gets a 64-bit accumulator when a magnitude prescan proves
    // every partial sum fits; the scan is O(mk + kn) against the matmul's
    // O(mnk). Everything else — every other integer dtype, non-AVX2 hardware,
    // and operands the scan rejects — takes the exact i128 scalar path. Both
    // produce the same clamped result, so which one ran is invisible.
    if T::DTYPE.is_int() {
        #[cfg(target_arch = "x86_64")]
        {
            use crate::dtype::DType;
            if T::DTYPE == DType::I32 && std::arch::is_x86_feature_detected!("avx2") {
                let (ai, bi) = (a as *const i32, b as *const i32);
                if super::super::simd::matmul::int32::matmul_i32_fits_i64(ai, bi, m, n, k, lda, ldb)
                {
                    super::super::simd::matmul::int32::matmul_i32_avx2(
                        ai,
                        bi,
                        out as *mut i32,
                        m,
                        n,
                        k,
                        lda,
                        ldb,
                        ldc,
                    );
                    return;
                }
            }
        }
        matmul_scalar_acc::<T, i128, T>(a, b, out, m, n, k, lda, ldb, ldc);
        return;
    }

    // Dispatch to SIMD for f32/f64, f16/bf16 via f32 conversion.
    //
    // aarch64 MUST be here, not just x86_64. `matmul_bt_kernel` already gates on
    // both, and `matmul_bt_matches_contiguous` promises the two agree bit for
    // bit wherever the tiled path runs. Leaving ARM out sent contiguous down the
    // scalar path while transposed ran the NEON tiled kernel: same maths, a
    // different summation order, and a 1-ULP disagreement that broke that
    // promise. `matmul_f32` already dispatches per architecture internally.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::super::simd::matmul;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                matmul::matmul_f32(
                    a as *const f32,
                    b as *const f32,
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
                matmul::matmul_f64(
                    a as *const f64,
                    b as *const f64,
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
                matmul::half_convert::matmul_via_f32(a, b, out, m, n, k, lda, ldb, ldc);
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // FP8 has no SIMD path on any architecture, and F16/BF16 reach here on
    // architectures without the block above. All of them saturate long before a
    // dot product ends if they accumulate in themselves.
    if T::DTYPE.is_narrow_float() {
        matmul_scalar_acc::<T, f32, T>(a, b, out, m, n, k, lda, ldb, ldc);
        return;
    }

    // Scalar fallback for non-SIMD types or non-x86 platforms
    matmul_scalar(a, b, out, m, n, k, lda, ldb, ldc);
}

/// Matmul with a wide accumulator, for element types that cannot hold the
/// running dot product.
///
/// Keeps the `ikj` loop order of [`matmul_scalar`] for cache locality by
/// holding one output row of accumulators, then storing that row once. The
/// store narrows to `O`; `matmul_kernel` passes `O = T`, and `matmul_wide`
/// passes the accumulator type itself so nothing is narrowed.
///
/// # Safety
/// Same as [`matmul_kernel`], with `out` valid for `m * ldc` elements of `O`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn matmul_scalar_acc<T: Element, A: WideAcc, O: Element>(
    a: *const T,
    b: *const T,
    out: *mut O,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    let mut row_acc = vec![A::ZERO; n];

    for i in 0..m {
        for slot in row_acc.iter_mut() {
            *slot = A::ZERO;
        }

        for kk in 0..k {
            let a_val = A::from_elem(*a.add(i * lda + kk));
            for (j, slot) in row_acc.iter_mut().enumerate() {
                let prod = a_val.wide_mul(A::from_elem(*b.add(kk * ldb + j)));
                *slot = slot.wide_add(prod);
            }
        }

        for (j, slot) in row_acc.iter().enumerate() {
            *out.add(i * ldc + j) = slot.to_elem::<O>();
        }
    }
}

/// Scalar matmul implementation for all Element types
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_scalar<T: Element>(
    a: *const T,
    b: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    // Zero output first
    for i in 0..m {
        for j in 0..n {
            *out.add(i * ldc + j) = T::zero();
        }
    }

    // ikj order: better cache locality for B
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

    #[test]
    fn test_matmul_i32_basic() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = [[19, 22], [43, 50]]
        let a = [1i32, 2, 3, 4];
        let b = [5i32, 6, 7, 8];
        let mut c = [0i32; 4];

        unsafe { matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 2, 2, 2, 2, 2) };
        assert_eq!(c, [19, 22, 43, 50]);
    }

    #[test]
    fn test_matmul_i32_non_square() {
        // A(3x2) @ B(2x4) = C(3x4)
        let a = [1i32, 2, 3, 4, 5, 6];
        let b = [1i32, 2, 3, 4, 5, 6, 7, 8];
        let mut c = [0i32; 12];

        unsafe { matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 3, 4, 2, 2, 4, 4) };
        assert_eq!(c, [11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68]);
    }

    #[test]
    fn test_matmul_i32_wide() {
        // n > 8: the width that used to select the AVX2 i32 microkernel.
        let (m, n, k) = (2, 16, 3);
        let a: Vec<i32> = (0..m * k).map(|i| (i + 1) as i32).collect();
        let b: Vec<i32> = (0..k * n).map(|i| (i + 1) as i32).collect();
        let mut c = vec![0i32; m * n];

        unsafe { matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, k, n, n) };

        let mut expected = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                for kk in 0..k {
                    expected[i * n + j] += a[i * k + kk] * b[kk * n + j];
                }
            }
        }
        assert_eq!(c, expected);
    }

    /// Catches an i32 matmul accumulator.
    ///
    /// Column 0's dot product is 4_000_000_000, which i32 cannot hold. An i32
    /// accumulator panics on the overflow in a debug build, and in a release
    /// build wraps to -294_967_296 where the documented answer is the saturated
    /// `i32::MAX`. Column 1 stays in range and pins that ordinary results are
    /// untouched.
    #[test]
    fn test_matmul_i32_saturates_instead_of_wrapping() {
        let a = [2_000_000_000i32, 2_000_000_000];
        let b = [1i32, 1, 1, -1];
        let mut c = [0i32; 2];

        unsafe { matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 1, 2, 2, 2, 2, 2) };
        assert_eq!(c, [i32::MAX, 0]);
    }

    /// Catches an FP8 matmul accumulator.
    ///
    /// A length-32 dot product of ones is 32. Accumulated in FP8E4M3 the
    /// running sum stalls at 16, because above 16 the format's spacing is 2 and
    /// `16 + 1` rounds back to 16.
    #[test]
    fn test_matmul_fp8_accumulates_in_f32() {
        use crate::dtype::FP8E4M3;

        let a = [FP8E4M3::from_f32(1.0); 32];
        let b = [FP8E4M3::from_f32(1.0); 32];
        let mut c = [FP8E4M3::from_f32(0.0); 1];

        unsafe { matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 1, 1, 32, 32, 1, 1) };
        assert_eq!(c[0].to_f32(), 32.0);
    }

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
