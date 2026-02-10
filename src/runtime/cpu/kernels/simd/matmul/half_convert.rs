//! f16/bf16 matmul via f32 conversion
//!
//! f16 and bf16 are storage formats. Computation happens in f32 using the
//! optimized SIMD tiled matmul, with conversion at the boundaries:
//!
//! 1. Convert A (f16/bf16) → A_f32
//! 2. Convert B (f16/bf16) → B_f32
//! 3. Compute C_f32 = A_f32 @ B_f32 (uses full SIMD pipeline)
//! 4. Convert C_f32 → C (f16/bf16)
//!
//! This is significantly faster than scalar f16 arithmetic because:
//! - The inner matmul uses AVX-512/AVX2 SIMD with cache-aware tiling
//! - Conversion overhead is O(m*k + k*n + m*n) vs O(m*n*k) compute

use crate::dtype::Element;

/// Convert a strided half-precision matrix to a contiguous f32 buffer.
///
/// # Safety
/// - `src` must be valid for reads of `rows * ld` elements
/// - `dst` must be valid for writes of `rows * cols` elements
#[inline]
unsafe fn convert_to_f32<T: Element>(
    src: *const T,
    dst: *mut f32,
    rows: usize,
    cols: usize,
    ld: usize,
) {
    for i in 0..rows {
        let src_row = src.add(i * ld);
        let dst_row = dst.add(i * cols);
        for j in 0..cols {
            *dst_row.add(j) = (*src_row.add(j)).to_f32();
        }
    }
}

/// Convert a contiguous f32 buffer back to a strided half-precision matrix.
///
/// # Safety
/// - `src` must be valid for reads of `rows * cols` elements
/// - `dst` must be valid for writes of `rows * ld` elements
#[inline]
unsafe fn convert_from_f32<T: Element>(
    src: *const f32,
    dst: *mut T,
    rows: usize,
    cols: usize,
    ld: usize,
) {
    for i in 0..rows {
        let src_row = src.add(i * cols);
        let dst_row = dst.add(i * ld);
        for j in 0..cols {
            *dst_row.add(j) = T::from_f32(*src_row.add(j));
        }
    }
}

/// f16/bf16 matmul via f32 SIMD path: C = A @ B
///
/// Converts inputs to f32 via `Element::to_f32()`, runs the optimized SIMD
/// matmul, then converts back via `Element::from_f32()`. Direct f32 conversion
/// is lossless for f16 (10-bit mantissa) and bf16 (7-bit mantissa).
///
/// # Safety
/// - `a` must be valid for reads of `m * lda` elements
/// - `b` must be valid for reads of `k * ldb` elements
/// - `out` must be valid for writes of `m * ldc` elements
/// - `out` must not alias with `a` or `b`
/// - All pointers must be aligned for type `T`
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_via_f32<T: Element>(
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
    let mut a_f32 = vec![0.0f32; m * k];
    let mut b_f32 = vec![0.0f32; k * n];
    let mut c_f32 = vec![0.0f32; m * n];

    convert_to_f32(a, a_f32.as_mut_ptr(), m, k, lda);
    convert_to_f32(b, b_f32.as_mut_ptr(), k, n, ldb);

    super::matmul_f32(
        a_f32.as_ptr(),
        b_f32.as_ptr(),
        c_f32.as_mut_ptr(),
        m,
        n,
        k,
        k,
        n,
        n,
    );

    convert_from_f32(c_f32.as_ptr(), out, m, n, ldc);
}

/// f16/bf16 fused matmul + bias via f32 SIMD path: C = A @ B + bias
///
/// See [`matmul_via_f32`] for details on the conversion approach.
///
/// # Safety
/// - `a` must be valid for reads of `m * lda` elements
/// - `b` must be valid for reads of `k * ldb` elements
/// - `bias` must be valid for reads of `n` elements
/// - `out` must be valid for writes of `m * ldc` elements
/// - `out` must not alias with `a`, `b`, or `bias`
/// - All pointers must be aligned for type `T`
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bias_via_f32<T: Element>(
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
    let mut a_f32 = vec![0.0f32; m * k];
    let mut b_f32 = vec![0.0f32; k * n];
    let mut bias_f32 = vec![0.0f32; n];
    let mut c_f32 = vec![0.0f32; m * n];

    convert_to_f32(a, a_f32.as_mut_ptr(), m, k, lda);
    convert_to_f32(b, b_f32.as_mut_ptr(), k, n, ldb);
    convert_to_f32(bias, bias_f32.as_mut_ptr(), 1, n, n);

    super::matmul_bias_f32(
        a_f32.as_ptr(),
        b_f32.as_ptr(),
        bias_f32.as_ptr(),
        c_f32.as_mut_ptr(),
        m,
        n,
        k,
        k,
        n,
        n,
    );

    convert_from_f32(c_f32.as_ptr(), out, m, n, ldc);
}
