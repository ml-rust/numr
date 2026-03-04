//! Matrix multiplication kernels
//!
//! This module provides matrix multiplication with automatic SIMD dispatch.
//! On x86-64, f32 and f64 matmuls use AVX-512 or AVX2+FMA when available.

use crate::dtype::{DType, Element};

/// SIMD-accelerated f32 dot product for use in half-precision GEMV-BT.
///
/// Dispatches to AVX-512 or AVX2+FMA based on detected SIMD level.
///
/// # Safety
/// - `a` and `b` must be valid pointers to `len` f32 elements
#[cfg(all(feature = "f16", target_arch = "x86_64"))]
#[inline]
unsafe fn simd_dot_f32(
    a: *const f32,
    b: *const f32,
    len: usize,
    level: super::simd::SimdLevel,
) -> f32 {
    use super::simd::SimdLevel;

    match level {
        SimdLevel::Avx512 => {
            use std::arch::x86_64::*;
            let mut offset = 0;
            let mut acc = _mm512_setzero_ps();
            while offset + 16 <= len {
                let av = _mm512_loadu_ps(a.add(offset));
                let bv = _mm512_loadu_ps(b.add(offset));
                acc = _mm512_fmadd_ps(av, bv, acc);
                offset += 16;
            }
            let mut sum = _mm512_reduce_add_ps(acc);
            while offset < len {
                sum += *a.add(offset) * *b.add(offset);
                offset += 1;
            }
            sum
        }
        SimdLevel::Avx2Fma => {
            use std::arch::x86_64::*;
            let mut offset = 0;
            let mut acc = _mm256_setzero_ps();
            while offset + 8 <= len {
                let av = _mm256_loadu_ps(a.add(offset));
                let bv = _mm256_loadu_ps(b.add(offset));
                acc = _mm256_fmadd_ps(av, bv, acc);
                offset += 8;
            }
            // Horizontal sum of 256-bit accumulator
            let hi = _mm256_extractf128_ps(acc, 1);
            let lo = _mm256_castps256_ps128(acc);
            let sum128 = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            let sums2 = _mm_add_ss(sums, shuf2);
            let mut sum = _mm_cvtss_f32(sums2);
            while offset < len {
                sum += *a.add(offset) * *b.add(offset);
                offset += 1;
            }
            sum
        }
        _ => {
            let mut sum = 0.0f32;
            for i in 0..len {
                sum += *a.add(i) * *b.add(i);
            }
            sum
        }
    }
}

/// GEMV-BT kernel: C[M,N] = A[M,K] @ B^T where B is stored as contiguous [N,K]
///
/// This avoids the costly contiguous copy of transposed weight matrices during
/// decode (M=1). Both A rows and B rows are contiguous, making this ideal for
/// SIMD dot products.
///
/// # Arguments
/// * `a` - Pointer to matrix A (m × k), contiguous row-major
/// * `b_nk` - Pointer to B in [N,K] layout (NOT the transposed view)
/// * `out` - Pointer to output C (m × n), row-major with leading dimension ldc
/// * `m`, `n`, `k` - Matrix dimensions
/// * `ldc` - Leading dimension of output
///
/// # Safety
/// - `a` must be valid for m*k contiguous reads
/// - `b_nk` must be valid for n*k contiguous reads
/// - `out` must be valid for m*ldc writes
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_bt_kernel<T: Element>(
    a: *const T,
    b_nk: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        use super::simd::detect_simd;
        use super::simd::matmul::gemv_bt;

        match T::DTYPE {
            DType::F32 => {
                let level = detect_simd();
                gemv_bt::gemv_bt_f32(
                    a as *const f32,
                    b_nk as *const f32,
                    out as *mut f32,
                    m,
                    n,
                    k,
                    ldc,
                    level,
                );
                return;
            }
            DType::F64 => {
                let level = detect_simd();
                gemv_bt::gemv_bt_f64(
                    a as *const f64,
                    b_nk as *const f64,
                    out as *mut f64,
                    m,
                    n,
                    k,
                    ldc,
                    level,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => {
                gemv_bt_via_f32(a, b_nk, out, m, n, k, ldc);
                return;
            }
            _ => {}
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        match T::DTYPE {
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => {
                gemv_bt_via_f32(a, b_nk, out, m, n, k, ldc);
                return;
            }
            _ => {}
        }
    }

    // Scalar fallback
    gemv_bt_scalar(a, b_nk, out, m, n, k, ldc);
}

/// Scalar GEMV-BT fallback
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_bt_scalar<T: Element>(
    a: *const T,
    b_nk: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    for row in 0..m {
        let a_row = a.add(row * k);
        let out_row = out.add(row * ldc);
        for col in 0..n {
            let b_row = b_nk.add(col * k);
            let mut sum = T::zero();
            for i in 0..k {
                sum = sum + *a_row.add(i) * *b_row.add(i);
            }
            *out_row.add(col) = sum;
        }
    }
}

/// GEMV-BT for f16/bf16 via f32 conversion
///
/// Converts A row to f32 (batch SIMD conversion), then converts each B row
/// to f32 in SIMD chunks and uses the f32 AVX2/AVX-512 dot product.
#[cfg(feature = "f16")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_bt_via_f32<T: Element>(
    a: *const T,
    b_nk: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    // Convert A row to f32 once (small buffer, reused per row)
    let mut a_f32 = vec![0.0f32; k];
    let mut b_f32 = vec![0.0f32; k];

    #[cfg(target_arch = "x86_64")]
    let level = super::simd::detect_simd();

    for row in 0..m {
        let a_row = a.add(row * k);
        // Batch convert A row to f32
        batch_half_to_f32::<T>(a_row, a_f32.as_mut_ptr(), k);

        let out_row = out.add(row * ldc);

        for col in 0..n {
            let b_row = b_nk.add(col * k);
            // Batch convert B row to f32
            batch_half_to_f32::<T>(b_row, b_f32.as_mut_ptr(), k);

            // Use SIMD f32 dot product
            #[cfg(target_arch = "x86_64")]
            {
                let dot = simd_dot_f32(a_f32.as_ptr(), b_f32.as_ptr(), k, level);
                *out_row.add(col) = T::from_f32(dot);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let mut sum = 0.0f32;
                for i in 0..k {
                    sum += a_f32[i] * b_f32[i];
                }
                *out_row.add(col) = T::from_f32(sum);
            }
        }
    }
}

/// Batch convert half-precision (f16/bf16) elements to f32 using SIMD when available.
#[cfg(feature = "f16")]
#[inline]
unsafe fn batch_half_to_f32<T: Element>(src: *const T, dst: *mut f32, len: usize) {
    match T::DTYPE {
        #[cfg(target_arch = "x86_64")]
        DType::BF16 => {
            // BF16 → f32: shift left by 16 bits (bf16 is upper 16 bits of f32)
            batch_bf16_to_f32(src as *const u16, dst, len);
        }
        #[cfg(target_arch = "x86_64")]
        DType::F16 => {
            // F16 → f32: use F16C instruction if available
            batch_f16_to_f32(src as *const u16, dst, len);
        }
        _ => {
            for i in 0..len {
                *dst.add(i) = (*src.add(i)).to_f32();
            }
        }
    }
}

/// BF16 → f32 conversion using SIMD bit-shift (bf16 is just f32 with lower 16 bits zeroed)
#[cfg(all(feature = "f16", target_arch = "x86_64"))]
#[inline]
unsafe fn batch_bf16_to_f32(src: *const u16, dst: *mut f32, len: usize) {
    let mut i = 0usize;

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        while i + 8 <= len {
            use std::arch::x86_64::*;
            // Load 8 bf16 values (16-bit each)
            let bf16_vals = _mm_loadu_si128(src.add(i) as *const __m128i);
            // Zero-extend to 32-bit
            let i32_vals = _mm256_cvtepu16_epi32(bf16_vals);
            // Shift left by 16 to get f32 bit pattern
            let f32_bits = _mm256_slli_epi32(i32_vals, 16);
            // Store as f32
            _mm256_storeu_ps(dst.add(i), _mm256_castsi256_ps(f32_bits));
            i += 8;
        }
    }

    // Scalar tail
    while i < len {
        let bits = (*src.add(i) as u32) << 16;
        *dst.add(i) = f32::from_bits(bits);
        i += 1;
    }
}

/// F16 → f32 conversion using F16C instructions (vcvtph2ps)
#[cfg(all(feature = "f16", target_arch = "x86_64"))]
#[inline]
unsafe fn batch_f16_to_f32(src: *const u16, dst: *mut f32, len: usize) {
    let mut i = 0usize;

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("f16c") {
        while i + 8 <= len {
            use std::arch::x86_64::*;
            let f16_vals = _mm_loadu_si128(src.add(i) as *const __m128i);
            let f32_vals = _mm256_cvtph_ps(f16_vals);
            _mm256_storeu_ps(dst.add(i), f32_vals);
            i += 8;
        }
    }

    // Scalar tail
    while i < len {
        *dst.add(i) = half::f16::from_bits(*src.add(i)).to_f32();
        i += 1;
    }
}

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
    // Dispatch to SIMD for f32/f64 on x86-64, f16/bf16 via f32 conversion
    #[cfg(target_arch = "x86_64")]
    {
        use super::simd::matmul;

        match T::DTYPE {
            DType::I32 => {
                matmul::int32::matmul_i32(
                    a as *const i32,
                    b as *const i32,
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

    // Scalar fallback for non-SIMD types or non-x86 platforms
    matmul_scalar(a, b, out, m, n, k, lda, ldb, ldc);
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
    // Dispatch to fused SIMD for f32/f64 on x86-64, f16/bf16 via f32 conversion
    #[cfg(target_arch = "x86_64")]
    {
        use super::simd::matmul;

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

    // Scalar fallback with fused bias
    matmul_bias_scalar(a, b, bias, out, m, n, k, lda, ldb, ldc);
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
