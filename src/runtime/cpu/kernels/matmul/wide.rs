//! Half matmul written as its F32 accumulator: `matmul_wide_kernel`.
//!
//! The same arithmetic as [`super::matmul_kernel`] on F16/BF16 — convert,
//! multiply in F32, accumulate in F32 — minus the final narrowing store. The
//! kernel exists so `matmul_wide` shares the algorithm and only changes the
//! store; see the module docs of `ops/matmul_dtype.rs` for why a caller wants
//! the accumulator.

use super::kernel::matmul_scalar_acc;
use crate::dtype::Element;

/// `C = A @ B` for a half element type, stored as f32.
///
/// # Arguments
/// * `a` - Pointer to matrix A (m × k), row-major with leading dimension lda
/// * `b` - Pointer to matrix B (k × n), row-major with leading dimension ldb
/// * `out` - Pointer to the f32 output (m × n), row-major with leading dimension ldc
/// * `m`, `n`, `k` - Matrix dimensions
/// * `lda`, `ldb`, `ldc` - Leading dimensions (row stride in elements)
///
/// # Safety
/// - `a` and `b` must be valid for the specified dimensions and strides
/// - `out` must be valid for `m * ldc` f32 writes
/// - `out` must not alias with `a` or `b`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_wide_kernel<T: Element>(
    a: *const T,
    b: *const T,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    // Same SIMD gate as `matmul_kernel`: both architectures, so the wide and
    // the narrowed result come from the same tiled kernel and agree bit for
    // bit before the narrowing.
    #[cfg(all(feature = "f16", any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        use super::super::simd::matmul::half_convert::matmul_wide_via_f32;
        use crate::dtype::DType;

        if matches!(T::DTYPE, DType::F16 | DType::BF16) {
            matmul_wide_via_f32(a, b, out, m, n, k, lda, ldb, ldc);
            return;
        }
    }

    matmul_scalar_acc::<T, f32, f32>(a, b, out, m, n, k, lda, ldb, ldc);
}
