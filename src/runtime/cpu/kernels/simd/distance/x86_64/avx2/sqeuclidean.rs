//! AVX2+FMA squared Euclidean distance.

use crate::runtime::cpu::kernels::simd::math::avx2::{hsum_f32, hsum_f64};
use std::arch::x86_64::*;

/// Generates one width of the squared Euclidean kernel.
///
/// Each step computes `d = a - b`, then `acc = fma(d, d, acc)`. The loop
/// structure matches the dot kernel: 4 accumulators, a single-register loop,
/// then a scalar tail that never reads past `len`.
macro_rules! sqeuclidean_avx2 {
    (
        $(#[$meta:meta])*
        $name:ident, $t:ty, $lanes:expr,
        $setzero:ident, $loadu:ident, $add:ident, $sub:ident, $fmadd:ident, $hsum:ident
    ) => {
        $(#[$meta])*
        #[target_feature(enable = "avx2,fma")]
        pub unsafe fn $name(a: *const $t, b: *const $t, len: usize) -> $t {
            const L: usize = $lanes;
            let mut acc0 = $setzero();
            let mut acc1 = $setzero();
            let mut acc2 = $setzero();
            let mut acc3 = $setzero();
            let mut i = 0usize;
            while i + 4 * L <= len {
                let d0 = $sub($loadu(a.add(i)), $loadu(b.add(i)));
                let d1 = $sub($loadu(a.add(i + L)), $loadu(b.add(i + L)));
                let d2 = $sub($loadu(a.add(i + 2 * L)), $loadu(b.add(i + 2 * L)));
                let d3 = $sub($loadu(a.add(i + 3 * L)), $loadu(b.add(i + 3 * L)));
                acc0 = $fmadd(d0, d0, acc0);
                acc1 = $fmadd(d1, d1, acc1);
                acc2 = $fmadd(d2, d2, acc2);
                acc3 = $fmadd(d3, d3, acc3);
                i += 4 * L;
            }
            let mut acc = $add($add(acc0, acc1), $add(acc2, acc3));
            while i + L <= len {
                let d = $sub($loadu(a.add(i)), $loadu(b.add(i)));
                acc = $fmadd(d, d, acc);
                i += L;
            }
            let mut sum = $hsum(acc);
            while i < len {
                let d = *a.add(i) - *b.add(i);
                sum += d * d;
                i += 1;
            }
            sum
        }
    };
}

sqeuclidean_avx2!(
    /// AVX2+FMA f32 squared Euclidean distance, 32 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support AVX2 and FMA.
    /// - `a` and `b` must each be valid for `len` reads.
    sqeuclidean_f32, f32, 8,
    _mm256_setzero_ps, _mm256_loadu_ps, _mm256_add_ps, _mm256_sub_ps, _mm256_fmadd_ps, hsum_f32
);

sqeuclidean_avx2!(
    /// AVX2+FMA f64 squared Euclidean distance, 16 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support AVX2 and FMA.
    /// - `a` and `b` must each be valid for `len` reads.
    sqeuclidean_f64, f64, 4,
    _mm256_setzero_pd, _mm256_loadu_pd, _mm256_add_pd, _mm256_sub_pd, _mm256_fmadd_pd, hsum_f64
);
