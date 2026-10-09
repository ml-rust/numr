//! AVX2 Manhattan (L1) distance.

use crate::runtime::cpu::kernels::simd::math::avx2::{hsum_f32, hsum_f64};
use std::arch::x86_64::*;

/// Generates one width of the Manhattan kernel.
///
/// `|d|` is `andnot(-0.0, d)`: it clears the sign bit and keeps NaN a NaN.
/// The loop structure matches the dot kernel: 4 accumulators, a
/// single-register loop, then a scalar tail that never reads past `len`.
macro_rules! manhattan_avx2 {
    (
        $(#[$meta:meta])*
        $name:ident, $t:ty, $lanes:expr,
        $setzero:ident, $set1:ident, $loadu:ident, $add:ident, $sub:ident, $andnot:ident,
        $hsum:ident
    ) => {
        $(#[$meta])*
        #[target_feature(enable = "avx2,fma")]
        pub unsafe fn $name(a: *const $t, b: *const $t, len: usize) -> $t {
            const L: usize = $lanes;
            let sign = $set1(-0.0);
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
                acc0 = $add(acc0, $andnot(sign, d0));
                acc1 = $add(acc1, $andnot(sign, d1));
                acc2 = $add(acc2, $andnot(sign, d2));
                acc3 = $add(acc3, $andnot(sign, d3));
                i += 4 * L;
            }
            let mut acc = $add($add(acc0, acc1), $add(acc2, acc3));
            while i + L <= len {
                let d = $sub($loadu(a.add(i)), $loadu(b.add(i)));
                acc = $add(acc, $andnot(sign, d));
                i += L;
            }
            let mut sum = $hsum(acc);
            while i < len {
                sum += (*a.add(i) - *b.add(i)).abs();
                i += 1;
            }
            sum
        }
    };
}

manhattan_avx2!(
    /// AVX2 f32 Manhattan distance, 32 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support AVX2 and FMA.
    /// - `a` and `b` must each be valid for `len` reads.
    manhattan_f32, f32, 8,
    _mm256_setzero_ps, _mm256_set1_ps, _mm256_loadu_ps, _mm256_add_ps, _mm256_sub_ps,
    _mm256_andnot_ps, hsum_f32
);

manhattan_avx2!(
    /// AVX2 f64 Manhattan distance, 16 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support AVX2 and FMA.
    /// - `a` and `b` must each be valid for `len` reads.
    manhattan_f64, f64, 4,
    _mm256_setzero_pd, _mm256_set1_pd, _mm256_loadu_pd, _mm256_add_pd, _mm256_sub_pd,
    _mm256_andnot_pd, hsum_f64
);
