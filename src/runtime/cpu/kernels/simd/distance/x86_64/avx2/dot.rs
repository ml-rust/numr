//! AVX2+FMA dot product.

use crate::runtime::cpu::kernels::simd::math::avx2::{hsum_f32, hsum_f64};
use std::arch::x86_64::*;

/// Generates one width of the dot kernel.
///
/// The main loop keeps 4 independent accumulators. FMA latency is 4 cycles and
/// 2 FMA ports issue per cycle, so 4 chains keep the units busy while loads
/// stay at 2 per cycle. A single-register loop then takes the next full
/// vectors. A scalar loop takes the last `lanes - 1` elements, so no load reads
/// past `len`.
macro_rules! dot_avx2 {
    (
        $(#[$meta:meta])*
        $name:ident, $t:ty, $lanes:expr,
        $setzero:ident, $loadu:ident, $add:ident, $fmadd:ident, $hsum:ident
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
                acc0 = $fmadd($loadu(a.add(i)), $loadu(b.add(i)), acc0);
                acc1 = $fmadd($loadu(a.add(i + L)), $loadu(b.add(i + L)), acc1);
                acc2 = $fmadd($loadu(a.add(i + 2 * L)), $loadu(b.add(i + 2 * L)), acc2);
                acc3 = $fmadd($loadu(a.add(i + 3 * L)), $loadu(b.add(i + 3 * L)), acc3);
                i += 4 * L;
            }
            let mut acc = $add($add(acc0, acc1), $add(acc2, acc3));
            while i + L <= len {
                acc = $fmadd($loadu(a.add(i)), $loadu(b.add(i)), acc);
                i += L;
            }
            let mut sum = $hsum(acc);
            while i < len {
                sum += *a.add(i) * *b.add(i);
                i += 1;
            }
            sum
        }
    };
}

dot_avx2!(
    /// AVX2+FMA f32 dot product, 32 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support AVX2 and FMA.
    /// - `a` and `b` must each be valid for `len` reads.
    dot_f32, f32, 8,
    _mm256_setzero_ps, _mm256_loadu_ps, _mm256_add_ps, _mm256_fmadd_ps, hsum_f32
);

dot_avx2!(
    /// AVX2+FMA f64 dot product, 16 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support AVX2 and FMA.
    /// - `a` and `b` must each be valid for `len` reads.
    dot_f64, f64, 4,
    _mm256_setzero_pd, _mm256_loadu_pd, _mm256_add_pd, _mm256_fmadd_pd, hsum_f64
);
