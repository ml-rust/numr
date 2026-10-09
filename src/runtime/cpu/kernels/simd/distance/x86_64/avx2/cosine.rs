//! AVX2+FMA one-pass cosine sums.

use super::super::super::sums::CosineSums;
use crate::runtime::cpu::kernels::simd::math::avx2::{hsum_f32, hsum_f64};
use std::arch::x86_64::*;

/// Generates one width of the cosine-sums kernel.
///
/// Register budget: AVX2 has 16 ymm registers. The main loop unrolls 3 times.
/// That needs 9 accumulators (dot, norm_a, norm_b per step) plus 6 loaded
/// vectors, so 15 registers and no spills. An unroll of 4 needs 20 and spills.
/// One step issues 3 FMAs, which take 1.5 cycles on 2 FMA ports. Each
/// accumulator is updated once per 3 steps, so 4.5 cycles apart. That exceeds
/// the 4-cycle FMA latency, so no chain stalls the pipeline.
///
/// A single-register loop then takes the next full vectors. A scalar loop
/// takes the rest, so no load reads past `len`.
macro_rules! cosine_avx2 {
    (
        $(#[$meta:meta])*
        $name:ident, $t:ty, $lanes:expr,
        $setzero:ident, $loadu:ident, $add:ident, $fmadd:ident, $hsum:ident
    ) => {
        $(#[$meta])*
        #[target_feature(enable = "avx2,fma")]
        pub unsafe fn $name(a: *const $t, b: *const $t, len: usize) -> CosineSums<$t> {
            const L: usize = $lanes;
            let (mut dot0, mut dot1, mut dot2) = ($setzero(), $setzero(), $setzero());
            let (mut na0, mut na1, mut na2) = ($setzero(), $setzero(), $setzero());
            let (mut nb0, mut nb1, mut nb2) = ($setzero(), $setzero(), $setzero());
            let mut i = 0usize;
            while i + 3 * L <= len {
                let (a0, b0) = ($loadu(a.add(i)), $loadu(b.add(i)));
                let (a1, b1) = ($loadu(a.add(i + L)), $loadu(b.add(i + L)));
                let (a2, b2) = ($loadu(a.add(i + 2 * L)), $loadu(b.add(i + 2 * L)));
                dot0 = $fmadd(a0, b0, dot0);
                na0 = $fmadd(a0, a0, na0);
                nb0 = $fmadd(b0, b0, nb0);
                dot1 = $fmadd(a1, b1, dot1);
                na1 = $fmadd(a1, a1, na1);
                nb1 = $fmadd(b1, b1, nb1);
                dot2 = $fmadd(a2, b2, dot2);
                na2 = $fmadd(a2, a2, na2);
                nb2 = $fmadd(b2, b2, nb2);
                i += 3 * L;
            }
            let mut dot = $add($add(dot0, dot1), dot2);
            let mut na = $add($add(na0, na1), na2);
            let mut nb = $add($add(nb0, nb1), nb2);
            while i + L <= len {
                let (av, bv) = ($loadu(a.add(i)), $loadu(b.add(i)));
                dot = $fmadd(av, bv, dot);
                na = $fmadd(av, av, na);
                nb = $fmadd(bv, bv, nb);
                i += L;
            }
            let mut sums = CosineSums {
                dot: $hsum(dot),
                norm_a: $hsum(na),
                norm_b: $hsum(nb),
            };
            while i < len {
                let (ak, bk) = (*a.add(i), *b.add(i));
                sums.dot += ak * bk;
                sums.norm_a += ak * ak;
                sums.norm_b += bk * bk;
                i += 1;
            }
            sums
        }
    };
}

cosine_avx2!(
    /// AVX2+FMA f32 cosine sums, 24 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support AVX2 and FMA.
    /// - `a` and `b` must each be valid for `len` reads.
    cosine_sums_f32, f32, 8,
    _mm256_setzero_ps, _mm256_loadu_ps, _mm256_add_ps, _mm256_fmadd_ps, hsum_f32
);

cosine_avx2!(
    /// AVX2+FMA f64 cosine sums, 12 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support AVX2 and FMA.
    /// - `a` and `b` must each be valid for `len` reads.
    cosine_sums_f64, f64, 4,
    _mm256_setzero_pd, _mm256_loadu_pd, _mm256_add_pd, _mm256_fmadd_pd, hsum_f64
);
