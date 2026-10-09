//! AVX-512 one-pass cosine sums.

use super::super::super::sums::CosineSums;
use std::arch::x86_64::*;

/// Generates one width of the cosine-sums kernel.
///
/// Register budget: 4 steps x 3 accumulators = 12, plus 8 loaded vectors, is 20 of 32 zmm.
/// One step issues 3 FMAs. Each accumulator is updated once per 4 steps, so
/// the 4-cycle FMA latency never stalls a chain.
///
/// A single-register loop then takes the next full vectors. The last
/// `lanes - 1` elements go through one masked load per input. Masked-off
/// lanes read as zero, add `0 * 0`, and never fault.
macro_rules! cosine_avx512 {
    (
        $(#[$meta:meta])*
        $name:ident, $t:ty, $lanes:expr, $mask:ty,
        $setzero:ident, $loadu:ident, $maskz_loadu:ident, $add:ident, $fmadd:ident,
        $reduce:ident
    ) => {
        $(#[$meta])*
        #[target_feature(enable = "avx512f")]
        pub unsafe fn $name(a: *const $t, b: *const $t, len: usize) -> CosineSums<$t> {
            const L: usize = $lanes;
            let (mut dot0, mut dot1) = ($setzero(), $setzero());
            let (mut dot2, mut dot3) = ($setzero(), $setzero());
            let (mut na0, mut na1) = ($setzero(), $setzero());
            let (mut na2, mut na3) = ($setzero(), $setzero());
            let (mut nb0, mut nb1) = ($setzero(), $setzero());
            let (mut nb2, mut nb3) = ($setzero(), $setzero());
            let mut i = 0usize;
            while i + 4 * L <= len {
                let (a0, b0) = ($loadu(a.add(i)), $loadu(b.add(i)));
                let (a1, b1) = ($loadu(a.add(i + L)), $loadu(b.add(i + L)));
                let (a2, b2) = ($loadu(a.add(i + 2 * L)), $loadu(b.add(i + 2 * L)));
                let (a3, b3) = ($loadu(a.add(i + 3 * L)), $loadu(b.add(i + 3 * L)));
                dot0 = $fmadd(a0, b0, dot0);
                na0 = $fmadd(a0, a0, na0);
                nb0 = $fmadd(b0, b0, nb0);
                dot1 = $fmadd(a1, b1, dot1);
                na1 = $fmadd(a1, a1, na1);
                nb1 = $fmadd(b1, b1, nb1);
                dot2 = $fmadd(a2, b2, dot2);
                na2 = $fmadd(a2, a2, na2);
                nb2 = $fmadd(b2, b2, nb2);
                dot3 = $fmadd(a3, b3, dot3);
                na3 = $fmadd(a3, a3, na3);
                nb3 = $fmadd(b3, b3, nb3);
                i += 4 * L;
            }
            let mut dot = $add($add(dot0, dot1), $add(dot2, dot3));
            let mut na = $add($add(na0, na1), $add(na2, na3));
            let mut nb = $add($add(nb0, nb1), $add(nb2, nb3));
            while i + L <= len {
                let (av, bv) = ($loadu(a.add(i)), $loadu(b.add(i)));
                dot = $fmadd(av, bv, dot);
                na = $fmadd(av, av, na);
                nb = $fmadd(bv, bv, nb);
                i += L;
            }
            if i < len {
                let k = ((1u32 << (len - i)) - 1) as $mask;
                let (av, bv) = ($maskz_loadu(k, a.add(i)), $maskz_loadu(k, b.add(i)));
                dot = $fmadd(av, bv, dot);
                na = $fmadd(av, av, na);
                nb = $fmadd(bv, bv, nb);
            }
            CosineSums {
                dot: $reduce(dot),
                norm_a: $reduce(na),
                norm_b: $reduce(nb),
            }
        }
    };
}

cosine_avx512!(
    /// AVX-512 f32 cosine sums, 64 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support AVX-512F.
    /// - `a` and `b` must each be valid for `len` reads.
    cosine_sums_f32, f32, 16, __mmask16,
    _mm512_setzero_ps, _mm512_loadu_ps, _mm512_maskz_loadu_ps, _mm512_add_ps, _mm512_fmadd_ps,
    _mm512_reduce_add_ps
);

cosine_avx512!(
    /// AVX-512 f64 cosine sums, 32 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support AVX-512F.
    /// - `a` and `b` must each be valid for `len` reads.
    cosine_sums_f64, f64, 8, __mmask8,
    _mm512_setzero_pd, _mm512_loadu_pd, _mm512_maskz_loadu_pd, _mm512_add_pd, _mm512_fmadd_pd,
    _mm512_reduce_add_pd
);
