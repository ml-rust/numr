//! AVX-512 squared Euclidean distance.

use std::arch::x86_64::*;

/// Generates one width of the squared Euclidean kernel.
///
/// Each step computes `d = a - b`, then `acc = fma(d, d, acc)`. The loop
/// structure matches the dot kernel: 4 accumulators, a single-register loop,
/// then one masked step. Masked-off lanes give `d = 0` and never fault.
macro_rules! sqeuclidean_avx512 {
    (
        $(#[$meta:meta])*
        $name:ident, $t:ty, $lanes:expr, $mask:ty,
        $setzero:ident, $loadu:ident, $maskz_loadu:ident, $add:ident, $sub:ident,
        $fmadd:ident, $reduce:ident
    ) => {
        $(#[$meta])*
        #[target_feature(enable = "avx512f")]
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
            if i < len {
                let k = ((1u32 << (len - i)) - 1) as $mask;
                let d = $sub($maskz_loadu(k, a.add(i)), $maskz_loadu(k, b.add(i)));
                acc = $fmadd(d, d, acc);
            }
            $reduce(acc)
        }
    };
}

sqeuclidean_avx512!(
    /// AVX-512 f32 squared Euclidean distance, 64 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support AVX-512F.
    /// - `a` and `b` must each be valid for `len` reads.
    sqeuclidean_f32, f32, 16, __mmask16,
    _mm512_setzero_ps, _mm512_loadu_ps, _mm512_maskz_loadu_ps, _mm512_add_ps, _mm512_sub_ps,
    _mm512_fmadd_ps, _mm512_reduce_add_ps
);

sqeuclidean_avx512!(
    /// AVX-512 f64 squared Euclidean distance, 32 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support AVX-512F.
    /// - `a` and `b` must each be valid for `len` reads.
    sqeuclidean_f64, f64, 8, __mmask8,
    _mm512_setzero_pd, _mm512_loadu_pd, _mm512_maskz_loadu_pd, _mm512_add_pd, _mm512_sub_pd,
    _mm512_fmadd_pd, _mm512_reduce_add_pd
);
