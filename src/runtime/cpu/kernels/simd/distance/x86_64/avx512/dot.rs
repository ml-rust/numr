//! AVX-512 dot product.

use std::arch::x86_64::*;

/// Generates one width of the dot kernel.
///
/// The main loop keeps 4 independent accumulators. FMA latency is 4 cycles and
/// 2 FMA ports issue per cycle, so 4 chains keep the units busy. A
/// single-register loop then takes the next full vectors. The last
/// `lanes - 1` elements go through one masked load per input and one FMA.
/// Masked-off lanes read as zero, add `0 * 0`, and never fault.
macro_rules! dot_avx512 {
    (
        $(#[$meta:meta])*
        $name:ident, $t:ty, $lanes:expr, $mask:ty,
        $setzero:ident, $loadu:ident, $maskz_loadu:ident, $add:ident, $fmadd:ident,
        $reduce:ident
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
            if i < len {
                let k = ((1u32 << (len - i)) - 1) as $mask;
                acc = $fmadd($maskz_loadu(k, a.add(i)), $maskz_loadu(k, b.add(i)), acc);
            }
            $reduce(acc)
        }
    };
}

dot_avx512!(
    /// AVX-512 f32 dot product, 64 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support AVX-512F.
    /// - `a` and `b` must each be valid for `len` reads.
    dot_f32, f32, 16, __mmask16,
    _mm512_setzero_ps, _mm512_loadu_ps, _mm512_maskz_loadu_ps, _mm512_add_ps, _mm512_fmadd_ps,
    _mm512_reduce_add_ps
);

dot_avx512!(
    /// AVX-512 f64 dot product, 32 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support AVX-512F.
    /// - `a` and `b` must each be valid for `len` reads.
    dot_f64, f64, 8, __mmask8,
    _mm512_setzero_pd, _mm512_loadu_pd, _mm512_maskz_loadu_pd, _mm512_add_pd, _mm512_fmadd_pd,
    _mm512_reduce_add_pd
);
