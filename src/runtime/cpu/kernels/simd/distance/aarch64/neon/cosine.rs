//! NEON one-pass cosine sums.

use super::super::super::sums::CosineSums;
use std::arch::aarch64::*;

/// Generates one width of the cosine-sums kernel.
///
/// The main loop unrolls 4 times with 3 accumulators per step (dot, norm_a,
/// norm_b).
///
/// Register budget: 12 accumulators plus 8 loaded vectors fit the 32 NEON
/// registers, so no spills.
///
/// A single-register loop then takes the next full vectors. A scalar loop
/// takes the rest, so no load reads past `len`.
macro_rules! cosine_neon {
    (
        $(#[$meta:meta])*
        $name:ident, $t:ty, $lanes:expr,
        $dup:ident, $ld:ident, $add:ident, $fma:ident, $hsum:ident
    ) => {
        $(#[$meta])*
        #[target_feature(enable = "neon")]
        pub unsafe fn $name(a: *const $t, b: *const $t, len: usize) -> CosineSums<$t> {
            const L: usize = $lanes;
            let z = $dup(0.0);
            let (mut dot0, mut dot1, mut dot2, mut dot3) = (z, z, z, z);
            let (mut na0, mut na1, mut na2, mut na3) = (z, z, z, z);
            let (mut nb0, mut nb1, mut nb2, mut nb3) = (z, z, z, z);
            let mut i = 0usize;
            while i + 4 * L <= len {
                let (a0, b0) = ($ld(a.add(i)), $ld(b.add(i)));
                let (a1, b1) = ($ld(a.add(i + L)), $ld(b.add(i + L)));
                let (a2, b2) = ($ld(a.add(i + 2 * L)), $ld(b.add(i + 2 * L)));
                let (a3, b3) = ($ld(a.add(i + 3 * L)), $ld(b.add(i + 3 * L)));
                dot0 = $fma(dot0, a0, b0);
                na0 = $fma(na0, a0, a0);
                nb0 = $fma(nb0, b0, b0);
                dot1 = $fma(dot1, a1, b1);
                na1 = $fma(na1, a1, a1);
                nb1 = $fma(nb1, b1, b1);
                dot2 = $fma(dot2, a2, b2);
                na2 = $fma(na2, a2, a2);
                nb2 = $fma(nb2, b2, b2);
                dot3 = $fma(dot3, a3, b3);
                na3 = $fma(na3, a3, a3);
                nb3 = $fma(nb3, b3, b3);
                i += 4 * L;
            }
            let mut dot = $add($add(dot0, dot1), $add(dot2, dot3));
            let mut na = $add($add(na0, na1), $add(na2, na3));
            let mut nb = $add($add(nb0, nb1), $add(nb2, nb3));
            while i + L <= len {
                let (av, bv) = ($ld(a.add(i)), $ld(b.add(i)));
                dot = $fma(dot, av, bv);
                na = $fma(na, av, av);
                nb = $fma(nb, bv, bv);
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

cosine_neon!(
    /// NEON f32 cosine sums, 16 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support NEON. Every AArch64 CPU does.
    /// - `a` and `b` must each be valid for `len` reads.
    cosine_sums_f32, f32, 4,
    vdupq_n_f32, vld1q_f32, vaddq_f32, vfmaq_f32, vaddvq_f32
);

cosine_neon!(
    /// NEON f64 cosine sums, 8 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support NEON. Every AArch64 CPU does.
    /// - `a` and `b` must each be valid for `len` reads.
    cosine_sums_f64, f64, 2,
    vdupq_n_f64, vld1q_f64, vaddq_f64, vfmaq_f64, vaddvq_f64
);
