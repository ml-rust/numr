//! NEON squared Euclidean distance.

use std::arch::aarch64::*;

/// Generates one width of the squared Euclidean kernel.
///
/// Each step computes `d = a - b`, then `acc = fma(acc, d, d)`. The loop
/// structure matches the dot kernel: 4 accumulators, a single-register loop,
/// then a scalar tail that never reads past `len`.
macro_rules! sqeuclidean_neon {
    (
        $(#[$meta:meta])*
        $name:ident, $t:ty, $lanes:expr,
        $dup:ident, $ld:ident, $add:ident, $sub:ident, $fma:ident, $hsum:ident
    ) => {
        $(#[$meta])*
        #[target_feature(enable = "neon")]
        pub unsafe fn $name(a: *const $t, b: *const $t, len: usize) -> $t {
            const L: usize = $lanes;
            let mut acc0 = $dup(0.0);
            let mut acc1 = $dup(0.0);
            let mut acc2 = $dup(0.0);
            let mut acc3 = $dup(0.0);
            let mut i = 0usize;
            while i + 4 * L <= len {
                let d0 = $sub($ld(a.add(i)), $ld(b.add(i)));
                let d1 = $sub($ld(a.add(i + L)), $ld(b.add(i + L)));
                let d2 = $sub($ld(a.add(i + 2 * L)), $ld(b.add(i + 2 * L)));
                let d3 = $sub($ld(a.add(i + 3 * L)), $ld(b.add(i + 3 * L)));
                acc0 = $fma(acc0, d0, d0);
                acc1 = $fma(acc1, d1, d1);
                acc2 = $fma(acc2, d2, d2);
                acc3 = $fma(acc3, d3, d3);
                i += 4 * L;
            }
            let mut acc = $add($add(acc0, acc1), $add(acc2, acc3));
            while i + L <= len {
                let d = $sub($ld(a.add(i)), $ld(b.add(i)));
                acc = $fma(acc, d, d);
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

sqeuclidean_neon!(
    /// NEON f32 squared Euclidean distance, 16 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support NEON. Every AArch64 CPU does.
    /// - `a` and `b` must each be valid for `len` reads.
    sqeuclidean_f32, f32, 4,
    vdupq_n_f32, vld1q_f32, vaddq_f32, vsubq_f32, vfmaq_f32, vaddvq_f32
);

sqeuclidean_neon!(
    /// NEON f64 squared Euclidean distance, 8 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support NEON. Every AArch64 CPU does.
    /// - `a` and `b` must each be valid for `len` reads.
    sqeuclidean_f64, f64, 2,
    vdupq_n_f64, vld1q_f64, vaddq_f64, vsubq_f64, vfmaq_f64, vaddvq_f64
);
