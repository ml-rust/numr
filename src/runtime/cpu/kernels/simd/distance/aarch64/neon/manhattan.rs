//! NEON Manhattan (L1) distance.

use std::arch::aarch64::*;

/// Generates one width of the Manhattan kernel.
///
/// `vabsq` clears the sign bit and keeps NaN a NaN. The loop structure matches
/// the dot kernel: 4 accumulators, a single-register loop, then a scalar tail
/// that never reads past `len`.
macro_rules! manhattan_neon {
    (
        $(#[$meta:meta])*
        $name:ident, $t:ty, $lanes:expr,
        $dup:ident, $ld:ident, $add:ident, $sub:ident, $abs:ident, $hsum:ident
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
                acc0 = $add(acc0, $abs(d0));
                acc1 = $add(acc1, $abs(d1));
                acc2 = $add(acc2, $abs(d2));
                acc3 = $add(acc3, $abs(d3));
                i += 4 * L;
            }
            let mut acc = $add($add(acc0, acc1), $add(acc2, acc3));
            while i + L <= len {
                let d = $sub($ld(a.add(i)), $ld(b.add(i)));
                acc = $add(acc, $abs(d));
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

manhattan_neon!(
    /// NEON f32 Manhattan distance, 16 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support NEON. Every AArch64 CPU does.
    /// - `a` and `b` must each be valid for `len` reads.
    manhattan_f32, f32, 4,
    vdupq_n_f32, vld1q_f32, vaddq_f32, vsubq_f32, vabsq_f32, vaddvq_f32
);

manhattan_neon!(
    /// NEON f64 Manhattan distance, 8 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support NEON. Every AArch64 CPU does.
    /// - `a` and `b` must each be valid for `len` reads.
    manhattan_f64, f64, 2,
    vdupq_n_f64, vld1q_f64, vaddq_f64, vsubq_f64, vabsq_f64, vaddvq_f64
);
