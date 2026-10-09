//! NEON dot product.

use std::arch::aarch64::*;

/// Generates one width of the dot kernel.
///
/// The main loop keeps 4 independent accumulators. That hides the FMA latency
/// of the 128-bit pipes. A single-register loop then takes the next full
/// vectors. A scalar loop takes the last `len % lanes` elements, so no load
/// reads past `len`.
macro_rules! dot_neon {
    (
        $(#[$meta:meta])*
        $name:ident, $t:ty, $lanes:expr,
        $dup:ident, $ld:ident, $add:ident, $fma:ident, $hsum:ident
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
                acc0 = $fma(acc0, $ld(a.add(i)), $ld(b.add(i)));
                acc1 = $fma(acc1, $ld(a.add(i + L)), $ld(b.add(i + L)));
                acc2 = $fma(acc2, $ld(a.add(i + 2 * L)), $ld(b.add(i + 2 * L)));
                acc3 = $fma(acc3, $ld(a.add(i + 3 * L)), $ld(b.add(i + 3 * L)));
                i += 4 * L;
            }
            let mut acc = $add($add(acc0, acc1), $add(acc2, acc3));
            while i + L <= len {
                acc = $fma(acc, $ld(a.add(i)), $ld(b.add(i)));
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

dot_neon!(
    /// NEON f32 dot product, 16 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support NEON. Every AArch64 CPU does.
    /// - `a` and `b` must each be valid for `len` reads.
    dot_f32, f32, 4,
    vdupq_n_f32, vld1q_f32, vaddq_f32, vfmaq_f32, vaddvq_f32
);

dot_neon!(
    /// NEON f64 dot product, 8 elements per main-loop iteration.
    ///
    /// # Safety
    /// - The CPU must support NEON. Every AArch64 CPU does.
    /// - `a` and `b` must each be valid for `len` reads.
    dot_f64, f64, 2,
    vdupq_n_f64, vld1q_f64, vaddq_f64, vfmaq_f64, vaddvq_f64
);
