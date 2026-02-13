//! Matrix packing functions for microkernel consumption
//!
//! These functions reorder matrix data into a layout optimized for the
//! 6×NR microkernels. Packing improves cache utilization by ensuring
//! sequential memory access in the innermost loop.

use super::MR;

/// Generate pack_a function for a given type
macro_rules! define_pack_a {
    ($name:ident, $ty:ty) => {
        /// Pack A matrix panel for microkernel consumption
        ///
        /// Layout: For each MR-row block, for each k: MR consecutive elements
        /// `[a[0,0], a[1,0], ..., a[MR-1,0], a[0,1], a[1,1], ..., a[MR-1,1], ...]`
        ///
        /// # Safety
        /// - `a` must be valid for reading `mc * kc` elements with stride `lda`
        /// - `packed` must be valid for writing `(mc.div_ceil(MR) * MR) * kc` elements
        #[inline]
        pub unsafe fn $name(a: *const $ty, packed: *mut $ty, mc: usize, kc: usize, lda: usize) {
            let mut p = 0;
            for ir in (0..mc).step_by(MR) {
                let mr_actual = (mc - ir).min(MR);
                if mr_actual == MR {
                    // Full MR block - no padding needed
                    for k in 0..kc {
                        for i in 0..MR {
                            *packed.add(p) = *a.add((ir + i) * lda + k);
                            p += 1;
                        }
                    }
                } else {
                    // Partial block - pad with zeros
                    for k in 0..kc {
                        for i in 0..mr_actual {
                            *packed.add(p) = *a.add((ir + i) * lda + k);
                            p += 1;
                        }
                        for _ in mr_actual..MR {
                            *packed.add(p) = 0.0;
                            p += 1;
                        }
                    }
                }
            }
        }
    };
}

/// Generate pack_b function for a given type
macro_rules! define_pack_b {
    ($name:ident, $ty:ty) => {
        /// Pack B matrix panel for microkernel consumption
        ///
        /// Layout: For each NR-column block, for each k: NR consecutive elements.
        /// Uses bulk copy for full NR blocks since B is row-major.
        ///
        /// # Safety
        /// - `b` must be valid for reading `kc * nc` elements with stride `ldb`
        /// - `packed` must be valid for writing `(nc.div_ceil(NR) * NR) * kc` elements
        #[inline]
        pub unsafe fn $name<const NR: usize>(
            b: *const $ty,
            packed: *mut $ty,
            nc: usize,
            kc: usize,
            ldb: usize,
        ) {
            let mut p = 0;
            for jr in (0..nc).step_by(NR) {
                let nr_actual = (nc - jr).min(NR);
                if nr_actual == NR {
                    // Full NR block: B elements are contiguous in each row
                    for k in 0..kc {
                        std::ptr::copy_nonoverlapping(b.add(k * ldb + jr), packed.add(p), NR);
                        p += NR;
                    }
                } else {
                    // Partial block - copy + zero-pad
                    for k in 0..kc {
                        for j in 0..nr_actual {
                            *packed.add(p) = *b.add(k * ldb + jr + j);
                            p += 1;
                        }
                        for _ in nr_actual..NR {
                            *packed.add(p) = 0.0;
                            p += 1;
                        }
                    }
                }
            }
        }
    };
}

define_pack_a!(pack_a_f32, f32);
define_pack_a!(pack_a_f64, f64);
define_pack_b!(pack_b_f32, f32);
define_pack_b!(pack_b_f64, f64);
