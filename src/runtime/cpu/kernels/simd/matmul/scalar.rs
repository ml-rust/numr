//! Scalar matmul fallback implementations
//!
//! These are used for small matrices (below tiling threshold) and as
//! edge-case handlers within the tiled algorithm.

use super::MR;

/// Generate scalar matmul function for a given type
macro_rules! define_scalar_matmul {
    ($name:ident, $ty:ty) => {
        /// Scalar matmul: C = A @ B
        ///
        /// Uses ikj loop order for better cache locality on B.
        ///
        /// # Safety
        /// - All pointers must be valid for the specified dimensions
        /// - `out` must not alias with `a` or `b`
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            a: *const $ty,
            b: *const $ty,
            out: *mut $ty,
            m: usize,
            n: usize,
            k: usize,
            lda: usize,
            ldb: usize,
            ldc: usize,
        ) {
            // Zero output first
            for i in 0..m {
                for j in 0..n {
                    *out.add(i * ldc + j) = 0.0;
                }
            }

            // ikj loop order for better cache locality
            for i in 0..m {
                for kk in 0..k {
                    let a_val = *a.add(i * lda + kk);
                    for j in 0..n {
                        let out_ptr = out.add(i * ldc + j);
                        *out_ptr += a_val * *b.add(kk * ldb + j);
                    }
                }
            }
        }
    };
}

/// Generate scalar matmul with fused bias for a given type
macro_rules! define_scalar_matmul_bias {
    ($name:ident, $ty:ty) => {
        /// Scalar matmul with fused bias: C = A @ B + bias
        ///
        /// Single-pass: initializes C with bias, then accumulates matmul.
        ///
        /// # Safety
        /// - All pointers must be valid for the specified dimensions
        /// - `out` must not alias with `a`, `b`, or `bias`
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            a: *const $ty,
            b: *const $ty,
            bias: *const $ty,
            out: *mut $ty,
            m: usize,
            n: usize,
            k: usize,
            lda: usize,
            ldb: usize,
            ldc: usize,
        ) {
            // Initialize with bias (single write pass)
            for i in 0..m {
                for j in 0..n {
                    *out.add(i * ldc + j) = *bias.add(j);
                }
            }

            // Accumulate matmul (ikj order for cache locality)
            for i in 0..m {
                for kk in 0..k {
                    let a_val = *a.add(i * lda + kk);
                    for j in 0..n {
                        let out_ptr = out.add(i * ldc + j);
                        *out_ptr += a_val * *b.add(kk * ldb + j);
                    }
                }
            }
        }
    };
}

/// Generate edge microkernel for partial tiles
macro_rules! define_microkernel_edge {
    ($name:ident, $ty:ty) => {
        /// Scalar microkernel for edge tiles (partial MR×NR blocks)
        ///
        /// Packed layout: For each k, MR consecutive A elements, NR consecutive B elements
        ///
        /// # Safety
        /// - `a` must be valid for `k * MR` elements (packed format)
        /// - `b` must be valid for `k * nr` elements (packed format)
        /// - `c` must be valid for `mr * ldc` elements
        #[inline]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            a: *const $ty,
            b: *const $ty,
            c: *mut $ty,
            mr: usize,
            nr: usize,
            k: usize,
            ldc: usize,
        ) {
            for kk in 0..k {
                for i in 0..mr {
                    let a_val = *a.add(kk * MR + i);
                    for j in 0..nr {
                        let b_val = *b.add(kk * nr + j);
                        let c_ptr = c.add(i * ldc + j);
                        *c_ptr += a_val * b_val;
                    }
                }
            }
        }
    };
}

define_scalar_matmul!(matmul_scalar_f32, f32);
define_scalar_matmul!(matmul_scalar_f64, f64);
define_scalar_matmul_bias!(matmul_bias_scalar_f32, f32);
define_scalar_matmul_bias!(matmul_bias_scalar_f64, f64);
define_microkernel_edge!(microkernel_edge_f32, f32);
define_microkernel_edge!(microkernel_edge_f64, f64);
