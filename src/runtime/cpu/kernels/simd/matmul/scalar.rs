//! Scalar matmul fallback implementations
//!
//! These are used for small matrices (below tiling threshold) and as
//! edge-case handlers within the tiled algorithm.

use super::MR;

/// Generate scalar matmul function for a given type
macro_rules! define_scalar_matmul {
    ($name:ident, $ty:ty) => {
        /// Matmul: C = A @ B
        ///
        /// Uses ikj loop order with slice-based access for auto-vectorization.
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
            // Zero output
            let out_slice = std::slice::from_raw_parts_mut(out, m * ldc);
            for i in 0..m {
                out_slice[i * ldc..i * ldc + n].fill(0.0);
            }

            // ikj loop with slice access enables auto-vectorization
            for i in 0..m {
                let c_row = &mut std::slice::from_raw_parts_mut(out.add(i * ldc), n)[..n];
                for kk in 0..k {
                    let a_val = *a.add(i * lda + kk);
                    let b_row = std::slice::from_raw_parts(b.add(kk * ldb), n);
                    for j in 0..n {
                        c_row[j] += a_val * b_row[j];
                    }
                }
            }
        }
    };
}

/// Generate scalar matmul with fused bias for a given type
macro_rules! define_scalar_matmul_bias {
    ($name:ident, $ty:ty) => {
        /// Matmul with fused bias: C = A @ B + bias
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
            let bias_slice = std::slice::from_raw_parts(bias, n);
            for i in 0..m {
                let c_row = &mut std::slice::from_raw_parts_mut(out.add(i * ldc), n)[..n];
                c_row.copy_from_slice(bias_slice);
            }

            for i in 0..m {
                let c_row = &mut std::slice::from_raw_parts_mut(out.add(i * ldc), n)[..n];
                for kk in 0..k {
                    let a_val = *a.add(i * lda + kk);
                    let b_row = std::slice::from_raw_parts(b.add(kk * ldb), n);
                    for j in 0..n {
                        c_row[j] += a_val * b_row[j];
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
        /// When `first_k` is true, C tile is zeroed before accumulation.
        /// When false, C is loaded and accumulated into.
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
            first_k: bool,
        ) {
            if first_k {
                for i in 0..mr {
                    for j in 0..nr {
                        *c.add(i * ldc + j) = 0.0;
                    }
                }
            }

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
