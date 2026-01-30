//! Cache-aware tiled matmul algorithm
//!
//! Implements BLIS-style 3-level blocking:
//! - L3 cache: NC blocks on N dimension
//! - L2 cache: KC blocks on K dimension, MC blocks on M dimension
//! - Registers: MR×NR microkernels

use super::packing::{pack_a_f32, pack_a_f64, pack_b_f32, pack_b_f64};
use super::scalar::{microkernel_edge_f32, microkernel_edge_f64};
use super::{KC, MC, MR, NC};
use super::{call_microkernel_f32, call_microkernel_f64};
use crate::runtime::cpu::kernels::simd::SimdLevel;

/// Tiled matmul: C = A @ B (f32)
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_tiled_f32<const NR: usize>(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
) {
    let mut packed_a = vec![0.0f32; MC * KC];
    let mut packed_b = vec![0.0f32; KC * NC];

    // Zero output matrix
    for i in 0..m {
        for j in 0..n {
            *c.add(i * ldc + j) = 0.0;
        }
    }

    tiled_loop_f32::<NR>(
        a,
        b,
        c,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        level,
        &mut packed_a,
        &mut packed_b,
    );
}

/// Tiled matmul with bias: C = A @ B + bias (f32)
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bias_tiled_f32<const NR: usize>(
    a: *const f32,
    b: *const f32,
    bias: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
) {
    let mut packed_a = vec![0.0f32; MC * KC];
    let mut packed_b = vec![0.0f32; KC * NC];

    // Initialize C with bias (broadcast across rows)
    for i in 0..m {
        for j in 0..n {
            *c.add(i * ldc + j) = *bias.add(j);
        }
    }

    tiled_loop_f32::<NR>(
        a,
        b,
        c,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        level,
        &mut packed_a,
        &mut packed_b,
    );
}

/// Core tiled loop for f32 (shared between matmul and matmul_bias)
#[allow(clippy::too_many_arguments)]
unsafe fn tiled_loop_f32<const NR: usize>(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
    packed_a: &mut [f32],
    packed_b: &mut [f32],
) {
    // L3 blocking over N
    for jc in (0..n).step_by(NC) {
        let nc = (n - jc).min(NC);

        // L2 blocking over K
        for pc in (0..k).step_by(KC) {
            let kc = (k - pc).min(KC);

            pack_b_f32::<NR>(b.add(pc * ldb + jc), packed_b.as_mut_ptr(), nc, kc, ldb);

            // L2 blocking over M
            for ic in (0..m).step_by(MC) {
                let mc = (m - ic).min(MC);

                pack_a_f32(a.add(ic * lda + pc), packed_a.as_mut_ptr(), mc, kc, lda);

                // Microkernel loops
                for jr in (0..nc).step_by(NR) {
                    let nr_actual = (nc - jr).min(NR);

                    for ir in (0..mc).step_by(MR) {
                        let mr_actual = (mc - ir).min(MR);

                        if mr_actual == MR && nr_actual == NR {
                            call_microkernel_f32(
                                packed_a.as_ptr().add(ir * kc),
                                packed_b.as_ptr().add(jr * kc),
                                c.add((ic + ir) * ldc + jc + jr),
                                kc,
                                ldc,
                                level,
                            );
                        } else {
                            microkernel_edge_f32(
                                packed_a.as_ptr().add(ir * kc),
                                packed_b.as_ptr().add(jr * kc),
                                c.add((ic + ir) * ldc + jc + jr),
                                mr_actual,
                                nr_actual,
                                kc,
                                ldc,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Tiled matmul: C = A @ B (f64)
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_tiled_f64<const NR: usize>(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
) {
    let mut packed_a = vec![0.0f64; MC * KC];
    let mut packed_b = vec![0.0f64; KC * NC];

    for i in 0..m {
        for j in 0..n {
            *c.add(i * ldc + j) = 0.0;
        }
    }

    tiled_loop_f64::<NR>(
        a,
        b,
        c,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        level,
        &mut packed_a,
        &mut packed_b,
    );
}

/// Tiled matmul with bias: C = A @ B + bias (f64)
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bias_tiled_f64<const NR: usize>(
    a: *const f64,
    b: *const f64,
    bias: *const f64,
    c: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
) {
    let mut packed_a = vec![0.0f64; MC * KC];
    let mut packed_b = vec![0.0f64; KC * NC];

    for i in 0..m {
        for j in 0..n {
            *c.add(i * ldc + j) = *bias.add(j);
        }
    }

    tiled_loop_f64::<NR>(
        a,
        b,
        c,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        level,
        &mut packed_a,
        &mut packed_b,
    );
}

/// Core tiled loop for f64
#[allow(clippy::too_many_arguments)]
unsafe fn tiled_loop_f64<const NR: usize>(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
    packed_a: &mut [f64],
    packed_b: &mut [f64],
) {
    for jc in (0..n).step_by(NC) {
        let nc = (n - jc).min(NC);

        for pc in (0..k).step_by(KC) {
            let kc = (k - pc).min(KC);

            pack_b_f64::<NR>(b.add(pc * ldb + jc), packed_b.as_mut_ptr(), nc, kc, ldb);

            for ic in (0..m).step_by(MC) {
                let mc = (m - ic).min(MC);

                pack_a_f64(a.add(ic * lda + pc), packed_a.as_mut_ptr(), mc, kc, lda);

                for jr in (0..nc).step_by(NR) {
                    let nr_actual = (nc - jr).min(NR);

                    for ir in (0..mc).step_by(MR) {
                        let mr_actual = (mc - ir).min(MR);

                        if mr_actual == MR && nr_actual == NR {
                            call_microkernel_f64(
                                packed_a.as_ptr().add(ir * kc),
                                packed_b.as_ptr().add(jr * kc),
                                c.add((ic + ir) * ldc + jc + jr),
                                kc,
                                ldc,
                                level,
                            );
                        } else {
                            microkernel_edge_f64(
                                packed_a.as_ptr().add(ir * kc),
                                packed_b.as_ptr().add(jr * kc),
                                c.add((ic + ir) * ldc + jc + jr),
                                mr_actual,
                                nr_actual,
                                kc,
                                ldc,
                            );
                        }
                    }
                }
            }
        }
    }
}
