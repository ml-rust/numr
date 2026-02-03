//! NEON cumulative operation kernels for ARM64
//!
//! Provides vectorized cumsum and cumprod operations using 128-bit NEON registers.
//!
//! # SIMD Strategy
//!
//! For strided cumulative ops, we vectorize over the inner_size dimension.
//! Each SIMD lane maintains its own independent accumulator.
//!
//! - f32: 4 lanes (4 independent cumsum/cumprod streams)
//! - f64: 2 lanes (2 independent cumsum/cumprod streams)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================================
// Cumsum Strided
// ============================================================================

/// NEON cumsum strided for f32
///
/// Vectorizes over inner_size dimension - each SIMD lane maintains
/// its own running sum independently.
///
/// # Safety
/// - All pointers must be valid for the specified sizes and strides
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn cumsum_strided_f32(
    a: *const f32,
    out: *mut f32,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    let lanes = 4;
    let chunks = inner_size / lanes;
    let remainder = inner_size % lanes;

    for o in 0..outer_size {
        let outer_offset = o * scan_size * inner_size;

        // Process 4 inner positions at a time
        for chunk in 0..chunks {
            let i_base = chunk * lanes;
            let mut acc = vdupq_n_f32(0.0);

            for s in 0..scan_size {
                let idx = outer_offset + s * inner_size + i_base;
                let v = vld1q_f32(a.add(idx));
                acc = vaddq_f32(acc, v);
                vst1q_f32(out.add(idx), acc);
            }
        }

        // Scalar tail for remaining inner positions
        for i in (chunks * lanes)..inner_size {
            let mut acc = 0.0f32;
            for s in 0..scan_size {
                let idx = outer_offset + s * inner_size + i;
                acc += *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}

/// NEON cumsum strided for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn cumsum_strided_f64(
    a: *const f64,
    out: *mut f64,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    let lanes = 2;
    let chunks = inner_size / lanes;

    for o in 0..outer_size {
        let outer_offset = o * scan_size * inner_size;

        // Process 2 inner positions at a time
        for chunk in 0..chunks {
            let i_base = chunk * lanes;
            let mut acc = vdupq_n_f64(0.0);

            for s in 0..scan_size {
                let idx = outer_offset + s * inner_size + i_base;
                let v = vld1q_f64(a.add(idx));
                acc = vaddq_f64(acc, v);
                vst1q_f64(out.add(idx), acc);
            }
        }

        // Scalar tail
        for i in (chunks * lanes)..inner_size {
            let mut acc = 0.0f64;
            for s in 0..scan_size {
                let idx = outer_offset + s * inner_size + i;
                acc += *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}

// ============================================================================
// Cumprod Strided
// ============================================================================

/// NEON cumprod strided for f32
///
/// # Safety
/// - All pointers must be valid for the specified sizes and strides
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn cumprod_strided_f32(
    a: *const f32,
    out: *mut f32,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    let lanes = 4;
    let chunks = inner_size / lanes;

    for o in 0..outer_size {
        let outer_offset = o * scan_size * inner_size;

        // Process 4 inner positions at a time
        for chunk in 0..chunks {
            let i_base = chunk * lanes;
            let mut acc = vdupq_n_f32(1.0);

            for s in 0..scan_size {
                let idx = outer_offset + s * inner_size + i_base;
                let v = vld1q_f32(a.add(idx));
                acc = vmulq_f32(acc, v);
                vst1q_f32(out.add(idx), acc);
            }
        }

        // Scalar tail
        for i in (chunks * lanes)..inner_size {
            let mut acc = 1.0f32;
            for s in 0..scan_size {
                let idx = outer_offset + s * inner_size + i;
                acc *= *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}

/// NEON cumprod strided for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn cumprod_strided_f64(
    a: *const f64,
    out: *mut f64,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    let lanes = 2;
    let chunks = inner_size / lanes;

    for o in 0..outer_size {
        let outer_offset = o * scan_size * inner_size;

        // Process 2 inner positions at a time
        for chunk in 0..chunks {
            let i_base = chunk * lanes;
            let mut acc = vdupq_n_f64(1.0);

            for s in 0..scan_size {
                let idx = outer_offset + s * inner_size + i_base;
                let v = vld1q_f64(a.add(idx));
                acc = vmulq_f64(acc, v);
                vst1q_f64(out.add(idx), acc);
            }
        }

        // Scalar tail
        for i in (chunks * lanes)..inner_size {
            let mut acc = 1.0f64;
            for s in 0..scan_size {
                let idx = outer_offset + s * inner_size + i;
                acc *= *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}
