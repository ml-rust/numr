//! NEON clamp kernels for ARM64
//!
//! Provides vectorized clamp operation using 128-bit NEON registers.
//!
//! clamp(x, min, max) = min(max(x, min_val), max_val)
//!
//! # SIMD Strategy
//!
//! 1. Broadcast min and max values to NEON registers
//! 2. Use vmaxq to clamp from below
//! 3. Use vminq to clamp from above

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const F32_LANES: usize = 4;
const F64_LANES: usize = 2;

/// NEON clamp for f32
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a` and `out` must point to `len` valid f32 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn clamp_f32(a: *const f32, out: *mut f32, len: usize, min_val: f32, max_val: f32) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    let v_min = vdupq_n_f32(min_val);
    let v_max = vdupq_n_f32(max_val);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let v = vld1q_f32(a.add(offset));
        // clamp = min(max(v, min_val), max_val)
        let clamped = vminq_f32(vmaxq_f32(v, v_min), v_max);
        vst1q_f32(out.add(offset), clamped);
    }

    // Scalar tail
    if remainder > 0 {
        let offset = chunks * F32_LANES;
        super::super::clamp_scalar_f32(a.add(offset), out.add(offset), remainder, min_val, max_val);
    }
}

/// NEON clamp for f64
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a` and `out` must point to `len` valid f64 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn clamp_f64(a: *const f64, out: *mut f64, len: usize, min_val: f64, max_val: f64) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    let v_min = vdupq_n_f64(min_val);
    let v_max = vdupq_n_f64(max_val);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let v = vld1q_f64(a.add(offset));
        let clamped = vminq_f64(vmaxq_f64(v, v_min), v_max);
        vst1q_f64(out.add(offset), clamped);
    }

    // Scalar tail
    if remainder > 0 {
        let offset = chunks * F64_LANES;
        super::super::clamp_scalar_f64(a.add(offset), out.add(offset), remainder, min_val, max_val);
    }
}
