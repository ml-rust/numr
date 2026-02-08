//! NEON where/select kernels for ARM64
//!
//! Provides vectorized conditional selection using 128-bit NEON registers.
//!
//! where(cond, x, y): out[i] = cond[i] ? x[i] : y[i]
//!
//! # SIMD Strategy
//!
//! 1. Load condition bytes (u8)
//! 2. Expand conditions to 32-bit or 64-bit masks
//! 3. Use vbslq (bit select) to blend x and y based on mask

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const F32_LANES: usize = 4;
const F64_LANES: usize = 2;

/// NEON where for f32
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `cond` must point to `len` valid u8 elements
/// - `x`, `y`, and `out` must point to `len` valid f32 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn where_f32(cond: *const u8, x: *const f32, y: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    for i in 0..chunks {
        let offset = i * F32_LANES;

        // Load 4 condition bytes
        let c0 = *cond.add(offset);
        let c1 = *cond.add(offset + 1);
        let c2 = *cond.add(offset + 2);
        let c3 = *cond.add(offset + 3);

        // Build full-width lane masks required by vbsl:
        // 0xFFFF_FFFF selects from x, 0x0000_0000 selects from y.
        let cond_mask_u32 = vld1q_u32(
            [
                if c0 != 0 { !0u32 } else { 0u32 },
                if c1 != 0 { !0u32 } else { 0u32 },
                if c2 != 0 { !0u32 } else { 0u32 },
                if c3 != 0 { !0u32 } else { 0u32 },
            ]
            .as_ptr(),
        );

        // Load x and y vectors
        let vx = vld1q_f32(x.add(offset));
        let vy = vld1q_f32(y.add(offset));

        // Blend: where mask is all-ones, select x; else select y
        let result = vbslq_f32(cond_mask_u32, vx, vy);
        vst1q_f32(out.add(offset), result);
    }

    // Scalar tail
    if remainder > 0 {
        let offset = chunks * F32_LANES;
        super::super::where_scalar_f32(
            cond.add(offset),
            x.add(offset),
            y.add(offset),
            out.add(offset),
            remainder,
        );
    }
}

/// NEON where for f64
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `cond` must point to `len` valid u8 elements
/// - `x`, `y`, and `out` must point to `len` valid f64 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn where_f64(cond: *const u8, x: *const f64, y: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    for i in 0..chunks {
        let offset = i * F64_LANES;

        // Load 2 condition bytes
        let c0 = *cond.add(offset);
        let c1 = *cond.add(offset + 1);

        // Create u64x2 mask: 0xFFFF...FF for non-zero condition, 0 for zero
        let m0: u64 = if c0 != 0 { !0u64 } else { 0u64 };
        let m1: u64 = if c1 != 0 { !0u64 } else { 0u64 };

        let mask = vld1q_u64([m0, m1].as_ptr());

        // Load x and y vectors
        let vx = vld1q_f64(x.add(offset));
        let vy = vld1q_f64(y.add(offset));

        // Blend
        let result = vbslq_f64(mask, vx, vy);
        vst1q_f64(out.add(offset), result);
    }

    // Scalar tail
    if remainder > 0 {
        let offset = chunks * F64_LANES;
        super::super::where_scalar_f64(
            cond.add(offset),
            x.add(offset),
            y.add(offset),
            out.add(offset),
            remainder,
        );
    }
}
