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

    let zero_u8 = vdup_n_u8(0);

    for i in 0..chunks {
        let offset = i * F32_LANES;

        // Load 4 condition bytes
        let c0 = *cond.add(offset);
        let c1 = *cond.add(offset + 1);
        let c2 = *cond.add(offset + 2);
        let c3 = *cond.add(offset + 3);

        // Create u8x8 vector with condition bytes in first 4 positions
        let cond_bytes = vcreate_u8(
            (c0 as u64) | ((c1 as u64) << 8) | ((c2 as u64) << 16) | ((c3 as u64) << 24),
        );

        // Compare with zero to get mask: 0xFF for non-zero, 0x00 for zero
        let cond_mask_u8 = vcgt_u8(cond_bytes, zero_u8);

        // Expand u8 mask to u32 mask (4 elements)
        // First expand u8x8 -> u16x8
        let cond_mask_u16 = vmovl_u8(cond_mask_u8);
        // Then expand low u16x4 -> u32x4
        let cond_mask_u32 = vmovl_u16(vget_low_u16(cond_mask_u16));

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

        let mask = vcombine_u64(vcreate_u64(m0), vcreate_u64(m1));

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
