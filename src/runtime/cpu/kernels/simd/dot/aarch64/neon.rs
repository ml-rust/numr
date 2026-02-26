//! NEON i8 dot product kernels for ARM64
//!
//! Uses vmull_s8 + vpadalq_s16 for i8 x i8 → i32 accumulation.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const I8_LANES: usize = 16; // 128-bit / 8-bit (process 8 at a time via vmull)

/// Dot product of signed i8 vectors, accumulated in i32.
///
/// Processes 16 i8 elements per iteration using two vmull_s8 (low/high halves).
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - Pointers must be valid for `len` elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn i8xi8_dot_i32(a: *const i8, b: *const i8, len: usize) -> i32 {
    let chunks = len / I8_LANES;
    let remainder = len % I8_LANES;

    let mut acc = vdupq_n_s32(0);

    for i in 0..chunks {
        let offset = i * I8_LANES;
        let va = vld1q_s8(a.add(offset));
        let vb = vld1q_s8(b.add(offset));

        // Multiply low 8 elements: i8 x i8 → 8x i16
        let prod_lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        // Multiply high 8 elements: i8 x i8 → 8x i16
        let prod_hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));

        // Pairwise add and accumulate i16 → i32
        acc = vpadalq_s16(acc, prod_lo);
        acc = vpadalq_s16(acc, prod_hi);
    }

    // Horizontal sum of 4 i32 lanes
    let mut result = vaddvq_s32(acc);

    // Scalar tail
    for i in 0..remainder {
        let offset = chunks * I8_LANES + i;
        result += (*a.add(offset) as i32) * (*b.add(offset) as i32);
    }

    result
}

/// Scaled dot product of signed i8 vectors, returning f32.
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - Pointers must be valid for `len` elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn i8xi8_dot_f32(a: *const i8, b: *const i8, scale: f32, len: usize) -> f32 {
    (i8xi8_dot_i32(a, b, len) as f32) * scale
}
