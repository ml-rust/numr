//! NEON binary operation kernels for i32 on ARM64
//!
//! Processes 4 i32s per iteration using 128-bit vectors.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::binary_scalar_i32;
use crate::ops::BinaryOp;

const I32_LANES: usize = 4;

/// NEON binary operation for i32
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - All pointers must be valid for `len` elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn binary_i32(op: BinaryOp, a: *const i32, b: *const i32, out: *mut i32, len: usize) {
    let chunks = len / I32_LANES;
    let remainder = len % I32_LANES;

    // Ops without SIMD integer support
    if !matches!(
        op,
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Max | BinaryOp::Min
    ) {
        binary_scalar_i32(op, a, b, out, len);
        return;
    }

    match op {
        BinaryOp::Add => binary_add_i32(a, b, out, chunks),
        BinaryOp::Sub => binary_sub_i32(a, b, out, chunks),
        BinaryOp::Mul => binary_mul_i32(a, b, out, chunks),
        BinaryOp::Max => binary_max_i32(a, b, out, chunks),
        BinaryOp::Min => binary_min_i32(a, b, out, chunks),
        _ => unreachable!(),
    }

    if remainder > 0 {
        let offset = chunks * I32_LANES;
        binary_scalar_i32(op, a.add(offset), b.add(offset), out.add(offset), remainder);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_add_i32(a: *const i32, b: *const i32, out: *mut i32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * I32_LANES;
        let va = vld1q_s32(a.add(offset));
        let vb = vld1q_s32(b.add(offset));
        let vr = vaddq_s32(va, vb);
        vst1q_s32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_sub_i32(a: *const i32, b: *const i32, out: *mut i32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * I32_LANES;
        let va = vld1q_s32(a.add(offset));
        let vb = vld1q_s32(b.add(offset));
        let vr = vsubq_s32(va, vb);
        vst1q_s32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_mul_i32(a: *const i32, b: *const i32, out: *mut i32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * I32_LANES;
        let va = vld1q_s32(a.add(offset));
        let vb = vld1q_s32(b.add(offset));
        let vr = vmulq_s32(va, vb);
        vst1q_s32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_max_i32(a: *const i32, b: *const i32, out: *mut i32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * I32_LANES;
        let va = vld1q_s32(a.add(offset));
        let vb = vld1q_s32(b.add(offset));
        let vr = vmaxq_s32(va, vb);
        vst1q_s32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_min_i32(a: *const i32, b: *const i32, out: *mut i32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * I32_LANES;
        let va = vld1q_s32(a.add(offset));
        let vb = vld1q_s32(b.add(offset));
        let vr = vminq_s32(va, vb);
        vst1q_s32(out.add(offset), vr);
    }
}
