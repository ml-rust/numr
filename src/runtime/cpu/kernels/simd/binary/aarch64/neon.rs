//! NEON binary operation kernels for ARM64
//!
//! Processes 4 f32s or 2 f64s per iteration using 128-bit vectors.
//!
//! NEON is the baseline SIMD instruction set for all ARM64 processors,
//! providing 128-bit vector operations with hardware support for:
//! - Single and double precision floating point
//! - Integer operations (8, 16, 32, 64-bit)
//! - Polynomial operations

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::{binary_scalar_f32, binary_scalar_f64};
use crate::ops::BinaryOp;

const F32_LANES: usize = 4;
const F64_LANES: usize = 2;

/// NEON binary operation for f32
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - All pointers must be valid for `len` elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn binary_f32(op: BinaryOp, a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    // Atan2 and Pow have no simple SIMD implementation - use scalar fallback
    if matches!(op, BinaryOp::Atan2 | BinaryOp::Pow) {
        binary_scalar_f32(op, a, b, out, len);
        return;
    }

    match op {
        BinaryOp::Add => binary_add_f32(a, b, out, chunks),
        BinaryOp::Sub => binary_sub_f32(a, b, out, chunks),
        BinaryOp::Mul => binary_mul_f32(a, b, out, chunks),
        BinaryOp::Div => binary_div_f32(a, b, out, chunks),
        BinaryOp::Max => binary_max_f32(a, b, out, chunks),
        BinaryOp::Min => binary_min_f32(a, b, out, chunks),
        BinaryOp::Pow | BinaryOp::Atan2 => unreachable!(), // Handled above
    }

    // Handle tail with scalar
    if remainder > 0 {
        let offset = chunks * F32_LANES;
        binary_scalar_f32(op, a.add(offset), b.add(offset), out.add(offset), remainder);
    }
}

/// NEON binary operation for f64
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - All pointers must be valid for `len` elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn binary_f64(op: BinaryOp, a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    // Atan2 and Pow have no simple SIMD implementation - use scalar fallback
    if matches!(op, BinaryOp::Atan2 | BinaryOp::Pow) {
        binary_scalar_f64(op, a, b, out, len);
        return;
    }

    match op {
        BinaryOp::Add => binary_add_f64(a, b, out, chunks),
        BinaryOp::Sub => binary_sub_f64(a, b, out, chunks),
        BinaryOp::Mul => binary_mul_f64(a, b, out, chunks),
        BinaryOp::Div => binary_div_f64(a, b, out, chunks),
        BinaryOp::Max => binary_max_f64(a, b, out, chunks),
        BinaryOp::Min => binary_min_f64(a, b, out, chunks),
        BinaryOp::Pow | BinaryOp::Atan2 => unreachable!(), // Handled above
    }

    // Handle tail with scalar
    if remainder > 0 {
        let offset = chunks * F64_LANES;
        binary_scalar_f64(op, a.add(offset), b.add(offset), out.add(offset), remainder);
    }
}

// ============================================================================
// f32 kernels
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_add_f32(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vb = vld1q_f32(b.add(offset));
        let vr = vaddq_f32(va, vb);
        vst1q_f32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_sub_f32(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vb = vld1q_f32(b.add(offset));
        let vr = vsubq_f32(va, vb);
        vst1q_f32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_mul_f32(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vb = vld1q_f32(b.add(offset));
        let vr = vmulq_f32(va, vb);
        vst1q_f32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_div_f32(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vb = vld1q_f32(b.add(offset));
        let vr = vdivq_f32(va, vb);
        vst1q_f32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_max_f32(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vb = vld1q_f32(b.add(offset));
        let vr = vmaxq_f32(va, vb);
        vst1q_f32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_min_f32(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vb = vld1q_f32(b.add(offset));
        let vr = vminq_f32(va, vb);
        vst1q_f32(out.add(offset), vr);
    }
}

// ============================================================================
// f64 kernels
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_add_f64(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vb = vld1q_f64(b.add(offset));
        let vr = vaddq_f64(va, vb);
        vst1q_f64(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_sub_f64(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vb = vld1q_f64(b.add(offset));
        let vr = vsubq_f64(va, vb);
        vst1q_f64(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_mul_f64(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vb = vld1q_f64(b.add(offset));
        let vr = vmulq_f64(va, vb);
        vst1q_f64(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_div_f64(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vb = vld1q_f64(b.add(offset));
        let vr = vdivq_f64(va, vb);
        vst1q_f64(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_max_f64(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vb = vld1q_f64(b.add(offset));
        let vr = vmaxq_f64(va, vb);
        vst1q_f64(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn binary_min_f64(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vb = vld1q_f64(b.add(offset));
        let vr = vminq_f64(va, vb);
        vst1q_f64(out.add(offset), vr);
    }
}
