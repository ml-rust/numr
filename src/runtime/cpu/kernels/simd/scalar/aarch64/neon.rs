//! NEON scalar operation kernels for ARM64
//!
//! Provides vectorized tensor-scalar operations using 128-bit NEON registers.
//!
//! # Supported Operations
//!
//! - Add: a + scalar
//! - Sub: a - scalar
//! - Mul: a * scalar
//! - Div: a / scalar
//! - Max: max(a, scalar)
//! - Min: min(a, scalar)
//!
//! # Unsupported (scalar fallback)
//!
//! - Pow: requires libm, no direct SIMD instruction
//! - Atan2: requires libm

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::ops::BinaryOp;

const F32_LANES: usize = 4;
const F64_LANES: usize = 2;

/// NEON scalar operation for f32
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a` and `out` must point to `len` valid f32 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn scalar_f32(op: BinaryOp, a: *const f32, scalar: f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    let vs = vdupq_n_f32(scalar);

    match op {
        BinaryOp::Add => {
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let va = vld1q_f32(a.add(offset));
                let result = vaddq_f32(va, vs);
                vst1q_f32(out.add(offset), result);
            }
        }
        BinaryOp::Sub => {
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let va = vld1q_f32(a.add(offset));
                let result = vsubq_f32(va, vs);
                vst1q_f32(out.add(offset), result);
            }
        }
        BinaryOp::Mul => {
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let va = vld1q_f32(a.add(offset));
                let result = vmulq_f32(va, vs);
                vst1q_f32(out.add(offset), result);
            }
        }
        BinaryOp::Div => {
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let va = vld1q_f32(a.add(offset));
                let result = vdivq_f32(va, vs);
                vst1q_f32(out.add(offset), result);
            }
        }
        BinaryOp::Max => {
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let va = vld1q_f32(a.add(offset));
                let result = vmaxq_f32(va, vs);
                vst1q_f32(out.add(offset), result);
            }
        }
        BinaryOp::Min => {
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let va = vld1q_f32(a.add(offset));
                let result = vminq_f32(va, vs);
                vst1q_f32(out.add(offset), result);
            }
        }
        BinaryOp::Pow | BinaryOp::Atan2 => {
            // Fallback to scalar for unsupported ops
            super::super::scalar_scalar_f32(op, a, scalar, out, len);
            return;
        }
    }

    // Scalar tail
    if remainder > 0 {
        let offset = chunks * F32_LANES;
        super::super::scalar_scalar_f32(op, a.add(offset), scalar, out.add(offset), remainder);
    }
}

/// NEON scalar operation for f64
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a` and `out` must point to `len` valid f64 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn scalar_f64(op: BinaryOp, a: *const f64, scalar: f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    let vs = vdupq_n_f64(scalar);

    match op {
        BinaryOp::Add => {
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let va = vld1q_f64(a.add(offset));
                let result = vaddq_f64(va, vs);
                vst1q_f64(out.add(offset), result);
            }
        }
        BinaryOp::Sub => {
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let va = vld1q_f64(a.add(offset));
                let result = vsubq_f64(va, vs);
                vst1q_f64(out.add(offset), result);
            }
        }
        BinaryOp::Mul => {
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let va = vld1q_f64(a.add(offset));
                let result = vmulq_f64(va, vs);
                vst1q_f64(out.add(offset), result);
            }
        }
        BinaryOp::Div => {
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let va = vld1q_f64(a.add(offset));
                let result = vdivq_f64(va, vs);
                vst1q_f64(out.add(offset), result);
            }
        }
        BinaryOp::Max => {
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let va = vld1q_f64(a.add(offset));
                let result = vmaxq_f64(va, vs);
                vst1q_f64(out.add(offset), result);
            }
        }
        BinaryOp::Min => {
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let va = vld1q_f64(a.add(offset));
                let result = vminq_f64(va, vs);
                vst1q_f64(out.add(offset), result);
            }
        }
        BinaryOp::Pow | BinaryOp::Atan2 => {
            // Fallback to scalar for unsupported ops
            super::super::scalar_scalar_f64(op, a, scalar, out, len);
            return;
        }
    }

    // Scalar tail
    if remainder > 0 {
        let offset = chunks * F64_LANES;
        super::super::scalar_scalar_f64(op, a.add(offset), scalar, out.add(offset), remainder);
    }
}

/// NEON reverse scalar subtract for f32: out[i] = scalar - a[i]
pub unsafe fn rsub_scalar_f32(a: *const f32, scalar: f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;
    let vs = vdupq_n_f32(scalar);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vr = vsubq_f32(vs, va);
        vst1q_f32(out.add(offset), vr);
    }

    for i in 0..remainder {
        let offset = chunks * F32_LANES + i;
        *out.add(offset) = scalar - *a.add(offset);
    }
}

/// NEON reverse scalar subtract for f64: out[i] = scalar - a[i]
pub unsafe fn rsub_scalar_f64(a: *const f64, scalar: f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;
    let vs = vdupq_n_f64(scalar);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vr = vsubq_f64(vs, va);
        vst1q_f64(out.add(offset), vr);
    }

    for i in 0..remainder {
        let offset = chunks * F64_LANES + i;
        *out.add(offset) = scalar - *a.add(offset);
    }
}
