//! NEON comparison kernels for ARM64
//!
//! Provides vectorized element-wise comparison operations using 128-bit NEON registers.
//!
//! # Supported Operations
//!
//! - Eq: a == b
//! - Ne: a != b
//! - Lt: a < b
//! - Le: a <= b
//! - Gt: a > b
//! - Ge: a >= b
//!
//! # Output
//!
//! Results are stored as floats: 1.0 for true, 0.0 for false.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::ops::CompareOp;

const F32_LANES: usize = 4;
const F64_LANES: usize = 2;

/// NEON comparison for f32
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a`, `b`, and `out` must point to `len` valid f32 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn compare_f32(op: CompareOp, a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);

    match op {
        CompareOp::Eq => {
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let va = vld1q_f32(a.add(offset));
                let vb = vld1q_f32(b.add(offset));
                let mask = vceqq_f32(va, vb);
                let result = vbslq_f32(mask, one, zero);
                vst1q_f32(out.add(offset), result);
            }
        }
        CompareOp::Ne => {
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let va = vld1q_f32(a.add(offset));
                let vb = vld1q_f32(b.add(offset));
                let mask = vceqq_f32(va, vb);
                // Ne is !Eq, so swap one and zero in blend
                let result = vbslq_f32(mask, zero, one);
                vst1q_f32(out.add(offset), result);
            }
        }
        CompareOp::Lt => {
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let va = vld1q_f32(a.add(offset));
                let vb = vld1q_f32(b.add(offset));
                let mask = vcltq_f32(va, vb);
                let result = vbslq_f32(mask, one, zero);
                vst1q_f32(out.add(offset), result);
            }
        }
        CompareOp::Le => {
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let va = vld1q_f32(a.add(offset));
                let vb = vld1q_f32(b.add(offset));
                let mask = vcleq_f32(va, vb);
                let result = vbslq_f32(mask, one, zero);
                vst1q_f32(out.add(offset), result);
            }
        }
        CompareOp::Gt => {
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let va = vld1q_f32(a.add(offset));
                let vb = vld1q_f32(b.add(offset));
                let mask = vcgtq_f32(va, vb);
                let result = vbslq_f32(mask, one, zero);
                vst1q_f32(out.add(offset), result);
            }
        }
        CompareOp::Ge => {
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let va = vld1q_f32(a.add(offset));
                let vb = vld1q_f32(b.add(offset));
                let mask = vcgeq_f32(va, vb);
                let result = vbslq_f32(mask, one, zero);
                vst1q_f32(out.add(offset), result);
            }
        }
    }

    // Scalar tail
    if remainder > 0 {
        let offset = chunks * F32_LANES;
        super::super::compare_scalar_f32(
            op,
            a.add(offset),
            b.add(offset),
            out.add(offset),
            remainder,
        );
    }
}

/// NEON comparison for f64
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a`, `b`, and `out` must point to `len` valid f64 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn compare_f64(op: CompareOp, a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    let one = vdupq_n_f64(1.0);
    let zero = vdupq_n_f64(0.0);

    match op {
        CompareOp::Eq => {
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let va = vld1q_f64(a.add(offset));
                let vb = vld1q_f64(b.add(offset));
                let mask = vceqq_f64(va, vb);
                let result = vbslq_f64(mask, one, zero);
                vst1q_f64(out.add(offset), result);
            }
        }
        CompareOp::Ne => {
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let va = vld1q_f64(a.add(offset));
                let vb = vld1q_f64(b.add(offset));
                let mask = vceqq_f64(va, vb);
                let result = vbslq_f64(mask, zero, one);
                vst1q_f64(out.add(offset), result);
            }
        }
        CompareOp::Lt => {
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let va = vld1q_f64(a.add(offset));
                let vb = vld1q_f64(b.add(offset));
                let mask = vcltq_f64(va, vb);
                let result = vbslq_f64(mask, one, zero);
                vst1q_f64(out.add(offset), result);
            }
        }
        CompareOp::Le => {
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let va = vld1q_f64(a.add(offset));
                let vb = vld1q_f64(b.add(offset));
                let mask = vcleq_f64(va, vb);
                let result = vbslq_f64(mask, one, zero);
                vst1q_f64(out.add(offset), result);
            }
        }
        CompareOp::Gt => {
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let va = vld1q_f64(a.add(offset));
                let vb = vld1q_f64(b.add(offset));
                let mask = vcgtq_f64(va, vb);
                let result = vbslq_f64(mask, one, zero);
                vst1q_f64(out.add(offset), result);
            }
        }
        CompareOp::Ge => {
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let va = vld1q_f64(a.add(offset));
                let vb = vld1q_f64(b.add(offset));
                let mask = vcgeq_f64(va, vb);
                let result = vbslq_f64(mask, one, zero);
                vst1q_f64(out.add(offset), result);
            }
        }
    }

    // Scalar tail
    if remainder > 0 {
        let offset = chunks * F64_LANES;
        super::super::compare_scalar_f64(
            op,
            a.add(offset),
            b.add(offset),
            out.add(offset),
            remainder,
        );
    }
}
