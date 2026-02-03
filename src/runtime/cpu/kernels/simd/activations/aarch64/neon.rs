//! NEON activation function kernels for ARM64
//!
//! Provides vectorized implementations of common neural network activations
//! using 128-bit NEON registers.
//!
//! # Supported Activations
//!
//! - Sigmoid: 1 / (1 + exp(-x))
//! - SiLU (Swish): x * sigmoid(x)
//! - GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//! - Leaky ReLU: max(negative_slope * x, x)
//! - ELU: x if x > 0, else alpha * (exp(x) - 1)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::super::math::aarch64::neon::{exp_f32, exp_f64, tanh_f32, tanh_f64};

const F32_LANES: usize = 4;
const F64_LANES: usize = 2;

// ============================================================================
// Sigmoid: 1 / (1 + exp(-x))
// ============================================================================

/// NEON sigmoid for f32
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a` and `out` must point to `len` valid elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn sigmoid_f32(a: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;
    let one = vdupq_n_f32(1.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let x = vld1q_f32(a.add(offset));
        let neg_x = vnegq_f32(x);
        let exp_neg_x = exp_f32(neg_x);
        let result = vdivq_f32(one, vaddq_f32(one, exp_neg_x));
        vst1q_f32(out.add(offset), result);
    }

    // Scalar tail
    if remainder > 0 {
        let offset = chunks * F32_LANES;
        super::super::sigmoid_scalar_f32(a.add(offset), out.add(offset), remainder);
    }
}

/// NEON sigmoid for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn sigmoid_f64(a: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;
    let one = vdupq_n_f64(1.0);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let x = vld1q_f64(a.add(offset));
        let neg_x = vnegq_f64(x);
        let exp_neg_x = exp_f64(neg_x);
        let result = vdivq_f64(one, vaddq_f64(one, exp_neg_x));
        vst1q_f64(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        super::super::sigmoid_scalar_f64(a.add(offset), out.add(offset), remainder);
    }
}

// ============================================================================
// SiLU (Swish): x * sigmoid(x) = x / (1 + exp(-x))
// ============================================================================

/// NEON SiLU for f32
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn silu_f32(a: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;
    let one = vdupq_n_f32(1.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let x = vld1q_f32(a.add(offset));
        let neg_x = vnegq_f32(x);
        let exp_neg_x = exp_f32(neg_x);
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let result = vdivq_f32(x, vaddq_f32(one, exp_neg_x));
        vst1q_f32(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        super::super::silu_scalar_f32(a.add(offset), out.add(offset), remainder);
    }
}

/// NEON SiLU for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn silu_f64(a: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;
    let one = vdupq_n_f64(1.0);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let x = vld1q_f64(a.add(offset));
        let neg_x = vnegq_f64(x);
        let exp_neg_x = exp_f64(neg_x);
        let result = vdivq_f64(x, vaddq_f64(one, exp_neg_x));
        vst1q_f64(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        super::super::silu_scalar_f64(a.add(offset), out.add(offset), remainder);
    }
}

// ============================================================================
// GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ============================================================================

/// NEON GELU for f32
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gelu_f32(a: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    let sqrt_2_over_pi = vdupq_n_f32(0.7978845608);
    let coef = vdupq_n_f32(0.044715);
    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let x = vld1q_f32(a.add(offset));

        // x^3 = x * x * x
        let x2 = vmulq_f32(x, x);
        let x3 = vmulq_f32(x2, x);

        // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
        let inner = vmulq_f32(sqrt_2_over_pi, vfmaq_f32(x, coef, x3));

        // tanh(inner)
        let tanh_inner = tanh_f32(inner);

        // result = 0.5 * x * (1 + tanh(inner))
        let result = vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, tanh_inner));

        vst1q_f32(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        super::super::gelu_scalar_f32(a.add(offset), out.add(offset), remainder);
    }
}

/// NEON GELU for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gelu_f64(a: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    let sqrt_2_over_pi = vdupq_n_f64(0.7978845608028654);
    let coef = vdupq_n_f64(0.044715);
    let half = vdupq_n_f64(0.5);
    let one = vdupq_n_f64(1.0);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let x = vld1q_f64(a.add(offset));

        let x2 = vmulq_f64(x, x);
        let x3 = vmulq_f64(x2, x);

        let inner = vmulq_f64(sqrt_2_over_pi, vfmaq_f64(x, coef, x3));
        let tanh_inner = tanh_f64(inner);

        let result = vmulq_f64(vmulq_f64(half, x), vaddq_f64(one, tanh_inner));

        vst1q_f64(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        super::super::gelu_scalar_f64(a.add(offset), out.add(offset), remainder);
    }
}

// ============================================================================
// Leaky ReLU: max(negative_slope * x, x) = x > 0 ? x : negative_slope * x
// ============================================================================

/// NEON Leaky ReLU for f32
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn leaky_relu_f32(a: *const f32, out: *mut f32, len: usize, negative_slope: f32) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    let v_slope = vdupq_n_f32(negative_slope);
    let zero = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let x = vld1q_f32(a.add(offset));

        // neg_part = negative_slope * x
        let neg_part = vmulq_f32(x, v_slope);

        // mask = x > 0
        let mask = vcgtq_f32(x, zero);

        // result = mask ? x : neg_part
        let result = vbslq_f32(mask, x, neg_part);

        vst1q_f32(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        super::super::leaky_relu_scalar_f32(
            a.add(offset),
            out.add(offset),
            remainder,
            negative_slope,
        );
    }
}

/// NEON Leaky ReLU for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn leaky_relu_f64(a: *const f64, out: *mut f64, len: usize, negative_slope: f64) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    let v_slope = vdupq_n_f64(negative_slope);
    let zero = vdupq_n_f64(0.0);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let x = vld1q_f64(a.add(offset));

        let neg_part = vmulq_f64(x, v_slope);
        let mask = vcgtq_f64(x, zero);
        let result = vbslq_f64(mask, x, neg_part);

        vst1q_f64(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        super::super::leaky_relu_scalar_f64(
            a.add(offset),
            out.add(offset),
            remainder,
            negative_slope,
        );
    }
}

// ============================================================================
// ELU: x if x > 0, else alpha * (exp(x) - 1)
// ============================================================================

/// NEON ELU for f32
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn elu_f32(a: *const f32, out: *mut f32, len: usize, alpha: f32) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    let v_alpha = vdupq_n_f32(alpha);
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let x = vld1q_f32(a.add(offset));

        // neg_part = alpha * (exp(x) - 1)
        let exp_x = exp_f32(x);
        let neg_part = vmulq_f32(v_alpha, vsubq_f32(exp_x, one));

        // mask = x > 0
        let mask = vcgtq_f32(x, zero);

        // result = mask ? x : neg_part
        let result = vbslq_f32(mask, x, neg_part);

        vst1q_f32(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        super::super::elu_scalar_f32(a.add(offset), out.add(offset), remainder, alpha);
    }
}

/// NEON ELU for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn elu_f64(a: *const f64, out: *mut f64, len: usize, alpha: f64) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    let v_alpha = vdupq_n_f64(alpha);
    let zero = vdupq_n_f64(0.0);
    let one = vdupq_n_f64(1.0);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let x = vld1q_f64(a.add(offset));

        let exp_x = exp_f64(x);
        let neg_part = vmulq_f64(v_alpha, vsubq_f64(exp_x, one));
        let mask = vcgtq_f64(x, zero);
        let result = vbslq_f64(mask, x, neg_part);

        vst1q_f64(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        super::super::elu_scalar_f64(a.add(offset), out.add(offset), remainder, alpha);
    }
}
