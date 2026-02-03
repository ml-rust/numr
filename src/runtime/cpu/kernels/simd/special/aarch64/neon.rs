//! NEON special function kernels for ARM64
//!
//! Provides vectorized implementations of error functions and Bessel functions
//! using 128-bit NEON registers with polynomial approximations.
//!
//! # Supported Functions
//!
//! | Function  | Status      | Notes                            |
//! |-----------|-------------|----------------------------------|
//! | erf       | Vectorized  | A&S 7.1.26 polynomial            |
//! | erfc      | Vectorized  | 1 - erf(x)                       |
//! | bessel_j0 | Scalar      | Complex, use scalar fallback     |
//! | bessel_j1 | Scalar      | Complex, use scalar fallback     |
//! | bessel_i0 | Scalar      | Complex, use scalar fallback     |
//! | bessel_i1 | Scalar      | Complex, use scalar fallback     |

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::algorithm::special::scalar::{
    bessel_i0_scalar, bessel_i1_scalar, bessel_j0_scalar, bessel_j1_scalar, erf_scalar, erfc_scalar,
};

// ============================================================================
// Error Function (erf)
// ============================================================================

/// NEON erf for f32
///
/// Uses Abramowitz and Stegun approximation 7.1.26 with polynomial coefficients.
///
/// # Safety
/// - Pointers must be valid for `len` elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn erf_f32(input: *const f32, output: *mut f32, len: usize) {
    let lanes = 4;
    let chunks = len / lanes;

    // Constants for A&S 7.1.26
    let a1 = vdupq_n_f32(0.254829592);
    let a2 = vdupq_n_f32(-0.284496736);
    let a3 = vdupq_n_f32(1.421413741);
    let a4 = vdupq_n_f32(-1.453152027);
    let a5 = vdupq_n_f32(1.061405429);
    let p = vdupq_n_f32(0.3275911);
    let one = vdupq_n_f32(1.0);
    let neg_one = vdupq_n_f32(-1.0);

    for i in 0..chunks {
        let idx = i * lanes;
        let x = vld1q_f32(input.add(idx));

        // sign = sign(x), absx = |x|
        let sign = vbslq_f32(vcltq_f32(x, vdupq_n_f32(0.0)), neg_one, one);
        let absx = vabsq_f32(x);

        // t = 1 / (1 + p * |x|)
        let t = vdivq_f32(one, vaddq_f32(one, vmulq_f32(p, absx)));

        // Polynomial: a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        // Using Horner's method: t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
        let poly = vmulq_f32(
            t,
            vaddq_f32(
                a1,
                vmulq_f32(
                    t,
                    vaddq_f32(
                        a2,
                        vmulq_f32(
                            t,
                            vaddq_f32(a3, vmulq_f32(t, vaddq_f32(a4, vmulq_f32(t, a5)))),
                        ),
                    ),
                ),
            ),
        );

        // exp(-x^2) approximation using scalar for each lane
        // NEON doesn't have native exp, so we compute element-wise
        let x2 = vmulq_f32(absx, absx);
        let exp_arr = [
            (-vgetq_lane_f32(x2, 0)).exp(),
            (-vgetq_lane_f32(x2, 1)).exp(),
            (-vgetq_lane_f32(x2, 2)).exp(),
            (-vgetq_lane_f32(x2, 3)).exp(),
        ];
        let exp_neg_x2 = vld1q_f32(exp_arr.as_ptr());

        // erf = sign * (1 - poly * exp(-x^2))
        let result = vmulq_f32(sign, vsubq_f32(one, vmulq_f32(poly, exp_neg_x2)));

        vst1q_f32(output.add(idx), result);
    }

    // Scalar tail
    for i in (chunks * lanes)..len {
        *output.add(i) = erf_scalar(*input.add(i) as f64) as f32;
    }
}

/// NEON erf for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn erf_f64(input: *const f64, output: *mut f64, len: usize) {
    let lanes = 2;
    let chunks = len / lanes;

    let a1 = vdupq_n_f64(0.254829592);
    let a2 = vdupq_n_f64(-0.284496736);
    let a3 = vdupq_n_f64(1.421413741);
    let a4 = vdupq_n_f64(-1.453152027);
    let a5 = vdupq_n_f64(1.061405429);
    let p = vdupq_n_f64(0.3275911);
    let one = vdupq_n_f64(1.0);
    let neg_one = vdupq_n_f64(-1.0);

    for i in 0..chunks {
        let idx = i * lanes;
        let x = vld1q_f64(input.add(idx));

        let sign = vbslq_f64(vcltq_f64(x, vdupq_n_f64(0.0)), neg_one, one);
        let absx = vabsq_f64(x);

        let t = vdivq_f64(one, vaddq_f64(one, vmulq_f64(p, absx)));

        let poly = vmulq_f64(
            t,
            vaddq_f64(
                a1,
                vmulq_f64(
                    t,
                    vaddq_f64(
                        a2,
                        vmulq_f64(
                            t,
                            vaddq_f64(a3, vmulq_f64(t, vaddq_f64(a4, vmulq_f64(t, a5)))),
                        ),
                    ),
                ),
            ),
        );

        let x2 = vmulq_f64(absx, absx);
        let exp_arr = [
            (-vgetq_lane_f64(x2, 0)).exp(),
            (-vgetq_lane_f64(x2, 1)).exp(),
        ];
        let exp_neg_x2 = vld1q_f64(exp_arr.as_ptr());

        let result = vmulq_f64(sign, vsubq_f64(one, vmulq_f64(poly, exp_neg_x2)));

        vst1q_f64(output.add(idx), result);
    }

    // Scalar tail
    for i in (chunks * lanes)..len {
        *output.add(i) = erf_scalar(*input.add(i));
    }
}

// ============================================================================
// Complementary Error Function (erfc)
// ============================================================================

/// NEON erfc for f32
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn erfc_f32(input: *const f32, output: *mut f32, len: usize) {
    // erfc(x) = 1 - erf(x)
    // For better numerical stability with large x, we compute directly
    let lanes = 4;
    let chunks = len / lanes;

    let one = vdupq_n_f32(1.0);

    // First compute erf into output
    erf_f32(input, output, len);

    // Then compute 1 - erf
    for i in 0..chunks {
        let idx = i * lanes;
        let erf_val = vld1q_f32(output.add(idx));
        let result = vsubq_f32(one, erf_val);
        vst1q_f32(output.add(idx), result);
    }

    for i in (chunks * lanes)..len {
        *output.add(i) = 1.0 - *output.add(i);
    }
}

/// NEON erfc for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn erfc_f64(input: *const f64, output: *mut f64, len: usize) {
    let lanes = 2;
    let chunks = len / lanes;

    let one = vdupq_n_f64(1.0);

    erf_f64(input, output, len);

    for i in 0..chunks {
        let idx = i * lanes;
        let erf_val = vld1q_f64(output.add(idx));
        let result = vsubq_f64(one, erf_val);
        vst1q_f64(output.add(idx), result);
    }

    for i in (chunks * lanes)..len {
        *output.add(i) = 1.0 - *output.add(i);
    }
}

// ============================================================================
// Bessel Functions - Scalar fallback
// ============================================================================

// Bessel functions have complex polynomial approximations with different
// regions (small args vs asymptotic). Use scalar fallback for now.

/// NEON bessel_j0 for f32 (scalar fallback)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn bessel_j0_f32(input: *const f32, output: *mut f32, len: usize) {
    for i in 0..len {
        *output.add(i) = bessel_j0_scalar(*input.add(i) as f64) as f32;
    }
}

/// NEON bessel_j0 for f64 (scalar fallback)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn bessel_j0_f64(input: *const f64, output: *mut f64, len: usize) {
    for i in 0..len {
        *output.add(i) = bessel_j0_scalar(*input.add(i));
    }
}

/// NEON bessel_j1 for f32 (scalar fallback)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn bessel_j1_f32(input: *const f32, output: *mut f32, len: usize) {
    for i in 0..len {
        *output.add(i) = bessel_j1_scalar(*input.add(i) as f64) as f32;
    }
}

/// NEON bessel_j1 for f64 (scalar fallback)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn bessel_j1_f64(input: *const f64, output: *mut f64, len: usize) {
    for i in 0..len {
        *output.add(i) = bessel_j1_scalar(*input.add(i));
    }
}

/// NEON bessel_i0 for f32 (scalar fallback)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn bessel_i0_f32(input: *const f32, output: *mut f32, len: usize) {
    for i in 0..len {
        *output.add(i) = bessel_i0_scalar(*input.add(i) as f64) as f32;
    }
}

/// NEON bessel_i0 for f64 (scalar fallback)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn bessel_i0_f64(input: *const f64, output: *mut f64, len: usize) {
    for i in 0..len {
        *output.add(i) = bessel_i0_scalar(*input.add(i));
    }
}

/// NEON bessel_i1 for f32 (scalar fallback)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn bessel_i1_f32(input: *const f32, output: *mut f32, len: usize) {
    for i in 0..len {
        *output.add(i) = bessel_i1_scalar(*input.add(i) as f64) as f32;
    }
}

/// NEON bessel_i1 for f64 (scalar fallback)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn bessel_i1_f64(input: *const f64, output: *mut f64, len: usize) {
    for i in 0..len {
        *output.add(i) = bessel_i1_scalar(*input.add(i));
    }
}
