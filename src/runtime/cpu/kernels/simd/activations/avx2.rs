//! AVX2 activation kernels
//!
//! Vectorized implementations using 256-bit registers.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::math::avx2::{exp_f32, exp_f64, tanh_f32, tanh_f64};
use super::{
    elu_scalar_f32, elu_scalar_f64, gelu_scalar_f32, gelu_scalar_f64, leaky_relu_scalar_f32,
    leaky_relu_scalar_f64, sigmoid_scalar_f32, sigmoid_scalar_f64, silu_scalar_f32,
    silu_scalar_f64,
};

const F32_LANES: usize = 8;
const F64_LANES: usize = 4;

/// AVX2 sigmoid for f32
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn sigmoid_f32(a: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let one = _mm256_set1_ps(1.0);

    for c in 0..chunks {
        let offset = c * F32_LANES;
        let x = _mm256_loadu_ps(a.add(offset));
        let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        let exp_neg_x = exp_f32(neg_x);
        let result = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x));
        _mm256_storeu_ps(out.add(offset), result);
    }

    let processed = chunks * F32_LANES;
    if processed < len {
        sigmoid_scalar_f32(a.add(processed), out.add(processed), len - processed);
    }
}

/// AVX2 sigmoid for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn sigmoid_f64(a: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let one = _mm256_set1_pd(1.0);

    for c in 0..chunks {
        let offset = c * F64_LANES;
        let x = _mm256_loadu_pd(a.add(offset));
        let neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
        let exp_neg_x = exp_f64(neg_x);
        let result = _mm256_div_pd(one, _mm256_add_pd(one, exp_neg_x));
        _mm256_storeu_pd(out.add(offset), result);
    }

    let processed = chunks * F64_LANES;
    if processed < len {
        sigmoid_scalar_f64(a.add(processed), out.add(processed), len - processed);
    }
}

/// AVX2 SiLU for f32
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn silu_f32(a: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let one = _mm256_set1_ps(1.0);

    for c in 0..chunks {
        let offset = c * F32_LANES;
        let x = _mm256_loadu_ps(a.add(offset));
        let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        let exp_neg_x = exp_f32(neg_x);
        let result = _mm256_div_ps(x, _mm256_add_ps(one, exp_neg_x));
        _mm256_storeu_ps(out.add(offset), result);
    }

    let processed = chunks * F32_LANES;
    if processed < len {
        silu_scalar_f32(a.add(processed), out.add(processed), len - processed);
    }
}

/// AVX2 SiLU for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn silu_f64(a: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let one = _mm256_set1_pd(1.0);

    for c in 0..chunks {
        let offset = c * F64_LANES;
        let x = _mm256_loadu_pd(a.add(offset));
        let neg_x = _mm256_sub_pd(_mm256_setzero_pd(), x);
        let exp_neg_x = exp_f64(neg_x);
        let result = _mm256_div_pd(x, _mm256_add_pd(one, exp_neg_x));
        _mm256_storeu_pd(out.add(offset), result);
    }

    let processed = chunks * F64_LANES;
    if processed < len {
        silu_scalar_f64(a.add(processed), out.add(processed), len - processed);
    }
}

/// AVX2 GELU for f32
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gelu_f32(a: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let sqrt_2_over_pi = _mm256_set1_ps(0.7978845608);
    let tanh_coef = _mm256_set1_ps(0.044715);

    for c in 0..chunks {
        let offset = c * F32_LANES;
        let x = _mm256_loadu_ps(a.add(offset));

        let x_cubed = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
        let inner = _mm256_mul_ps(sqrt_2_over_pi, _mm256_fmadd_ps(tanh_coef, x_cubed, x));

        let tanh_inner = tanh_f32(inner);
        let result = _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, tanh_inner)));

        _mm256_storeu_ps(out.add(offset), result);
    }

    let processed = chunks * F32_LANES;
    if processed < len {
        gelu_scalar_f32(a.add(processed), out.add(processed), len - processed);
    }
}

/// AVX2 GELU for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gelu_f64(a: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let half = _mm256_set1_pd(0.5);
    let one = _mm256_set1_pd(1.0);
    let sqrt_2_over_pi = _mm256_set1_pd(0.7978845608028654);
    let tanh_coef = _mm256_set1_pd(0.044715);

    for c in 0..chunks {
        let offset = c * F64_LANES;
        let x = _mm256_loadu_pd(a.add(offset));

        let x_cubed = _mm256_mul_pd(_mm256_mul_pd(x, x), x);
        let inner = _mm256_mul_pd(sqrt_2_over_pi, _mm256_fmadd_pd(tanh_coef, x_cubed, x));

        let tanh_inner = tanh_f64(inner);
        let result = _mm256_mul_pd(half, _mm256_mul_pd(x, _mm256_add_pd(one, tanh_inner)));

        _mm256_storeu_pd(out.add(offset), result);
    }

    let processed = chunks * F64_LANES;
    if processed < len {
        gelu_scalar_f64(a.add(processed), out.add(processed), len - processed);
    }
}

/// AVX2 Leaky ReLU for f32
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn leaky_relu_f32(a: *const f32, out: *mut f32, len: usize, negative_slope: f32) {
    let chunks = len / F32_LANES;
    let v_slope = _mm256_set1_ps(negative_slope);
    let zero = _mm256_setzero_ps();

    for c in 0..chunks {
        let offset = c * F32_LANES;
        let x = _mm256_loadu_ps(a.add(offset));
        let mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
        let scaled = _mm256_mul_ps(v_slope, x);
        let result = _mm256_blendv_ps(scaled, x, mask);
        _mm256_storeu_ps(out.add(offset), result);
    }

    let processed = chunks * F32_LANES;
    if processed < len {
        leaky_relu_scalar_f32(
            a.add(processed),
            out.add(processed),
            len - processed,
            negative_slope,
        );
    }
}

/// AVX2 Leaky ReLU for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn leaky_relu_f64(a: *const f64, out: *mut f64, len: usize, negative_slope: f64) {
    let chunks = len / F64_LANES;
    let v_slope = _mm256_set1_pd(negative_slope);
    let zero = _mm256_setzero_pd();

    for c in 0..chunks {
        let offset = c * F64_LANES;
        let x = _mm256_loadu_pd(a.add(offset));
        let mask = _mm256_cmp_pd(x, zero, _CMP_GT_OQ);
        let scaled = _mm256_mul_pd(v_slope, x);
        let result = _mm256_blendv_pd(scaled, x, mask);
        _mm256_storeu_pd(out.add(offset), result);
    }

    let processed = chunks * F64_LANES;
    if processed < len {
        leaky_relu_scalar_f64(
            a.add(processed),
            out.add(processed),
            len - processed,
            negative_slope,
        );
    }
}

/// AVX2 ELU for f32
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn elu_f32(a: *const f32, out: *mut f32, len: usize, alpha: f32) {
    let chunks = len / F32_LANES;
    let v_alpha = _mm256_set1_ps(alpha);
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();

    for c in 0..chunks {
        let offset = c * F32_LANES;
        let x = _mm256_loadu_ps(a.add(offset));
        let mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
        let exp_x = exp_f32(x);
        let neg_result = _mm256_mul_ps(v_alpha, _mm256_sub_ps(exp_x, one));
        let result = _mm256_blendv_ps(neg_result, x, mask);
        _mm256_storeu_ps(out.add(offset), result);
    }

    let processed = chunks * F32_LANES;
    if processed < len {
        elu_scalar_f32(a.add(processed), out.add(processed), len - processed, alpha);
    }
}

/// AVX2 ELU for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn elu_f64(a: *const f64, out: *mut f64, len: usize, alpha: f64) {
    let chunks = len / F64_LANES;
    let v_alpha = _mm256_set1_pd(alpha);
    let one = _mm256_set1_pd(1.0);
    let zero = _mm256_setzero_pd();

    for c in 0..chunks {
        let offset = c * F64_LANES;
        let x = _mm256_loadu_pd(a.add(offset));
        let mask = _mm256_cmp_pd(x, zero, _CMP_GT_OQ);
        let exp_x = exp_f64(x);
        let neg_result = _mm256_mul_pd(v_alpha, _mm256_sub_pd(exp_x, one));
        let result = _mm256_blendv_pd(neg_result, x, mask);
        _mm256_storeu_pd(out.add(offset), result);
    }

    let processed = chunks * F64_LANES;
    if processed < len {
        elu_scalar_f64(a.add(processed), out.add(processed), len - processed, alpha);
    }
}
