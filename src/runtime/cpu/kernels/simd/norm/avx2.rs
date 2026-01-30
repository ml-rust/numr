//! AVX2 normalization kernels
//!
//! SIMD-optimized RMS norm and layer norm with manual horizontal reductions.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{
    layer_norm_scalar_f32, layer_norm_scalar_f64, rms_norm_scalar_f32, rms_norm_scalar_f64,
};

const F32_LANES: usize = 8;
const F64_LANES: usize = 4;

// ============================================================================
// Horizontal reduction helpers
// ============================================================================

#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn hsum_f32(v: __m256) -> f32 {
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(low, high);
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
}

#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn hsum_f64(v: __m256d) -> f64 {
    let high = _mm256_extractf128_pd(v, 1);
    let low = _mm256_castpd256_pd128(v);
    let sum128 = _mm_add_pd(low, high);
    let shuf = _mm_unpackhi_pd(sum128, sum128);
    let sum64 = _mm_add_sd(sum128, shuf);
    _mm_cvtsd_f64(sum64)
}

// ============================================================================
// RMS Norm
// ============================================================================

/// AVX2 RMS normalization for f32
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn rms_norm_f32(
    input: *const f32,
    weight: *const f32,
    out: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let chunks = hidden_size / F32_LANES;

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        // SIMD sum of squares using FMA
        let mut acc = _mm256_setzero_ps();
        for c in 0..chunks {
            let offset = row_start + c * F32_LANES;
            let v = _mm256_loadu_ps(input.add(offset));
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        let mut sum_sq = hsum_f32(acc);

        for i in (chunks * F32_LANES)..hidden_size {
            let x = *input.add(row_start + i);
            sum_sq += x * x;
        }

        let inv_rms = 1.0 / (sum_sq / hidden_size as f32 + eps).sqrt();
        let v_inv_rms = _mm256_set1_ps(inv_rms);

        for c in 0..chunks {
            let offset = row_start + c * F32_LANES;
            let w_offset = c * F32_LANES;
            let v_input = _mm256_loadu_ps(input.add(offset));
            let v_weight = _mm256_loadu_ps(weight.add(w_offset));
            let v_result = _mm256_mul_ps(_mm256_mul_ps(v_input, v_inv_rms), v_weight);
            _mm256_storeu_ps(out.add(offset), v_result);
        }

        for i in (chunks * F32_LANES)..hidden_size {
            let x = *input.add(row_start + i);
            let w = *weight.add(i);
            *out.add(row_start + i) = x * inv_rms * w;
        }
    }
}

/// AVX2 RMS normalization for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn rms_norm_f64(
    input: *const f64,
    weight: *const f64,
    out: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    let chunks = hidden_size / F64_LANES;

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        let mut acc = _mm256_setzero_pd();
        for c in 0..chunks {
            let offset = row_start + c * F64_LANES;
            let v = _mm256_loadu_pd(input.add(offset));
            acc = _mm256_fmadd_pd(v, v, acc);
        }
        let mut sum_sq = hsum_f64(acc);

        for i in (chunks * F64_LANES)..hidden_size {
            let x = *input.add(row_start + i);
            sum_sq += x * x;
        }

        let inv_rms = 1.0 / (sum_sq / hidden_size as f64 + eps).sqrt();
        let v_inv_rms = _mm256_set1_pd(inv_rms);

        for c in 0..chunks {
            let offset = row_start + c * F64_LANES;
            let w_offset = c * F64_LANES;
            let v_input = _mm256_loadu_pd(input.add(offset));
            let v_weight = _mm256_loadu_pd(weight.add(w_offset));
            let v_result = _mm256_mul_pd(_mm256_mul_pd(v_input, v_inv_rms), v_weight);
            _mm256_storeu_pd(out.add(offset), v_result);
        }

        for i in (chunks * F64_LANES)..hidden_size {
            let x = *input.add(row_start + i);
            let w = *weight.add(i);
            *out.add(row_start + i) = x * inv_rms * w;
        }
    }
}

// ============================================================================
// Layer Norm
// ============================================================================

/// AVX2 Layer normalization for f32
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn layer_norm_f32(
    input: *const f32,
    weight: *const f32,
    bias: *const f32,
    out: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let chunks = hidden_size / F32_LANES;

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        // SIMD sum for mean
        let mut sum_acc = _mm256_setzero_ps();
        for c in 0..chunks {
            let v = _mm256_loadu_ps(input.add(row_start + c * F32_LANES));
            sum_acc = _mm256_add_ps(sum_acc, v);
        }
        let mut sum = hsum_f32(sum_acc);

        for i in (chunks * F32_LANES)..hidden_size {
            sum += *input.add(row_start + i);
        }
        let mean = sum / hidden_size as f32;
        let v_mean = _mm256_set1_ps(mean);

        // SIMD variance computation
        let mut var_acc = _mm256_setzero_ps();
        for c in 0..chunks {
            let v = _mm256_loadu_ps(input.add(row_start + c * F32_LANES));
            let diff = _mm256_sub_ps(v, v_mean);
            var_acc = _mm256_fmadd_ps(diff, diff, var_acc);
        }
        let mut var_sum = hsum_f32(var_acc);

        for i in (chunks * F32_LANES)..hidden_size {
            let diff = *input.add(row_start + i) - mean;
            var_sum += diff * diff;
        }
        let inv_std = 1.0 / (var_sum / hidden_size as f32 + eps).sqrt();
        let v_inv_std = _mm256_set1_ps(inv_std);

        // SIMD normalization with weight and bias
        for c in 0..chunks {
            let offset = row_start + c * F32_LANES;
            let w_offset = c * F32_LANES;
            let v_input = _mm256_loadu_ps(input.add(offset));
            let v_weight = _mm256_loadu_ps(weight.add(w_offset));
            let v_bias = _mm256_loadu_ps(bias.add(w_offset));

            let diff = _mm256_sub_ps(v_input, v_mean);
            let normalized = _mm256_mul_ps(diff, v_inv_std);
            let scaled = _mm256_mul_ps(normalized, v_weight);
            let result = _mm256_add_ps(scaled, v_bias);

            _mm256_storeu_ps(out.add(offset), result);
        }

        for i in (chunks * F32_LANES)..hidden_size {
            let x = *input.add(row_start + i);
            let w = *weight.add(i);
            let b = *bias.add(i);
            *out.add(row_start + i) = (x - mean) * inv_std * w + b;
        }
    }
}

/// AVX2 Layer normalization for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn layer_norm_f64(
    input: *const f64,
    weight: *const f64,
    bias: *const f64,
    out: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    let chunks = hidden_size / F64_LANES;

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        let mut sum_acc = _mm256_setzero_pd();
        for c in 0..chunks {
            let v = _mm256_loadu_pd(input.add(row_start + c * F64_LANES));
            sum_acc = _mm256_add_pd(sum_acc, v);
        }
        let mut sum = hsum_f64(sum_acc);

        for i in (chunks * F64_LANES)..hidden_size {
            sum += *input.add(row_start + i);
        }
        let mean = sum / hidden_size as f64;
        let v_mean = _mm256_set1_pd(mean);

        let mut var_acc = _mm256_setzero_pd();
        for c in 0..chunks {
            let v = _mm256_loadu_pd(input.add(row_start + c * F64_LANES));
            let diff = _mm256_sub_pd(v, v_mean);
            var_acc = _mm256_fmadd_pd(diff, diff, var_acc);
        }
        let mut var_sum = hsum_f64(var_acc);

        for i in (chunks * F64_LANES)..hidden_size {
            let diff = *input.add(row_start + i) - mean;
            var_sum += diff * diff;
        }
        let inv_std = 1.0 / (var_sum / hidden_size as f64 + eps).sqrt();
        let v_inv_std = _mm256_set1_pd(inv_std);

        for c in 0..chunks {
            let offset = row_start + c * F64_LANES;
            let w_offset = c * F64_LANES;
            let v_input = _mm256_loadu_pd(input.add(offset));
            let v_weight = _mm256_loadu_pd(weight.add(w_offset));
            let v_bias = _mm256_loadu_pd(bias.add(w_offset));

            let diff = _mm256_sub_pd(v_input, v_mean);
            let normalized = _mm256_mul_pd(diff, v_inv_std);
            let scaled = _mm256_mul_pd(normalized, v_weight);
            let result = _mm256_add_pd(scaled, v_bias);

            _mm256_storeu_pd(out.add(offset), result);
        }

        for i in (chunks * F64_LANES)..hidden_size {
            let x = *input.add(row_start + i);
            let w = *weight.add(i);
            let b = *bias.add(i);
            *out.add(row_start + i) = (x - mean) * inv_std * w + b;
        }
    }
}

// Suppress unused warnings for scalar fallback imports used in dispatch
const _: () = {
    let _ = rms_norm_scalar_f32 as unsafe fn(*const f32, *const f32, *mut f32, usize, usize, f32);
    let _ = rms_norm_scalar_f64 as unsafe fn(*const f64, *const f64, *mut f64, usize, usize, f64);
    let _ = layer_norm_scalar_f32
        as unsafe fn(*const f32, *const f32, *const f32, *mut f32, usize, usize, f32);
    let _ = layer_norm_scalar_f64
        as unsafe fn(*const f64, *const f64, *const f64, *mut f64, usize, usize, f64);
};
