//! AVX2 layer normalization kernels

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{F32_LANES, F64_LANES, hsum_f32, hsum_f64};

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
