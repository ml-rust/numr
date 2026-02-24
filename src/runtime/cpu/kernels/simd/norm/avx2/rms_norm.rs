//! AVX2 RMS normalization kernels

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{F32_LANES, F64_LANES, hsum_f32, hsum_f64};

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
