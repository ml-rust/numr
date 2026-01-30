//! AVX2 softmax kernels
//!
//! Uses SIMD for max-reduce, sum-reduce, and final normalization.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::math::avx2::{exp_f32, exp_f64, hmax_f32, hmax_f64, hsum_f32, hsum_f64};

const F32_LANES: usize = 8;
const F64_LANES: usize = 4;

/// AVX2 softmax for f32
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn softmax_f32(a: *const f32, out: *mut f32, outer_size: usize, dim_size: usize) {
    let chunks = dim_size / F32_LANES;

    for o in 0..outer_size {
        let base = o * dim_size;

        // Step 1: SIMD max-reduce
        let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
        for c in 0..chunks {
            let offset = base + c * F32_LANES;
            let v = _mm256_loadu_ps(a.add(offset));
            max_vec = _mm256_max_ps(max_vec, v);
        }
        let mut max_val = hmax_f32(max_vec);

        // Scalar tail for max
        for d in (chunks * F32_LANES)..dim_size {
            let val = *a.add(base + d);
            if val > max_val {
                max_val = val;
            }
        }

        // Step 2: Compute exp(x - max) and accumulate sum
        let v_max = _mm256_set1_ps(max_val);
        let mut sum_vec = _mm256_setzero_ps();

        for c in 0..chunks {
            let offset = base + c * F32_LANES;
            let v = _mm256_loadu_ps(a.add(offset));
            let diff = _mm256_sub_ps(v, v_max);
            let exp_v = exp_f32(diff);
            _mm256_storeu_ps(out.add(offset), exp_v);
            sum_vec = _mm256_add_ps(sum_vec, exp_v);
        }

        let mut sum = hsum_f32(sum_vec);

        // Scalar tail for exp and sum
        for d in (chunks * F32_LANES)..dim_size {
            let val = *a.add(base + d);
            let exp_val = (val - max_val).exp();
            *out.add(base + d) = exp_val;
            sum += exp_val;
        }

        // Step 3: SIMD normalize
        let v_inv_sum = _mm256_set1_ps(1.0 / sum);

        for c in 0..chunks {
            let offset = base + c * F32_LANES;
            let v = _mm256_loadu_ps(out.add(offset));
            let normalized = _mm256_mul_ps(v, v_inv_sum);
            _mm256_storeu_ps(out.add(offset), normalized);
        }

        // Scalar tail for normalization
        let inv_sum = 1.0 / sum;
        for d in (chunks * F32_LANES)..dim_size {
            *out.add(base + d) *= inv_sum;
        }
    }
}

/// AVX2 softmax for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn softmax_f64(a: *const f64, out: *mut f64, outer_size: usize, dim_size: usize) {
    let chunks = dim_size / F64_LANES;

    for o in 0..outer_size {
        let base = o * dim_size;

        // Step 1: SIMD max-reduce
        let mut max_vec = _mm256_set1_pd(f64::NEG_INFINITY);
        for c in 0..chunks {
            let offset = base + c * F64_LANES;
            let v = _mm256_loadu_pd(a.add(offset));
            max_vec = _mm256_max_pd(max_vec, v);
        }
        let mut max_val = hmax_f64(max_vec);

        // Scalar tail for max
        for d in (chunks * F64_LANES)..dim_size {
            let val = *a.add(base + d);
            if val > max_val {
                max_val = val;
            }
        }

        // Step 2: Compute exp(x - max) and accumulate sum
        let v_max = _mm256_set1_pd(max_val);
        let mut sum_vec = _mm256_setzero_pd();

        for c in 0..chunks {
            let offset = base + c * F64_LANES;
            let v = _mm256_loadu_pd(a.add(offset));
            let diff = _mm256_sub_pd(v, v_max);
            let exp_v = exp_f64(diff);
            _mm256_storeu_pd(out.add(offset), exp_v);
            sum_vec = _mm256_add_pd(sum_vec, exp_v);
        }

        let mut sum = hsum_f64(sum_vec);

        // Scalar tail for exp and sum
        for d in (chunks * F64_LANES)..dim_size {
            let val = *a.add(base + d);
            let exp_val = (val - max_val).exp();
            *out.add(base + d) = exp_val;
            sum += exp_val;
        }

        // Step 3: SIMD normalize
        let v_inv_sum = _mm256_set1_pd(1.0 / sum);

        for c in 0..chunks {
            let offset = base + c * F64_LANES;
            let v = _mm256_loadu_pd(out.add(offset));
            let normalized = _mm256_mul_pd(v, v_inv_sum);
            _mm256_storeu_pd(out.add(offset), normalized);
        }

        // Scalar tail for normalization
        let inv_sum = 1.0 / sum;
        for d in (chunks * F64_LANES)..dim_size {
            *out.add(base + d) *= inv_sum;
        }
    }
}
