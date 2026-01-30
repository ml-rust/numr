//! AVX-512 softmax kernels
//!
//! Uses SIMD for max-reduce, sum-reduce, and final normalization.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::math::avx512::{exp_f32, exp_f64};

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

/// AVX-512 softmax for f32
#[target_feature(enable = "avx512f")]
pub unsafe fn softmax_f32(a: *const f32, out: *mut f32, outer_size: usize, dim_size: usize) {
    let chunks = dim_size / F32_LANES;

    for o in 0..outer_size {
        let base = o * dim_size;

        // Step 1: SIMD max-reduce
        let mut max_vec = _mm512_set1_ps(f32::NEG_INFINITY);
        for c in 0..chunks {
            let offset = base + c * F32_LANES;
            let v = _mm512_loadu_ps(a.add(offset));
            max_vec = _mm512_max_ps(max_vec, v);
        }
        let mut max_val = _mm512_reduce_max_ps(max_vec);

        // Scalar tail for max
        for d in (chunks * F32_LANES)..dim_size {
            let val = *a.add(base + d);
            if val > max_val {
                max_val = val;
            }
        }

        // Step 2: Compute exp(x - max) and accumulate sum
        let v_max = _mm512_set1_ps(max_val);
        let mut sum_vec = _mm512_setzero_ps();

        for c in 0..chunks {
            let offset = base + c * F32_LANES;
            let v = _mm512_loadu_ps(a.add(offset));
            let diff = _mm512_sub_ps(v, v_max);
            let exp_v = exp_f32(diff);
            _mm512_storeu_ps(out.add(offset), exp_v);
            sum_vec = _mm512_add_ps(sum_vec, exp_v);
        }

        let mut sum = _mm512_reduce_add_ps(sum_vec);

        // Scalar tail for exp and sum
        for d in (chunks * F32_LANES)..dim_size {
            let val = *a.add(base + d);
            let exp_val = (val - max_val).exp();
            *out.add(base + d) = exp_val;
            sum += exp_val;
        }

        // Step 3: SIMD normalize by 1/sum
        let v_inv_sum = _mm512_set1_ps(1.0 / sum);

        for c in 0..chunks {
            let offset = base + c * F32_LANES;
            let v = _mm512_loadu_ps(out.add(offset));
            let normalized = _mm512_mul_ps(v, v_inv_sum);
            _mm512_storeu_ps(out.add(offset), normalized);
        }

        // Scalar tail for normalization
        let inv_sum = 1.0 / sum;
        for d in (chunks * F32_LANES)..dim_size {
            *out.add(base + d) *= inv_sum;
        }
    }
}

/// AVX-512 softmax for f64
#[target_feature(enable = "avx512f")]
pub unsafe fn softmax_f64(a: *const f64, out: *mut f64, outer_size: usize, dim_size: usize) {
    let chunks = dim_size / F64_LANES;

    for o in 0..outer_size {
        let base = o * dim_size;

        // Step 1: SIMD max-reduce
        let mut max_vec = _mm512_set1_pd(f64::NEG_INFINITY);
        for c in 0..chunks {
            let offset = base + c * F64_LANES;
            let v = _mm512_loadu_pd(a.add(offset));
            max_vec = _mm512_max_pd(max_vec, v);
        }
        let mut max_val = _mm512_reduce_max_pd(max_vec);

        // Scalar tail for max
        for d in (chunks * F64_LANES)..dim_size {
            let val = *a.add(base + d);
            if val > max_val {
                max_val = val;
            }
        }

        // Step 2: Compute exp(x - max) and accumulate sum
        let v_max = _mm512_set1_pd(max_val);
        let mut sum_vec = _mm512_setzero_pd();

        for c in 0..chunks {
            let offset = base + c * F64_LANES;
            let v = _mm512_loadu_pd(a.add(offset));
            let diff = _mm512_sub_pd(v, v_max);
            let exp_v = exp_f64(diff);
            _mm512_storeu_pd(out.add(offset), exp_v);
            sum_vec = _mm512_add_pd(sum_vec, exp_v);
        }

        let mut sum = _mm512_reduce_add_pd(sum_vec);

        // Scalar tail for exp and sum
        for d in (chunks * F64_LANES)..dim_size {
            let val = *a.add(base + d);
            let exp_val = (val - max_val).exp();
            *out.add(base + d) = exp_val;
            sum += exp_val;
        }

        // Step 3: SIMD normalize
        let v_inv_sum = _mm512_set1_pd(1.0 / sum);

        for c in 0..chunks {
            let offset = base + c * F64_LANES;
            let v = _mm512_loadu_pd(out.add(offset));
            let normalized = _mm512_mul_pd(v, v_inv_sum);
            _mm512_storeu_pd(out.add(offset), normalized);
        }

        // Scalar tail for normalization
        let inv_sum = 1.0 / sum;
        for d in (chunks * F64_LANES)..dim_size {
            *out.add(base + d) *= inv_sum;
        }
    }
}
