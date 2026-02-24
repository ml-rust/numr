//! AVX2 softmax backward kernels.
//!
//! Fused 2-pass: SIMD dot product, then SIMD elementwise output * (grad - dot).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::math::avx2::{hsum_f32, hsum_f64};

const F32_LANES: usize = 8;
const F64_LANES: usize = 4;

/// AVX2 softmax backward for f32.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn softmax_bwd_f32(
    grad: *const f32,
    output: *const f32,
    d_input: *mut f32,
    outer_size: usize,
    dim_size: usize,
) {
    let chunks = dim_size / F32_LANES;

    for o in 0..outer_size {
        let base = o * dim_size;

        // Pass 1: SIMD dot product — dot = sum(grad * output)
        let mut dot_vec = _mm256_setzero_ps();
        for c in 0..chunks {
            let offset = base + c * F32_LANES;
            let g = _mm256_loadu_ps(grad.add(offset));
            let out = _mm256_loadu_ps(output.add(offset));
            dot_vec = _mm256_fmadd_ps(g, out, dot_vec);
        }
        let mut dot = hsum_f32(dot_vec);

        // Scalar tail for dot
        for d in (chunks * F32_LANES)..dim_size {
            dot += *grad.add(base + d) * *output.add(base + d);
        }

        // Pass 2: SIMD d_input = output * (grad - dot)
        let v_dot = _mm256_set1_ps(dot);
        for c in 0..chunks {
            let offset = base + c * F32_LANES;
            let g = _mm256_loadu_ps(grad.add(offset));
            let out = _mm256_loadu_ps(output.add(offset));
            let shifted = _mm256_sub_ps(g, v_dot);
            let result = _mm256_mul_ps(out, shifted);
            _mm256_storeu_ps(d_input.add(offset), result);
        }

        // Scalar tail
        for d in (chunks * F32_LANES)..dim_size {
            let idx = base + d;
            *d_input.add(idx) = *output.add(idx) * (*grad.add(idx) - dot);
        }
    }
}

/// AVX2 softmax backward for f64.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn softmax_bwd_f64(
    grad: *const f64,
    output: *const f64,
    d_input: *mut f64,
    outer_size: usize,
    dim_size: usize,
) {
    let chunks = dim_size / F64_LANES;

    for o in 0..outer_size {
        let base = o * dim_size;

        // Pass 1: SIMD dot product
        let mut dot_vec = _mm256_setzero_pd();
        for c in 0..chunks {
            let offset = base + c * F64_LANES;
            let g = _mm256_loadu_pd(grad.add(offset));
            let out = _mm256_loadu_pd(output.add(offset));
            dot_vec = _mm256_fmadd_pd(g, out, dot_vec);
        }
        let mut dot = hsum_f64(dot_vec);

        for d in (chunks * F64_LANES)..dim_size {
            dot += *grad.add(base + d) * *output.add(base + d);
        }

        // Pass 2: d_input = output * (grad - dot)
        let v_dot = _mm256_set1_pd(dot);
        for c in 0..chunks {
            let offset = base + c * F64_LANES;
            let g = _mm256_loadu_pd(grad.add(offset));
            let out = _mm256_loadu_pd(output.add(offset));
            let shifted = _mm256_sub_pd(g, v_dot);
            let result = _mm256_mul_pd(out, shifted);
            _mm256_storeu_pd(d_input.add(offset), result);
        }

        for d in (chunks * F64_LANES)..dim_size {
            let idx = base + d;
            *d_input.add(idx) = *output.add(idx) * (*grad.add(idx) - dot);
        }
    }
}
