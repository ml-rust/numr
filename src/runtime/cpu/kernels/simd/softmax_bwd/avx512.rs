//! AVX-512 softmax backward kernels.
//!
//! Fused 2-pass: SIMD dot product, then SIMD elementwise output * (grad - dot).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

/// AVX-512 softmax backward for f32.
#[target_feature(enable = "avx512f")]
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

        // Pass 1: SIMD dot product
        let mut dot_vec = _mm512_setzero_ps();
        for c in 0..chunks {
            let offset = base + c * F32_LANES;
            let g = _mm512_loadu_ps(grad.add(offset));
            let out = _mm512_loadu_ps(output.add(offset));
            dot_vec = _mm512_fmadd_ps(g, out, dot_vec);
        }
        let mut dot = _mm512_reduce_add_ps(dot_vec);

        for d in (chunks * F32_LANES)..dim_size {
            dot += *grad.add(base + d) * *output.add(base + d);
        }

        // Pass 2: d_input = output * (grad - dot)
        let v_dot = _mm512_set1_ps(dot);
        for c in 0..chunks {
            let offset = base + c * F32_LANES;
            let g = _mm512_loadu_ps(grad.add(offset));
            let out = _mm512_loadu_ps(output.add(offset));
            let shifted = _mm512_sub_ps(g, v_dot);
            let result = _mm512_mul_ps(out, shifted);
            _mm512_storeu_ps(d_input.add(offset), result);
        }

        for d in (chunks * F32_LANES)..dim_size {
            let idx = base + d;
            *d_input.add(idx) = *output.add(idx) * (*grad.add(idx) - dot);
        }
    }
}

/// AVX-512 softmax backward for f64.
#[target_feature(enable = "avx512f")]
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
        let mut dot_vec = _mm512_setzero_pd();
        for c in 0..chunks {
            let offset = base + c * F64_LANES;
            let g = _mm512_loadu_pd(grad.add(offset));
            let out = _mm512_loadu_pd(output.add(offset));
            dot_vec = _mm512_fmadd_pd(g, out, dot_vec);
        }
        let mut dot = _mm512_reduce_add_pd(dot_vec);

        for d in (chunks * F64_LANES)..dim_size {
            dot += *grad.add(base + d) * *output.add(base + d);
        }

        // Pass 2: d_input = output * (grad - dot)
        let v_dot = _mm512_set1_pd(dot);
        for c in 0..chunks {
            let offset = base + c * F64_LANES;
            let g = _mm512_loadu_pd(grad.add(offset));
            let out = _mm512_loadu_pd(output.add(offset));
            let shifted = _mm512_sub_pd(g, v_dot);
            let result = _mm512_mul_pd(out, shifted);
            _mm512_storeu_pd(d_input.add(offset), result);
        }

        for d in (chunks * F64_LANES)..dim_size {
            let idx = base + d;
            *d_input.add(idx) = *output.add(idx) * (*grad.add(idx) - dot);
        }
    }
}
