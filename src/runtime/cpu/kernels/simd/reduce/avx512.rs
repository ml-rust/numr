//! AVX-512 reduction kernels
//!
//! Uses horizontal reduction intrinsics for optimal performance.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::reduce_scalar_f32;
use super::reduce_scalar_f64;
use crate::ops::ReduceOp;

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

/// AVX-512 reduction for f32
#[target_feature(enable = "avx512f")]
pub unsafe fn reduce_f32(
    op: ReduceOp,
    a: *const f32,
    out: *mut f32,
    reduce_size: usize,
    outer_size: usize,
) {
    match op {
        ReduceOp::Sum => reduce_sum_f32(a, out, reduce_size, outer_size),
        ReduceOp::Max => reduce_max_f32(a, out, reduce_size, outer_size),
        ReduceOp::Min => reduce_min_f32(a, out, reduce_size, outer_size),
        ReduceOp::Prod => reduce_prod_f32(a, out, reduce_size, outer_size),
        _ => reduce_scalar_f32(op, a, out, reduce_size, outer_size),
    }
}

/// AVX-512 reduction for f64
#[target_feature(enable = "avx512f")]
pub unsafe fn reduce_f64(
    op: ReduceOp,
    a: *const f64,
    out: *mut f64,
    reduce_size: usize,
    outer_size: usize,
) {
    match op {
        ReduceOp::Sum => reduce_sum_f64(a, out, reduce_size, outer_size),
        ReduceOp::Max => reduce_max_f64(a, out, reduce_size, outer_size),
        ReduceOp::Min => reduce_min_f64(a, out, reduce_size, outer_size),
        ReduceOp::Prod => reduce_prod_f64(a, out, reduce_size, outer_size),
        _ => reduce_scalar_f64(op, a, out, reduce_size, outer_size),
    }
}

// ============================================================================
// f32 reductions
// ============================================================================

#[target_feature(enable = "avx512f")]
unsafe fn reduce_sum_f32(a: *const f32, out: *mut f32, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F32_LANES;
    let remainder = reduce_size % F32_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;
        let mut acc = _mm512_setzero_ps();

        // SIMD accumulation
        for c in 0..chunks {
            let v = _mm512_loadu_ps(a.add(base + c * F32_LANES));
            acc = _mm512_add_ps(acc, v);
        }

        // Horizontal reduction
        let mut sum = _mm512_reduce_add_ps(acc);

        // Scalar tail
        for r in (chunks * F32_LANES)..reduce_size {
            sum += *a.add(base + r);
        }
        let _ = remainder; // Used in loop above

        *out.add(o) = sum;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn reduce_max_f32(a: *const f32, out: *mut f32, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F32_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;

        // Initialize with first vector or first element
        let mut acc = if chunks > 0 {
            _mm512_loadu_ps(a.add(base))
        } else {
            _mm512_set1_ps(*a.add(base))
        };

        // SIMD accumulation (start from 1 if we used first chunk)
        for c in 1..chunks {
            let v = _mm512_loadu_ps(a.add(base + c * F32_LANES));
            acc = _mm512_max_ps(acc, v);
        }

        // Horizontal reduction
        let mut max_val = _mm512_reduce_max_ps(acc);

        // Scalar tail
        for r in (chunks * F32_LANES)..reduce_size {
            let val = *a.add(base + r);
            if val > max_val {
                max_val = val;
            }
        }

        *out.add(o) = max_val;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn reduce_min_f32(a: *const f32, out: *mut f32, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F32_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;

        // Initialize with first vector or first element
        let mut acc = if chunks > 0 {
            _mm512_loadu_ps(a.add(base))
        } else {
            _mm512_set1_ps(*a.add(base))
        };

        // SIMD accumulation
        for c in 1..chunks {
            let v = _mm512_loadu_ps(a.add(base + c * F32_LANES));
            acc = _mm512_min_ps(acc, v);
        }

        // Horizontal reduction
        let mut min_val = _mm512_reduce_min_ps(acc);

        // Scalar tail
        for r in (chunks * F32_LANES)..reduce_size {
            let val = *a.add(base + r);
            if val < min_val {
                min_val = val;
            }
        }

        *out.add(o) = min_val;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn reduce_prod_f32(a: *const f32, out: *mut f32, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F32_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;
        let mut acc = _mm512_set1_ps(1.0);

        // SIMD accumulation
        for c in 0..chunks {
            let v = _mm512_loadu_ps(a.add(base + c * F32_LANES));
            acc = _mm512_mul_ps(acc, v);
        }

        // Horizontal reduction (multiply all elements)
        let mut prod = _mm512_reduce_mul_ps(acc);

        // Scalar tail
        for r in (chunks * F32_LANES)..reduce_size {
            prod *= *a.add(base + r);
        }

        *out.add(o) = prod;
    }
}

// ============================================================================
// f64 reductions
// ============================================================================

#[target_feature(enable = "avx512f")]
unsafe fn reduce_sum_f64(a: *const f64, out: *mut f64, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F64_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;
        let mut acc = _mm512_setzero_pd();

        // SIMD accumulation
        for c in 0..chunks {
            let v = _mm512_loadu_pd(a.add(base + c * F64_LANES));
            acc = _mm512_add_pd(acc, v);
        }

        // Horizontal reduction
        let mut sum = _mm512_reduce_add_pd(acc);

        // Scalar tail
        for r in (chunks * F64_LANES)..reduce_size {
            sum += *a.add(base + r);
        }

        *out.add(o) = sum;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn reduce_max_f64(a: *const f64, out: *mut f64, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F64_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;

        let mut acc = if chunks > 0 {
            _mm512_loadu_pd(a.add(base))
        } else {
            _mm512_set1_pd(*a.add(base))
        };

        for c in 1..chunks {
            let v = _mm512_loadu_pd(a.add(base + c * F64_LANES));
            acc = _mm512_max_pd(acc, v);
        }

        let mut max_val = _mm512_reduce_max_pd(acc);

        for r in (chunks * F64_LANES)..reduce_size {
            let val = *a.add(base + r);
            if val > max_val {
                max_val = val;
            }
        }

        *out.add(o) = max_val;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn reduce_min_f64(a: *const f64, out: *mut f64, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F64_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;

        let mut acc = if chunks > 0 {
            _mm512_loadu_pd(a.add(base))
        } else {
            _mm512_set1_pd(*a.add(base))
        };

        for c in 1..chunks {
            let v = _mm512_loadu_pd(a.add(base + c * F64_LANES));
            acc = _mm512_min_pd(acc, v);
        }

        let mut min_val = _mm512_reduce_min_pd(acc);

        for r in (chunks * F64_LANES)..reduce_size {
            let val = *a.add(base + r);
            if val < min_val {
                min_val = val;
            }
        }

        *out.add(o) = min_val;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn reduce_prod_f64(a: *const f64, out: *mut f64, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F64_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;
        let mut acc = _mm512_set1_pd(1.0);

        for c in 0..chunks {
            let v = _mm512_loadu_pd(a.add(base + c * F64_LANES));
            acc = _mm512_mul_pd(acc, v);
        }

        let mut prod = _mm512_reduce_mul_pd(acc);

        for r in (chunks * F64_LANES)..reduce_size {
            prod *= *a.add(base + r);
        }

        *out.add(o) = prod;
    }
}
