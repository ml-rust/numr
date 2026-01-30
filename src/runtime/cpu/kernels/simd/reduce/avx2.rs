//! AVX2 reduction kernels
//!
//! Manual horizontal reductions since AVX2 lacks _mm256_reduce_* intrinsics.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::reduce_scalar_f32;
use super::reduce_scalar_f64;
use crate::ops::ReduceOp;

const F32_LANES: usize = 8;
const F64_LANES: usize = 4;

/// AVX2 reduction for f32
#[target_feature(enable = "avx2")]
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

/// AVX2 reduction for f64
#[target_feature(enable = "avx2")]
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
// Horizontal reduction helpers for AVX2
// ============================================================================

/// Horizontal sum of 8 f32s in __m256
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_f32(v: __m256) -> f32 {
    // Add high 128 bits to low 128 bits
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(low, high);

    // Horizontal add within 128 bits
    let shuf = _mm_movehdup_ps(sum128); // [1,1,3,3]
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);

    _mm_cvtss_f32(sum32)
}

/// Horizontal max of 8 f32s in __m256
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmax_f32(v: __m256) -> f32 {
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let max128 = _mm_max_ps(low, high);

    let shuf = _mm_movehdup_ps(max128);
    let max64 = _mm_max_ps(max128, shuf);
    let shuf2 = _mm_movehl_ps(max64, max64);
    let max32 = _mm_max_ss(max64, shuf2);

    _mm_cvtss_f32(max32)
}

/// Horizontal min of 8 f32s in __m256
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmin_f32(v: __m256) -> f32 {
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let min128 = _mm_min_ps(low, high);

    let shuf = _mm_movehdup_ps(min128);
    let min64 = _mm_min_ps(min128, shuf);
    let shuf2 = _mm_movehl_ps(min64, min64);
    let min32 = _mm_min_ss(min64, shuf2);

    _mm_cvtss_f32(min32)
}

/// Horizontal product of 8 f32s in __m256
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hprod_f32(v: __m256) -> f32 {
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let prod128 = _mm_mul_ps(low, high);

    let shuf = _mm_movehdup_ps(prod128);
    let prod64 = _mm_mul_ps(prod128, shuf);
    let shuf2 = _mm_movehl_ps(prod64, prod64);
    let prod32 = _mm_mul_ss(prod64, shuf2);

    _mm_cvtss_f32(prod32)
}

/// Horizontal sum of 4 f64s in __m256d
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_f64(v: __m256d) -> f64 {
    let high = _mm256_extractf128_pd(v, 1);
    let low = _mm256_castpd256_pd128(v);
    let sum128 = _mm_add_pd(low, high);

    let shuf = _mm_unpackhi_pd(sum128, sum128);
    let sum64 = _mm_add_sd(sum128, shuf);

    _mm_cvtsd_f64(sum64)
}

/// Horizontal max of 4 f64s in __m256d
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmax_f64(v: __m256d) -> f64 {
    let high = _mm256_extractf128_pd(v, 1);
    let low = _mm256_castpd256_pd128(v);
    let max128 = _mm_max_pd(low, high);

    let shuf = _mm_unpackhi_pd(max128, max128);
    let max64 = _mm_max_sd(max128, shuf);

    _mm_cvtsd_f64(max64)
}

/// Horizontal min of 4 f64s in __m256d
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmin_f64(v: __m256d) -> f64 {
    let high = _mm256_extractf128_pd(v, 1);
    let low = _mm256_castpd256_pd128(v);
    let min128 = _mm_min_pd(low, high);

    let shuf = _mm_unpackhi_pd(min128, min128);
    let min64 = _mm_min_sd(min128, shuf);

    _mm_cvtsd_f64(min64)
}

/// Horizontal product of 4 f64s in __m256d
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hprod_f64(v: __m256d) -> f64 {
    let high = _mm256_extractf128_pd(v, 1);
    let low = _mm256_castpd256_pd128(v);
    let prod128 = _mm_mul_pd(low, high);

    let shuf = _mm_unpackhi_pd(prod128, prod128);
    let prod64 = _mm_mul_sd(prod128, shuf);

    _mm_cvtsd_f64(prod64)
}

// ============================================================================
// f32 reductions
// ============================================================================

#[target_feature(enable = "avx2")]
unsafe fn reduce_sum_f32(a: *const f32, out: *mut f32, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F32_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;
        let mut acc = _mm256_setzero_ps();

        for c in 0..chunks {
            let v = _mm256_loadu_ps(a.add(base + c * F32_LANES));
            acc = _mm256_add_ps(acc, v);
        }

        let mut sum = hsum_f32(acc);

        for r in (chunks * F32_LANES)..reduce_size {
            sum += *a.add(base + r);
        }

        *out.add(o) = sum;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn reduce_max_f32(a: *const f32, out: *mut f32, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F32_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;

        let mut acc = if chunks > 0 {
            _mm256_loadu_ps(a.add(base))
        } else {
            _mm256_set1_ps(*a.add(base))
        };

        for c in 1..chunks {
            let v = _mm256_loadu_ps(a.add(base + c * F32_LANES));
            acc = _mm256_max_ps(acc, v);
        }

        let mut max_val = hmax_f32(acc);

        for r in (chunks * F32_LANES)..reduce_size {
            let val = *a.add(base + r);
            if val > max_val {
                max_val = val;
            }
        }

        *out.add(o) = max_val;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn reduce_min_f32(a: *const f32, out: *mut f32, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F32_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;

        let mut acc = if chunks > 0 {
            _mm256_loadu_ps(a.add(base))
        } else {
            _mm256_set1_ps(*a.add(base))
        };

        for c in 1..chunks {
            let v = _mm256_loadu_ps(a.add(base + c * F32_LANES));
            acc = _mm256_min_ps(acc, v);
        }

        let mut min_val = hmin_f32(acc);

        for r in (chunks * F32_LANES)..reduce_size {
            let val = *a.add(base + r);
            if val < min_val {
                min_val = val;
            }
        }

        *out.add(o) = min_val;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn reduce_prod_f32(a: *const f32, out: *mut f32, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F32_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;
        let mut acc = _mm256_set1_ps(1.0);

        for c in 0..chunks {
            let v = _mm256_loadu_ps(a.add(base + c * F32_LANES));
            acc = _mm256_mul_ps(acc, v);
        }

        let mut prod = hprod_f32(acc);

        for r in (chunks * F32_LANES)..reduce_size {
            prod *= *a.add(base + r);
        }

        *out.add(o) = prod;
    }
}

// ============================================================================
// f64 reductions
// ============================================================================

#[target_feature(enable = "avx2")]
unsafe fn reduce_sum_f64(a: *const f64, out: *mut f64, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F64_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;
        let mut acc = _mm256_setzero_pd();

        for c in 0..chunks {
            let v = _mm256_loadu_pd(a.add(base + c * F64_LANES));
            acc = _mm256_add_pd(acc, v);
        }

        let mut sum = hsum_f64(acc);

        for r in (chunks * F64_LANES)..reduce_size {
            sum += *a.add(base + r);
        }

        *out.add(o) = sum;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn reduce_max_f64(a: *const f64, out: *mut f64, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F64_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;

        let mut acc = if chunks > 0 {
            _mm256_loadu_pd(a.add(base))
        } else {
            _mm256_set1_pd(*a.add(base))
        };

        for c in 1..chunks {
            let v = _mm256_loadu_pd(a.add(base + c * F64_LANES));
            acc = _mm256_max_pd(acc, v);
        }

        let mut max_val = hmax_f64(acc);

        for r in (chunks * F64_LANES)..reduce_size {
            let val = *a.add(base + r);
            if val > max_val {
                max_val = val;
            }
        }

        *out.add(o) = max_val;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn reduce_min_f64(a: *const f64, out: *mut f64, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F64_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;

        let mut acc = if chunks > 0 {
            _mm256_loadu_pd(a.add(base))
        } else {
            _mm256_set1_pd(*a.add(base))
        };

        for c in 1..chunks {
            let v = _mm256_loadu_pd(a.add(base + c * F64_LANES));
            acc = _mm256_min_pd(acc, v);
        }

        let mut min_val = hmin_f64(acc);

        for r in (chunks * F64_LANES)..reduce_size {
            let val = *a.add(base + r);
            if val < min_val {
                min_val = val;
            }
        }

        *out.add(o) = min_val;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn reduce_prod_f64(a: *const f64, out: *mut f64, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F64_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;
        let mut acc = _mm256_set1_pd(1.0);

        for c in 0..chunks {
            let v = _mm256_loadu_pd(a.add(base + c * F64_LANES));
            acc = _mm256_mul_pd(acc, v);
        }

        let mut prod = hprod_f64(acc);

        for r in (chunks * F64_LANES)..reduce_size {
            prod *= *a.add(base + r);
        }

        *out.add(o) = prod;
    }
}
