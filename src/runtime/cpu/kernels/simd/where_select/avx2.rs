//! AVX2 conditional select (where) kernels
//!
//! Uses blend operations based on expanded condition masks.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{where_scalar_f32, where_scalar_f64};

const F32_LANES: usize = 8;
const F64_LANES: usize = 4;

/// AVX2 where for f32
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn where_f32(cond: *const u8, x: *const f32, y: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;

    for c in 0..chunks {
        let offset = c * F32_LANES;

        // Load 8 condition bytes and expand to 32-bit integers
        let cond_ptr = cond.add(offset);
        let mut mask_arr = [0i32; 8];

        for i in 0..8 {
            // Non-zero condition -> all 1s (0xFFFFFFFF), zero -> all 0s
            mask_arr[i] = if *cond_ptr.add(i) != 0 {
                -1i32 // 0xFFFFFFFF
            } else {
                0i32
            };
        }

        let mask = _mm256_loadu_si256(mask_arr.as_ptr() as *const __m256i);
        let mask_ps = _mm256_castsi256_ps(mask);

        // Load x and y values
        let vx = _mm256_loadu_ps(x.add(offset));
        let vy = _mm256_loadu_ps(y.add(offset));

        // Blend: where mask is all 1s, select x; where all 0s, select y
        let result = _mm256_blendv_ps(vy, vx, mask_ps);
        _mm256_storeu_ps(out.add(offset), result);
    }

    // Scalar tail
    let processed = chunks * F32_LANES;
    if processed < len {
        where_scalar_f32(
            cond.add(processed),
            x.add(processed),
            y.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

/// AVX2 where for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn where_f64(cond: *const u8, x: *const f64, y: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;

    for c in 0..chunks {
        let offset = c * F64_LANES;

        // Load 4 condition bytes and expand to 64-bit integers
        let cond_ptr = cond.add(offset);
        let mut mask_arr = [0i64; 4];

        for i in 0..4 {
            mask_arr[i] = if *cond_ptr.add(i) != 0 {
                -1i64 // 0xFFFFFFFFFFFFFFFF
            } else {
                0i64
            };
        }

        let mask = _mm256_loadu_si256(mask_arr.as_ptr() as *const __m256i);
        let mask_pd = _mm256_castsi256_pd(mask);

        // Load x and y values
        let vx = _mm256_loadu_pd(x.add(offset));
        let vy = _mm256_loadu_pd(y.add(offset));

        // Blend
        let result = _mm256_blendv_pd(vy, vx, mask_pd);
        _mm256_storeu_pd(out.add(offset), result);
    }

    // Scalar tail
    let processed = chunks * F64_LANES;
    if processed < len {
        where_scalar_f64(
            cond.add(processed),
            x.add(processed),
            y.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

// Suppress unused warnings
const _: () = {
    let _ = where_scalar_f32 as unsafe fn(*const u8, *const f32, *const f32, *mut f32, usize);
    let _ = where_scalar_f64 as unsafe fn(*const u8, *const f64, *const f64, *mut f64, usize);
};
