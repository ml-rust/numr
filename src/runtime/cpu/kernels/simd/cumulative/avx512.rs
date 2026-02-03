//! AVX-512 SIMD implementations for cumulative operations
//!
//! Vectorizes over the inner_size dimension - each SIMD lane maintains
//! its own independent accumulator through the scan dimension.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================================
// Cumsum Strided - AVX-512
// ============================================================================

/// AVX-512 strided cumsum for f32.
///
/// Processes 16 inner positions simultaneously, each with its own accumulator.
#[target_feature(enable = "avx512f")]
pub unsafe fn cumsum_strided_f32(
    a: *const f32,
    out: *mut f32,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    const LANES: usize = 16;
    let chunks = inner_size / LANES;

    for o in 0..outer_size {
        let outer_base = o * scan_size * inner_size;

        // Process SIMD chunks
        for chunk in 0..chunks {
            let i_base = chunk * LANES;

            // Initialize 16 accumulators to zero
            let mut acc = _mm512_setzero_ps();

            // Scan through the scan dimension
            for s in 0..scan_size {
                let idx = outer_base + s * inner_size + i_base;

                // Load 16 consecutive inner elements
                let val = _mm512_loadu_ps(a.add(idx));

                // Add to running sum
                acc = _mm512_add_ps(acc, val);

                // Store cumulative sum
                _mm512_storeu_ps(out.add(idx), acc);
            }
        }

        // Handle remainder with scalar
        let i_start = chunks * LANES;
        for i in i_start..inner_size {
            let mut acc = 0.0f32;
            for s in 0..scan_size {
                let idx = outer_base + s * inner_size + i;
                acc += *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}

/// AVX-512 strided cumsum for f64.
#[target_feature(enable = "avx512f")]
pub unsafe fn cumsum_strided_f64(
    a: *const f64,
    out: *mut f64,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    const LANES: usize = 8;
    let chunks = inner_size / LANES;

    for o in 0..outer_size {
        let outer_base = o * scan_size * inner_size;

        // Process SIMD chunks
        for chunk in 0..chunks {
            let i_base = chunk * LANES;

            // Initialize 8 accumulators to zero
            let mut acc = _mm512_setzero_pd();

            // Scan through the scan dimension
            for s in 0..scan_size {
                let idx = outer_base + s * inner_size + i_base;

                // Load 8 consecutive inner elements
                let val = _mm512_loadu_pd(a.add(idx));

                // Add to running sum
                acc = _mm512_add_pd(acc, val);

                // Store cumulative sum
                _mm512_storeu_pd(out.add(idx), acc);
            }
        }

        // Handle remainder with scalar
        let i_start = chunks * LANES;
        for i in i_start..inner_size {
            let mut acc = 0.0f64;
            for s in 0..scan_size {
                let idx = outer_base + s * inner_size + i;
                acc += *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}

// ============================================================================
// Cumprod Strided - AVX-512
// ============================================================================

/// AVX-512 strided cumprod for f32.
#[target_feature(enable = "avx512f")]
pub unsafe fn cumprod_strided_f32(
    a: *const f32,
    out: *mut f32,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    const LANES: usize = 16;
    let chunks = inner_size / LANES;

    for o in 0..outer_size {
        let outer_base = o * scan_size * inner_size;

        // Process SIMD chunks
        for chunk in 0..chunks {
            let i_base = chunk * LANES;

            // Initialize 16 accumulators to one
            let mut acc = _mm512_set1_ps(1.0);

            // Scan through the scan dimension
            for s in 0..scan_size {
                let idx = outer_base + s * inner_size + i_base;

                // Load 16 consecutive inner elements
                let val = _mm512_loadu_ps(a.add(idx));

                // Multiply with running product
                acc = _mm512_mul_ps(acc, val);

                // Store cumulative product
                _mm512_storeu_ps(out.add(idx), acc);
            }
        }

        // Handle remainder with scalar
        let i_start = chunks * LANES;
        for i in i_start..inner_size {
            let mut acc = 1.0f32;
            for s in 0..scan_size {
                let idx = outer_base + s * inner_size + i;
                acc *= *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}

/// AVX-512 strided cumprod for f64.
#[target_feature(enable = "avx512f")]
pub unsafe fn cumprod_strided_f64(
    a: *const f64,
    out: *mut f64,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    const LANES: usize = 8;
    let chunks = inner_size / LANES;

    for o in 0..outer_size {
        let outer_base = o * scan_size * inner_size;

        // Process SIMD chunks
        for chunk in 0..chunks {
            let i_base = chunk * LANES;

            // Initialize 8 accumulators to one
            let mut acc = _mm512_set1_pd(1.0);

            // Scan through the scan dimension
            for s in 0..scan_size {
                let idx = outer_base + s * inner_size + i_base;

                // Load 8 consecutive inner elements
                let val = _mm512_loadu_pd(a.add(idx));

                // Multiply with running product
                acc = _mm512_mul_pd(acc, val);

                // Store cumulative product
                _mm512_storeu_pd(out.add(idx), acc);
            }
        }

        // Handle remainder with scalar
        let i_start = chunks * LANES;
        for i in i_start..inner_size {
            let mut acc = 1.0f64;
            for s in 0..scan_size {
                let idx = outer_base + s * inner_size + i;
                acc *= *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn has_avx512() -> bool {
        is_x86_feature_detected!("avx512f")
    }

    #[test]
    fn test_cumsum_strided_f32_avx512() {
        if !has_avx512() {
            return;
        }

        // 1 outer segment, scan_size=4, inner_size=32 (will use SIMD)
        let input: Vec<f32> = (0..128).map(|x| x as f32).collect();
        let mut output = vec![0.0f32; 128];

        unsafe {
            cumsum_strided_f32(input.as_ptr(), output.as_mut_ptr(), 4, 1, 32);
        }

        // Verify i=0: values at 0, 32, 64, 96
        // cumsum: 0, 0+32=32, 32+64=96, 96+96=192
        assert_eq!(output[0], 0.0);
        assert_eq!(output[32], 32.0);
        assert_eq!(output[64], 96.0);
        assert_eq!(output[96], 192.0);

        // Verify i=1: values at 1, 33, 65, 97
        // cumsum: 1, 1+33=34, 34+65=99, 99+97=196
        assert_eq!(output[1], 1.0);
        assert_eq!(output[33], 34.0);
        assert_eq!(output[65], 99.0);
        assert_eq!(output[97], 196.0);
    }

    #[test]
    fn test_cumprod_strided_f32_avx512() {
        if !has_avx512() {
            return;
        }

        // 1 outer segment, scan_size=3, inner_size=16
        let input = vec![
            2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            2.0, // s=0
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, // s=1
            4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
            4.0, // s=2
        ];
        let mut output = vec![0.0f32; 48];

        unsafe {
            cumprod_strided_f32(input.as_ptr(), output.as_mut_ptr(), 3, 1, 16);
        }

        // Each column should be: 2, 2*3=6, 6*4=24
        for i in 0..16 {
            assert_eq!(output[i], 2.0, "s=0, i={}", i);
            assert_eq!(output[16 + i], 6.0, "s=1, i={}", i);
            assert_eq!(output[32 + i], 24.0, "s=2, i={}", i);
        }
    }
}
