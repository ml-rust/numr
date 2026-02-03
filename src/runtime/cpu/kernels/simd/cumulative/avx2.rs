//! AVX2 SIMD implementations for cumulative operations
//!
//! Vectorizes over the inner_size dimension - each SIMD lane maintains
//! its own independent accumulator through the scan dimension.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================================
// Cumsum Strided - AVX2
// ============================================================================

/// AVX2 strided cumsum for f32.
///
/// Processes 8 inner positions simultaneously, each with its own accumulator.
#[target_feature(enable = "avx2")]
pub unsafe fn cumsum_strided_f32(
    a: *const f32,
    out: *mut f32,
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
            let mut acc = _mm256_setzero_ps();

            // Scan through the scan dimension
            for s in 0..scan_size {
                let idx = outer_base + s * inner_size + i_base;

                // Load 8 consecutive inner elements
                let val = _mm256_loadu_ps(a.add(idx));

                // Add to running sum
                acc = _mm256_add_ps(acc, val);

                // Store cumulative sum
                _mm256_storeu_ps(out.add(idx), acc);
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

/// AVX2 strided cumsum for f64.
#[target_feature(enable = "avx2")]
pub unsafe fn cumsum_strided_f64(
    a: *const f64,
    out: *mut f64,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    const LANES: usize = 4;
    let chunks = inner_size / LANES;

    for o in 0..outer_size {
        let outer_base = o * scan_size * inner_size;

        // Process SIMD chunks
        for chunk in 0..chunks {
            let i_base = chunk * LANES;

            // Initialize 4 accumulators to zero
            let mut acc = _mm256_setzero_pd();

            // Scan through the scan dimension
            for s in 0..scan_size {
                let idx = outer_base + s * inner_size + i_base;

                // Load 4 consecutive inner elements
                let val = _mm256_loadu_pd(a.add(idx));

                // Add to running sum
                acc = _mm256_add_pd(acc, val);

                // Store cumulative sum
                _mm256_storeu_pd(out.add(idx), acc);
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
// Cumprod Strided - AVX2
// ============================================================================

/// AVX2 strided cumprod for f32.
#[target_feature(enable = "avx2")]
pub unsafe fn cumprod_strided_f32(
    a: *const f32,
    out: *mut f32,
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
            let mut acc = _mm256_set1_ps(1.0);

            // Scan through the scan dimension
            for s in 0..scan_size {
                let idx = outer_base + s * inner_size + i_base;

                // Load 8 consecutive inner elements
                let val = _mm256_loadu_ps(a.add(idx));

                // Multiply with running product
                acc = _mm256_mul_ps(acc, val);

                // Store cumulative product
                _mm256_storeu_ps(out.add(idx), acc);
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

/// AVX2 strided cumprod for f64.
#[target_feature(enable = "avx2")]
pub unsafe fn cumprod_strided_f64(
    a: *const f64,
    out: *mut f64,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    const LANES: usize = 4;
    let chunks = inner_size / LANES;

    for o in 0..outer_size {
        let outer_base = o * scan_size * inner_size;

        // Process SIMD chunks
        for chunk in 0..chunks {
            let i_base = chunk * LANES;

            // Initialize 4 accumulators to one
            let mut acc = _mm256_set1_pd(1.0);

            // Scan through the scan dimension
            for s in 0..scan_size {
                let idx = outer_base + s * inner_size + i_base;

                // Load 4 consecutive inner elements
                let val = _mm256_loadu_pd(a.add(idx));

                // Multiply with running product
                acc = _mm256_mul_pd(acc, val);

                // Store cumulative product
                _mm256_storeu_pd(out.add(idx), acc);
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

    fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }

    #[test]
    fn test_cumsum_strided_f32_avx2() {
        if !has_avx2() {
            return;
        }

        // 1 outer segment, scan_size=4, inner_size=16 (will use SIMD)
        let input: Vec<f32> = (0..64).map(|x| x as f32).collect();
        let mut output = vec![0.0f32; 64];

        unsafe {
            cumsum_strided_f32(input.as_ptr(), output.as_mut_ptr(), 4, 1, 16);
        }

        // Verify i=0: values at 0, 16, 32, 48
        // cumsum: 0, 0+16=16, 16+32=48, 48+48=96
        assert_eq!(output[0], 0.0);
        assert_eq!(output[16], 16.0);
        assert_eq!(output[32], 48.0);
        assert_eq!(output[48], 96.0);

        // Verify i=1: values at 1, 17, 33, 49
        // cumsum: 1, 1+17=18, 18+33=51, 51+49=100
        assert_eq!(output[1], 1.0);
        assert_eq!(output[17], 18.0);
        assert_eq!(output[33], 51.0);
        assert_eq!(output[49], 100.0);
    }

    #[test]
    fn test_cumprod_strided_f32_avx2() {
        if !has_avx2() {
            return;
        }

        // 1 outer segment, scan_size=3, inner_size=8
        let input = vec![
            2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, // s=0
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, // s=1
            4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, // s=2
        ];
        let mut output = vec![0.0f32; 24];

        unsafe {
            cumprod_strided_f32(input.as_ptr(), output.as_mut_ptr(), 3, 1, 8);
        }

        // Each column should be: 2, 2*3=6, 6*4=24
        for i in 0..8 {
            assert_eq!(output[i], 2.0, "s=0, i={}", i);
            assert_eq!(output[8 + i], 6.0, "s=1, i={}", i);
            assert_eq!(output[16 + i], 24.0, "s=2, i={}", i);
        }
    }

    #[test]
    fn test_cumsum_strided_f64_avx2() {
        if !has_avx2() {
            return;
        }

        // 1 outer segment, scan_size=3, inner_size=8
        let input: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let mut output = vec![0.0f64; 24];

        unsafe {
            cumsum_strided_f64(input.as_ptr(), output.as_mut_ptr(), 3, 1, 8);
        }

        // Verify i=0: values at 0, 8, 16
        // cumsum: 0, 0+8=8, 8+16=24
        assert_eq!(output[0], 0.0);
        assert_eq!(output[8], 8.0);
        assert_eq!(output[16], 24.0);
    }
}
