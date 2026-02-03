//! AVX-512 SIMD implementations for index operations
//!
//! Uses 512-bit vectors (16 f32s or 8 f64s per iteration).
//! AVX-512 has native mask registers and compress/store instructions.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================================
// Masked Fill - AVX-512
// ============================================================================

/// AVX-512 masked fill for f32.
///
/// Uses native mask registers and _mm512_mask_blend_ps.
#[target_feature(enable = "avx512f")]
pub unsafe fn masked_fill_f32(
    input: *const f32,
    mask: *const u8,
    output: *mut f32,
    len: usize,
    value: f32,
) {
    const LANES: usize = 16;
    let chunks = len / LANES;

    // Broadcast fill value to all lanes
    let fill_vec = _mm512_set1_ps(value);

    for i in 0..chunks {
        let offset = i * LANES;

        // Load 16 f32 values
        let input_vec = _mm512_loadu_ps(input.add(offset));

        // Build 16-bit mask from 16 bytes
        let k = build_mask16(mask.add(offset));

        // blend: selects from fill_vec where mask is 1, from input_vec where 0
        let result = _mm512_mask_blend_ps(k, input_vec, fill_vec);

        _mm512_storeu_ps(output.add(offset), result);
    }

    // Handle remainder with scalar
    let start = chunks * LANES;
    for i in start..len {
        *output.add(i) = if *mask.add(i) != 0 {
            value
        } else {
            *input.add(i)
        };
    }
}

/// AVX-512 masked fill for f64.
#[target_feature(enable = "avx512f")]
pub unsafe fn masked_fill_f64(
    input: *const f64,
    mask: *const u8,
    output: *mut f64,
    len: usize,
    value: f64,
) {
    const LANES: usize = 8;
    let chunks = len / LANES;

    // Broadcast fill value to all lanes
    let fill_vec = _mm512_set1_pd(value);

    for i in 0..chunks {
        let offset = i * LANES;

        // Load 8 f64 values
        let input_vec = _mm512_loadu_pd(input.add(offset));

        // Build 8-bit mask from 8 bytes
        let k = build_mask8(mask.add(offset));

        // blend: selects from fill_vec where mask is 1, from input_vec where 0
        let result = _mm512_mask_blend_pd(k, input_vec, fill_vec);

        _mm512_storeu_pd(output.add(offset), result);
    }

    // Handle remainder with scalar
    let start = chunks * LANES;
    for i in start..len {
        *output.add(i) = if *mask.add(i) != 0 {
            value
        } else {
            *input.add(i)
        };
    }
}

// ============================================================================
// Masked Select - AVX-512
// ============================================================================

/// AVX-512 masked select for f32.
///
/// Uses native _mm512_mask_compressstoreu_ps for efficient compression.
#[target_feature(enable = "avx512f")]
pub unsafe fn masked_select_f32(
    input: *const f32,
    mask: *const u8,
    output: *mut f32,
    len: usize,
) -> usize {
    const LANES: usize = 16;
    let chunks = len / LANES;
    let mut out_idx = 0;

    for i in 0..chunks {
        let offset = i * LANES;

        // Load 16 f32 values
        let input_vec = _mm512_loadu_ps(input.add(offset));

        // Build 16-bit mask from 16 bytes
        let k = build_mask16(mask.add(offset));

        // Count how many will be selected
        let count = (k as u32).count_ones() as usize;

        if count > 0 {
            // Use native compress store - only writes selected elements
            _mm512_mask_compressstoreu_ps(output.add(out_idx) as *mut _, k, input_vec);
            out_idx += count;
        }
    }

    // Handle remainder with scalar
    let start = chunks * LANES;
    for i in start..len {
        if *mask.add(i) != 0 {
            *output.add(out_idx) = *input.add(i);
            out_idx += 1;
        }
    }

    out_idx
}

/// AVX-512 masked select for f64.
#[target_feature(enable = "avx512f")]
pub unsafe fn masked_select_f64(
    input: *const f64,
    mask: *const u8,
    output: *mut f64,
    len: usize,
) -> usize {
    const LANES: usize = 8;
    let chunks = len / LANES;
    let mut out_idx = 0;

    for i in 0..chunks {
        let offset = i * LANES;

        // Load 8 f64 values
        let input_vec = _mm512_loadu_pd(input.add(offset));

        // Build 8-bit mask from 8 bytes
        let k = build_mask8(mask.add(offset));

        // Count how many will be selected
        let count = (k as u32).count_ones() as usize;

        if count > 0 {
            // Use native compress store - only writes selected elements
            _mm512_mask_compressstoreu_pd(output.add(out_idx) as *mut _, k, input_vec);
            out_idx += count;
        }
    }

    // Handle remainder with scalar
    let start = chunks * LANES;
    for i in start..len {
        if *mask.add(i) != 0 {
            *output.add(out_idx) = *input.add(i);
            out_idx += 1;
        }
    }

    out_idx
}

// ============================================================================
// Masked Count - AVX-512
// ============================================================================

/// AVX-512 mask count (popcount).
///
/// Uses SIMD to process 64 bytes at a time.
#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn masked_count(mask: *const u8, len: usize) -> usize {
    const LANES: usize = 64; // Process 64 bytes at a time
    let chunks = len / LANES;
    let mut count = 0usize;

    let zero = _mm512_setzero_si512();

    for i in 0..chunks {
        let offset = i * LANES;

        // Load 64 bytes
        let mask_vec = _mm512_loadu_si512(mask.add(offset) as *const _);

        // Compare against zero: returns mask where non-zero
        let k = _mm512_cmpneq_epi8_mask(mask_vec, zero);

        // Count bits in the 64-bit mask
        count += (k as u64).count_ones() as usize;
    }

    // Handle remainder with scalar
    let start = chunks * LANES;
    for i in start..len {
        if *mask.add(i) != 0 {
            count += 1;
        }
    }

    count
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Build a 16-bit mask from 16 u8 bytes.
///
/// Each non-zero byte sets the corresponding bit in the mask.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn build_mask16(mask: *const u8) -> __mmask16 {
    let mut k: u16 = 0;
    for j in 0..16 {
        if *mask.add(j) != 0 {
            k |= 1 << j;
        }
    }
    k
}

/// Build an 8-bit mask from 8 u8 bytes.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn build_mask8(mask: *const u8) -> __mmask8 {
    let mut k: u8 = 0;
    for j in 0..8 {
        if *mask.add(j) != 0 {
            k |= 1 << j;
        }
    }
    k
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

    fn has_avx512bw() -> bool {
        is_x86_feature_detected!("avx512bw")
    }

    #[test]
    fn test_masked_fill_f32_avx512() {
        if !has_avx512() {
            return;
        }

        let input: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let mask: Vec<u8> = (0..64).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
        let mut output = vec![0.0f32; 64];
        let fill_value = -1.0f32;

        unsafe {
            masked_fill_f32(
                input.as_ptr(),
                mask.as_ptr(),
                output.as_mut_ptr(),
                64,
                fill_value,
            );
        }

        for i in 0..64 {
            let expected = if i % 2 == 0 { fill_value } else { i as f32 };
            assert_eq!(output[i], expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_masked_select_f32_avx512() {
        if !has_avx512() {
            return;
        }

        let input: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let mask: Vec<u8> = (0..64).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        let mut output = vec![0.0f32; 64];

        let count =
            unsafe { masked_select_f32(input.as_ptr(), mask.as_ptr(), output.as_mut_ptr(), 64) };

        // Expected: 0, 3, 6, ..., 63 = 22 elements
        let expected_count = (0..64).filter(|i| i % 3 == 0).count();
        assert_eq!(count, expected_count);

        let expected: Vec<f32> = (0..64).filter(|i| i % 3 == 0).map(|i| i as f32).collect();
        for (j, &exp) in expected.iter().enumerate() {
            assert_eq!(output[j], exp, "mismatch at output index {}", j);
        }
    }

    #[test]
    fn test_masked_count_avx512() {
        if !has_avx512() || !has_avx512bw() {
            return;
        }

        let mask: Vec<u8> = (0..256).map(|i| if i % 5 == 0 { 1 } else { 0 }).collect();

        let count = unsafe { masked_count(mask.as_ptr(), 256) };

        // 256 / 5 = 51.2 so 52 elements (0, 5, 10, ..., 255)
        let expected = (0..256).filter(|i| i % 5 == 0).count();
        assert_eq!(count, expected);
    }
}
