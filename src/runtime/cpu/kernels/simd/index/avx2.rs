//! AVX2 SIMD implementations for index operations
//!
//! Uses 256-bit vectors (8 f32s or 4 f64s per iteration).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================================
// Masked Fill - AVX2
// ============================================================================

/// AVX2 masked fill for f32.
///
/// Uses _mm256_blendv_ps for conditional selection.
#[target_feature(enable = "avx2")]
pub unsafe fn masked_fill_f32(
    input: *const f32,
    mask: *const u8,
    output: *mut f32,
    len: usize,
    value: f32,
) {
    const LANES: usize = 8;
    let chunks = len / LANES;

    // Broadcast fill value to all lanes
    let fill_vec = _mm256_set1_ps(value);

    for i in 0..chunks {
        let offset = i * LANES;

        // Load 8 f32 values
        let input_vec = _mm256_loadu_ps(input.add(offset));

        // Load 8 mask bytes and expand to f32 mask
        // mask byte != 0 means fill, == 0 means keep input
        let mask_vec = expand_mask_u8_to_f32_avx2(mask.add(offset));

        // blendv: selects from fill_vec where mask MSB is 1, from input_vec where 0
        let result = _mm256_blendv_ps(input_vec, fill_vec, mask_vec);

        _mm256_storeu_ps(output.add(offset), result);
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

/// AVX2 masked fill for f64.
#[target_feature(enable = "avx2")]
pub unsafe fn masked_fill_f64(
    input: *const f64,
    mask: *const u8,
    output: *mut f64,
    len: usize,
    value: f64,
) {
    const LANES: usize = 4;
    let chunks = len / LANES;

    // Broadcast fill value to all lanes
    let fill_vec = _mm256_set1_pd(value);

    for i in 0..chunks {
        let offset = i * LANES;

        // Load 4 f64 values
        let input_vec = _mm256_loadu_pd(input.add(offset));

        // Load 4 mask bytes and expand to f64 mask
        let mask_vec = expand_mask_u8_to_f64_avx2(mask.add(offset));

        // blendv: selects from fill_vec where mask MSB is 1, from input_vec where 0
        let result = _mm256_blendv_pd(input_vec, fill_vec, mask_vec);

        _mm256_storeu_pd(output.add(offset), result);
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
// Masked Select - AVX2
// ============================================================================

/// AVX2 masked select for f32.
///
/// AVX2 doesn't have native compress/store, so we use a lookup table approach
/// with PSHUFB for efficient compression.
#[target_feature(enable = "avx2")]
pub unsafe fn masked_select_f32(
    input: *const f32,
    mask: *const u8,
    output: *mut f32,
    len: usize,
) -> usize {
    const LANES: usize = 8;
    let chunks = len / LANES;
    let mut out_idx = 0;

    for i in 0..chunks {
        let offset = i * LANES;

        // Load 8 f32 values
        let input_vec = _mm256_loadu_ps(input.add(offset));

        // Build mask from 8 bytes
        let mut mask_bits: u32 = 0;
        for j in 0..LANES {
            if *mask.add(offset + j) != 0 {
                mask_bits |= 1 << j;
            }
        }

        // Count how many will be selected
        let count = mask_bits.count_ones() as usize;

        if count == 0 {
            continue;
        }

        if count == LANES {
            // All selected - direct store
            _mm256_storeu_ps(output.add(out_idx), input_vec);
        } else {
            // Partial selection - use LUT-based compression
            // Split into two 128-bit halves for compression
            let lo = _mm256_castps256_ps128(input_vec);
            let hi = _mm256_extractf128_ps(input_vec, 1);

            let lo_mask = (mask_bits & 0x0F) as usize;
            let hi_mask = ((mask_bits >> 4) & 0x0F) as usize;

            // Compress lower 4 elements
            let lo_count = compress_store_f32_128(lo, lo_mask, output.add(out_idx));

            // Compress upper 4 elements
            compress_store_f32_128(hi, hi_mask, output.add(out_idx + lo_count));
        }

        out_idx += count;
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

/// AVX2 masked select for f64.
#[target_feature(enable = "avx2")]
pub unsafe fn masked_select_f64(
    input: *const f64,
    mask: *const u8,
    output: *mut f64,
    len: usize,
) -> usize {
    const LANES: usize = 4;
    let chunks = len / LANES;
    let mut out_idx = 0;

    for i in 0..chunks {
        let offset = i * LANES;

        // Load 4 f64 values
        let input_vec = _mm256_loadu_pd(input.add(offset));

        // Build mask from 4 bytes
        let mut mask_bits: u32 = 0;
        for j in 0..LANES {
            if *mask.add(offset + j) != 0 {
                mask_bits |= 1 << j;
            }
        }

        // Count how many will be selected
        let count = mask_bits.count_ones() as usize;

        if count == 0 {
            continue;
        }

        if count == LANES {
            // All selected - direct store
            _mm256_storeu_pd(output.add(out_idx), input_vec);
        } else {
            // Partial selection - use LUT-based compression
            let lo = _mm256_castpd256_pd128(input_vec);
            let hi = _mm256_extractf128_pd(input_vec, 1);

            let lo_mask = (mask_bits & 0x03) as usize;
            let hi_mask = ((mask_bits >> 2) & 0x03) as usize;

            // Compress lower 2 elements
            let lo_count = compress_store_f64_128(lo, lo_mask, output.add(out_idx));

            // Compress upper 2 elements
            compress_store_f64_128(hi, hi_mask, output.add(out_idx + lo_count));
        }

        out_idx += count;
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
// Masked Count - AVX2
// ============================================================================

/// AVX2 mask count (popcount).
///
/// Uses SIMD to process 32 bytes at a time.
#[target_feature(enable = "avx2")]
pub unsafe fn masked_count(mask: *const u8, len: usize) -> usize {
    const LANES: usize = 32; // Process 32 bytes at a time with AVX2
    let chunks = len / LANES;
    let mut count = 0usize;

    // AVX2 doesn't have native popcount for vectors, but we can use
    // comparison and horizontal sum
    let zero = _mm256_setzero_si256();

    for i in 0..chunks {
        let offset = i * LANES;

        // Load 32 bytes
        let mask_vec = _mm256_loadu_si256(mask.add(offset) as *const __m256i);

        // Compare against zero: result is 0xFF where non-zero, 0x00 where zero
        let cmp = _mm256_cmpeq_epi8(mask_vec, zero);

        // movemask gives us a 32-bit mask where bit i is set if byte i was 0
        let zero_mask = _mm256_movemask_epi8(cmp) as u32;

        // Count non-zero bytes (invert the mask and count ones)
        count += (!zero_mask).count_ones() as usize;
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

/// Expand 8 mask bytes to 8 f32 values suitable for blendv.
///
/// Non-zero bytes become 0xFFFFFFFF (negative float), zero bytes become 0x00000000.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn expand_mask_u8_to_f32_avx2(mask: *const u8) -> __m256 {
    // Load 8 bytes as i32 (zero-extended via gather or manual expansion)
    // We need each byte to become a full 32-bit mask

    // Load the 8 mask bytes into lower 64 bits
    let mask_bytes = _mm_loadl_epi64(mask as *const __m128i);

    // Zero-extend bytes to 32-bit integers
    let mask_i32 = _mm256_cvtepu8_epi32(mask_bytes);

    // Compare against zero: non-zero becomes 0xFFFFFFFF
    let zero = _mm256_setzero_si256();
    let cmp = _mm256_cmpeq_epi32(mask_i32, zero);

    // Invert: we want 0xFFFFFFFF where mask != 0
    let inverted = _mm256_xor_si256(cmp, _mm256_set1_epi32(-1));

    _mm256_castsi256_ps(inverted)
}

/// Expand 4 mask bytes to 4 f64 values suitable for blendv.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn expand_mask_u8_to_f64_avx2(mask: *const u8) -> __m256d {
    // Load 4 bytes manually and expand to 64-bit
    let m0 = if *mask.add(0) != 0 { -1i64 } else { 0i64 };
    let m1 = if *mask.add(1) != 0 { -1i64 } else { 0i64 };
    let m2 = if *mask.add(2) != 0 { -1i64 } else { 0i64 };
    let m3 = if *mask.add(3) != 0 { -1i64 } else { 0i64 };

    let mask_i64 = _mm256_set_epi64x(m3, m2, m1, m0);
    _mm256_castsi256_pd(mask_i64)
}

/// Compress and store 4 f32 values based on a 4-bit mask.
///
/// Returns the number of elements stored.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn compress_store_f32_128(vec: __m128, mask: usize, output: *mut f32) -> usize {
    // Lookup table for shuffle indices based on 4-bit mask
    // Each entry contains 4 indices (or 0xFF for unused slots)
    static SHUFFLE_LUT: [[u8; 4]; 16] = [
        [0xFF, 0xFF, 0xFF, 0xFF], // 0b0000: none
        [0, 0xFF, 0xFF, 0xFF],    // 0b0001: [0]
        [1, 0xFF, 0xFF, 0xFF],    // 0b0010: [1]
        [0, 1, 0xFF, 0xFF],       // 0b0011: [0,1]
        [2, 0xFF, 0xFF, 0xFF],    // 0b0100: [2]
        [0, 2, 0xFF, 0xFF],       // 0b0101: [0,2]
        [1, 2, 0xFF, 0xFF],       // 0b0110: [1,2]
        [0, 1, 2, 0xFF],          // 0b0111: [0,1,2]
        [3, 0xFF, 0xFF, 0xFF],    // 0b1000: [3]
        [0, 3, 0xFF, 0xFF],       // 0b1001: [0,3]
        [1, 3, 0xFF, 0xFF],       // 0b1010: [1,3]
        [0, 1, 3, 0xFF],          // 0b1011: [0,1,3]
        [2, 3, 0xFF, 0xFF],       // 0b1100: [2,3]
        [0, 2, 3, 0xFF],          // 0b1101: [0,2,3]
        [1, 2, 3, 0xFF],          // 0b1110: [1,2,3]
        [0, 1, 2, 3],             // 0b1111: [0,1,2,3]
    ];

    let count = (mask as u32).count_ones() as usize;
    if count == 0 {
        return 0;
    }

    let indices = &SHUFFLE_LUT[mask & 0xF];

    // Extract and store selected elements
    let arr: [f32; 4] = std::mem::transmute(vec);
    for (j, &idx) in indices.iter().take(count).enumerate() {
        *output.add(j) = arr[idx as usize];
    }

    count
}

/// Compress and store 2 f64 values based on a 2-bit mask.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn compress_store_f64_128(vec: __m128d, mask: usize, output: *mut f64) -> usize {
    let count = (mask as u32 & 0x3).count_ones() as usize;
    if count == 0 {
        return 0;
    }

    let arr: [f64; 2] = std::mem::transmute(vec);
    let mut out_idx = 0;

    if mask & 1 != 0 {
        *output.add(out_idx) = arr[0];
        out_idx += 1;
    }
    if mask & 2 != 0 {
        *output.add(out_idx) = arr[1];
    }

    count
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
    fn test_masked_fill_f32_avx2() {
        if !has_avx2() {
            return;
        }

        let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mask: Vec<u8> = (0..32).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
        let mut output = vec![0.0f32; 32];
        let fill_value = -1.0f32;

        unsafe {
            masked_fill_f32(
                input.as_ptr(),
                mask.as_ptr(),
                output.as_mut_ptr(),
                32,
                fill_value,
            );
        }

        for i in 0..32 {
            let expected = if i % 2 == 0 { fill_value } else { i as f32 };
            assert_eq!(output[i], expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_masked_select_f32_avx2() {
        if !has_avx2() {
            return;
        }

        let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mask: Vec<u8> = (0..32).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        let mut output = vec![0.0f32; 32];

        let count =
            unsafe { masked_select_f32(input.as_ptr(), mask.as_ptr(), output.as_mut_ptr(), 32) };

        // Expected: 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30 = 11 elements
        assert_eq!(count, 11);

        let expected: Vec<f32> = (0..32).filter(|i| i % 3 == 0).map(|i| i as f32).collect();
        for (j, &exp) in expected.iter().enumerate() {
            assert_eq!(output[j], exp, "mismatch at output index {}", j);
        }
    }

    #[test]
    fn test_masked_count_avx2() {
        if !has_avx2() {
            return;
        }

        let mask: Vec<u8> = (0..128).map(|i| if i % 7 == 0 { 1 } else { 0 }).collect();

        let count = unsafe { masked_count(mask.as_ptr(), 128) };

        // 128 / 7 = 18.28... so 19 elements (0, 7, 14, ..., 126)
        assert_eq!(count, 19);
    }
}
