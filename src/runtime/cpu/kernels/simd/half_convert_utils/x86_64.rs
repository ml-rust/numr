//! x86_64 SIMD implementations for f16/bf16 ↔ f32 conversion
//!
//! - f16: F16C instructions (`_mm256_cvtph_ps` / `_mm256_cvtps_ph`)
//! - bf16: AVX2 integer bit-shift (`u32 << 16` / rounded `>> 16`)

// ---------------------------------------------------------------------------
// F16C: f16 ↔ f32
// ---------------------------------------------------------------------------

#[target_feature(enable = "f16c,avx")]
pub(super) unsafe fn convert_f16_to_f32_f16c(src: *const u16, dst: *mut f32, len: usize) {
    use std::arch::x86_64::*;

    let mut i = 0usize;

    // Process 8 elements at a time
    while i + 8 <= len {
        let half_vec = _mm_loadu_si128(src.add(i) as *const __m128i);
        let float_vec = _mm256_cvtph_ps(half_vec);
        _mm256_storeu_ps(dst.add(i), float_vec);
        i += 8;
    }

    // Scalar tail
    while i < len {
        *dst.add(i) = half::f16::from_bits(*src.add(i)).to_f32();
        i += 1;
    }
}

#[target_feature(enable = "f16c,avx")]
pub(super) unsafe fn convert_f32_to_f16_f16c(src: *const f32, dst: *mut u16, len: usize) {
    use std::arch::x86_64::*;

    let mut i = 0usize;

    // _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC = 0x08
    while i + 8 <= len {
        let float_vec = _mm256_loadu_ps(src.add(i));
        let half_vec = _mm256_cvtps_ph(float_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(dst.add(i) as *mut __m128i, half_vec);
        i += 8;
    }

    // Scalar tail
    while i < len {
        *dst.add(i) = half::f16::from_f32(*src.add(i)).to_bits();
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// AVX2: bf16 ↔ f32 (integer bit-shift)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
pub(super) unsafe fn convert_bf16_to_f32_avx2(src: *const u16, dst: *mut f32, len: usize) {
    use std::arch::x86_64::*;

    let mut i = 0usize;

    // bf16 → f32: zero-extend u16 to u32, shift left by 16
    while i + 8 <= len {
        let bf16_vec = _mm_loadu_si128(src.add(i) as *const __m128i);
        let u32_vec = _mm256_cvtepu16_epi32(bf16_vec);
        let f32_bits = _mm256_slli_epi32(u32_vec, 16);
        _mm256_storeu_ps(dst.add(i), _mm256_castsi256_ps(f32_bits));
        i += 8;
    }

    // Scalar tail
    while i < len {
        let bits = (*src.add(i) as u32) << 16;
        *dst.add(i) = f32::from_bits(bits);
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn convert_f32_to_bf16_avx2(src: *const f32, dst: *mut u16, len: usize) {
    use std::arch::x86_64::*;

    let mut i = 0usize;

    // f32 → bf16 with round-to-nearest-even:
    // Add rounding bias 0x7FFF + ((bits >> 16) & 1), then shift right 16
    let rounding_bias = _mm256_set1_epi32(0x7FFF);
    let one = _mm256_set1_epi32(1);

    while i + 8 <= len {
        let f32_vec = _mm256_loadu_ps(src.add(i));
        let bits = _mm256_castps_si256(f32_vec);

        // Round-to-nearest-even: bias = 0x7FFF + ((bits >> 16) & 1)
        let shifted = _mm256_srli_epi32(bits, 16);
        let lsb = _mm256_and_si256(shifted, one);
        let bias = _mm256_add_epi32(rounding_bias, lsb);

        // Add bias and shift right
        let rounded = _mm256_add_epi32(bits, bias);
        let bf16_u32 = _mm256_srli_epi32(rounded, 16);

        // Pack 8 u32 values down to 8 u16 values
        let lo = _mm256_castsi256_si128(bf16_u32);
        let hi = _mm256_extracti128_si256(bf16_u32, 1);
        let packed = _mm_packus_epi32(lo, hi);

        _mm_storeu_si128(dst.add(i) as *mut __m128i, packed);
        i += 8;
    }

    // Scalar tail with same rounding
    while i < len {
        let bits = (*src.add(i)).to_bits();
        let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
        *dst.add(i) = (rounded >> 16) as u16;
        i += 1;
    }
}
