//! AVX2 integer-only adjacent encoding kernels.

use std::arch::x86_64::*;

/// Step raw binary32 encodings.
///
/// # Safety
/// Pointers must address `len` readable input and writable output elements.
/// Output must not overlap unread input. The CPU must support AVX2.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn next_f32(src: *const u32, dst: *mut u32, len: usize, up: bool) {
    unsafe {
        let zero = _mm256_setzero_si256();
        let one = _mm256_set1_epi32(1);
        let magnitude_mask = _mm256_set1_epi32(0x7fffffffi32);
        let infinity = _mm256_set1_epi32(0x7f800000i32);
        let outward = _mm256_set1_epi32(if up { 0x7f800000u32 } else { 0xff800000u32 } as i32);
        let zero_result = _mm256_set1_epi32(if up { 1 } else { 0x80000001u32 } as i32);
        let mut index = 0;
        while len - index >= 8 {
            let value = _mm256_loadu_si256(src.add(index).cast());
            let magnitude = _mm256_and_si256(value, magnitude_mask);
            let negative = _mm256_cmpgt_epi32(zero, value);
            let delta = _mm256_or_si256(negative, one);
            let stepped = if up {
                _mm256_add_epi32(value, delta)
            } else {
                _mm256_sub_epi32(value, delta)
            };
            let is_zero = _mm256_cmpeq_epi32(magnitude, zero);
            let preserve = _mm256_or_si256(
                _mm256_cmpgt_epi32(magnitude, infinity),
                _mm256_cmpeq_epi32(value, outward),
            );
            let result = _mm256_blendv_epi8(stepped, zero_result, is_zero);
            let result = _mm256_blendv_epi8(result, value, preserve);
            _mm256_storeu_si256(dst.add(index).cast(), result);
            index += 8;
        }
        super::scalar::next_f32(src.add(index), dst.add(index), len - index, up);
    }
}

/// Step raw binary64 encodings.
///
/// # Safety
/// Pointers must address `len` readable input and writable output elements.
/// Output must not overlap unread input. The CPU must support AVX2.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn next_f64(src: *const u64, dst: *mut u64, len: usize, up: bool) {
    unsafe {
        let zero = _mm256_setzero_si256();
        let one = _mm256_set1_epi64x(1);
        let magnitude_mask = _mm256_set1_epi64x(0x7fffffffffffffffi64);
        let infinity = _mm256_set1_epi64x(0x7ff0000000000000i64);
        let outward = _mm256_set1_epi64x(if up {
            0x7ff0000000000000u64
        } else {
            0xfff0000000000000u64
        } as i64);
        let zero_result = _mm256_set1_epi64x(if up { 1 } else { 0x8000000000000001u64 } as i64);
        let mut index = 0;
        while len - index >= 4 {
            let value = _mm256_loadu_si256(src.add(index).cast());
            let magnitude = _mm256_and_si256(value, magnitude_mask);
            let negative = _mm256_cmpgt_epi64(zero, value);
            let delta = _mm256_or_si256(negative, one);
            let stepped = if up {
                _mm256_add_epi64(value, delta)
            } else {
                _mm256_sub_epi64(value, delta)
            };
            let is_zero = _mm256_cmpeq_epi64(magnitude, zero);
            let preserve = _mm256_or_si256(
                _mm256_cmpgt_epi64(magnitude, infinity),
                _mm256_cmpeq_epi64(value, outward),
            );
            let result = _mm256_blendv_epi8(stepped, zero_result, is_zero);
            let result = _mm256_blendv_epi8(result, value, preserve);
            _mm256_storeu_si256(dst.add(index).cast(), result);
            index += 4;
        }
        super::scalar::next_f64(src.add(index), dst.add(index), len - index, up);
    }
}
