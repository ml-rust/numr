//! AVX-512 integer-only adjacent encoding kernels.

use std::arch::x86_64::*;

/// Step raw binary32 encodings.
///
/// # Safety
/// Pointers must address `len` readable input and writable output elements.
/// Output must not overlap unread input. The CPU must support AVX512.
#[target_feature(enable = "avx512f")]
pub unsafe fn next_f32(src: *const u32, dst: *mut u32, len: usize, up: bool) {
    unsafe {
        let zero = _mm512_setzero_si512();
        let one = _mm512_set1_epi32(1);
        let magnitude_mask = _mm512_set1_epi32(0x7fffffffi32);
        let infinity = _mm512_set1_epi32(0x7f800000i32);
        let outward = _mm512_set1_epi32(if up { 0x7f800000u32 } else { 0xff800000u32 } as i32);
        let zero_result = _mm512_set1_epi32(if up { 1 } else { 0x80000001u32 } as i32);
        let mut index = 0;
        while len - index >= 16 {
            let value = _mm512_loadu_si512(src.add(index).cast());
            let magnitude = _mm512_and_si512(value, magnitude_mask);
            let negative = _mm512_cmpgt_epi32_mask(zero, value);
            let delta = _mm512_mask_blend_epi32(negative, one, _mm512_set1_epi32(-1));
            let stepped = if up {
                _mm512_add_epi32(value, delta)
            } else {
                _mm512_sub_epi32(value, delta)
            };
            let is_zero = _mm512_cmpeq_epi32_mask(magnitude, zero);
            let preserve = _mm512_cmpgt_epi32_mask(magnitude, infinity)
                | _mm512_cmpeq_epi32_mask(value, outward);
            let result = _mm512_mask_blend_epi32(is_zero, stepped, zero_result);
            let result = _mm512_mask_blend_epi32(preserve, result, value);
            _mm512_storeu_si512(dst.add(index).cast(), result);
            index += 16;
        }
        super::scalar::next_f32(src.add(index), dst.add(index), len - index, up);
    }
}

/// Step raw binary64 encodings.
///
/// # Safety
/// Pointers must address `len` readable input and writable output elements.
/// Output must not overlap unread input. The CPU must support AVX512.
#[target_feature(enable = "avx512f")]
pub unsafe fn next_f64(src: *const u64, dst: *mut u64, len: usize, up: bool) {
    unsafe {
        let zero = _mm512_setzero_si512();
        let one = _mm512_set1_epi64(1);
        let magnitude_mask = _mm512_set1_epi64(0x7fffffffffffffffi64);
        let infinity = _mm512_set1_epi64(0x7ff0000000000000i64);
        let outward = _mm512_set1_epi64(if up {
            0x7ff0000000000000u64
        } else {
            0xfff0000000000000u64
        } as i64);
        let zero_result = _mm512_set1_epi64(if up { 1 } else { 0x8000000000000001u64 } as i64);
        let mut index = 0;
        while len - index >= 8 {
            let value = _mm512_loadu_si512(src.add(index).cast());
            let magnitude = _mm512_and_si512(value, magnitude_mask);
            let negative = _mm512_cmpgt_epi64_mask(zero, value);
            let delta = _mm512_mask_blend_epi64(negative, one, _mm512_set1_epi64(-1));
            let stepped = if up {
                _mm512_add_epi64(value, delta)
            } else {
                _mm512_sub_epi64(value, delta)
            };
            let is_zero = _mm512_cmpeq_epi64_mask(magnitude, zero);
            let preserve = _mm512_cmpgt_epi64_mask(magnitude, infinity)
                | _mm512_cmpeq_epi64_mask(value, outward);
            let result = _mm512_mask_blend_epi64(is_zero, stepped, zero_result);
            let result = _mm512_mask_blend_epi64(preserve, result, value);
            _mm512_storeu_si512(dst.add(index).cast(), result);
            index += 8;
        }
        super::scalar::next_f64(src.add(index), dst.add(index), len - index, up);
    }
}
