//! NEON integer-only adjacent encoding kernels.

use std::arch::aarch64::*;

/// Step raw binary32 encodings on either NEON capability level.
///
/// # Safety
/// Pointers must address `len` readable input and writable output elements.
/// Output must not overlap unread input. The CPU must support NEON.
#[target_feature(enable = "neon")]
pub unsafe fn next_f32(src: *const u32, dst: *mut u32, len: usize, up: bool) {
    unsafe {
        let zero = vdupq_n_u32(0);
        let one = vdupq_n_u32(1);
        let sign = vdupq_n_u32(0x80000000);
        let infinity = vdupq_n_u32(0x7f800000);
        let outward = vdupq_n_u32(if up { 0x7f800000 } else { 0xff800000 });
        let zero_result = vdupq_n_u32(if up { 1 } else { 0x80000001 });
        let mut index = 0;
        while len - index >= 4 {
            let value = vld1q_u32(src.add(index));
            let magnitude = vbicq_u32(value, sign);
            let negative = vceqq_u32(vandq_u32(value, sign), sign);
            let delta = vorrq_u32(negative, one);
            let stepped = if up {
                vaddq_u32(value, delta)
            } else {
                vsubq_u32(value, delta)
            };
            let is_zero = vceqq_u32(magnitude, zero);
            let preserve = vorrq_u32(vcgtq_u32(magnitude, infinity), vceqq_u32(value, outward));
            let result = vbslq_u32(is_zero, zero_result, stepped);
            let result = vbslq_u32(preserve, value, result);
            vst1q_u32(dst.add(index), result);
            index += 4;
        }
        super::scalar::next_f32(src.add(index), dst.add(index), len - index, up);
    }
}

/// Step raw binary64 encodings on either NEON capability level.
///
/// # Safety
/// Pointers must address `len` readable input and writable output elements.
/// Output must not overlap unread input. The CPU must support NEON.
#[target_feature(enable = "neon")]
pub unsafe fn next_f64(src: *const u64, dst: *mut u64, len: usize, up: bool) {
    unsafe {
        let zero = vdupq_n_u64(0);
        let one = vdupq_n_u64(1);
        let sign = vdupq_n_u64(0x8000000000000000);
        let infinity = vdupq_n_u64(0x7ff0000000000000);
        let outward = vdupq_n_u64(if up {
            0x7ff0000000000000
        } else {
            0xfff0000000000000
        });
        let zero_result = vdupq_n_u64(if up { 1 } else { 0x8000000000000001 });
        let mut index = 0;
        while len - index >= 2 {
            let value = vld1q_u64(src.add(index));
            let magnitude = vbicq_u64(value, sign);
            let negative = vceqq_u64(vandq_u64(value, sign), sign);
            let delta = vorrq_u64(negative, one);
            let stepped = if up {
                vaddq_u64(value, delta)
            } else {
                vsubq_u64(value, delta)
            };
            let is_zero = vceqq_u64(magnitude, zero);
            let preserve = vorrq_u64(vcgtq_u64(magnitude, infinity), vceqq_u64(value, outward));
            let result = vbslq_u64(is_zero, zero_result, stepped);
            let result = vbslq_u64(preserve, value, result);
            vst1q_u64(dst.add(index), result);
            index += 2;
        }
        super::scalar::next_f64(src.add(index), dst.add(index), len - index, up);
    }
}
