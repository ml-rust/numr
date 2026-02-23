//! aarch64 NEON implementations for f16/bf16 ↔ f32 conversion
//!
//! - f16: NEON `vcvt_f32_f16` / `vcvt_f16_f32`
//! - bf16: NEON integer bit-shift

// ---------------------------------------------------------------------------
// NEON: f16 ↔ f32
// ---------------------------------------------------------------------------

pub(super) unsafe fn convert_f16_to_f32_neon(src: *const u16, dst: *mut f32, len: usize) {
    use std::arch::aarch64::*;

    let mut i = 0usize;

    // Process 4 elements at a time using vcvt_f32_f16
    while i + 4 <= len {
        let half_vec = vld1_u16(src.add(i));
        let half_f16 = vreinterpret_f16_u16(half_vec);
        let float_vec = vcvt_f32_f16(half_f16);
        vst1q_f32(dst.add(i), float_vec);
        i += 4;
    }

    // Scalar tail
    while i < len {
        *dst.add(i) = half::f16::from_bits(*src.add(i)).to_f32();
        i += 1;
    }
}

pub(super) unsafe fn convert_f32_to_f16_neon(src: *const f32, dst: *mut u16, len: usize) {
    use std::arch::aarch64::*;

    let mut i = 0usize;

    // Process 4 elements at a time using vcvt_f16_f32
    while i + 4 <= len {
        let float_vec = vld1q_f32(src.add(i));
        let half_f16 = vcvt_f16_f32(float_vec);
        let half_u16 = vreinterpret_u16_f16(half_f16);
        vst1_u16(dst.add(i), half_u16);
        i += 4;
    }

    // Scalar tail
    while i < len {
        *dst.add(i) = half::f16::from_f32(*src.add(i)).to_bits();
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// NEON: bf16 ↔ f32 (integer bit-shift)
// ---------------------------------------------------------------------------

pub(super) unsafe fn convert_bf16_to_f32_neon(src: *const u16, dst: *mut f32, len: usize) {
    use std::arch::aarch64::*;

    let mut i = 0usize;

    // Process 4 bf16 values at a time: zero-extend to u32, shift left 16
    while i + 4 <= len {
        let bf16_vec = vld1_u16(src.add(i));
        // vmovl_u16: uint16x4 → uint32x4 (zero-extend)
        let u32_vec = vmovl_u16(bf16_vec);
        let shifted = vshlq_n_u32(u32_vec, 16);
        let f32_vec = vreinterpretq_f32_u32(shifted);
        vst1q_f32(dst.add(i), f32_vec);
        i += 4;
    }

    // Scalar tail
    while i < len {
        let bits = (*src.add(i) as u32) << 16;
        *dst.add(i) = f32::from_bits(bits);
        i += 1;
    }
}

pub(super) unsafe fn convert_f32_to_bf16_neon(src: *const f32, dst: *mut u16, len: usize) {
    use std::arch::aarch64::*;

    let mut i = 0usize;

    let rounding_bias = vdupq_n_u32(0x7FFF);
    let one = vdupq_n_u32(1);

    // Process 4 f32 values at a time
    while i + 4 <= len {
        let f32_vec = vld1q_f32(src.add(i));
        let bits = vreinterpretq_u32_f32(f32_vec);

        // Round-to-nearest-even
        let shifted = vshrq_n_u32(bits, 16);
        let lsb = vandq_u32(shifted, one);
        let bias = vaddq_u32(rounding_bias, lsb);
        let rounded = vaddq_u32(bits, bias);
        let bf16_u32 = vshrq_n_u32(rounded, 16);

        // Narrow u32x4 → u16x4
        let bf16_u16 = vmovn_u32(bf16_u32);
        vst1_u16(dst.add(i), bf16_u16);
        i += 4;
    }

    // Scalar tail with same rounding
    while i < len {
        let bits = (*src.add(i)).to_bits();
        let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
        *dst.add(i) = (rounded >> 16) as u16;
        i += 1;
    }
}
