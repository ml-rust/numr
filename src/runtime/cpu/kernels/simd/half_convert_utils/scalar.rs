//! Scalar fallbacks for f16/bf16 <-> f32 conversion (non-x86_64/aarch64 targets)

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
pub(super) unsafe fn convert_f16_to_f32_scalar(src: *const u16, dst: *mut f32, len: usize) {
    for i in 0..len {
        *dst.add(i) = half::f16::from_bits(*src.add(i)).to_f32();
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
pub(super) unsafe fn convert_f32_to_f16_scalar(src: *const f32, dst: *mut u16, len: usize) {
    for i in 0..len {
        *dst.add(i) = half::f16::from_f32(*src.add(i)).to_bits();
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
pub(super) unsafe fn convert_bf16_to_f32_scalar(src: *const u16, dst: *mut f32, len: usize) {
    for i in 0..len {
        *dst.add(i) = half::bf16::from_bits(*src.add(i)).to_f32();
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
pub(super) unsafe fn convert_f32_to_bf16_scalar(src: *const f32, dst: *mut u16, len: usize) {
    for i in 0..len {
        let bits = (*src.add(i)).to_bits();
        let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
        *dst.add(i) = (rounded >> 16) as u16;
    }
}
