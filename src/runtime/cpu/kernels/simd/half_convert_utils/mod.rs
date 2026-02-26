//! SIMD-accelerated f16/bf16 ↔ f32 conversion utilities
//!
//! These are the building blocks for the block-convert-compute pattern:
//! convert half-precision data to f32 in L1-sized blocks, run existing
//! f32 SIMD kernels, then convert back.
//!
//! # Conversion strategies
//!
//! - **x86 f16**: F16C instructions (`_mm256_cvtph_ps` / `_mm256_cvtps_ph`)
//! - **x86 bf16**: SIMD integer bit-shift (`u32 << 16` for load, rounded `>> 16` for store)
//! - **ARM f16**: NEON `vcvt_f32_f16` / `vcvt_f16_f32`
//! - **ARM bf16**: NEON integer bit-shift
//! - **Fallback**: `half` crate scalar conversion

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

/// Block size for stack-allocated conversion buffers.
/// 256 f32s = 1024 bytes, fits comfortably in L1 cache.
pub const HALF_BLOCK: usize = 256;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Convert f16 values to f32 using SIMD when available.
///
/// # Safety
/// - `src` must be valid for reads of `len` u16 values (f16 bit patterns)
/// - `dst` must be valid for writes of `len` f32 values
#[inline]
pub unsafe fn convert_f16_to_f32(src: *const u16, dst: *mut f32, len: usize) {
    if len == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") {
            return x86_64::convert_f16_to_f32_f16c(src, dst, len);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return aarch64::convert_f16_to_f32_neon(src, dst, len);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    convert_f16_to_f32_scalar(src, dst, len);
}

/// Convert f32 values to f16 using SIMD when available.
///
/// # Safety
/// - `src` must be valid for reads of `len` f32 values
/// - `dst` must be valid for writes of `len` u16 values (f16 bit patterns)
#[inline]
pub unsafe fn convert_f32_to_f16(src: *const f32, dst: *mut u16, len: usize) {
    if len == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") {
            return x86_64::convert_f32_to_f16_f16c(src, dst, len);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return aarch64::convert_f32_to_f16_neon(src, dst, len);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    convert_f32_to_f16_scalar(src, dst, len);
}

/// Convert bf16 values to f32 using SIMD when available.
///
/// # Safety
/// - `src` must be valid for reads of `len` u16 values (bf16 bit patterns)
/// - `dst` must be valid for writes of `len` f32 values
#[inline]
pub unsafe fn convert_bf16_to_f32(src: *const u16, dst: *mut f32, len: usize) {
    if len == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return x86_64::convert_bf16_to_f32_avx2(src, dst, len);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return aarch64::convert_bf16_to_f32_neon(src, dst, len);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    convert_bf16_to_f32_scalar(src, dst, len);
}

/// Convert f32 values to bf16 using SIMD when available (with round-to-nearest-even).
///
/// # Safety
/// - `src` must be valid for reads of `len` f32 values
/// - `dst` must be valid for writes of `len` u16 values (bf16 bit patterns)
#[inline]
pub unsafe fn convert_f32_to_bf16(src: *const f32, dst: *mut u16, len: usize) {
    if len == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return x86_64::convert_f32_to_bf16_avx2(src, dst, len);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return aarch64::convert_f32_to_bf16_neon(src, dst, len);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    convert_f32_to_bf16_scalar(src, dst, len);
}

// ---------------------------------------------------------------------------
// Scalar fallbacks
// ---------------------------------------------------------------------------

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
unsafe fn convert_f16_to_f32_scalar(src: *const u16, dst: *mut f32, len: usize) {
    for i in 0..len {
        *dst.add(i) = half::f16::from_bits(*src.add(i)).to_f32();
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
unsafe fn convert_f32_to_f16_scalar(src: *const f32, dst: *mut u16, len: usize) {
    for i in 0..len {
        *dst.add(i) = half::f16::from_f32(*src.add(i)).to_bits();
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
unsafe fn convert_bf16_to_f32_scalar(src: *const u16, dst: *mut f32, len: usize) {
    for i in 0..len {
        *dst.add(i) = half::bf16::from_bits(*src.add(i)).to_f32();
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
unsafe fn convert_f32_to_bf16_scalar(src: *const f32, dst: *mut u16, len: usize) {
    for i in 0..len {
        let bits = (*src.add(i)).to_bits();
        let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
        *dst.add(i) = (rounded >> 16) as u16;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_roundtrip() {
        let values: Vec<f32> = vec![
            0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            65504.0,
            -65504.0,
            0.000061035156,
            3.15,
        ];
        let f16_bits: Vec<u16> = values
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let mut f32_out = vec![0.0f32; values.len()];
        let mut f16_out = vec![0u16; values.len()];

        unsafe {
            convert_f16_to_f32(f16_bits.as_ptr(), f32_out.as_mut_ptr(), values.len());
            convert_f32_to_f16(f32_out.as_ptr(), f16_out.as_mut_ptr(), f32_out.len());
        }

        for i in 0..values.len() {
            assert_eq!(
                f16_bits[i], f16_out[i],
                "f16 roundtrip failed at index {}: input bits {:04x}, output bits {:04x}",
                i, f16_bits[i], f16_out[i]
            );
        }
    }

    #[test]
    fn test_bf16_roundtrip() {
        let values: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 3.15];
        let bf16_bits: Vec<u16> = values
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_bits())
            .collect();
        let mut f32_out = vec![0.0f32; values.len()];
        let mut bf16_out = vec![0u16; values.len()];

        unsafe {
            convert_bf16_to_f32(bf16_bits.as_ptr(), f32_out.as_mut_ptr(), values.len());
            convert_f32_to_bf16(f32_out.as_ptr(), bf16_out.as_mut_ptr(), f32_out.len());
        }

        for i in 0..values.len() {
            assert_eq!(
                bf16_bits[i], bf16_out[i],
                "bf16 roundtrip failed at index {}: input bits {:04x}, output bits {:04x}",
                i, bf16_bits[i], bf16_out[i]
            );
        }
    }

    #[test]
    fn test_f16_conversion_accuracy() {
        let f16_bits: Vec<u16> = (0..512)
            .map(|i| half::f16::from_f32((i as f32 - 256.0) * 0.1).to_bits())
            .collect();
        let mut f32_out = vec![0.0f32; f16_bits.len()];
        unsafe { convert_f16_to_f32(f16_bits.as_ptr(), f32_out.as_mut_ptr(), f16_bits.len()) }

        for i in 0..f16_bits.len() {
            let expected = half::f16::from_bits(f16_bits[i]).to_f32();
            assert_eq!(f32_out[i], expected, "f16→f32 mismatch at index {}", i);
        }
    }

    #[test]
    fn test_bf16_conversion_accuracy() {
        let bf16_bits: Vec<u16> = (0..512)
            .map(|i| half::bf16::from_f32((i as f32 - 256.0) * 0.1).to_bits())
            .collect();
        let mut f32_out = vec![0.0f32; bf16_bits.len()];
        unsafe { convert_bf16_to_f32(bf16_bits.as_ptr(), f32_out.as_mut_ptr(), bf16_bits.len()) }

        for i in 0..bf16_bits.len() {
            let expected = half::bf16::from_bits(bf16_bits[i]).to_f32();
            assert_eq!(f32_out[i], expected, "bf16→f32 mismatch at index {}", i);
        }
    }

    #[test]
    fn test_empty_conversion() {
        unsafe {
            convert_f16_to_f32(std::ptr::null(), std::ptr::null_mut(), 0);
            convert_f32_to_f16(std::ptr::null(), std::ptr::null_mut(), 0);
            convert_bf16_to_f32(std::ptr::null(), std::ptr::null_mut(), 0);
            convert_f32_to_bf16(std::ptr::null(), std::ptr::null_mut(), 0);
        }
    }

    #[test]
    fn test_unaligned_lengths() {
        for len in [1, 3, 5, 7, 9, 15, 17, 31, 33] {
            let f16_bits: Vec<u16> = (0..len)
                .map(|i| half::f16::from_f32(i as f32).to_bits())
                .collect();
            let mut f32_out = vec![0.0f32; len];

            unsafe { convert_f16_to_f32(f16_bits.as_ptr(), f32_out.as_mut_ptr(), len) }

            for i in 0..len {
                let expected = half::f16::from_bits(f16_bits[i]).to_f32();
                assert_eq!(f32_out[i], expected, "mismatch at len={}, index={}", len, i);
            }
        }
    }
}
