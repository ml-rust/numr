//! Dot product dispatch.

#[cfg(target_arch = "aarch64")]
use super::super::aarch64::neon;
use super::super::scalar;
#[cfg(target_arch = "x86_64")]
use super::super::x86_64::{avx2, avx512};
use crate::runtime::cpu::kernels::simd::SimdLevel;

/// f32 `sum(a[i] * b[i])` on an explicit level.
///
/// `SimdLevel::Scalar` keeps one sequential accumulator, in index order.
///
/// # Safety
/// - `level` must not exceed `detect_simd()`. A higher level runs instructions
///   this CPU lacks.
/// - `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn dot_f32_with(level: SimdLevel, a: *const f32, b: *const f32, len: usize) -> f32 {
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => avx512::dot_f32(a, b, len),
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2Fma => avx2::dot_f32(a, b, len),
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon | SimdLevel::NeonFp16 => neon::dot_f32(a, b, len),
        _ => scalar::dot_f32(a, b, len),
    }
}

/// f64 `sum(a[i] * b[i])` on an explicit level.
///
/// `SimdLevel::Scalar` keeps one sequential accumulator, in index order.
///
/// # Safety
/// - `level` must not exceed `detect_simd()`. A higher level runs instructions
///   this CPU lacks.
/// - `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn dot_f64_with(level: SimdLevel, a: *const f64, b: *const f64, len: usize) -> f64 {
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => avx512::dot_f64(a, b, len),
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2Fma => avx2::dot_f64(a, b, len),
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon | SimdLevel::NeonFp16 => neon::dot_f64(a, b, len),
        _ => scalar::dot_f64(a, b, len),
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_avx512::{F32_MAX_LEN, F64_MAX_LEN, check_avx512_every_length};
    use super::super::test_support::{Side, check_all};
    use super::*;
    use crate::runtime::cpu::kernels::simd::detect_simd;

    fn term(a: f64, b: f64) -> f64 {
        a * b
    }

    fn k32(level: SimdLevel, a: *const f32, b: *const f32, len: usize) -> f32 {
        unsafe { dot_f32_with(level, a, b, len) }
    }

    fn k64(level: SimdLevel, a: *const f64, b: *const f64, len: usize) -> f64 {
        unsafe { dot_f64_with(level, a, b, len) }
    }

    #[test]
    fn f32_every_level_within_bound_and_specials() {
        check_all::<f32>("dot_f32", k32, term, &[Side::A, Side::B]);
    }

    #[test]
    fn f64_every_level_within_bound_and_specials() {
        check_all::<f64>("dot_f64", k64, term, &[Side::A, Side::B]);
    }

    #[test]
    fn avx512_every_length_within_bound() {
        check_avx512_every_length::<f32>("dot_f32", k32, term, F32_MAX_LEN);
        check_avx512_every_length::<f64>("dot_f64", k64, term, F64_MAX_LEN);
    }

    #[test]
    fn public_dispatch_matches_the_detected_level() {
        let a: Vec<f32> = (0..45).map(|i| i as f32 * 0.25 - 3.0).collect();
        let c: Vec<f64> = (0..45).map(|i| i as f64 * 0.25 - 3.0).collect();
        let level = detect_simd();
        let want32 = unsafe { dot_f32_with(level, a.as_ptr(), a.as_ptr(), 45) };
        let want64 = unsafe { dot_f64_with(level, c.as_ptr(), c.as_ptr(), 45) };
        assert_eq!(
            crate::distance::dot_f32(&a, &a).to_bits(),
            want32.to_bits(),
            "f32: level {level}, len 45, offset 0"
        );
        assert_eq!(
            crate::distance::dot_f64(&c, &c).to_bits(),
            want64.to_bits(),
            "f64: level {level}, len 45, offset 0"
        );
    }
}
