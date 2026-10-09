//! Squared Euclidean distance dispatch.

use super::super::scalar;
#[cfg(target_arch = "x86_64")]
use super::super::x86_64::{avx2, avx512};
use crate::runtime::cpu::kernels::simd::{SimdLevel, detect_simd};

/// f32 `sum((a[i] - b[i])^2)` on the best level this CPU supports.
///
/// # Safety
/// `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn sqeuclidean_f32(a: *const f32, b: *const f32, len: usize) -> f32 {
    sqeuclidean_f32_with(detect_simd(), a, b, len)
}

/// f64 `sum((a[i] - b[i])^2)` on the best level this CPU supports.
///
/// # Safety
/// `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn sqeuclidean_f64(a: *const f64, b: *const f64, len: usize) -> f64 {
    sqeuclidean_f64_with(detect_simd(), a, b, len)
}

/// f32 `sum((a[i] - b[i])^2)` on an explicit level.
///
/// `SimdLevel::Scalar` reproduces `metrics::sqeuclidean` bit for bit.
///
/// # Safety
/// - `level` must not exceed `detect_simd()`. A higher level runs instructions
///   this CPU lacks.
/// - `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn sqeuclidean_f32_with(
    level: SimdLevel,
    a: *const f32,
    b: *const f32,
    len: usize,
) -> f32 {
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => avx512::sqeuclidean_f32(a, b, len),
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2Fma => avx2::sqeuclidean_f32(a, b, len),
        // The NEON kernel lands in a later unit.
        SimdLevel::Neon | SimdLevel::NeonFp16 => scalar::sqeuclidean_f32(a, b, len),
        _ => scalar::sqeuclidean_f32(a, b, len),
    }
}

/// f64 `sum((a[i] - b[i])^2)` on an explicit level.
///
/// `SimdLevel::Scalar` reproduces `metrics::sqeuclidean` bit for bit.
///
/// # Safety
/// - `level` must not exceed `detect_simd()`. A higher level runs instructions
///   this CPU lacks.
/// - `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn sqeuclidean_f64_with(
    level: SimdLevel,
    a: *const f64,
    b: *const f64,
    len: usize,
) -> f64 {
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => avx512::sqeuclidean_f64(a, b, len),
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2Fma => avx2::sqeuclidean_f64(a, b, len),
        // The NEON kernel lands in a later unit.
        SimdLevel::Neon | SimdLevel::NeonFp16 => scalar::sqeuclidean_f64(a, b, len),
        _ => scalar::sqeuclidean_f64(a, b, len),
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_avx512::{F32_MAX_LEN, F64_MAX_LEN, check_avx512_every_length};
    use super::super::test_support::{Lcg, Side, check_all, lengths, levels, random_vec};
    use super::*;

    fn term(a: f64, b: f64) -> f64 {
        (a - b) * (a - b)
    }

    fn k32(level: SimdLevel, a: *const f32, b: *const f32, len: usize) -> f32 {
        unsafe { sqeuclidean_f32_with(level, a, b, len) }
    }

    fn k64(level: SimdLevel, a: *const f64, b: *const f64, len: usize) -> f64 {
        unsafe { sqeuclidean_f64_with(level, a, b, len) }
    }

    #[test]
    fn f32_every_level_within_bound_and_specials() {
        check_all::<f32>("sqeuclidean_f32", k32, term, &[Side::A, Side::B]);
    }

    #[test]
    fn f64_every_level_within_bound_and_specials() {
        check_all::<f64>("sqeuclidean_f64", k64, term, &[Side::A, Side::B]);
    }

    #[test]
    fn avx512_every_length_within_bound() {
        check_avx512_every_length::<f32>("sqeuclidean_f32", k32, term, F32_MAX_LEN);
        check_avx512_every_length::<f64>("sqeuclidean_f64", k64, term, F64_MAX_LEN);
    }

    #[test]
    fn identical_vectors_give_exact_zero() {
        let mut rng = Lcg::new(7);
        for len in lengths(8) {
            let a = random_vec::<f32>(&mut rng, len, 1.0);
            let c = random_vec::<f64>(&mut rng, len, 1.0);
            for level in levels() {
                let got32 = k32(level, a.as_ptr(), a.as_ptr(), len);
                let got64 = k64(level, c.as_ptr(), c.as_ptr(), len);
                assert_eq!(got32, 0.0, "f32: level {level}, len {len}, offset 0");
                assert_eq!(got64, 0.0, "f64: level {level}, len {len}, offset 0");
            }
        }
    }
}
