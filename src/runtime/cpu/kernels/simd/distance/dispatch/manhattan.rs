//! Manhattan (L1) distance dispatch.

use super::super::scalar;
#[cfg(target_arch = "x86_64")]
use super::super::x86_64::avx2;
use crate::runtime::cpu::kernels::simd::{SimdLevel, detect_simd};

/// f32 `sum(|a[i] - b[i]|)` on the best level this CPU supports.
///
/// # Safety
/// `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn manhattan_f32(a: *const f32, b: *const f32, len: usize) -> f32 {
    manhattan_f32_with(detect_simd(), a, b, len)
}

/// f64 `sum(|a[i] - b[i]|)` on the best level this CPU supports.
///
/// # Safety
/// `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn manhattan_f64(a: *const f64, b: *const f64, len: usize) -> f64 {
    manhattan_f64_with(detect_simd(), a, b, len)
}

/// f32 `sum(|a[i] - b[i]|)` on an explicit level.
///
/// `SimdLevel::Scalar` reproduces `metrics::manhattan` bit for bit.
///
/// # Safety
/// - `level` must not exceed `detect_simd()`. A higher level runs instructions
///   this CPU lacks.
/// - `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn manhattan_f32_with(
    level: SimdLevel,
    a: *const f32,
    b: *const f32,
    len: usize,
) -> f32 {
    match level {
        // Every AVX-512 CPU also runs AVX2+FMA. The AVX-512 kernel lands in a later unit.
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 | SimdLevel::Avx2Fma => avx2::manhattan_f32(a, b, len),
        // The NEON kernel lands in a later unit.
        SimdLevel::Neon | SimdLevel::NeonFp16 => scalar::manhattan_f32(a, b, len),
        _ => scalar::manhattan_f32(a, b, len),
    }
}

/// f64 `sum(|a[i] - b[i]|)` on an explicit level.
///
/// `SimdLevel::Scalar` reproduces `metrics::manhattan` bit for bit.
///
/// # Safety
/// - `level` must not exceed `detect_simd()`. A higher level runs instructions
///   this CPU lacks.
/// - `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn manhattan_f64_with(
    level: SimdLevel,
    a: *const f64,
    b: *const f64,
    len: usize,
) -> f64 {
    match level {
        // Every AVX-512 CPU also runs AVX2+FMA. The AVX-512 kernel lands in a later unit.
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 | SimdLevel::Avx2Fma => avx2::manhattan_f64(a, b, len),
        // The NEON kernel lands in a later unit.
        SimdLevel::Neon | SimdLevel::NeonFp16 => scalar::manhattan_f64(a, b, len),
        _ => scalar::manhattan_f64(a, b, len),
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{Lcg, Side, check_all, lengths, levels, random_vec};
    use super::*;

    fn term(a: f64, b: f64) -> f64 {
        (a - b).abs()
    }

    fn k32(level: SimdLevel, a: *const f32, b: *const f32, len: usize) -> f32 {
        unsafe { manhattan_f32_with(level, a, b, len) }
    }

    fn k64(level: SimdLevel, a: *const f64, b: *const f64, len: usize) -> f64 {
        unsafe { manhattan_f64_with(level, a, b, len) }
    }

    #[test]
    fn f32_every_level_within_bound_and_specials() {
        check_all::<f32>("manhattan_f32", k32, term, &[Side::A, Side::B]);
    }

    #[test]
    fn f64_every_level_within_bound_and_specials() {
        check_all::<f64>("manhattan_f64", k64, term, &[Side::A, Side::B]);
    }

    #[test]
    fn identical_vectors_give_exact_zero() {
        let mut rng = Lcg::new(11);
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

    #[test]
    fn negative_zero_differences_add_no_sign() {
        // `andnot(-0.0, d)` must clear the sign of every `-0.0` difference.
        let a = vec![-0.0f32; 37];
        let b = vec![0.0f32; 37];
        for level in levels() {
            let got = k32(level, a.as_ptr(), b.as_ptr(), 37);
            assert!(
                got == 0.0 && got.is_sign_positive(),
                "level {level}, len 37, offset 0: got {got:e}"
            );
        }
    }
}
