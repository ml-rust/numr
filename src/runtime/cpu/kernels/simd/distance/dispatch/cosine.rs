//! One-pass cosine sums dispatch.

use super::super::scalar;
use super::super::sums::CosineSums;
#[cfg(target_arch = "x86_64")]
use super::super::x86_64::{avx2, avx512};
use crate::runtime::cpu::kernels::simd::{SimdLevel, detect_simd};

/// f32 cosine sums (`a·b`, `a·a`, `b·b`) on the best level this CPU supports.
///
/// # Safety
/// `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn cosine_sums_f32(a: *const f32, b: *const f32, len: usize) -> CosineSums<f32> {
    cosine_sums_f32_with(detect_simd(), a, b, len)
}

/// f64 cosine sums (`a·b`, `a·a`, `b·b`) on the best level this CPU supports.
///
/// # Safety
/// `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn cosine_sums_f64(a: *const f64, b: *const f64, len: usize) -> CosineSums<f64> {
    cosine_sums_f64_with(detect_simd(), a, b, len)
}

/// f32 cosine sums on an explicit level.
///
/// `SimdLevel::Scalar` reproduces the sums of `metrics::cosine` bit for bit.
///
/// # Safety
/// - `level` must not exceed `detect_simd()`. A higher level runs instructions
///   this CPU lacks.
/// - `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn cosine_sums_f32_with(
    level: SimdLevel,
    a: *const f32,
    b: *const f32,
    len: usize,
) -> CosineSums<f32> {
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => avx512::cosine_sums_f32(a, b, len),
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2Fma => avx2::cosine_sums_f32(a, b, len),
        // The NEON kernel lands in a later unit.
        SimdLevel::Neon | SimdLevel::NeonFp16 => scalar::cosine_sums_f32(a, b, len),
        _ => scalar::cosine_sums_f32(a, b, len),
    }
}

/// f64 cosine sums on an explicit level.
///
/// `SimdLevel::Scalar` reproduces the sums of `metrics::cosine` bit for bit.
///
/// # Safety
/// - `level` must not exceed `detect_simd()`. A higher level runs instructions
///   this CPU lacks.
/// - `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn cosine_sums_f64_with(
    level: SimdLevel,
    a: *const f64,
    b: *const f64,
    len: usize,
) -> CosineSums<f64> {
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => avx512::cosine_sums_f64(a, b, len),
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2Fma => avx2::cosine_sums_f64(a, b, len),
        // The NEON kernel lands in a later unit.
        SimdLevel::Neon | SimdLevel::NeonFp16 => scalar::cosine_sums_f64(a, b, len),
        _ => scalar::cosine_sums_f64(a, b, len),
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_avx512::{F32_MAX_LEN, F64_MAX_LEN, check_avx512_every_length};
    use super::super::test_support::{Lcg, Side, check_all, lengths, levels, random_vec};
    use super::*;

    fn dot_term(a: f64, b: f64) -> f64 {
        a * b
    }

    fn norm_a_term(a: f64, _b: f64) -> f64 {
        a * a
    }

    fn norm_b_term(_a: f64, b: f64) -> f64 {
        b * b
    }

    fn dot32(level: SimdLevel, a: *const f32, b: *const f32, len: usize) -> f32 {
        unsafe { cosine_sums_f32_with(level, a, b, len).dot }
    }

    fn norm_a32(level: SimdLevel, a: *const f32, b: *const f32, len: usize) -> f32 {
        unsafe { cosine_sums_f32_with(level, a, b, len).norm_a }
    }

    fn norm_b32(level: SimdLevel, a: *const f32, b: *const f32, len: usize) -> f32 {
        unsafe { cosine_sums_f32_with(level, a, b, len).norm_b }
    }

    fn dot64(level: SimdLevel, a: *const f64, b: *const f64, len: usize) -> f64 {
        unsafe { cosine_sums_f64_with(level, a, b, len).dot }
    }

    fn norm_a64(level: SimdLevel, a: *const f64, b: *const f64, len: usize) -> f64 {
        unsafe { cosine_sums_f64_with(level, a, b, len).norm_a }
    }

    fn norm_b64(level: SimdLevel, a: *const f64, b: *const f64, len: usize) -> f64 {
        unsafe { cosine_sums_f64_with(level, a, b, len).norm_b }
    }

    #[test]
    fn f32_every_level_within_bound_and_specials() {
        check_all::<f32>("cosine_f32.dot", dot32, dot_term, &[Side::A, Side::B]);
        check_all::<f32>("cosine_f32.norm_a", norm_a32, norm_a_term, &[Side::A]);
        check_all::<f32>("cosine_f32.norm_b", norm_b32, norm_b_term, &[Side::B]);
    }

    #[test]
    fn f64_every_level_within_bound_and_specials() {
        check_all::<f64>("cosine_f64.dot", dot64, dot_term, &[Side::A, Side::B]);
        check_all::<f64>("cosine_f64.norm_a", norm_a64, norm_a_term, &[Side::A]);
        check_all::<f64>("cosine_f64.norm_b", norm_b64, norm_b_term, &[Side::B]);
    }

    #[test]
    fn avx512_every_length_within_bound() {
        check_avx512_every_length::<f32>("cosine_f32.dot", dot32, dot_term, F32_MAX_LEN);
        check_avx512_every_length::<f32>("cosine_f32.norm_a", norm_a32, norm_a_term, F32_MAX_LEN);
        check_avx512_every_length::<f32>("cosine_f32.norm_b", norm_b32, norm_b_term, F32_MAX_LEN);
        check_avx512_every_length::<f64>("cosine_f64.dot", dot64, dot_term, F64_MAX_LEN);
        check_avx512_every_length::<f64>("cosine_f64.norm_a", norm_a64, norm_a_term, F64_MAX_LEN);
        check_avx512_every_length::<f64>("cosine_f64.norm_b", norm_b64, norm_b_term, F64_MAX_LEN);
    }

    #[test]
    fn opposite_vectors_give_consistent_sums() {
        // `b = -a`. Negation is exact and round-to-nearest is symmetric, so
        // `dot == -norm_a` and `norm_b == norm_a` hold bit for bit at every level.
        let mut rng = Lcg::new(13);
        for len in lengths(8) {
            let a = random_vec::<f32>(&mut rng, len, 1.0);
            let b: Vec<f32> = a.iter().map(|x| -x).collect();
            let c = random_vec::<f64>(&mut rng, len, 1.0);
            let d: Vec<f64> = c.iter().map(|x| -x).collect();
            for level in levels() {
                let s = unsafe { cosine_sums_f32_with(level, a.as_ptr(), b.as_ptr(), len) };
                assert_eq!(s.dot, -s.norm_a, "f32: level {level}, len {len}, offset 0");
                assert_eq!(
                    s.norm_b, s.norm_a,
                    "f32: level {level}, len {len}, offset 0"
                );
                let t = unsafe { cosine_sums_f64_with(level, c.as_ptr(), d.as_ptr(), len) };
                assert_eq!(t.dot, -t.norm_a, "f64: level {level}, len {len}, offset 0");
                assert_eq!(
                    t.norm_b, t.norm_a,
                    "f64: level {level}, len {len}, offset 0"
                );
            }
        }
    }
}
