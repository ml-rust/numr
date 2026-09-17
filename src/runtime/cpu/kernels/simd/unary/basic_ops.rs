//! Basic unary ops (neg, abs, sign, sqrt, rsqrt, cbrt, square, recip, floor,
//! ceil, round, round-ties-even, trunc) — the catch-all family for every
//! SIMD-supported [`UnaryOp`] not owned by a more specific family module.
//!
//! Owns the same threshold-check/ISA-selection dispatch as every other
//! family (see `dispatch.rs`'s module doc), so it behaves identically
//! whether reached through `dispatch::unary_f32`/`unary_f64` or called
//! directly, as this file's own tests do.

use super::super::{SimdLevel, detect_simd};
use super::dispatch::{SIMD_THRESHOLD, is_simd_supported};
use super::{unary_scalar_f32, unary_scalar_f64};
use crate::ops::UnaryOp;

/// Dispatch a basic unary op for f32.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub(super) unsafe fn basic_f32(op: UnaryOp, a: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar || !is_simd_supported(op) {
        unsafe { unary_scalar_f32(op, a, out, len) };
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => unsafe { super::x86_64::avx512::unary_f32(op, a, out, len) },
        SimdLevel::Avx2Fma => unsafe { super::x86_64::avx2::unary_f32(op, a, out, len) },
        _ => unsafe { unary_scalar_f32(op, a, out, len) },
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => unsafe {
            super::aarch64::neon::unary_f32(op, a, out, len)
        },
        _ => unsafe { unary_scalar_f32(op, a, out, len) },
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    unsafe {
        unary_scalar_f32(op, a, out, len)
    };
}

/// f64 counterpart of [`basic_f32`].
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub(super) unsafe fn basic_f64(op: UnaryOp, a: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar || !is_simd_supported(op) {
        unsafe { unary_scalar_f64(op, a, out, len) };
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => unsafe { super::x86_64::avx512::unary_f64(op, a, out, len) },
        SimdLevel::Avx2Fma => unsafe { super::x86_64::avx2::unary_f64(op, a, out, len) },
        _ => unsafe { unary_scalar_f64(op, a, out, len) },
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => unsafe {
            super::aarch64::neon::unary_f64(op, a, out, len)
        },
        _ => unsafe { unary_scalar_f64(op, a, out, len) },
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    unsafe {
        unary_scalar_f64(op, a, out, len)
    };
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{fill_range_f32, rel_err_f32};
    use super::*;

    #[test]
    fn test_unary_neg_f32() {
        let a: Vec<f32> = (0..100).map(|x| x as f32 - 50.0).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { basic_f32(UnaryOp::Neg, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert_eq!(out[i], -a[i], "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_unary_abs_f32() {
        let a: Vec<f32> = (0..100).map(|x| x as f32 - 50.0).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { basic_f32(UnaryOp::Abs, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert_eq!(out[i], a[i].abs(), "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_unary_sign_f32() {
        let a: Vec<f32> = (0..100).map(|x| x as f32 - 50.0).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { basic_f32(UnaryOp::Sign, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = if a[i] > 0.0 {
                1.0
            } else if a[i] < 0.0 {
                -1.0
            } else {
                0.0
            };
            assert_eq!(out[i], expected, "sign mismatch at index {}", i);
        }
    }

    /// Inputs that stress both rounding modes: every tie between -4.5 and 4.5,
    /// the largest f32 below 0.5 (where a naive `floor(|x| + 0.5)` rounds the
    /// wrong way), values above 2^23 that are already integers, and infinities.
    fn rounding_probe_f32() -> Vec<f32> {
        let mut v = vec![
            -4.5,
            -3.5,
            -2.5,
            -1.5,
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            0.0,
            -0.0,
            1.1,
            -1.1,
            3.9,
            -4.7,
            f32::from_bits(0x3EFF_FFFF),
            -f32::from_bits(0x3EFF_FFFF),
            8_388_609.0,
            -8_388_609.0,
            1e30,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        // Pad past SIMD_THRESHOLD with a length that leaves a scalar tail for
        // every vector width in use (16, 8 and 4 lanes).
        while v.len() < 35 {
            v.push(v.len() as f32 * 0.25);
        }
        v
    }

    #[test]
    fn test_unary_round_f32_ties_away_from_zero() {
        let a = rounding_probe_f32();
        let len = a.len();
        let mut out = vec![0.0f32; len];

        unsafe { basic_f32(UnaryOp::Round, a.as_ptr(), out.as_mut_ptr(), len) }

        for i in 0..len {
            let expected = a[i].round();
            assert_eq!(
                out[i].to_bits(),
                expected.to_bits(),
                "round mismatch at {}: input {}, got {}, expected {}",
                i,
                a[i],
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_unary_round_ties_even_f32() {
        let a = rounding_probe_f32();
        let len = a.len();
        let mut out = vec![0.0f32; len];

        unsafe { basic_f32(UnaryOp::RoundTiesEven, a.as_ptr(), out.as_mut_ptr(), len) }

        for i in 0..len {
            let expected = a[i].round_ties_even();
            assert_eq!(
                out[i].to_bits(),
                expected.to_bits(),
                "round_ties_even mismatch at {}: input {}, got {}, expected {}",
                i,
                a[i],
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_unary_round_f64_ties_away_from_zero() {
        let a: Vec<f64> = rounding_probe_f32().iter().map(|&x| f64::from(x)).collect();
        let len = a.len();
        let mut out = vec![0.0f64; len];

        unsafe { basic_f64(UnaryOp::Round, a.as_ptr(), out.as_mut_ptr(), len) }

        for i in 0..len {
            let expected = a[i].round();
            assert_eq!(
                out[i].to_bits(),
                expected.to_bits(),
                "round mismatch at {}: input {}, got {}, expected {}",
                i,
                a[i],
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_unary_round_ties_even_f64() {
        let a: Vec<f64> = rounding_probe_f32().iter().map(|&x| f64::from(x)).collect();
        let len = a.len();
        let mut out = vec![0.0f64; len];

        unsafe { basic_f64(UnaryOp::RoundTiesEven, a.as_ptr(), out.as_mut_ptr(), len) }

        for i in 0..len {
            let expected = a[i].round_ties_even();
            assert_eq!(
                out[i].to_bits(),
                expected.to_bits(),
                "round_ties_even mismatch at {}: input {}, got {}, expected {}",
                i,
                a[i],
                out[i],
                expected
            );
        }
    }

    /// The SIMD kernels only run at or above `SIMD_THRESHOLD`, so a short input
    /// silently tests the scalar path alone. Sweep the lengths around and past
    /// the threshold to cover both.
    #[test]
    fn test_round_ops_across_simd_threshold_f32() {
        let probe = rounding_probe_f32();
        for len in [1usize, 7, 31, 32, 33, 35, 64, 65] {
            let a: Vec<f32> = probe.iter().copied().cycle().take(len).collect();
            let mut away = vec![0.0f32; len];
            let mut even = vec![0.0f32; len];

            unsafe {
                basic_f32(UnaryOp::Round, a.as_ptr(), away.as_mut_ptr(), len);
                basic_f32(UnaryOp::RoundTiesEven, a.as_ptr(), even.as_mut_ptr(), len);
            }

            for i in 0..len {
                assert_eq!(
                    away[i].to_bits(),
                    a[i].round().to_bits(),
                    "round mismatch at len {} index {} (input {})",
                    len,
                    i,
                    a[i]
                );
                assert_eq!(
                    even[i].to_bits(),
                    a[i].round_ties_even().to_bits(),
                    "round_ties_even mismatch at len {} index {} (input {})",
                    len,
                    i,
                    a[i]
                );
            }
        }
    }

    /// f32 cbrt over every binade, both signs. Seeding the iteration from the
    /// exponent alone leaves the mantissa unaccounted for: the seed is off by
    /// up to 37%, and two Newton steps square that to 5e-2, not to 1e-7.
    #[test]
    fn test_unary_cbrt_f32_single_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);

        // One value per binade over the full exponent range, powers of two
        // included, subnormals below 2^-126 among them. Both are built from
        // bit patterns: a subnormal power of two has no exponent field to set.
        for k in -149i32..=127 {
            let p = if k >= -126 {
                f32::from_bits(((k + 127) as u32) << 23)
            } else {
                f32::from_bits(1u32 << (k + 149))
            };
            a.push(p);
            a.push(-p);
            a.push(1.5 * p);
            a.push(-1.5 * p);
        }
        a.push(f32::MAX);
        a.push(-f32::MAX);
        a.push(-31.965_813);
        fill_range_f32(&mut a, LEN, -100.0, 100.0);

        let mut out = vec![0.0f32; LEN];
        unsafe { basic_f32(UnaryOp::Cbrt, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).cbrt() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "cbrt({:e}) = {:e}, expected {:e}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// cbrt is odd, so the two halves of the line must agree exactly.
    #[test]
    fn test_unary_cbrt_f32_odd_symmetry() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);
        fill_range_f32(&mut a, LEN / 2, 1e-30, 1e30);
        let positive = a.clone();
        for &v in &positive {
            a.push(-v);
        }

        let mut out = vec![0.0f32; LEN];
        unsafe { basic_f32(UnaryOp::Cbrt, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN / 2 {
            assert_eq!(
                out[i],
                -out[i + LEN / 2],
                "cbrt({:e}) = {:e} but cbrt of its negation is {:e}",
                a[i],
                out[i],
                out[i + LEN / 2]
            );
        }
    }
}
