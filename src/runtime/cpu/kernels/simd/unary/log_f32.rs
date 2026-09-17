//! Log family (log, log2, log10, log1p) f32 dispatch.
//! Same threshold/ISA-selection logic as every unary family (`dispatch.rs`).

use super::super::{SimdLevel, detect_simd};
use super::dispatch::{SIMD_THRESHOLD, is_simd_supported};
use super::unary_scalar_f32;
use crate::ops::UnaryOp;

/// Dispatch a log-family op for f32.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub(super) unsafe fn log_f32(op: UnaryOp, a: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();
    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar || !is_simd_supported(op) {
        return unsafe { unary_scalar_f32(op, a, out, len) };
    }
    unsafe {
        #[cfg(target_arch = "x86_64")]
        match level {
            SimdLevel::Avx512 => super::x86_64::avx512::unary_f32(op, a, out, len),
            SimdLevel::Avx2Fma => super::x86_64::avx2::unary_f32(op, a, out, len),
            _ => unary_scalar_f32(op, a, out, len),
        }
        #[cfg(target_arch = "aarch64")]
        match level {
            SimdLevel::Neon | SimdLevel::NeonFp16 => {
                super::aarch64::neon::unary_f32(op, a, out, len)
            }
            _ => unary_scalar_f32(op, a, out, len),
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        unary_scalar_f32(op, a, out, len);
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{SUBNORMAL_STEP_F32, fill_range_f32, nudge_f32, rel_err_f32};
    use super::*;

    #[test]
    fn test_unary_log_f32() {
        let a: Vec<f32> = (1..101).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { log_f32(UnaryOp::Log, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = (a[i] as f64).ln() as f32;
            assert!(
                rel_err_f32(out[i], expected) < 1e-6,
                "log mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    /// `2^k` for k in [-149, 127], built from the bit pattern.
    ///
    /// `2.0f32.powi(k)` cannot serve here: for k below -126 it evaluates
    /// `1 / 2^|k|`, whose numerator overflows to infinity, so it returns zero
    /// for the whole subnormal range these tests exist to cover.
    fn pow2_f32(k: i32) -> f32 {
        if k >= -126 {
            f32::from_bits(((k + 127) as u32) << 23)
        } else {
            f32::from_bits(1u32 << (k + 149))
        }
    }

    /// Arguments that expose every weak point of an f32 log reduction: the
    /// region around 1 where `log` cancels, the sqrt(2) normalization
    /// breakpoint, one value per binade across the whole exponent range, and
    /// subnormals, which carry no implicit leading 1.
    fn log_probe_points_f32(len: usize) -> Vec<f32> {
        let mut a: Vec<f32> = Vec::with_capacity(len);

        // Near 1 the mantissa polynomial is the entire result, so a series that
        // is merely "close" over the reduction interval shows up here.
        for k in -40i32..=40 {
            a.push(nudge_f32(1.0, k));
            a.push(nudge_f32(std::f32::consts::SQRT_2, k));
            a.push(nudge_f32(std::f32::consts::FRAC_1_SQRT_2, k));
        }
        for k in -60i32..=60 {
            a.push(1.0 + (k as f32) * 1e-3);
        }

        // One value per binade over the full exponent range, powers of two
        // included, plus subnormals below f32::MIN_POSITIVE. The old kernel
        // decomposed a subnormal against an absent leading 1.
        for k in -149i32..=127 {
            if k % 3 == 0 {
                a.push(pow2_f32(k));
            }
        }
        a.push(f32::MIN_POSITIVE);
        a.push(f32::MIN_POSITIVE * 0.5);
        a.push(SUBNORMAL_STEP_F32);
        a.push(f32::MAX);

        fill_range_f32(&mut a, len, 1e-8, 1e8);
        a
    }

    /// The f32 log family must reach single precision over the whole positive
    /// range. The seven-term Mercator series it used leaves 1.1e-4 of absolute
    /// error at the top of the reduction interval, which is 1e-6 relative once
    /// the exponent term is large and far worse where it is not, and it
    /// propagates into asinh, acosh and atanh unchanged.
    #[test]
    fn test_unary_log_family_f32_single_precision() {
        const LEN: usize = 2048;
        let a = log_probe_points_f32(LEN);
        let mut out = vec![0.0f32; LEN];

        for (op, reference) in [
            (UnaryOp::Log, f64::ln as fn(f64) -> f64),
            (UnaryOp::Log2, f64::log2 as fn(f64) -> f64),
            (UnaryOp::Log10, f64::log10 as fn(f64) -> f64),
        ] {
            unsafe { log_f32(op, a.as_ptr(), out.as_mut_ptr(), LEN) }

            for i in 0..LEN {
                let expected = reference(a[i] as f64) as f32;
                let rel_err = rel_err_f32(out[i], expected);
                assert!(
                    rel_err < 1e-6,
                    "{:?}({:e}) = {:e}, expected {:e}, rel_err = {} at index {}",
                    op,
                    a[i],
                    out[i],
                    expected,
                    rel_err,
                    i
                );
            }
        }
    }

    /// log2 adds the exact integer exponent back rather than scaling `log(x)`,
    /// so every power of two comes out exact. Scaling by log2(e) rounds twice
    /// and misses them.
    #[test]
    fn test_unary_log2_f32_exact_powers_of_two() {
        const LEN: usize = 2048;
        let a: Vec<f32> = (0..LEN)
            .map(|i| pow2_f32(-149 + (i as i32) % 277))
            .collect();
        let mut out = vec![0.0f32; LEN];

        unsafe { log_f32(UnaryOp::Log2, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (-149 + (i as i32) % 277) as f32;
            assert_eq!(
                out[i], expected,
                "log2({:e}) = {}, expected exactly {}",
                a[i], out[i], expected
            );
        }
    }

    /// The f32 log domain edges. The old kernel had none: it fed the bit
    /// pattern of zero, of a negative number and of infinity straight through
    /// the exponent split, so log(0) came back finite and log(-1) was a number.
    #[test]
    fn test_unary_log_f32_domain_edges() {
        const LEN: usize = 2048;
        let edges = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            SUBNORMAL_STEP_F32,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -2.5,
            2.0,
            f32::MAX,
        ];
        let a: Vec<f32> = (0..LEN).map(|i| edges[i % edges.len()]).collect();
        let mut out = vec![0.0f32; LEN];

        for (op, reference) in [
            (UnaryOp::Log, f64::ln as fn(f64) -> f64),
            (UnaryOp::Log2, f64::log2 as fn(f64) -> f64),
            (UnaryOp::Log10, f64::log10 as fn(f64) -> f64),
        ] {
            unsafe { log_f32(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
            for i in 0..LEN {
                let expected = reference(a[i] as f64) as f32;
                let rel_err = rel_err_f32(out[i], expected);
                assert!(
                    rel_err < 1e-6,
                    "{:?}({}) = {}, expected {} at index {}",
                    op,
                    a[i],
                    out[i],
                    expected,
                    i
                );
            }
        }
    }

    /// f32 log1p at tiny |x|, where `1 + x` rounds the answer away entirely,
    /// and around -0.5, where the degree-4 Taylor series it used dropped
    /// `x⁵/5` — 6.1e-3, a hundredth of the result.
    #[test]
    fn test_unary_log1p_f32_single_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);

        for k in 1i32..=30 {
            a.push(10.0f32.powi(-k));
            a.push(-10.0f32.powi(-k));
        }
        for k in -100i32..=100 {
            a.push(-0.4980708 + (k as f32) * 1e-7);
        }
        // Both sides of the 0.5 breakpoint the old kernel switched on.
        for k in -20i32..=20 {
            a.push(nudge_f32(0.5, k));
            a.push(nudge_f32(-0.5, k));
        }
        // Approaching -1, where log1p -> -inf.
        for k in 1i32..=23 {
            a.push(-1.0 + 2.0f32.powi(-k));
        }
        for k in 1i32..=30 {
            a.push(10.0f32.powi(k));
        }
        fill_range_f32(&mut a, LEN, -0.9, 10.0);

        let mut out = vec![0.0f32; LEN];
        unsafe { log_f32(UnaryOp::Log1p, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).ln_1p() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "log1p({:e}) = {:e}, expected {:e}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }
}
