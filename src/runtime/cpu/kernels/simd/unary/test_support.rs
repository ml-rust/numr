//! Shared test helpers for the SIMD unary kernel test suites.
//!
//! Every function here is called from more than one thematic test file under
//! this module, so it lives here instead of being duplicated per file.

#![cfg(test)]

/// One f32 subnormal step, the whole precision available below
/// `f32::MIN_POSITIVE`. Relative error is meaningless there — the grid
/// itself is only 5e-6 fine near 2.6e-40 — so the subnormal tests bound the
/// absolute error in these units instead.
pub(crate) const SUBNORMAL_STEP_F32: f32 = 1.401_298_5e-45;

/// Relative error against a reference, safe when the reference is zero.
///
/// asin(0) and acos(1) are exactly zero, so a plain division would report
/// NaN for a bit-exact result.
pub(crate) fn rel_err_f64(got: f64, expected: f64) -> f64 {
    // An exact match reports no error even when both sides are infinite or
    // both NaN. Subtracting two equal infinities yields NaN, which would
    // otherwise fail a comparison the kernel got exactly right.
    if got == expected || (got.is_nan() && expected.is_nan()) {
        return 0.0;
    }
    if !got.is_finite() || !expected.is_finite() {
        return f64::INFINITY;
    }
    (got - expected).abs() / expected.abs().max(f64::MIN_POSITIVE)
}

/// Fill `a` up to `len` with a sweep over [-limit, limit].
pub(crate) fn fill_sweep_f64(a: &mut Vec<f64>, len: usize, limit: f64) {
    let start = a.len();
    let span = len - start;
    for i in 0..span {
        a.push(-limit + 2.0 * limit * (i as f64) / (span as f64 - 1.0));
    }
}

/// Fill `a` up to `len` with a linear sweep over [lo, hi].
pub(crate) fn fill_range_f64(a: &mut Vec<f64>, len: usize, lo: f64, hi: f64) {
    let start = a.len();
    let span = len - start;
    for i in 0..span {
        a.push(lo + (hi - lo) * (i as f64) / (span as f64 - 1.0));
    }
}

/// Arguments that expose every weak point of a log reduction: the region
/// around 1 where `log` cancels, one point per binade across the whole
/// exponent range, and subnormals, which carry no implicit leading 1.
pub(crate) fn log_probe_points_f64(len: usize) -> Vec<f64> {
    let mut a: Vec<f64> = Vec::with_capacity(len);

    // Near 1 the mantissa polynomial is the entire result, so a series that
    // is merely "close" over the reduction interval shows up here.
    for k in -60i32..=60 {
        a.push(1.0 + (k as f64) * 1e-12);
        a.push(1.0 + (k as f64) * 1e-3);
    }

    // sqrt(2) is the normalization breakpoint; a wrong branch lands here.
    for k in -40i32..=40 {
        a.push(std::f64::consts::SQRT_2 + (k as f64) * 1e-12);
        a.push(std::f64::consts::FRAC_1_SQRT_2 + (k as f64) * 1e-12);
    }

    // One value per binade over the full exponent range, powers of two
    // included, plus subnormals below f64::MIN_POSITIVE.
    for k in -1074i32..=1023 {
        if k % 3 == 0 {
            a.push(2.0f64.powi(k));
        }
    }
    a.push(f64::MIN_POSITIVE);
    a.push(f64::MIN_POSITIVE * 0.5);
    a.push(5e-324);
    a.push(f64::MAX);

    fill_range_f64(&mut a, len, 1e-8, 1e8);
    a
}

/// Relative error against a reference, safe when the reference is zero.
///
/// The f32 counterpart of `rel_err_f64`: the reference is always the f64
/// result rounded once to f32, so the bound below is a bound on the kernel
/// alone and not on the reference.
pub(crate) fn rel_err_f32(got: f32, expected: f32) -> f32 {
    // An exact match reports no error even when both sides are infinite or
    // both NaN. Subtracting two equal infinities yields NaN, which would
    // otherwise fail a comparison the kernel got exactly right.
    if got == expected || (got.is_nan() && expected.is_nan()) {
        return 0.0;
    }
    if !got.is_finite() || !expected.is_finite() {
        return f32::INFINITY;
    }
    (got - expected).abs() / expected.abs().max(f32::MIN_POSITIVE)
}

/// Fill `a` up to `len` with a linear sweep over [lo, hi].
pub(crate) fn fill_range_f32(a: &mut Vec<f32>, len: usize, lo: f32, hi: f32) {
    let start = a.len();
    let span = len - start;
    for i in 0..span {
        a.push(lo + (hi - lo) * (i as f32) / (span as f32 - 1.0));
    }
}

/// One ulp step away from `x`, used to straddle a branch point exactly.
pub(crate) fn nudge_f32(x: f32, steps: i32) -> f32 {
    f32::from_bits((x.to_bits() as i32 + steps) as u32)
}

/// Reference atanh, evaluated on the side where `f64::atanh` is accurate.
///
/// `f64::atanh` is the only one of these seven references not delegated to
/// libm: std computes `0.5 * ((2x)/(1-x)).ln_1p()` directly, which is not
/// odd-symmetric. For x >= 0 the small quantity is the denominator `1 - x`,
/// exact by Sterbenz on [0.5, 1), and the quotient is large and well
/// conditioned. For x -> -1 the quotient instead approaches -1, where
/// `ln_1p` amplifies its half-ulp rounding by `1/(1+q)` without bound: 107
/// ulps at x = -(1 - 2^-13), and 1.8e6 ulps at x = -(1 - 2^-26). atanh is
/// odd, so the negative side is referenced through the positive one.
pub(crate) fn atanh_reference(x: f64) -> f64 {
    if x < 0.0 { -(-x).atanh() } else { x.atanh() }
}
