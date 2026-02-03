//! AVX2 special function kernels
//!
//! Vectorized implementations using polynomial evaluation with FMA.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::math::avx2::{exp_f32, exp_f64};
use super::coefficients::{bessel_i0, bessel_j0, erf};

const F32_LANES: usize = 8;
const F64_LANES: usize = 4;

// ============================================================================
// Error Function (erf)
// ============================================================================

/// Vectorized erf for f32 using AVX2
///
/// Uses Abramowitz & Stegun approximation 7.1.26:
/// erf(x) = 1 - (a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵) * exp(-x²)
/// where t = 1/(1 + p*|x|)
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn erf_f32(input: *const f32, output: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    // Load constants
    let a1 = _mm256_set1_ps(erf::A1_F32);
    let a2 = _mm256_set1_ps(erf::A2_F32);
    let a3 = _mm256_set1_ps(erf::A3_F32);
    let a4 = _mm256_set1_ps(erf::A4_F32);
    let a5 = _mm256_set1_ps(erf::A5_F32);
    let p = _mm256_set1_ps(erf::P_F32);
    let one = _mm256_set1_ps(1.0);
    let sign_mask = _mm256_set1_ps(-0.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let x = _mm256_loadu_ps(input.add(offset));

        // Extract sign and work with absolute value
        let sign = _mm256_and_ps(x, sign_mask);
        let ax = _mm256_andnot_ps(sign_mask, x);

        // t = 1/(1 + p*|x|)
        let t = _mm256_div_ps(one, _mm256_fmadd_ps(p, ax, one));

        // Horner's method: ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
        let mut poly = a5;
        poly = _mm256_fmadd_ps(poly, t, a4);
        poly = _mm256_fmadd_ps(poly, t, a3);
        poly = _mm256_fmadd_ps(poly, t, a2);
        poly = _mm256_fmadd_ps(poly, t, a1);
        poly = _mm256_mul_ps(poly, t);

        // exp(-x²)
        let neg_x2 = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(ax, ax));
        let exp_term = exp_f32(neg_x2);

        // y = 1 - poly * exp(-x²)
        let y = _mm256_fnmadd_ps(poly, exp_term, one);

        // Restore sign
        let result = _mm256_or_ps(y, sign);

        _mm256_storeu_ps(output.add(offset), result);
    }

    // Handle remainder
    if remainder > 0 {
        let offset = chunks * F32_LANES;
        for i in 0..remainder {
            let x = *input.add(offset + i);
            *output.add(offset + i) =
                crate::algorithm::special::scalar::erf_scalar(x as f64) as f32;
        }
    }
}

/// Vectorized erf for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn erf_f64(input: *const f64, output: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    let a1 = _mm256_set1_pd(erf::A1);
    let a2 = _mm256_set1_pd(erf::A2);
    let a3 = _mm256_set1_pd(erf::A3);
    let a4 = _mm256_set1_pd(erf::A4);
    let a5 = _mm256_set1_pd(erf::A5);
    let p = _mm256_set1_pd(erf::P);
    let one = _mm256_set1_pd(1.0);
    let sign_mask = _mm256_set1_pd(-0.0);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let x = _mm256_loadu_pd(input.add(offset));

        let sign = _mm256_and_pd(x, sign_mask);
        let ax = _mm256_andnot_pd(sign_mask, x);

        let t = _mm256_div_pd(one, _mm256_fmadd_pd(p, ax, one));

        let mut poly = a5;
        poly = _mm256_fmadd_pd(poly, t, a4);
        poly = _mm256_fmadd_pd(poly, t, a3);
        poly = _mm256_fmadd_pd(poly, t, a2);
        poly = _mm256_fmadd_pd(poly, t, a1);
        poly = _mm256_mul_pd(poly, t);

        let neg_x2 = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_mul_pd(ax, ax));
        let exp_term = exp_f64(neg_x2);

        let y = _mm256_fnmadd_pd(poly, exp_term, one);
        let result = _mm256_or_pd(y, sign);

        _mm256_storeu_pd(output.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        for i in 0..remainder {
            let x = *input.add(offset + i);
            *output.add(offset + i) = crate::algorithm::special::scalar::erf_scalar(x);
        }
    }
}

// ============================================================================
// Complementary Error Function (erfc)
// ============================================================================

/// Vectorized erfc for f32
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn erfc_f32(input: *const f32, output: *mut f32, len: usize) {
    // Compute erf first, then subtract from 1
    erf_f32(input, output, len);

    let chunks = len / F32_LANES;
    let one = _mm256_set1_ps(1.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let erf_val = _mm256_loadu_ps(output.add(offset));
        let result = _mm256_sub_ps(one, erf_val);
        _mm256_storeu_ps(output.add(offset), result);
    }

    let remainder = len % F32_LANES;
    if remainder > 0 {
        let offset = chunks * F32_LANES;
        for i in 0..remainder {
            let val = *output.add(offset + i);
            *output.add(offset + i) = 1.0 - val;
        }
    }
}

/// Vectorized erfc for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn erfc_f64(input: *const f64, output: *mut f64, len: usize) {
    erf_f64(input, output, len);

    let chunks = len / F64_LANES;
    let one = _mm256_set1_pd(1.0);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let erf_val = _mm256_loadu_pd(output.add(offset));
        let result = _mm256_sub_pd(one, erf_val);
        _mm256_storeu_pd(output.add(offset), result);
    }

    let remainder = len % F64_LANES;
    if remainder > 0 {
        let offset = chunks * F64_LANES;
        for i in 0..remainder {
            let val = *output.add(offset + i);
            *output.add(offset + i) = 1.0 - val;
        }
    }
}

// ============================================================================
// Bessel J0
// ============================================================================

/// Vectorized bessel_j0 for f32
///
/// Note: The f32 coefficients cannot precisely represent values like 57568490574.0
/// (f32 only has ~7 significant digits). For accuracy, we use scalar f64 computation.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn bessel_j0_f32(input: *const f32, output: *mut f32, len: usize) {
    // Use scalar f64 computation for accuracy - the coefficients don't fit in f32
    for i in 0..len {
        let x = *input.add(i);
        *output.add(i) = crate::algorithm::special::scalar::bessel_j0_scalar(x as f64) as f32;
    }
}

/// Vectorized bessel_j0 for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn bessel_j0_f64(input: *const f64, output: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    let threshold = _mm256_set1_pd(8.0);
    let sign_mask = _mm256_set1_pd(-0.0);

    let sp0 = _mm256_set1_pd(bessel_j0::SMALL_P[0]);
    let sp1 = _mm256_set1_pd(bessel_j0::SMALL_P[1]);
    let sp2 = _mm256_set1_pd(bessel_j0::SMALL_P[2]);
    let sp3 = _mm256_set1_pd(bessel_j0::SMALL_P[3]);
    let sp4 = _mm256_set1_pd(bessel_j0::SMALL_P[4]);
    let sp5 = _mm256_set1_pd(bessel_j0::SMALL_P[5]);

    let sq0 = _mm256_set1_pd(bessel_j0::SMALL_Q[0]);
    let sq1 = _mm256_set1_pd(bessel_j0::SMALL_Q[1]);
    let sq2 = _mm256_set1_pd(bessel_j0::SMALL_Q[2]);
    let sq3 = _mm256_set1_pd(bessel_j0::SMALL_Q[3]);
    let sq4 = _mm256_set1_pd(bessel_j0::SMALL_Q[4]);
    let sq5 = _mm256_set1_pd(bessel_j0::SMALL_Q[5]);

    let ap0 = _mm256_set1_pd(bessel_j0::ASYMP_P[0]);
    let ap1 = _mm256_set1_pd(bessel_j0::ASYMP_P[1]);
    let ap2 = _mm256_set1_pd(bessel_j0::ASYMP_P[2]);
    let ap3 = _mm256_set1_pd(bessel_j0::ASYMP_P[3]);
    let ap4 = _mm256_set1_pd(bessel_j0::ASYMP_P[4]);

    let aq0 = _mm256_set1_pd(bessel_j0::ASYMP_Q[0]);
    let aq1 = _mm256_set1_pd(bessel_j0::ASYMP_Q[1]);
    let aq2 = _mm256_set1_pd(bessel_j0::ASYMP_Q[2]);
    let aq3 = _mm256_set1_pd(bessel_j0::ASYMP_Q[3]);
    let aq4 = _mm256_set1_pd(bessel_j0::ASYMP_Q[4]);

    let eight = _mm256_set1_pd(8.0);
    let frac_pi_4 = _mm256_set1_pd(bessel_j0::FRAC_PI_4);
    let two_over_pi = _mm256_set1_pd(bessel_j0::TWO_OVER_PI);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let x = _mm256_loadu_pd(input.add(offset));
        let ax = _mm256_andnot_pd(sign_mask, x);

        // Small argument
        let y = _mm256_mul_pd(x, x);

        let mut num = sp5;
        num = _mm256_fmadd_pd(num, y, sp4);
        num = _mm256_fmadd_pd(num, y, sp3);
        num = _mm256_fmadd_pd(num, y, sp2);
        num = _mm256_fmadd_pd(num, y, sp1);
        num = _mm256_fmadd_pd(num, y, sp0);

        let mut den = sq5;
        den = _mm256_fmadd_pd(den, y, sq4);
        den = _mm256_fmadd_pd(den, y, sq3);
        den = _mm256_fmadd_pd(den, y, sq2);
        den = _mm256_fmadd_pd(den, y, sq1);
        den = _mm256_fmadd_pd(den, y, sq0);

        let small_result = _mm256_div_pd(num, den);

        // Asymptotic
        let z = _mm256_div_pd(eight, ax);
        let y_asymp = _mm256_mul_pd(z, z);
        let xx = _mm256_sub_pd(ax, frac_pi_4);

        let mut p0 = ap4;
        p0 = _mm256_fmadd_pd(p0, y_asymp, ap3);
        p0 = _mm256_fmadd_pd(p0, y_asymp, ap2);
        p0 = _mm256_fmadd_pd(p0, y_asymp, ap1);
        p0 = _mm256_fmadd_pd(p0, y_asymp, ap0);

        let mut q0 = aq4;
        q0 = _mm256_fmadd_pd(q0, y_asymp, aq3);
        q0 = _mm256_fmadd_pd(q0, y_asymp, aq2);
        q0 = _mm256_fmadd_pd(q0, y_asymp, aq1);
        q0 = _mm256_fmadd_pd(q0, y_asymp, aq0);
        q0 = _mm256_mul_pd(q0, z);

        let scale = _mm256_sqrt_pd(_mm256_div_pd(two_over_pi, ax));

        let mut cos_vals = [0.0f64; F64_LANES];
        let mut sin_vals = [0.0f64; F64_LANES];
        let mut xx_arr = [0.0f64; F64_LANES];
        _mm256_storeu_pd(xx_arr.as_mut_ptr(), xx);
        for j in 0..F64_LANES {
            cos_vals[j] = xx_arr[j].cos();
            sin_vals[j] = xx_arr[j].sin();
        }
        let cos_xx = _mm256_loadu_pd(cos_vals.as_ptr());
        let sin_xx = _mm256_loadu_pd(sin_vals.as_ptr());

        let asymp_result = _mm256_mul_pd(
            scale,
            _mm256_fnmadd_pd(sin_xx, q0, _mm256_mul_pd(cos_xx, p0)),
        );

        let mask = _mm256_cmp_pd::<_CMP_LT_OQ>(ax, threshold);
        let result = _mm256_blendv_pd(asymp_result, small_result, mask);

        _mm256_storeu_pd(output.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        for i in 0..remainder {
            let x = *input.add(offset + i);
            *output.add(offset + i) = crate::algorithm::special::scalar::bessel_j0_scalar(x);
        }
    }
}

// ============================================================================
// Bessel J1
// ============================================================================

/// Vectorized bessel_j1 for f32
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn bessel_j1_f32(input: *const f32, output: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    // Use scalar fallback for simplicity - J1 has sign handling
    // Full SIMD implementation can be added later
    for i in 0..chunks {
        let offset = i * F32_LANES;
        for j in 0..F32_LANES {
            let x = *input.add(offset + j);
            *output.add(offset + j) =
                crate::algorithm::special::scalar::bessel_j1_scalar(x as f64) as f32;
        }
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        for i in 0..remainder {
            let x = *input.add(offset + i);
            *output.add(offset + i) =
                crate::algorithm::special::scalar::bessel_j1_scalar(x as f64) as f32;
        }
    }
}

/// Vectorized bessel_j1 for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn bessel_j1_f64(input: *const f64, output: *mut f64, len: usize) {
    for i in 0..len {
        let x = *input.add(i);
        *output.add(i) = crate::algorithm::special::scalar::bessel_j1_scalar(x);
    }
}

// ============================================================================
// Modified Bessel I0
// ============================================================================

/// Vectorized bessel_i0 for f32
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn bessel_i0_f32(input: *const f32, output: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    let sign_mask = _mm256_set1_ps(-0.0);
    let threshold = _mm256_set1_ps(bessel_i0::THRESHOLD_F32);
    let one = _mm256_set1_ps(1.0);
    let _four = _mm256_set1_ps(4.0); // Reserved for potential future use
    let two_pi = _mm256_set1_ps(2.0 * std::f32::consts::PI);

    // Asymptotic coefficients
    let ai0 = _mm256_set1_ps(bessel_i0::ASYMP_F32[0]);
    let ai1 = _mm256_set1_ps(bessel_i0::ASYMP_F32[1]);
    let ai2 = _mm256_set1_ps(bessel_i0::ASYMP_F32[2]);
    let ai3 = _mm256_set1_ps(bessel_i0::ASYMP_F32[3]);
    let ai4 = _mm256_set1_ps(bessel_i0::ASYMP_F32[4]);
    let ai5 = _mm256_set1_ps(bessel_i0::ASYMP_F32[5]);
    let ai6 = _mm256_set1_ps(bessel_i0::ASYMP_F32[6]);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let x = _mm256_loadu_ps(input.add(offset));
        let ax = _mm256_andnot_ps(sign_mask, x);

        // Small argument: power series
        // I0(x) = sum_{k=0}^N (x/2)^{2k} / (k!)^2
        let z = _mm256_mul_ps(ax, ax);
        let mut sum = one;
        let mut term = one;

        // Unroll a few iterations of the power series
        for k in 1..15 {
            let k_f = k as f32;
            let factor = _mm256_div_ps(z, _mm256_set1_ps(4.0 * k_f * k_f));
            term = _mm256_mul_ps(term, factor);
            sum = _mm256_add_ps(sum, term);
        }

        let small_result = sum;

        // Large argument: asymptotic expansion
        let z_inv = _mm256_div_ps(one, ax);

        let mut poly = ai6;
        poly = _mm256_fmadd_ps(poly, z_inv, ai5);
        poly = _mm256_fmadd_ps(poly, z_inv, ai4);
        poly = _mm256_fmadd_ps(poly, z_inv, ai3);
        poly = _mm256_fmadd_ps(poly, z_inv, ai2);
        poly = _mm256_fmadd_ps(poly, z_inv, ai1);
        poly = _mm256_fmadd_ps(poly, z_inv, ai0);

        // exp(ax) / sqrt(2*pi*ax) * poly
        let exp_ax = exp_f32(ax);
        let scale = _mm256_div_ps(one, _mm256_sqrt_ps(_mm256_mul_ps(two_pi, ax)));
        let asymp_result = _mm256_mul_ps(_mm256_mul_ps(exp_ax, scale), poly);

        // Blend based on threshold
        let mask = _mm256_cmp_ps::<_CMP_LE_OQ>(ax, threshold);
        let result = _mm256_blendv_ps(asymp_result, small_result, mask);

        _mm256_storeu_ps(output.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        for i in 0..remainder {
            let x = *input.add(offset + i);
            *output.add(offset + i) =
                crate::algorithm::special::scalar::bessel_i0_scalar(x as f64) as f32;
        }
    }
}

/// Vectorized bessel_i0 for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn bessel_i0_f64(input: *const f64, output: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    let sign_mask = _mm256_set1_pd(-0.0);
    let threshold = _mm256_set1_pd(bessel_i0::THRESHOLD);
    let one = _mm256_set1_pd(1.0);
    let two_pi = _mm256_set1_pd(2.0 * std::f64::consts::PI);

    let ai0 = _mm256_set1_pd(bessel_i0::ASYMP[0]);
    let ai1 = _mm256_set1_pd(bessel_i0::ASYMP[1]);
    let ai2 = _mm256_set1_pd(bessel_i0::ASYMP[2]);
    let ai3 = _mm256_set1_pd(bessel_i0::ASYMP[3]);
    let ai4 = _mm256_set1_pd(bessel_i0::ASYMP[4]);
    let ai5 = _mm256_set1_pd(bessel_i0::ASYMP[5]);
    let ai6 = _mm256_set1_pd(bessel_i0::ASYMP[6]);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let x = _mm256_loadu_pd(input.add(offset));
        let ax = _mm256_andnot_pd(sign_mask, x);

        // Small argument
        let z = _mm256_mul_pd(ax, ax);
        let mut sum = one;
        let mut term = one;

        for k in 1..20 {
            let k_f = k as f64;
            let factor = _mm256_div_pd(z, _mm256_set1_pd(4.0 * k_f * k_f));
            term = _mm256_mul_pd(term, factor);
            sum = _mm256_add_pd(sum, term);
        }

        let small_result = sum;

        // Asymptotic
        let z_inv = _mm256_div_pd(one, ax);

        let mut poly = ai6;
        poly = _mm256_fmadd_pd(poly, z_inv, ai5);
        poly = _mm256_fmadd_pd(poly, z_inv, ai4);
        poly = _mm256_fmadd_pd(poly, z_inv, ai3);
        poly = _mm256_fmadd_pd(poly, z_inv, ai2);
        poly = _mm256_fmadd_pd(poly, z_inv, ai1);
        poly = _mm256_fmadd_pd(poly, z_inv, ai0);

        let exp_ax = exp_f64(ax);
        let scale = _mm256_div_pd(one, _mm256_sqrt_pd(_mm256_mul_pd(two_pi, ax)));
        let asymp_result = _mm256_mul_pd(_mm256_mul_pd(exp_ax, scale), poly);

        let mask = _mm256_cmp_pd::<_CMP_LE_OQ>(ax, threshold);
        let result = _mm256_blendv_pd(asymp_result, small_result, mask);

        _mm256_storeu_pd(output.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        for i in 0..remainder {
            let x = *input.add(offset + i);
            *output.add(offset + i) = crate::algorithm::special::scalar::bessel_i0_scalar(x);
        }
    }
}

// ============================================================================
// Modified Bessel I1
// ============================================================================

/// Vectorized bessel_i1 for f32
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn bessel_i1_f32(input: *const f32, output: *mut f32, len: usize) {
    // I1 is an odd function, use scalar for now
    for i in 0..len {
        let x = *input.add(i);
        *output.add(i) = crate::algorithm::special::scalar::bessel_i1_scalar(x as f64) as f32;
    }
}

/// Vectorized bessel_i1 for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn bessel_i1_f64(input: *const f64, output: *mut f64, len: usize) {
    for i in 0..len {
        let x = *input.add(i);
        *output.add(i) = crate::algorithm::special::scalar::bessel_i1_scalar(x);
    }
}
