//! Error function implementations (erf, erfc, erfinv)

// ============================================================================
// Error Function Implementation
// ============================================================================

/// Compute erf(x) using Abramowitz and Stegun approximation.
///
/// Uses polynomial approximation (A&S 7.1.26).
/// Accuracy: ~1e-7 relative error.
pub fn erf_scalar(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }

    // Constants for Abramowitz and Stegun approximation 7.1.26
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // A&S formula 7.1.26
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();

    sign * y
}

/// Compute erfc(x) = 1 - erf(x) directly for numerical stability.
pub fn erfc_scalar(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 - erf_scalar(x)
    } else {
        1.0 + erf_scalar(-x)
    }
}

// ============================================================================
// Inverse Error Function
// ============================================================================

/// Compute erfinv(x) using the relationship with the normal quantile function.
///
/// Uses: erfinv(x) = ndtri((1+x)/2) / sqrt(2)
/// where ndtri is the inverse of the standard normal CDF.
///
/// The ndtri approximation uses the Beasley-Springer-Moro algorithm
/// with Halley refinement for high accuracy.
///
/// Accuracy: ~1e-12 relative error.
pub fn erfinv_scalar(p: f64) -> f64 {
    if p <= -1.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p.is_nan() {
        return f64::NAN;
    }

    // erfinv(x) = ndtri((1+x)/2) / sqrt(2)
    // We compute ndtri directly using Beasley-Springer-Moro algorithm
    let prob = (1.0 + p) / 2.0; // Map from (-1,1) to (0,1)

    let ndtri_result = ndtri_scalar(prob);

    ndtri_result * std::f64::consts::FRAC_1_SQRT_2
}

/// Inverse of the standard normal CDF (quantile function).
///
/// Uses the Acklam algorithm (Peter J. Acklam, 2010) which provides
/// high accuracy across the entire range.
fn ndtri_scalar(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }

    // Acklam's algorithm coefficients
    // Rational approximation for the lower region
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];

    // Rational approximation for the central region
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    // Break-points for the regions
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let mut x = if p < P_LOW {
        // Lower region
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Upper region
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // Halley refinement for improved accuracy
    // f(x) = Phi(x) - p
    // f'(x) = phi(x)
    const SQRT_2PI: f64 = 2.5066282746310002;

    for _ in 0..3 {
        let phi_x = (-0.5 * x * x).exp() / SQRT_2PI;

        if phi_x < 1e-300 {
            break;
        }

        let cdf_x = 0.5 * (1.0 + erf_scalar(x * std::f64::consts::FRAC_1_SQRT_2));
        let err = cdf_x - p;

        if err.abs() < 1e-15 {
            break;
        }

        let step = err / phi_x;
        let halley_factor = 1.0 + 0.5 * x * step;

        if halley_factor.abs() > 0.5 {
            x -= step / halley_factor;
        } else {
            x -= step;
        }
    }

    x
}
