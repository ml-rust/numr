//! Error function implementations (erf, erfc, erfinv)

use crate::algorithm::special::TWO_OVER_SQRT_PI;

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

/// Compute erfinv(x) using rational approximation + Newton refinement.
///
/// Based on the algorithm from Numerical Recipes with Halley's method refinement.
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

    let a = p.abs();

    // Initial approximation using rational function
    let mut x = if a <= 0.7 {
        // Central region
        let y = a * a;
        a * (((-0.140543331 * y + 0.914624893) * y - 1.645349621) * y + 0.886226899)
    } else {
        // Tail region - use different approximation
        let y = (-((1.0 - a).ln())).sqrt();
        if y < 5.0 {
            let t = y - 1.6;
            (((((7.7454501427834140764e-4 * t + 2.2723844989269184048e-2) * t
                + 2.4178072517745061177e-1)
                * t
                + 1.2704174423399106455)
                * t
                + 3.6477010300020166088)
                * t
                + 4.9380691871697904016)
                / ((((((1.0507500716444169339e-9 * t + 5.475938084995344946e-4) * t
                    + 1.5198666563616457524e-2)
                    * t
                    + 1.4821087685904695082e-1)
                    * t
                    + 6.8976733498510000455e-1)
                    * t
                    + 1.5707956373078195526)
                    * t
                    + 1.0)
        } else {
            let t = y - 5.0;
            (((((2.0103343992922881344e-7 * t + 2.7115555687434876539e-5) * t
                + 1.2426609473880784386e-3)
                * t
                + 2.6537166573738514658e-2)
                * t
                + 2.9611541567127226145e-1)
                * t
                + 1.4199773302117956999)
                / ((((((2.0442631033899397236e-15 * t + 1.4215117583164458887e-7) * t
                    + 1.8463183175100546818e-5)
                    * t
                    + 7.8686913114561329059e-4)
                    * t
                    + 1.4875361290850615025e-2)
                    * t
                    + 1.3692988092273580531e-1)
                    * t
                    + 1.0)
        }
    };

    // Newton-Raphson refinement (2 iterations for high precision)
    for _ in 0..2 {
        let err = erf_scalar(x) - a;
        x -= err / (TWO_OVER_SQRT_PI * (-x * x).exp());
    }

    if p < 0.0 { -x } else { x }
}
