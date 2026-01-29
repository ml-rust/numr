//! Scalar implementations of special mathematical functions
//!
//! This module provides high-precision scalar implementations that are
//! applied element-wise by the helper functions in `helpers.rs`.
//!
//! # Algorithms
//!
//! - **erf/erfc**: Abramowitz & Stegun approximation 7.1.26 (~1e-7 accuracy)
//! - **erfinv**: Rational approximation + Newton-Raphson refinement
//! - **gamma/lgamma**: Lanczos approximation (g=7, n=9)
//! - **digamma**: Asymptotic expansion with recurrence relation
//! - **beta**: Computed via lgamma for numerical stability
//! - **betainc**: Continued fraction (Lentz's method)
//! - **gammainc/gammaincc**: Series expansion + continued fraction

use crate::algorithm::special::{LANCZOS_COEFFICIENTS, LANCZOS_G, LN_SQRT_2PI, TWO_OVER_SQRT_PI};

// ============================================================================
// Numerical Constants for Convergence
// ============================================================================

/// Maximum iterations for continued fraction / series expansions
const MAX_ITER: usize = 200;

/// Convergence threshold for double precision
const EPSILON_F64: f64 = 1e-15;

/// Small value to prevent division by zero in continued fractions
const TINY: f64 = 1e-30;

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

// ============================================================================
// Gamma Function (Lanczos Approximation)
// ============================================================================

/// Compute gamma(x) using Lanczos approximation.
pub fn gamma_scalar(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        // Poles at non-positive integers
        return f64::NAN;
    }

    if x < 0.5 {
        // Reflection formula: Gamma(x) = pi / (sin(pi*x) * Gamma(1-x))
        let sin_pi_x = (std::f64::consts::PI * x).sin();
        if sin_pi_x.abs() < 1e-15 {
            return f64::NAN;
        }
        std::f64::consts::PI / (sin_pi_x * gamma_scalar(1.0 - x))
    } else {
        lgamma_positive(x).exp()
    }
}

/// Compute lgamma(x) = ln(|Gamma(x)|).
pub fn lgamma_scalar(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY;
    }

    if x < 0.5 {
        // Reflection: lgamma(x) = ln(pi) - ln|sin(pi*x)| - lgamma(1-x)
        let sin_pi_x = (std::f64::consts::PI * x).sin().abs();
        std::f64::consts::PI.ln() - sin_pi_x.ln() - lgamma_scalar(1.0 - x)
    } else {
        lgamma_positive(x)
    }
}

/// Compute lgamma for x >= 0.5 using Lanczos approximation.
fn lgamma_positive(x: f64) -> f64 {
    let x = x - 1.0;

    let mut ag = LANCZOS_COEFFICIENTS[0];
    for i in 1..LANCZOS_COEFFICIENTS.len() {
        ag += LANCZOS_COEFFICIENTS[i] / (x + i as f64);
    }

    let t = x + LANCZOS_G + 0.5;
    LN_SQRT_2PI + (x + 0.5) * t.ln() - t + ag.ln()
}

// ============================================================================
// Digamma Function
// ============================================================================

/// Compute digamma(x) = Gamma'(x)/Gamma(x).
pub fn digamma_scalar(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::NAN;
    }

    let mut x = x;
    let mut result = 0.0;

    // Use recurrence for x < 6
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic expansion for large x
    result += x.ln() - 0.5 / x;
    let x2 = 1.0 / (x * x);

    // Bernoulli numbers
    result -= x2 * (1.0 / 12.0 - x2 * (1.0 / 120.0 - x2 * (1.0 / 252.0 - x2 / 240.0)));

    result
}

// ============================================================================
// Beta Function
// ============================================================================

/// Compute beta(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b).
pub fn beta_scalar(a: f64, b: f64) -> f64 {
    // Use lgamma for numerical stability
    (lgamma_scalar(a) + lgamma_scalar(b) - lgamma_scalar(a + b)).exp()
}

// ============================================================================
// Regularized Incomplete Beta Function
// ============================================================================

/// Compute the regularized incomplete beta function I_x(a, b).
///
/// # Input Bounds
///
/// - `x` is clamped to [0, 1]: values <= 0 return 0, values >= 1 return 1
/// - `a` and `b` should be positive for meaningful results
pub fn betainc_scalar(a: f64, b: f64, x: f64) -> f64 {
    // Boundary conditions
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry relation for better convergence
    // I_x(a,b) = 1 - I_{1-x}(b,a)
    if x > (a + 1.0) / (a + b + 2.0) {
        1.0 - betainc_scalar(b, a, 1.0 - x)
    } else {
        // Compute using continued fraction
        let bt = if x == 0.0 || x == 1.0 {
            0.0
        } else {
            (lgamma_scalar(a + b) - lgamma_scalar(a) - lgamma_scalar(b)
                + a * x.ln()
                + b * (1.0 - x).ln())
            .exp()
        };

        bt * betainc_cf(a, b, x) / a
    }
}

/// Continued fraction for incomplete beta using Lentz's method.
fn betainc_cf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;

        // Even step
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < EPSILON_F64 {
            return h;
        }
    }

    h
}

// ============================================================================
// Regularized Incomplete Gamma Functions
// ============================================================================

/// Compute the lower regularized incomplete gamma P(a, x).
pub fn gammainc_scalar(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }

    if x < a + 1.0 {
        // Use series representation
        gammainc_series(a, x)
    } else {
        // Use continued fraction
        1.0 - gammaincc_cf(a, x)
    }
}

/// Compute the upper regularized incomplete gamma Q(a, x) = 1 - P(a, x).
pub fn gammaincc_scalar(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }

    if x < a + 1.0 {
        1.0 - gammainc_series(a, x)
    } else {
        gammaincc_cf(a, x)
    }
}

/// Series representation for P(a, x).
fn gammainc_series(a: f64, x: f64) -> f64 {
    let gln = lgamma_scalar(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;

    for _ in 0..MAX_ITER {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPSILON_F64 {
            break;
        }
    }

    sum * (-x + a * x.ln() - gln).exp()
}

/// Continued fraction for Q(a, x).
fn gammaincc_cf(a: f64, x: f64) -> f64 {
    let gln = lgamma_scalar(a);

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..=MAX_ITER {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPSILON_F64 {
            break;
        }
    }

    (-x + a * x.ln() - gln).exp() * h
}
