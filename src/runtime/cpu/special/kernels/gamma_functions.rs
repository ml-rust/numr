//! Gamma, beta, and incomplete gamma/beta function implementations

use crate::algorithm::special::{LANCZOS_COEFFICIENTS, LANCZOS_G, LN_SQRT_2PI};

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
