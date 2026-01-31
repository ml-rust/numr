//! Inverse incomplete gamma and beta function implementations
//!
//! These functions compute the inverse of regularized incomplete functions,
//! essential for quantile functions in probability distributions.

use super::gamma_functions::{
    EPSILON_F64, MAX_ITER, TINY, betainc_scalar, gammainc_scalar, lgamma_scalar,
};

// ============================================================================
// Inverse Incomplete Gamma Function
// ============================================================================

/// Compute the inverse of the lower regularized incomplete gamma function.
///
/// Returns x such that P(a, x) = p.
///
/// Uses Halley's method with an initial guess from asymptotic expansions.
///
/// # Algorithm
///
/// 1. Initial guess via normal approximation (a > 1) or power approximation (a <= 1)
/// 2. Halley's method iteration with cubic convergence
/// 3. Underflow/overflow protection for extreme values
///
/// # Accuracy
///
/// Typically achieves 1e-12 relative accuracy for well-conditioned inputs.
pub fn gammaincinv_scalar(a: f64, p: f64) -> f64 {
    // Handle boundary cases
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if a <= 0.0 {
        return f64::NAN;
    }

    // Initial guess using asymptotic approximations
    let gln = lgamma_scalar(a);

    let mut x = if a > 1.0 {
        // For a > 1, use normal approximation to gamma
        // P(a, x) ≈ Φ((x - a) / sqrt(a))
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let x0 = t - (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481));
        let x0 = if p < 0.5 { -x0 } else { x0 };
        // Map back to gamma scale
        (a + x0 * a.sqrt()).max(0.001)
    } else {
        // For small a, use power approximation
        // P(a, x) ≈ x^a / (a * Gamma(a))
        let t = 1.0 - a * (0.253 + a * 0.12);
        let pp = p;
        if pp < t {
            (pp / t).powf(1.0 / a) * a
        } else {
            1.0 - (1.0 - pp).ln() + (1.0 - a) * (1.0 - pp).ln().abs().ln()
        }
    };

    // Halley's method iteration
    for _ in 0..MAX_ITER {
        if x <= 0.0 {
            return 0.0;
        }

        let err = gammainc_scalar(a, x) - p;
        if err.abs() < EPSILON_F64 {
            break;
        }

        // f'(x) = x^(a-1) * exp(-x) / Gamma(a)
        let t = (a - 1.0) * x.ln() - x - gln;
        if t < -700.0 {
            // Underflow protection
            break;
        }
        let dfdx = t.exp();

        if dfdx.abs() < TINY {
            break;
        }

        // f''(x) / f'(x) = (a - 1) / x - 1
        let d2ratio = (a - 1.0) / x - 1.0;

        // Halley's method: x_new = x - f / (f' - f * f'' / (2 * f'))
        let dx = err / (dfdx * (1.0 - 0.5 * err * d2ratio / dfdx));
        x -= dx;

        if x <= 0.0 {
            x = 0.5 * (x + dx); // Backtrack if negative
        }

        if dx.abs() < x * EPSILON_F64 {
            break;
        }
    }

    x.max(0.0)
}

// ============================================================================
// Inverse Incomplete Beta Function
// ============================================================================

/// Compute the inverse of the regularized incomplete beta function.
///
/// Returns x such that I_x(a, b) = p.
///
/// Uses Newton's method with bisection fallback for robustness.
///
/// # Algorithm
///
/// 1. Initial guess via normal approximation (for a,b >= 1) or mean-based heuristic
/// 2. Bisection refinement if initial guess is poor
/// 3. Newton's method with damped steps for convergence
///
/// # Accuracy
///
/// Typically achieves 1e-12 relative accuracy for well-conditioned inputs.
pub fn betaincinv_scalar(a: f64, b: f64, p: f64) -> f64 {
    // Handle boundary cases
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    if a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }

    // Use symmetry for p > 0.5
    if p > 0.5 {
        return 1.0 - betaincinv_scalar(b, a, 1.0 - p);
    }

    let lnbeta = lgamma_scalar(a) + lgamma_scalar(b) - lgamma_scalar(a + b);

    // Initial guess: use mean of Beta distribution as starting point
    // Mean = a / (a + b), adjusted toward 0 or 1 based on p
    let mean = a / (a + b);
    let mut x = if p < 0.5 {
        // For small p, start below the mean
        mean * (2.0 * p).powf(1.0 / a.max(1.0))
    } else {
        mean
    };

    // For normal approximation when both a, b >= 1
    if a >= 1.0 && b >= 1.0 {
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let s = t - (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481));
        let s = if p < 0.5 { -s } else { s };

        let lam = (s * s - 3.0) / 6.0;
        let h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        let w = s * (h + lam).sqrt() / h
            - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (lam + 5.0 / 6.0 - 2.0 / (3.0 * h));

        x = a / (a + b * (2.0 * w).exp());
    }

    // Clamp initial guess
    x = x.clamp(1e-10, 1.0 - 1e-10);

    // Check initial guess quality - use bisection if way off
    let init_err = (betainc_scalar(a, b, x) - p).abs();
    if init_err > 0.3 {
        // Bisection to get a better starting point
        let mut lo = 0.0;
        let mut hi = 1.0;
        for _ in 0..20 {
            let mid = (lo + hi) / 2.0;
            if betainc_scalar(a, b, mid) < p {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        x = (lo + hi) / 2.0;
    }

    // Newton's method iteration (simpler than Halley, more robust)
    let afac = -lnbeta;
    for _ in 0..MAX_ITER {
        if x <= 0.0 || x >= 1.0 {
            break;
        }

        let err = betainc_scalar(a, b, x) - p;
        if err.abs() < EPSILON_F64 {
            break;
        }

        // f'(x) = x^(a-1) * (1-x)^(b-1) / Beta(a,b)
        let t = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() + afac;
        if t < -700.0 {
            break;
        }
        let dfdx = t.exp();

        if dfdx.abs() < TINY {
            break;
        }

        // Newton step with damping for robustness
        let mut dx = err / dfdx;

        // Limit step size to prevent overshooting
        if dx > x / 2.0 {
            dx = x / 2.0;
        }
        if dx < -(1.0 - x) / 2.0 {
            dx = -(1.0 - x) / 2.0;
        }

        x -= dx;

        // Keep x in (0, 1)
        x = x.clamp(1e-15, 1.0 - 1e-15);

        if dx.abs() < x * EPSILON_F64 {
            break;
        }
    }

    x.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected {} to be close to {}, diff = {}",
            a,
            b,
            (a - b).abs()
        );
    }

    #[test]
    fn test_gammaincinv_roundtrip() {
        // Test that gammaincinv is the inverse of gammainc
        let test_cases = [(1.0, 0.5), (2.0, 0.3), (5.0, 0.7), (10.0, 0.9), (0.5, 0.4)];

        for (a, p) in test_cases {
            let x = gammaincinv_scalar(a, p);
            let back = gammainc_scalar(a, x);
            assert_close(back, p, TOL);
        }
    }

    #[test]
    fn test_gammaincinv_bounds() {
        // gammaincinv(a, 0) = 0
        assert_close(gammaincinv_scalar(2.0, 0.0), 0.0, 1e-10);
        // gammaincinv(a, 1) = inf
        assert!(gammaincinv_scalar(2.0, 1.0).is_infinite());
    }

    #[test]
    fn test_betaincinv_roundtrip() {
        // Test that betaincinv is the inverse of betainc
        let test_cases = [
            (1.0, 1.0, 0.5),
            (2.0, 2.0, 0.3),
            (2.0, 5.0, 0.4),
            (5.0, 2.0, 0.6),
            (5.0, 0.5, 0.2),
            (0.5, 5.0, 0.8),
            (10.0, 10.0, 0.5),
        ];

        for (a, b, p) in test_cases {
            let x = betaincinv_scalar(a, b, p);
            let back = betainc_scalar(a, b, x);
            assert_close(back, p, TOL);
        }
    }

    #[test]
    fn test_betaincinv_bounds() {
        // betaincinv(a, b, 0) = 0
        assert_close(betaincinv_scalar(2.0, 3.0, 0.0), 0.0, 1e-10);
        // betaincinv(a, b, 1) = 1
        assert_close(betaincinv_scalar(2.0, 3.0, 1.0), 1.0, 1e-10);
    }
}
