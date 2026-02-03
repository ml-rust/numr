//! Fresnel integrals S(x) and C(x)
//!
//! Uses power series for small arguments and auxiliary functions for large arguments.
//!
//! # References
//! - DLMF 7.2: Fresnel Integrals
//! - Abramowitz & Stegun 7.3

use std::f64::consts::PI;

/// Maximum terms in power series
const MAX_SERIES_TERMS: usize = 100;

/// Convergence threshold
const EPSILON: f64 = 1e-15;

/// Threshold for switching to auxiliary function expansion
const AUXILIARY_THRESHOLD: f64 = 4.0;

/// Fresnel sine integral S(x).
///
/// ```text
/// S(x) = ∫₀ˣ sin(π t²/2) dt
/// ```
///
/// # Properties
/// - S(0) = 0
/// - S(∞) = 0.5
/// - S(-x) = -S(x) (odd function)
///
/// # Algorithm
/// - Power series for |x| < 4
/// - Auxiliary functions f(x), g(x) for |x| ≥ 4
pub fn fresnel_s_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();

    if ax < AUXILIARY_THRESHOLD {
        sign * fresnel_s_series(ax)
    } else {
        sign * fresnel_s_auxiliary(ax)
    }
}

/// Fresnel cosine integral C(x).
///
/// ```text
/// C(x) = ∫₀ˣ cos(π t²/2) dt
/// ```
///
/// # Properties
/// - C(0) = 0
/// - C(∞) = 0.5
/// - C(-x) = -C(x) (odd function)
///
/// # Algorithm
/// - Power series for |x| < 4
/// - Auxiliary functions f(x), g(x) for |x| ≥ 4
pub fn fresnel_c_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();

    if ax < AUXILIARY_THRESHOLD {
        sign * fresnel_c_series(ax)
    } else {
        sign * fresnel_c_auxiliary(ax)
    }
}

/// Power series for S(x)
///
/// S(x) = Σ_{n=0}^∞ (-1)^n (π/2)^(2n+1) x^(4n+3) / ((2n+1)! (4n+3))
fn fresnel_s_series(x: f64) -> f64 {
    let pi_2 = PI / 2.0;
    let x2 = x * x;
    let x4 = x2 * x2;

    let mut sum = x * x2 * pi_2 / 3.0; // First term: (π/2) x³ / 3
    let mut term = sum;
    let mut sign = -1.0;

    for n in 1..MAX_SERIES_TERMS {
        let n2 = 2 * n;
        let n4 = 4 * n;
        // term *= -1 * (π/2)² * x⁴ * (4n-1)! / (4n+3)! * (2n-1)! / (2n+1)!
        // Simplified: term *= -(π/2)² x⁴ / ((2n)(2n+1) * (4n)(4n+1)(4n+2)(4n+3))
        term *= pi_2 * pi_2 * x4
            / (((n2 * (n2 + 1)) as f64) * ((n4 * (n4 + 1) * (n4 + 2) * (n4 + 3)) as f64));
        term *= (((n4 - 1) * (n4 - 2) * (n4 - 3)) as f64) / (((n2 - 1) * (n2)) as f64); // Correction factor

        // Actually use recurrence relation
        // Better: compute directly
        sum += sign * term;
        sign = -sign;

        if term.abs() < EPSILON * sum.abs() {
            break;
        }
    }

    // Use direct series computation
    fresnel_s_series_direct(x)
}

/// Direct series computation for S(x)
fn fresnel_s_series_direct(x: f64) -> f64 {
    let pi_2 = PI / 2.0;
    let t = pi_2 * x * x;

    // S(x) = x * Σ (-1)^n t^(2n+1) / ((2n+1)! (4n+3))
    // Better form: S(x) = (π/2)^(1/2) * x³ * Σ
    let mut sum = 0.0;
    let mut term = x * t / 3.0; // First term
    sum += term;

    for n in 1..MAX_SERIES_TERMS {
        // Multiply by -t² / ((2n)(2n+1)) * (4n-1) / (4n+3)
        let n2 = 2 * n;
        let n4 = 4 * n;
        term *= -t * t / ((n2 * (n2 + 1)) as f64);
        term *= ((n4 - 1) as f64) / ((n4 + 3) as f64);
        sum += term;

        if term.abs() < EPSILON * sum.abs() {
            break;
        }
    }

    sum
}

/// Power series for C(x)
fn fresnel_c_series(x: f64) -> f64 {
    let pi_2 = PI / 2.0;
    let t = pi_2 * x * x;

    // C(x) = x * Σ (-1)^n t^(2n) / ((2n)! (4n+1))
    let mut sum = 0.0;
    let mut term = x; // First term: x / 1
    sum += term;

    for n in 1..MAX_SERIES_TERMS {
        // Multiply by -t² / ((2n-1)(2n)) * (4n-3) / (4n+1)
        let n2 = 2 * n;
        let n4 = 4 * n;
        term *= -t * t / (((n2 - 1) * n2) as f64);
        term *= ((n4 - 3) as f64) / ((n4 + 1) as f64);
        sum += term;

        if term.abs() < EPSILON * sum.abs() {
            break;
        }
    }

    sum
}

/// Auxiliary function expansion for S(x) with x ≥ 4
///
/// S(x) = 0.5 - f(x)cos(πx²/2) - g(x)sin(πx²/2)
fn fresnel_s_auxiliary(x: f64) -> f64 {
    let (f, g) = auxiliary_fg(x);
    let t = PI * x * x / 2.0;
    0.5 - f * t.cos() - g * t.sin()
}

/// Auxiliary function expansion for C(x) with x ≥ 4
///
/// C(x) = 0.5 + f(x)sin(πx²/2) - g(x)cos(πx²/2)
fn fresnel_c_auxiliary(x: f64) -> f64 {
    let (f, g) = auxiliary_fg(x);
    let t = PI * x * x / 2.0;
    0.5 + f * t.sin() - g * t.cos()
}

/// Compute auxiliary functions f(x) and g(x) for large x
///
/// f(x) ~ 1/(πx) * [1 - 1·3/(πx²)² + ...]
/// g(x) ~ 1/(π²x³) * [1 - 1·3·5·7/(πx²)² + ...]
fn auxiliary_fg(x: f64) -> (f64, f64) {
    let pix = PI * x;
    let pix2 = pix * pix;

    // Rational approximation coefficients for f(x) and g(x)
    // Using asymptotic series
    let z = 1.0 / pix2;

    // f(x) = 1/(πx) * P(z) / Q(z)
    // g(x) = 1/(π²x³) * R(z) / S(z)

    // Simplified asymptotic:
    // f ~ 1/(πx) * (1 - 3z²/4 + ...)
    // g ~ 1/(π²x³) * (1 - 15z²/4 + ...)

    // Use Chebyshev-like rational approximation
    let f_num = 1.0 + z * (-1.0 / 4.0 + z * 3.0 / 64.0);
    let f_den = 1.0 + z * (3.0 / 4.0 + z * 15.0 / 64.0);
    let f = f_num / (pix * f_den);

    let g_num = 1.0 + z * (-5.0 / 4.0 + z * 35.0 / 64.0);
    let g_den = 1.0 + z * (7.0 / 4.0 + z * 63.0 / 64.0);
    let g = g_num / (pix2 * x * g_den);

    (f, g)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-7;

    fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
        let diff = (a - b).abs();
        assert!(
            diff < tol || (a.is_nan() && b.is_nan()),
            "{}: expected {}, got {}, diff {}",
            msg,
            b,
            a,
            diff
        );
    }

    #[test]
    fn test_fresnel_s_special_values() {
        // S(0) = 0
        assert_close(fresnel_s_scalar(0.0), 0.0, TOL, "S(0)");

        // S(1) ≈ 0.4382591473903548
        assert_close(fresnel_s_scalar(1.0), 0.4382591473903548, TOL, "S(1)");

        // S(2) ≈ 0.3434156783636982
        assert_close(fresnel_s_scalar(2.0), 0.3434156783636982, TOL, "S(2)");

        // S approaches 0.5 for large x
        assert_close(fresnel_s_scalar(10.0), 0.5, 0.05, "S(10)");
        assert_close(fresnel_s_scalar(100.0), 0.5, 0.01, "S(100)");
    }

    #[test]
    fn test_fresnel_c_special_values() {
        // C(0) = 0
        assert_close(fresnel_c_scalar(0.0), 0.0, TOL, "C(0)");

        // C(1) ≈ 0.7798934003768228
        assert_close(fresnel_c_scalar(1.0), 0.7798934003768228, TOL, "C(1)");

        // C(2) ≈ 0.4882534060753408
        assert_close(fresnel_c_scalar(2.0), 0.4882534060753408, TOL, "C(2)");

        // C approaches 0.5 for large x
        assert_close(fresnel_c_scalar(10.0), 0.5, 0.05, "C(10)");
        assert_close(fresnel_c_scalar(100.0), 0.5, 0.01, "C(100)");
    }

    #[test]
    fn test_fresnel_odd_symmetry() {
        // S(-x) = -S(x)
        for &x in &[0.5, 1.0, 2.0, 5.0] {
            assert_close(
                fresnel_s_scalar(-x),
                -fresnel_s_scalar(x),
                TOL,
                &format!("S(-{}) = -S({})", x, x),
            );
            assert_close(
                fresnel_c_scalar(-x),
                -fresnel_c_scalar(x),
                TOL,
                &format!("C(-{}) = -C({})", x, x),
            );
        }
    }

    #[test]
    fn test_fresnel_nan() {
        assert!(fresnel_s_scalar(f64::NAN).is_nan());
        assert!(fresnel_c_scalar(f64::NAN).is_nan());
    }

    #[test]
    fn test_fresnel_cornu_spiral() {
        // The Cornu spiral (C(t), S(t)) spirals towards (0.5, 0.5) as t → ∞
        let (c, s) = (fresnel_c_scalar(100.0), fresnel_s_scalar(100.0));
        let dist = ((c - 0.5).powi(2) + (s - 0.5).powi(2)).sqrt();
        assert!(dist < 0.02, "Should converge to (0.5, 0.5)");
    }

    #[test]
    fn test_fresnel_series_auxiliary_continuity() {
        // Test that series and auxiliary expansions give same value at transition
        let x = AUXILIARY_THRESHOLD;

        // Compute both methods at exactly the threshold
        let s_series = fresnel_s_series_direct(x);
        let s_aux = fresnel_s_auxiliary(x);

        // Both methods should agree at the threshold (within tolerance)
        assert!(
            (s_series - s_aux).abs() < 0.05,
            "S series-auxiliary continuity: series={}, aux={}",
            s_series,
            s_aux
        );

        let c_at = fresnel_c_scalar(x);
        assert!(c_at > 0.0 && c_at < 1.0, "C at threshold in valid range");
    }
}
