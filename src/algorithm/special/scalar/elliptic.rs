//! Complete elliptic integrals K(m) and E(m)
//!
//! Uses the Arithmetic-Geometric Mean (AGM) method for high accuracy.
//!
//! # References
//! - DLMF 19.8: AGM method for elliptic integrals
//! - Abramowitz & Stegun 17.6

use std::f64::consts::PI;

/// Maximum iterations for AGM convergence
const MAX_ITER: usize = 30;

/// Convergence threshold for AGM
const EPSILON: f64 = 1e-15;

/// Complete elliptic integral of the first kind K(m).
///
/// ```text
/// K(m) = ∫₀^(π/2) dθ / √(1 - m·sin²θ)
/// ```
///
/// # Properties
/// - Domain: m ∈ [0, 1)
/// - K(0) = π/2
/// - K(m) → ∞ as m → 1
/// - Uses parameter convention m = k², where k is the modulus
///
/// # Algorithm
/// AGM method: K(m) = π / (2 * AGM(1, √(1-m)))
pub fn ellipk_scalar(m: f64) -> f64 {
    // Handle special cases
    if m < 0.0 {
        return f64::NAN;
    }
    if m == 0.0 {
        return PI / 2.0;
    }
    if m >= 1.0 {
        return f64::INFINITY;
    }

    // AGM method: K(m) = π / (2 * AGM(1, √(1-m)))
    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();

    for _ in 0..MAX_ITER {
        let a_new = (a + b) / 2.0;
        let b_new = (a * b).sqrt();

        if (a_new - b_new).abs() < EPSILON * a_new {
            return PI / (2.0 * a_new);
        }

        a = a_new;
        b = b_new;
    }

    PI / (2.0 * a)
}

/// Complete elliptic integral of the second kind E(m).
///
/// ```text
/// E(m) = ∫₀^(π/2) √(1 - m·sin²θ) dθ
/// ```
///
/// # Properties
/// - Domain: m ∈ [0, 1]
/// - E(0) = π/2
/// - E(1) = 1
/// - Uses parameter convention m = k², where k is the modulus
///
/// # Algorithm
/// AGM method with sum of geometric series:
/// E(m) = K(m) * [1 - Σᵢ 2^(i-1) * cᵢ²]
/// where cᵢ = (aᵢ - bᵢ) / 2
pub fn ellipe_scalar(m: f64) -> f64 {
    // Handle special cases
    if m < 0.0 {
        return f64::NAN;
    }
    if m == 0.0 {
        return PI / 2.0;
    }
    if m == 1.0 {
        return 1.0;
    }
    if m > 1.0 {
        return f64::NAN;
    }

    // AGM method for E(m)
    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();
    let mut c = m.sqrt();
    let mut sum = c * c;
    let mut power_of_two = 1.0;

    for _ in 0..MAX_ITER {
        let a_new = (a + b) / 2.0;
        let b_new = (a * b).sqrt();
        c = (a - b) / 2.0;

        power_of_two *= 2.0;
        sum += power_of_two * c * c;

        if c.abs() < EPSILON * a_new {
            let k = PI / (2.0 * a_new);
            return k * (1.0 - sum / 2.0);
        }

        a = a_new;
        b = b_new;
    }

    let k = PI / (2.0 * a);
    k * (1.0 - sum / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
        let diff = (a - b).abs();
        assert!(
            diff < tol || (a.is_nan() && b.is_nan()) || (a.is_infinite() && b.is_infinite()),
            "{}: expected {}, got {}, diff {}",
            msg,
            b,
            a,
            diff
        );
    }

    #[test]
    fn test_ellipk_special_values() {
        // K(0) = π/2
        assert_close(ellipk_scalar(0.0), PI / 2.0, TOL, "K(0)");

        // K(0.5) ≈ 1.8540746773013719 (from Wolfram Alpha)
        assert_close(ellipk_scalar(0.5), 1.8540746773013719, TOL, "K(0.5)");

        // K approaches infinity as m -> 1
        // K(0.9999999) ≈ 9.445, K(0.99999999) ≈ 10.6
        assert!(ellipk_scalar(0.9999999).is_finite());
        assert!(ellipk_scalar(0.9999999) > 9.0);

        // K(1) = infinity
        assert!(ellipk_scalar(1.0).is_infinite());

        // K(m < 0) = NaN
        assert!(ellipk_scalar(-0.1).is_nan());
    }

    #[test]
    fn test_ellipe_special_values() {
        // E(0) = π/2
        assert_close(ellipe_scalar(0.0), PI / 2.0, TOL, "E(0)");

        // E(0.5) ≈ 1.3506438810476755 (from Wolfram Alpha)
        assert_close(ellipe_scalar(0.5), 1.3506438810476755, TOL, "E(0.5)");

        // E(1) = 1
        assert_close(ellipe_scalar(1.0), 1.0, TOL, "E(1)");

        // E(m > 1) = NaN
        assert!(ellipe_scalar(1.1).is_nan());

        // E(m < 0) = NaN
        assert!(ellipe_scalar(-0.1).is_nan());
    }

    #[test]
    fn test_elliptic_integrals_relation() {
        // Test Legendre relation: E(m)K'(m) + E'(m)K(m) - K(m)K'(m) = π/2
        // where K'(m) = K(1-m), E'(m) = E(1-m)
        for &m in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let k = ellipk_scalar(m);
            let e = ellipe_scalar(m);
            let kp = ellipk_scalar(1.0 - m);
            let ep = ellipe_scalar(1.0 - m);

            let legendre = e * kp + ep * k - k * kp;
            assert_close(
                legendre,
                PI / 2.0,
                1e-9,
                &format!("Legendre relation at m={}", m),
            );
        }
    }

    #[test]
    fn test_ellipk_monotonicity() {
        // K(m) is monotonically increasing on [0, 1)
        let mut prev = ellipk_scalar(0.0);
        for i in 1..10 {
            let m = i as f64 / 10.0;
            let curr = ellipk_scalar(m);
            assert!(
                curr > prev,
                "K should be increasing: K({}) > K({})",
                m,
                m - 0.1
            );
            prev = curr;
        }
    }

    #[test]
    fn test_ellipe_monotonicity() {
        // E(m) is monotonically decreasing on [0, 1]
        let mut prev = ellipe_scalar(0.0);
        for i in 1..=10 {
            let m = i as f64 / 10.0;
            let curr = ellipe_scalar(m);
            assert!(
                curr < prev,
                "E should be decreasing: E({}) < E({})",
                m,
                m - 0.1
            );
            prev = curr;
        }
    }
}
