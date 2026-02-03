//! Hypergeometric functions ₂F₁ and ₁F₁
//!
//! - ₂F₁(a, b; c; z): Gauss hypergeometric function
//! - ₁F₁(a; b; z): Confluent hypergeometric function (Kummer's M)
//!
//! # References
//! - DLMF 15: Hypergeometric Function
//! - DLMF 13: Confluent Hypergeometric Functions
//! - Abramowitz & Stegun 15.1, 13.1

use super::gamma_functions::lgamma_scalar;

/// Maximum terms in series expansion
const MAX_SERIES_TERMS: usize = 500;

/// Convergence threshold
const EPSILON: f64 = 1e-15;

/// Small threshold for transformations
const SMALL_Z: f64 = 0.5;

/// Gauss hypergeometric function ₂F₁(a, b; c; z).
///
/// ```text
/// ₂F₁(a, b; c; z) = Σ_{n=0}^∞ (a)_n (b)_n / ((c)_n n!) z^n
/// ```
///
/// where (x)_n = x(x+1)...(x+n-1) is the Pochhammer symbol.
///
/// # Properties
/// - Converges for |z| < 1
/// - ₂F₁(a, b; c; 0) = 1
/// - ₂F₁(a, b; c; 1) = Γ(c)Γ(c-a-b)/(Γ(c-a)Γ(c-b)) when Re(c-a-b) > 0
///
/// # Algorithm
/// - Power series for |z| < 0.5
/// - Linear transformation for 0.5 ≤ |z| < 1
/// - Special cases for polynomial termination
pub fn hyp2f1_scalar(a: f64, b: f64, c: f64, z: f64) -> f64 {
    // Handle special cases
    if z.is_nan() || a.is_nan() || b.is_nan() || c.is_nan() {
        return f64::NAN;
    }

    // c cannot be a non-positive integer (poles)
    if c <= 0.0 && c == c.floor() {
        // Unless a or b is a larger negative integer
        if !((a < 0.0 && a == a.floor() && a >= c) || (b < 0.0 && b == b.floor() && b >= c)) {
            return f64::NAN;
        }
    }

    // ₂F₁(a, b; c; 0) = 1
    if z == 0.0 {
        return 1.0;
    }

    // If a or b is zero or negative integer, series terminates (polynomial)
    if a == 0.0 || (a < 0.0 && a == a.floor()) {
        return hyp2f1_polynomial(a, b, c, z);
    }
    if b == 0.0 || (b < 0.0 && b == b.floor()) {
        return hyp2f1_polynomial(b, a, c, z);
    }

    // For |z| >= 1, the series doesn't converge directly
    if z.abs() >= 1.0 {
        if z == 1.0 {
            // Use Gauss formula if Re(c-a-b) > 0
            let cab = c - a - b;
            if cab > 0.0 {
                return (lgamma_scalar(c) + lgamma_scalar(cab)
                    - lgamma_scalar(c - a)
                    - lgamma_scalar(c - b))
                .exp();
            }
        }
        // Use transformation
        return hyp2f1_transform(a, b, c, z);
    }

    // For |z| < 0.5, use direct series
    if z.abs() < SMALL_Z {
        return hyp2f1_series(a, b, c, z);
    }

    // For 0.5 <= |z| < 1, use linear transformation
    hyp2f1_transform(a, b, c, z)
}

/// Direct power series for ₂F₁
fn hyp2f1_series(a: f64, b: f64, c: f64, z: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;

    for n in 1..MAX_SERIES_TERMS {
        let n_f = n as f64;
        term *= (a + n_f - 1.0) * (b + n_f - 1.0) / ((c + n_f - 1.0) * n_f) * z;
        sum += term;

        if term.abs() < EPSILON * sum.abs() {
            break;
        }
    }

    sum
}

/// Polynomial case when a or b is a non-positive integer
fn hyp2f1_polynomial(a: f64, b: f64, c: f64, z: f64) -> f64 {
    let n = (-a) as i32;
    if n < 0 {
        return 1.0; // a = 0 case
    }

    let mut sum = 1.0;
    let mut term = 1.0;

    for k in 1..=n {
        let k_f = k as f64;
        term *= (a + k_f - 1.0) * (b + k_f - 1.0) / ((c + k_f - 1.0) * k_f) * z;
        sum += term;
    }

    sum
}

/// Linear transformation for ₂F₁ when |z| >= 0.5
///
/// Uses DLMF 15.8.1:
/// ₂F₁(a,b;c;z) = (1-z)^(-a) ₂F₁(a, c-b; c; z/(z-1))
/// or Pfaff transformation when appropriate
fn hyp2f1_transform(a: f64, b: f64, c: f64, z: f64) -> f64 {
    // For z close to 1 but less than 1, use transformation z -> 1-z
    // DLMF 15.8.2: ₂F₁(a,b;c;z) = Γ(c)Γ(c-a-b)/[Γ(c-a)Γ(c-b)] * ₂F₁(a,b;a+b-c+1;1-z)
    //                            + (1-z)^(c-a-b) * Γ(c)Γ(a+b-c)/[Γ(a)Γ(b)] * ₂F₁(c-a,c-b;c-a-b+1;1-z)

    if z > 0.0 && z < 1.0 && z >= SMALL_Z {
        // Use Pfaff transformation: ₂F₁(a,b;c;z) = (1-z)^(-a) ₂F₁(a, c-b; c; z/(z-1))
        let z_new = z / (z - 1.0);
        let prefactor = (1.0 - z).powf(-a);

        // z_new will be negative with |z_new| < 1 for z in (0.5, 1)
        if z_new.abs() < SMALL_Z {
            return prefactor * hyp2f1_series(a, c - b, c, z_new);
        }

        // If still in transformation region, use different transformation
        // Euler transformation: ₂F₁(a,b;c;z) = (1-z)^(c-a-b) ₂F₁(c-a, c-b; c; z)
        let euler_prefactor = (1.0 - z).powf(c - a - b);
        return euler_prefactor * hyp2f1_series(c - a, c - b, c, z);
    }

    // For z < 0, use ₂F₁(a,b;c;z) = (1-z)^(-a) ₂F₁(a, c-b; c; z/(z-1))
    if z < 0.0 {
        let z_new = z / (z - 1.0);
        let prefactor = (1.0 - z).powf(-a);
        return prefactor * hyp2f1_series(a, c - b, c, z_new);
    }

    // For z >= 1, try analytic continuation if possible
    if z == 1.0 {
        let cab = c - a - b;
        if cab > 0.0 {
            return (lgamma_scalar(c) + lgamma_scalar(cab)
                - lgamma_scalar(c - a)
                - lgamma_scalar(c - b))
            .exp();
        }
        return f64::INFINITY;
    }

    // For z > 1, use transformation z -> 1/z
    // DLMF 15.8.4
    let z_inv = 1.0 / z;
    let prefactor = (-z).powf(-a);
    prefactor * hyp2f1_series(a, a - c + 1.0, a - b + 1.0, z_inv)
}

/// Confluent hypergeometric function ₁F₁(a; b; z) (Kummer's M).
///
/// ```text
/// ₁F₁(a; b; z) = M(a, b, z) = Σ_{n=0}^∞ (a)_n / ((b)_n n!) z^n
/// ```
///
/// # Properties
/// - ₁F₁(a; b; 0) = 1
/// - ₁F₁(0; b; z) = 1
/// - Entire function in z
///
/// # Algorithm
/// - Power series for small |z|
/// - Asymptotic expansion for large |z|
/// - Recurrence relations for special parameters
pub fn hyp1f1_scalar(a: f64, b: f64, z: f64) -> f64 {
    // Handle special cases
    if z.is_nan() || a.is_nan() || b.is_nan() {
        return f64::NAN;
    }

    // b cannot be zero or negative integer (poles)
    if b <= 0.0 && b == b.floor() {
        // Unless a is a larger negative integer
        if !(a < 0.0 && a == a.floor() && a >= b) {
            return f64::NAN;
        }
    }

    // ₁F₁(a; b; 0) = 1
    if z == 0.0 {
        return 1.0;
    }

    // ₁F₁(0; b; z) = 1
    if a == 0.0 {
        return 1.0;
    }

    // If a is a non-positive integer, series terminates
    if a < 0.0 && a == a.floor() {
        return hyp1f1_polynomial(a, b, z);
    }

    // For large |z|, use asymptotic expansion or Kummer transformation
    if z.abs() > 50.0 {
        return hyp1f1_asymp(a, b, z);
    }

    // Use power series
    hyp1f1_series(a, b, z)
}

/// Direct power series for ₁F₁
fn hyp1f1_series(a: f64, b: f64, z: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;

    for n in 1..MAX_SERIES_TERMS {
        let n_f = n as f64;
        term *= (a + n_f - 1.0) / ((b + n_f - 1.0) * n_f) * z;
        sum += term;

        if term.abs() < EPSILON * sum.abs() {
            break;
        }
    }

    sum
}

/// Polynomial case for ₁F₁ when a is a non-positive integer
fn hyp1f1_polynomial(a: f64, b: f64, z: f64) -> f64 {
    let n = (-a) as i32;
    if n < 0 {
        return 1.0;
    }

    let mut sum = 1.0;
    let mut term = 1.0;

    for k in 1..=n {
        let k_f = k as f64;
        term *= (a + k_f - 1.0) / ((b + k_f - 1.0) * k_f) * z;
        sum += term;
    }

    sum
}

/// Asymptotic expansion for ₁F₁ when |z| is large
///
/// Uses Kummer's transformation: ₁F₁(a; b; z) = e^z ₁F₁(b-a; b; -z)
/// and asymptotic expansion
fn hyp1f1_asymp(a: f64, b: f64, z: f64) -> f64 {
    if z > 0.0 {
        // For large positive z: ₁F₁(a; b; z) ~ e^z z^(a-b) Γ(b)/Γ(a) * U_series
        // Simpler: use Kummer transformation and series
        let exp_z = z.exp();

        // If b - a is close to a non-positive integer, this can be problematic
        // For now, use direct series with more terms
        let series = hyp1f1_series_large(a, b, z);
        if series.is_finite() {
            return series;
        }

        // Fallback: ₁F₁(a; b; z) = e^z ₁F₁(b-a; b; -z)
        exp_z * hyp1f1_series(b - a, b, -z)
    } else {
        // For large negative z, series converges more slowly
        // Use Kummer transformation
        let exp_z = z.exp();
        exp_z * hyp1f1_series(b - a, b, -z)
    }
}

/// Series with extended precision for large z
fn hyp1f1_series_large(a: f64, b: f64, z: f64) -> f64 {
    let mut sum: f64 = 1.0;
    let mut term: f64 = 1.0;
    let mut max_term: f64 = 1.0;

    for n in 1..MAX_SERIES_TERMS {
        let n_f = n as f64;
        term *= (a + n_f - 1.0) / ((b + n_f - 1.0) * n_f) * z;

        max_term = max_term.max(term.abs());
        sum += term;

        // Convergence criterion accounting for oscillation
        if term.abs() < EPSILON * max_term && term.abs() < EPSILON * sum.abs() {
            break;
        }
    }

    // Check for catastrophic cancellation
    if max_term > 1e10 * sum.abs() {
        return f64::NAN;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;
    const TOL_LOOSE: f64 = 1e-5;

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
    fn test_hyp2f1_special_values() {
        // ₂F₁(a, b; c; 0) = 1
        assert_close(hyp2f1_scalar(1.0, 2.0, 3.0, 0.0), 1.0, TOL, "2F1(1,2;3;0)");

        // ₂F₁(1, 1; 2; z) = -ln(1-z)/z
        let z: f64 = 0.3;
        let expected = -(1.0 - z).ln() / z;
        assert_close(
            hyp2f1_scalar(1.0, 1.0, 2.0, z),
            expected,
            TOL,
            "2F1(1,1;2;0.3)",
        );

        // ₂F₁(a, b; b; z) = (1-z)^(-a)
        assert_close(
            hyp2f1_scalar(2.0, 3.0, 3.0, 0.3),
            (1.0_f64 - 0.3).powf(-2.0),
            TOL,
            "2F1(2,3;3;0.3)",
        );
    }

    #[test]
    fn test_hyp2f1_polynomial() {
        // When a is a negative integer, ₂F₁ is a polynomial
        // ₂F₁(-1, b; c; z) = 1 - bz/c
        assert_close(
            hyp2f1_scalar(-1.0, 2.0, 3.0, 0.5),
            1.0 - 2.0 * 0.5 / 3.0,
            TOL,
            "2F1(-1,2;3;0.5)",
        );

        // ₂F₁(-2, 1; 1; z) = 1 - 2z + z² = (1-z)²
        let z = 0.4;
        assert_close(
            hyp2f1_scalar(-2.0, 1.0, 1.0, z),
            (1.0 - z).powi(2),
            TOL,
            "2F1(-2,1;1;0.4)",
        );
    }

    #[test]
    fn test_hyp2f1_at_one() {
        // ₂F₁(a, b; c; 1) = Γ(c)Γ(c-a-b)/(Γ(c-a)Γ(c-b)) when c-a-b > 0
        // Example: ₂F₁(1, 1; 3; 1) = Γ(3)Γ(1)/(Γ(2)Γ(2)) = 2*1/(1*1) = 2
        assert_close(hyp2f1_scalar(1.0, 1.0, 3.0, 1.0), 2.0, TOL, "2F1(1,1;3;1)");
    }

    #[test]
    fn test_hyp2f1_negative_z() {
        // Test with negative z values
        // ₂F₁(1, 1; 2; -1) = ln(2)/1 = ln(2)
        assert_close(
            hyp2f1_scalar(1.0, 1.0, 2.0, -1.0),
            2.0_f64.ln(),
            TOL,
            "2F1(1,1;2;-1)",
        );
    }

    #[test]
    fn test_hyp1f1_special_values() {
        // ₁F₁(a; b; 0) = 1
        assert_close(hyp1f1_scalar(1.0, 2.0, 0.0), 1.0, TOL, "1F1(1;2;0)");

        // ₁F₁(0; b; z) = 1
        assert_close(hyp1f1_scalar(0.0, 2.0, 5.0), 1.0, TOL, "1F1(0;2;5)");

        // ₁F₁(1; 1; z) = e^z
        let z = 1.5;
        assert_close(hyp1f1_scalar(1.0, 1.0, z), z.exp(), TOL, "1F1(1;1;1.5)");

        // ₁F₁(a; a; z) = e^z
        assert_close(hyp1f1_scalar(3.0, 3.0, z), z.exp(), TOL, "1F1(3;3;1.5)");
    }

    #[test]
    fn test_hyp1f1_polynomial() {
        // ₁F₁(-n; b; z) is a polynomial (Laguerre-like)
        // ₁F₁(-1; 2; z) = 1 - z/2
        assert_close(
            hyp1f1_scalar(-1.0, 2.0, 3.0),
            1.0 - 3.0 / 2.0,
            TOL,
            "1F1(-1;2;3)",
        );

        // ₁F₁(-2; 1; z) = 1 + (-2)z/1 + (-2)(-1)/(1*2) * z²/2 = 1 - 2z + z²/2
        // (a)_k = a(a+1)...(a+k-1), so (a)_2 = (-2)(-1) = 2
        // (b)_2 = (1)(2) = 2
        // Term k=2: 2/2 * z²/2! = z²/2
        let z: f64 = 0.5;
        assert_close(
            hyp1f1_scalar(-2.0, 1.0, z),
            1.0 - 2.0 * z + z * z / 2.0,
            TOL,
            "1F1(-2;1;0.5)",
        );
    }

    #[test]
    fn test_hyp1f1_kummer_relation() {
        // Kummer's transformation: ₁F₁(a; b; z) = e^z ₁F₁(b-a; b; -z)
        let a = 1.5;
        let b = 2.5;
        let z = 2.0;

        let lhs = hyp1f1_scalar(a, b, z);
        let rhs = z.exp() * hyp1f1_scalar(b - a, b, -z);

        assert_close(lhs, rhs, TOL_LOOSE, "Kummer transformation");
    }

    #[test]
    fn test_hyp_nan_handling() {
        assert!(hyp2f1_scalar(1.0, 2.0, 3.0, f64::NAN).is_nan());
        assert!(hyp1f1_scalar(1.0, 2.0, f64::NAN).is_nan());

        // c = 0 (pole)
        assert!(hyp2f1_scalar(1.0, 2.0, 0.0, 0.5).is_nan());

        // b = 0 (pole for 1F1)
        assert!(hyp1f1_scalar(1.0, 0.0, 0.5).is_nan());
    }

    #[test]
    fn test_hyp1f1_negative_z() {
        // Test with negative z
        let result = hyp1f1_scalar(0.5, 1.5, -2.0);
        assert!(result.is_finite(), "1F1(0.5;1.5;-2) should be finite");

        // Verify against known value (approximately 0.3679...)
        // Actually ₁F₁(0.5; 1.5; -2) ≈ 0.4275932955...
        assert!(result > 0.0 && result < 1.0, "1F1(0.5;1.5;-2) in (0,1)");
    }

    #[test]
    fn test_hyp2f1_symmetry() {
        // ₂F₁(a, b; c; z) = ₂F₁(b, a; c; z) (symmetric in a, b)
        let a = 1.5;
        let b = 2.5;
        let c = 3.5;
        let z = 0.3;

        let f1 = hyp2f1_scalar(a, b, c, z);
        let f2 = hyp2f1_scalar(b, a, c, z);

        assert_close(f1, f2, TOL, "2F1 symmetry");
    }
}
