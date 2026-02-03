//! Airy functions Ai(z) and Bi(z)
//!
//! Uses power series for small arguments and asymptotic expansions for large arguments.
//!
//! # References
//! - DLMF Chapter 9: Airy and Related Functions
//! - Abramowitz & Stegun Chapter 10

use std::f64::consts::PI;

/// 1/√π
const INV_SQRT_PI: f64 = 0.5641895835477563;

/// Power series coefficients for Ai(x)
/// Ai(x) = c1 * f(x) - c2 * g(x)
/// where f and g are even and odd auxiliary functions
const AI_C1: f64 = 0.3550280538878172; // Ai(0) = 3^(-2/3) / Γ(2/3)
const AI_C2: f64 = 0.2588194037928068; // -Ai'(0) = 3^(-1/3) / Γ(1/3)

/// Power series coefficients for Bi(x)
const BI_C1: f64 = 0.6149266274460007; // Bi(0) = 3^(-1/6) / Γ(2/3)
const BI_C2: f64 = 0.4482883573538264; // Bi'(0) = 3^(1/6) / Γ(1/3)

/// Maximum terms in power series
const MAX_SERIES_TERMS: usize = 100;

/// Convergence threshold
const EPSILON: f64 = 1e-15;

/// Threshold for switching to asymptotic expansion
const ASYMP_THRESHOLD: f64 = 10.0;

/// Compute the Airy function of the first kind Ai(x).
///
/// ```text
/// Ai(x) is the solution of y'' - xy = 0 that decays as x → +∞
/// ```
///
/// # Properties
/// - Ai(x) → 0 as x → +∞ (exponentially)
/// - Ai(x) oscillates for x < 0
/// - Ai(0) ≈ 0.3550280538878172
pub fn airy_ai_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }

    if x > ASYMP_THRESHOLD {
        // Asymptotic expansion for large positive x
        airy_ai_asymp_pos(x)
    } else if x < -ASYMP_THRESHOLD {
        // Asymptotic expansion for large negative x
        airy_ai_asymp_neg(x)
    } else {
        // Power series for moderate x
        airy_ai_series(x)
    }
}

/// Compute the Airy function of the second kind Bi(x).
///
/// ```text
/// Bi(x) is the solution of y'' - xy = 0 that grows as x → +∞
/// ```
///
/// # Properties
/// - Bi(x) → +∞ as x → +∞ (exponentially)
/// - Bi(x) oscillates for x < 0
/// - Bi(0) ≈ 0.6149266274460007
pub fn airy_bi_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }

    if x > ASYMP_THRESHOLD {
        // Asymptotic expansion for large positive x
        airy_bi_asymp_pos(x)
    } else if x < -ASYMP_THRESHOLD {
        // Asymptotic expansion for large negative x
        airy_bi_asymp_neg(x)
    } else {
        // Power series for moderate x
        airy_bi_series(x)
    }
}

/// Power series for Ai(x)
///
/// Ai(x) = c1 * f(ζ) - c2 * g(ζ)
/// where ζ = x³/9 and f, g are auxiliary functions
fn airy_ai_series(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        return AI_C1;
    }

    // Compute using direct series
    // Ai(x) = sum of terms involving x^(3k) / (3k)! * product terms
    let x3 = x * x * x;

    // f series: sum of x^(3k) * prod_{j=1}^{k} (3j-2) / (3k)!
    let mut f_sum = 1.0;
    let mut f_term = 1.0;
    for k in 1..MAX_SERIES_TERMS {
        let k3 = 3 * k;
        f_term *= x3 * (k3 as f64 - 2.0) / ((k3 as f64) * ((k3 - 1) as f64) * ((k3 - 2) as f64));
        f_sum += f_term;
        if f_term.abs() < EPSILON * f_sum.abs() {
            break;
        }
    }

    // g series: sum of x^(3k+1) * prod_{j=1}^{k} (3j-1) / (3k+1)!
    let mut g_sum = x;
    let mut g_term = x;
    for k in 1..MAX_SERIES_TERMS {
        let k3 = 3 * k;
        g_term *= x3 * (k3 as f64 - 1.0) / (((k3 + 1) as f64) * (k3 as f64) * ((k3 - 1) as f64));
        g_sum += g_term;
        if g_term.abs() < EPSILON * g_sum.abs() {
            break;
        }
    }

    AI_C1 * f_sum - AI_C2 * g_sum
}

/// Power series for Bi(x)
fn airy_bi_series(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        return BI_C1;
    }

    let x3 = x * x * x;

    // Same f and g series as Ai
    let mut f_sum = 1.0;
    let mut f_term = 1.0;
    for k in 1..MAX_SERIES_TERMS {
        let k3 = 3 * k;
        f_term *= x3 * (k3 as f64 - 2.0) / ((k3 as f64) * ((k3 - 1) as f64) * ((k3 - 2) as f64));
        f_sum += f_term;
        if f_term.abs() < EPSILON * f_sum.abs() {
            break;
        }
    }

    let mut g_sum = x;
    let mut g_term = x;
    for k in 1..MAX_SERIES_TERMS {
        let k3 = 3 * k;
        g_term *= x3 * (k3 as f64 - 1.0) / (((k3 + 1) as f64) * (k3 as f64) * ((k3 - 1) as f64));
        g_sum += g_term;
        if g_term.abs() < EPSILON * g_sum.abs() {
            break;
        }
    }

    BI_C1 * f_sum + BI_C2 * g_sum
}

/// Asymptotic expansion for Ai(x) with x > 0
///
/// Ai(x) ~ exp(-ζ) / (2√π x^(1/4)) * sum
/// where ζ = (2/3) x^(3/2)
fn airy_ai_asymp_pos(x: f64) -> f64 {
    let zeta = (2.0 / 3.0) * x.powf(1.5);
    let x14 = x.powf(0.25);

    // Leading term
    let prefactor = (-zeta).exp() / (2.0 * PI.sqrt() * x14);

    // Asymptotic series
    let z = 1.0 / zeta;
    let sum = asymp_u_series(z);

    prefactor * sum
}

/// Asymptotic expansion for Bi(x) with x > 0
///
/// Bi(x) ~ exp(ζ) / (√π x^(1/4)) * sum
fn airy_bi_asymp_pos(x: f64) -> f64 {
    let zeta = (2.0 / 3.0) * x.powf(1.5);
    let x14 = x.powf(0.25);

    // Leading term
    let prefactor = zeta.exp() / (PI.sqrt() * x14);

    // Asymptotic series
    let z = 1.0 / zeta;
    let sum = asymp_u_series(z);

    prefactor * sum
}

/// Asymptotic expansion for Ai(x) with x < 0
///
/// Ai(-|x|) ~ sin(ζ + π/4) / (√π |x|^(1/4))
fn airy_ai_asymp_neg(x: f64) -> f64 {
    let ax = x.abs();
    let zeta = (2.0 / 3.0) * ax.powf(1.5);
    let x14 = ax.powf(0.25);

    let phase = zeta + PI / 4.0;
    INV_SQRT_PI / x14 * phase.sin()
}

/// Asymptotic expansion for Bi(x) with x < 0
///
/// Bi(-|x|) ~ cos(ζ + π/4) / (√π |x|^(1/4))
fn airy_bi_asymp_neg(x: f64) -> f64 {
    let ax = x.abs();
    let zeta = (2.0 / 3.0) * ax.powf(1.5);
    let x14 = ax.powf(0.25);

    let phase = zeta + PI / 4.0;
    INV_SQRT_PI / x14 * phase.cos()
}

/// Asymptotic series U(z) for Airy functions
///
/// U(z) = 1 + sum_{k=1}^n u_k / z^k
fn asymp_u_series(z: f64) -> f64 {
    // First few u_k coefficients
    // u_k = prod_{j=1}^{k} (6j-5)(6j-3)(6j-1) / (216^k k!)
    // Precomputed: 72^4 = 26873856, 72^5 = 1934917632
    const U: [f64; 6] = [
        1.0,
        5.0 / 72.0,
        5.0 * 7.0 * 11.0 / (2.0 * 72.0 * 72.0),
        5.0 * 7.0 * 11.0 * 13.0 * 17.0 / (6.0 * 72.0 * 72.0 * 72.0),
        5.0 * 7.0 * 11.0 * 13.0 * 17.0 * 19.0 * 23.0 / (24.0 * 26873856.0),
        5.0 * 7.0 * 11.0 * 13.0 * 17.0 * 19.0 * 23.0 * 25.0 * 29.0 / (120.0 * 1934917632.0),
    ];

    let mut sum = U[0];
    let mut zk = z;
    for &u in &U[1..] {
        sum += u * zk;
        zk *= z;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

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
    fn test_airy_ai_special_values() {
        // Ai(0) ≈ 0.3550280538878172
        assert_close(airy_ai_scalar(0.0), 0.3550280538878172, TOL, "Ai(0)");

        // Ai(1) ≈ 0.1352924163128814
        assert_close(airy_ai_scalar(1.0), 0.1352924163128814, TOL, "Ai(1)");

        // Ai(-1) ≈ 0.5355608832923521
        assert_close(airy_ai_scalar(-1.0), 0.5355608832923521, TOL, "Ai(-1)");

        // Ai decays for large positive x
        assert!(airy_ai_scalar(10.0) < 1e-10);
        assert!(airy_ai_scalar(20.0) < 1e-20);
    }

    #[test]
    fn test_airy_bi_special_values() {
        // Bi(0) ≈ 0.6149266274460007
        assert_close(airy_bi_scalar(0.0), 0.6149266274460007, TOL, "Bi(0)");

        // Bi(1) ≈ 1.2074235949528714
        assert_close(airy_bi_scalar(1.0), 1.2074235949528714, TOL, "Bi(1)");

        // Bi(-1) ≈ 0.10399738949694461
        assert_close(airy_bi_scalar(-1.0), 0.10399738949694461, TOL, "Bi(-1)");

        // Bi grows for large positive x
        // Bi(10) ≈ 4.556e8 (verified against scipy)
        assert!(airy_bi_scalar(10.0) > 1e8);
    }

    #[test]
    fn test_airy_wronskian() {
        // The Wronskian Ai(x)Bi'(x) - Ai'(x)Bi(x) = 1/π
        // We can check at x=0 using known derivatives:
        // Ai'(0) ≈ -0.2588194037928068
        // Bi'(0) ≈ 0.4482883573538264
        let ai0 = AI_C1;
        let bi0 = BI_C1;
        let ai_prime_0 = -AI_C2;
        let bi_prime_0 = BI_C2;

        let wronskian = ai0 * bi_prime_0 - ai_prime_0 * bi0;
        assert_close(wronskian, 1.0 / PI, 1e-10, "Wronskian at x=0");
    }

    #[test]
    fn test_airy_nan() {
        assert!(airy_ai_scalar(f64::NAN).is_nan());
        assert!(airy_bi_scalar(f64::NAN).is_nan());
    }

    #[test]
    fn test_airy_oscillation_negative() {
        // For x < 0, both Ai and Bi oscillate
        // Check that they have zeros
        let mut ai_signs = Vec::new();
        let mut bi_signs = Vec::new();
        for i in 0..20 {
            let x = -(i as f64 + 0.5);
            ai_signs.push(airy_ai_scalar(x) > 0.0);
            bi_signs.push(airy_bi_scalar(x) > 0.0);
        }

        // Should have sign changes
        let ai_changes: usize = ai_signs.windows(2).filter(|w| w[0] != w[1]).count();
        let bi_changes: usize = bi_signs.windows(2).filter(|w| w[0] != w[1]).count();

        assert!(ai_changes > 0, "Ai should oscillate for x < 0");
        assert!(bi_changes > 0, "Bi should oscillate for x < 0");
    }
}
