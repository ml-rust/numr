//! Orthogonal polynomials and spherical harmonics
//!
//! - Legendre polynomials P_n(x)
//! - Associated Legendre functions P_n^m(x)
//! - Spherical harmonics Y_n^m(θ, φ)
//!
//! # References
//! - DLMF 14: Legendre and Related Functions
//! - DLMF 18.9: Recurrence Relations
//! - Abramowitz & Stegun 8.6, 8.8, 22.7

use std::f64::consts::PI;

/// Maximum allowed order for Legendre polynomials
const MAX_ORDER: i32 = 100;

/// Legendre polynomial P_n(x).
///
/// ```text
/// P_n(x) is the solution of (1-x²)y'' - 2xy' + n(n+1)y = 0
/// ```
///
/// # Properties
/// - Domain: x ∈ [-1, 1]
/// - P_n(1) = 1
/// - P_n(-1) = (-1)^n
/// - P_0(x) = 1, P_1(x) = x
///
/// # Algorithm
/// Three-term recurrence: (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
pub fn legendre_p_scalar(n: i32, x: f64) -> f64 {
    if n < 0 {
        // P_{-n-1}(x) = P_n(x)
        return legendre_p_scalar(-n - 1, x);
    }
    if n > MAX_ORDER {
        return f64::NAN;
    }
    if x.is_nan() {
        return f64::NAN;
    }

    // Special cases
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }

    // Three-term recurrence
    let mut p_prev = 1.0; // P_0
    let mut p_curr = x; // P_1

    for k in 1..n {
        let k_f = k as f64;
        let p_next = ((2.0 * k_f + 1.0) * x * p_curr - k_f * p_prev) / (k_f + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
    }

    p_curr
}

/// Associated Legendre function P_n^m(x).
///
/// ```text
/// P_n^m(x) = (-1)^m (1-x²)^(m/2) d^m/dx^m P_n(x)
/// ```
///
/// Uses Condon-Shortley phase convention (factor of (-1)^m).
///
/// # Properties
/// - Domain: x ∈ [-1, 1], 0 ≤ m ≤ n
/// - P_n^0(x) = P_n(x)
/// - P_n^n(x) = (-1)^n (2n-1)!! (1-x²)^(n/2)
///
/// # Algorithm
/// 1. Start from P_m^m(x) using explicit formula
/// 2. Use recurrence to increase n: (n-m+1)P_{n+1}^m = (2n+1)xP_n^m - (n+m)P_{n-1}^m
pub fn legendre_p_assoc_scalar(n: i32, m: i32, x: f64) -> f64 {
    if n < 0 || m < 0 {
        return f64::NAN;
    }
    if m > n {
        return 0.0;
    }
    if n > MAX_ORDER {
        return f64::NAN;
    }
    if x.is_nan() {
        return f64::NAN;
    }
    if x.abs() > 1.0 {
        return f64::NAN;
    }

    // Special case m = 0
    if m == 0 {
        return legendre_p_scalar(n, x);
    }

    // Compute P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^(m/2)
    let sin_theta = (1.0 - x * x).sqrt();
    let mut pmm = 1.0;
    let mut fact = 1.0;

    for _ in 1..=m {
        pmm *= -fact * sin_theta;
        fact += 2.0;
    }

    // If n == m, we're done
    if n == m {
        return pmm;
    }

    // Compute P_{m+1}^m(x) = x(2m+1)P_m^m(x)
    let pmm1 = x * (2 * m + 1) as f64 * pmm;

    if n == m + 1 {
        return pmm1;
    }

    // Use recurrence to get P_n^m(x)
    let mut p_prev = pmm;
    let mut p_curr = pmm1;

    for k in (m + 1)..n {
        let k_f = k as f64;
        let m_f = m as f64;
        let p_next = ((2.0 * k_f + 1.0) * x * p_curr - (k_f + m_f) * p_prev) / (k_f - m_f + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
    }

    p_curr
}

/// Real spherical harmonic Y_n^m(θ, φ).
///
/// ```text
/// Y_n^m(θ, φ) = N_n^m P_n^|m|(cos θ) e^{imφ}
/// ```
///
/// where N_n^m is a normalization factor.
///
/// Returns the real part for m ≥ 0, imaginary part times i for m < 0.
/// More specifically:
/// - m > 0: Y_n^m ∝ P_n^m(cos θ) cos(mφ)
/// - m = 0: Y_n^0 ∝ P_n(cos θ)
/// - m < 0: Y_n^m ∝ P_n^|m|(cos θ) sin(|m|φ)
///
/// # Normalization
/// Uses Schmidt semi-normalization (common in geophysics):
/// ∫∫ |Y_n^m|² sin θ dθ dφ = 4π/(2n+1)
///
/// # Arguments
/// - n: degree (n ≥ 0)
/// - m: order (-n ≤ m ≤ n)
/// - theta: polar angle θ ∈ [0, π] (colatitude)
/// - phi: azimuthal angle φ ∈ [0, 2π)
pub fn sph_harm_scalar(n: i32, m: i32, theta: f64, phi: f64) -> f64 {
    if n < 0 {
        return f64::NAN;
    }
    if m.abs() > n {
        return 0.0;
    }
    if theta.is_nan() || phi.is_nan() {
        return f64::NAN;
    }

    let abs_m = m.abs();
    let cos_theta = theta.cos();

    // Compute associated Legendre function
    let p_nm = legendre_p_assoc_scalar(n, abs_m, cos_theta);

    // Normalization factor (Schmidt semi-normalization)
    let norm = sph_harm_norm(n, abs_m);

    // Angular part
    let angular = if m > 0 {
        (m as f64 * phi).cos()
    } else if m < 0 {
        (abs_m as f64 * phi).sin()
    } else {
        1.0
    };

    norm * p_nm * angular
}

/// Compute normalization factor for spherical harmonics
fn sph_harm_norm(n: i32, m: i32) -> f64 {
    // Schmidt semi-normalization:
    // N_n^m = sqrt((2 - δ_{m0}) * (n-m)! / (n+m)!)
    if m == 0 {
        1.0
    } else {
        // Compute (n-m)! / (n+m)! = 1/((n-m+1)(n-m+2)...(n+m))
        let mut ratio = 1.0;
        for k in (n - m + 1)..=(n + m) {
            ratio /= k as f64;
        }
        (2.0 * ratio).sqrt()
    }
}

/// Compute spherical harmonic with full complex normalization
///
/// Uses the physics/quantum mechanics convention with orthonormalization:
/// ∫∫ Y_l^m* Y_l'^m' sin θ dθ dφ = δ_{ll'} δ_{mm'}
///
/// Returns (real_part, imag_part).
pub fn sph_harm_complex_scalar(n: i32, m: i32, theta: f64, phi: f64) -> (f64, f64) {
    if n < 0 || m.abs() > n {
        return (f64::NAN, f64::NAN);
    }
    if theta.is_nan() || phi.is_nan() {
        return (f64::NAN, f64::NAN);
    }

    let abs_m = m.abs();
    let cos_theta = theta.cos();

    // Associated Legendre with |m|
    let p_nm = legendre_p_assoc_scalar(n, abs_m, cos_theta);

    // Full normalization factor
    // K_l^m = sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!)
    let norm = sph_harm_full_norm(n, abs_m);

    // Condon-Shortley phase for negative m
    let phase = if m < 0 && (abs_m % 2 != 0) { -1.0 } else { 1.0 };

    // e^{imφ} = cos(mφ) + i sin(mφ)
    let m_phi = m as f64 * phi;
    let real = phase * norm * p_nm * m_phi.cos();
    let imag = phase * norm * p_nm * m_phi.sin();

    (real, imag)
}

/// Full orthonormal normalization factor
fn sph_harm_full_norm(n: i32, m: i32) -> f64 {
    // K_n^m = sqrt((2n+1)/(4π) * (n-m)!/(n+m)!)
    let n_f = n as f64;

    // Compute (n-m)! / (n+m)!
    let mut ratio = 1.0;
    for k in (n - m + 1)..=(n + m) {
        ratio /= k as f64;
    }

    ((2.0 * n_f + 1.0) / (4.0 * PI) * ratio).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

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
    fn test_legendre_p_special_values() {
        // P_n(1) = 1 for all n
        for n in 0..=10 {
            assert_close(legendre_p_scalar(n, 1.0), 1.0, TOL, &format!("P_{}(1)", n));
        }

        // P_n(-1) = (-1)^n
        for n in 0..=10 {
            let expected = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert_close(
                legendre_p_scalar(n, -1.0),
                expected,
                TOL,
                &format!("P_{}(-1)", n),
            );
        }

        // P_n(0) = 0 for odd n
        for n in [1, 3, 5, 7, 9] {
            assert_close(legendre_p_scalar(n, 0.0), 0.0, TOL, &format!("P_{}(0)", n));
        }
    }

    #[test]
    fn test_legendre_p_explicit() {
        // P_0(x) = 1
        assert_close(legendre_p_scalar(0, 0.5), 1.0, TOL, "P_0(0.5)");

        // P_1(x) = x
        assert_close(legendre_p_scalar(1, 0.5), 0.5, TOL, "P_1(0.5)");

        // P_2(x) = (3x² - 1)/2
        let x = 0.5;
        assert_close(
            legendre_p_scalar(2, x),
            (3.0 * x * x - 1.0) / 2.0,
            TOL,
            "P_2(0.5)",
        );

        // P_3(x) = (5x³ - 3x)/2
        assert_close(
            legendre_p_scalar(3, x),
            (5.0 * x * x * x - 3.0 * x) / 2.0,
            TOL,
            "P_3(0.5)",
        );
    }

    #[test]
    fn test_legendre_p_assoc_m0() {
        // P_n^0(x) = P_n(x)
        for n in 0..=5 {
            for &x in &[-0.5, 0.0, 0.5, 0.9] {
                assert_close(
                    legendre_p_assoc_scalar(n, 0, x),
                    legendre_p_scalar(n, x),
                    TOL,
                    &format!("P_{}^0({}) = P_{}({})", n, x, n, x),
                );
            }
        }
    }

    #[test]
    fn test_legendre_p_assoc_explicit() {
        let x: f64 = 0.6;
        let sin_theta = (1.0 - x * x).sqrt();

        // P_1^1(x) = -sqrt(1-x²)
        assert_close(legendre_p_assoc_scalar(1, 1, x), -sin_theta, TOL, "P_1^1");

        // P_2^1(x) = -3x sqrt(1-x²)
        assert_close(
            legendre_p_assoc_scalar(2, 1, x),
            -3.0 * x * sin_theta,
            TOL,
            "P_2^1",
        );

        // P_2^2(x) = 3(1-x²)
        assert_close(
            legendre_p_assoc_scalar(2, 2, x),
            3.0 * (1.0 - x * x),
            TOL,
            "P_2^2",
        );
    }

    #[test]
    fn test_legendre_p_assoc_m_greater_n() {
        // P_n^m = 0 for m > n
        assert_close(legendre_p_assoc_scalar(2, 3, 0.5), 0.0, TOL, "P_2^3");
        assert_close(legendre_p_assoc_scalar(1, 5, 0.5), 0.0, TOL, "P_1^5");
    }

    #[test]
    fn test_sph_harm_orthogonality() {
        // Test that Y_0^0 is constant (spherically symmetric)
        let y00_1 = sph_harm_scalar(0, 0, 0.5, 0.0);
        let y00_2 = sph_harm_scalar(0, 0, 1.0, 1.0);
        let y00_3 = sph_harm_scalar(0, 0, 2.0, 3.0);

        assert_close(y00_1, y00_2, TOL, "Y_0^0 constant");
        assert_close(y00_2, y00_3, TOL, "Y_0^0 constant");
    }

    #[test]
    fn test_sph_harm_symmetry() {
        // Y_n^m(θ, φ + 2π) = Y_n^m(θ, φ)
        let theta = 0.8;
        let phi = 1.2;

        for n in 0..=3 {
            for m in -n..=n {
                let y1 = sph_harm_scalar(n, m, theta, phi);
                let y2 = sph_harm_scalar(n, m, theta, phi + 2.0 * PI);
                assert_close(y1, y2, 1e-9, &format!("Y_{}^{} 2π periodicity", n, m));
            }
        }
    }

    #[test]
    fn test_sph_harm_complex_normalization() {
        // |Y_1^0(θ, φ)|² integrated over sphere should be 1/(4π)
        // Y_1^0 = sqrt(3/(4π)) cos θ
        let (re, im) = sph_harm_complex_scalar(1, 0, 0.0, 0.0); // θ=0 -> cos θ = 1
        let expected = (3.0 / (4.0 * PI)).sqrt();
        assert_close(re, expected, 1e-9, "Y_1^0(0,0) normalization");
        assert_close(im, 0.0, 1e-9, "Y_1^0(0,0) imaginary part");
    }

    #[test]
    fn test_legendre_nan() {
        assert!(legendre_p_scalar(5, f64::NAN).is_nan());
        assert!(legendre_p_assoc_scalar(5, 2, f64::NAN).is_nan());
        assert!(sph_harm_scalar(5, 2, f64::NAN, 0.0).is_nan());
    }
}
