//! Unit tests for special function scalar implementations

use crate::algorithm::special::scalar::*;
use crate::algorithm::special::{EULER_MASCHERONI, SQRT_PI};

// A&S approximation has ~1e-7 relative error
const TOL: f64 = 1e-6;
const TOL_TIGHT: f64 = 1e-10;

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
fn test_erf_values() {
    // Exact at zero
    assert_close(erf_scalar(0.0), 0.0, TOL_TIGHT);
    // A&S approximation accuracy ~1e-7
    assert_close(erf_scalar(1.0), 0.8427007929497149, TOL);
    assert_close(erf_scalar(-1.0), -0.8427007929497149, TOL);
    assert_close(erf_scalar(2.0), 0.9953222650189527, TOL);
    assert_close(erf_scalar(0.5), 0.5204998778130465, TOL);
}

#[test]
fn test_erfc_values() {
    assert_close(erfc_scalar(0.0), 1.0, TOL_TIGHT);
    assert_close(erfc_scalar(1.0), 0.1572992070502851, TOL);
    assert_close(erfc_scalar(2.0), 0.004677734981047266, TOL);
    // Large x: check relative error instead of absolute
    let erfc5 = erfc_scalar(5.0);
    let expected = 1.5374597944280347e-12;
    let rel_err = (erfc5 - expected).abs() / expected;
    assert!(
        rel_err < 0.01,
        "erfc(5) relative error {} too large",
        rel_err
    );
}

#[test]
fn test_erfinv_values() {
    assert_close(erfinv_scalar(0.0), 0.0, TOL_TIGHT);
    assert_close(erfinv_scalar(0.5), 0.4769362762044699, 1e-4);
    assert_close(erfinv_scalar(-0.5), -0.4769362762044699, 1e-4);
    // Roundtrip test (limited by erf accuracy)
    assert_close(erf_scalar(erfinv_scalar(0.3)), 0.3, 1e-6);
}

#[test]
fn test_erfinv_full_range() {
    // Test that erfinv produces finite values for the full range
    // and that the roundtrip erf(erfinv(x)) ≈ x holds
    for &p in &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let x = erfinv_scalar(p);
        assert!(x.is_finite(), "erfinv({}) = {} is not finite", p, x);
        let back = erf_scalar(x);
        assert_close(back, p, 1e-6);

        // Test negative values too
        let x_neg = erfinv_scalar(-p);
        assert!(
            x_neg.is_finite(),
            "erfinv({}) = {} is not finite",
            -p,
            x_neg
        );
        let back_neg = erf_scalar(x_neg);
        assert_close(back_neg, -p, 1e-6);
    }

    // Test extreme tails with looser tolerance (erf precision limits)
    for &p in &[0.95, 0.99] {
        let x = erfinv_scalar(p);
        assert!(x.is_finite(), "erfinv({}) = {} is not finite", p, x);
        let back = erf_scalar(x);
        assert_close(back, p, 1e-3);
    }
}

#[test]
fn test_gamma_values() {
    assert_close(gamma_scalar(1.0), 1.0, TOL);
    assert_close(gamma_scalar(2.0), 1.0, TOL);
    assert_close(gamma_scalar(3.0), 2.0, TOL);
    assert_close(gamma_scalar(4.0), 6.0, TOL);
    assert_close(gamma_scalar(5.0), 24.0, TOL);
    assert_close(gamma_scalar(0.5), SQRT_PI, TOL);
}

#[test]
fn test_lgamma_values() {
    assert_close(lgamma_scalar(1.0), 0.0, TOL);
    assert_close(lgamma_scalar(2.0), 0.0, TOL);
    assert_close(lgamma_scalar(3.0), 2.0_f64.ln(), TOL);
    assert_close(lgamma_scalar(10.0), (362880.0_f64).ln(), TOL);
}

#[test]
fn test_digamma_values() {
    assert_close(digamma_scalar(1.0), -EULER_MASCHERONI, 1e-8);
    assert_close(digamma_scalar(2.0), 1.0 - EULER_MASCHERONI, 1e-8);
}

#[test]
fn test_beta_values() {
    assert_close(beta_scalar(1.0, 1.0), 1.0, TOL);
    assert_close(beta_scalar(2.0, 2.0), 1.0 / 6.0, TOL);
    assert_close(beta_scalar(0.5, 0.5), std::f64::consts::PI, TOL);
}

#[test]
fn test_betainc_values() {
    assert_close(betainc_scalar(1.0, 1.0, 0.5), 0.5, TOL);
    assert_close(betainc_scalar(2.0, 2.0, 0.5), 0.5, TOL);
    assert_close(betainc_scalar(1.0, 1.0, 0.0), 0.0, TOL);
    assert_close(betainc_scalar(1.0, 1.0, 1.0), 1.0, TOL);
}

#[test]
fn test_gammainc_values() {
    // P(1, x) = 1 - e^(-x)
    assert_close(gammainc_scalar(1.0, 1.0), 1.0 - (-1.0_f64).exp(), TOL);
    assert_close(gammainc_scalar(1.0, 2.0), 1.0 - (-2.0_f64).exp(), TOL);
    assert_close(gammainc_scalar(1.0, 0.0), 0.0, TOL);
}

#[test]
fn test_gammaincc_values() {
    // Q(1, x) = e^(-x)
    assert_close(gammaincc_scalar(1.0, 1.0), (-1.0_f64).exp(), TOL);
    assert_close(gammaincc_scalar(1.0, 2.0), (-2.0_f64).exp(), TOL);
    // P + Q = 1
    assert_close(
        gammainc_scalar(2.0, 3.0) + gammaincc_scalar(2.0, 3.0),
        1.0,
        TOL,
    );
}

#[test]
fn test_betaincinv_roundtrip() {
    // Test that betaincinv is the inverse of betainc
    let test_cases = [
        (1.0, 1.0, 0.5),
        (2.0, 2.0, 0.3),
        (2.0, 5.0, 0.4),
        (5.0, 2.0, 0.6),
        (5.0, 0.5, 0.2), // This is the problematic case for Student's t
        (0.5, 5.0, 0.8),
        (10.0, 10.0, 0.5),
    ];

    for (a, b, p) in test_cases {
        let x = betaincinv_scalar(a, b, p);
        let back = betainc_scalar(a, b, x);
        assert!(
            (back - p).abs() < 1e-6,
            "betaincinv roundtrip failed: betainc({}, {}, betaincinv({}, {}, {})) = {} (expected {})",
            a,
            b,
            a,
            b,
            p,
            back,
            p
        );
    }
}
