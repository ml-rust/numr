//! Polynomial coefficients for special functions
//!
//! These coefficients are used for vectorized polynomial evaluation.
//! Some coefficients are reserved for future SIMD implementations.

#![allow(dead_code)]

// ============================================================================
// Error Function Coefficients (Abramowitz & Stegun 7.1.26)
// ============================================================================

/// erf(x) constants
pub mod erf {
    /// Polynomial coefficients for A&S 7.1.26
    pub const A1: f64 = 0.254829592;
    pub const A2: f64 = -0.284496736;
    pub const A3: f64 = 1.421413741;
    pub const A4: f64 = -1.453152027;
    pub const A5: f64 = 1.061405429;
    pub const P: f64 = 0.3275911;

    // f32 versions
    pub const A1_F32: f32 = 0.254829592;
    pub const A2_F32: f32 = -0.284496736;
    pub const A3_F32: f32 = 1.421413741;
    pub const A4_F32: f32 = -1.453152027;
    pub const A5_F32: f32 = 1.061405429;
    pub const P_F32: f32 = 0.3275911;
}

// ============================================================================
// Bessel J0 Coefficients
// ============================================================================

pub mod bessel_j0 {
    /// Small argument polynomial (|x| < 8): numerator
    pub const SMALL_P: [f64; 6] = [
        57568490574.0,
        -13362590354.0,
        651619640.7,
        -11214424.18,
        77392.33017,
        -184.9052456,
    ];

    /// Small argument polynomial (|x| < 8): denominator
    pub const SMALL_Q: [f64; 6] = [
        57568490411.0,
        1029532985.0,
        9494680.718,
        59272.64853,
        267.8532712,
        1.0,
    ];

    /// Asymptotic expansion (|x| >= 8): P polynomial
    pub const ASYMP_P: [f64; 5] = [
        1.0,
        -0.001098628627e-2,
        0.2734510407e-4,
        -0.2073370639e-5,
        0.2093887211e-6,
    ];

    /// Asymptotic expansion (|x| >= 8): Q polynomial
    pub const ASYMP_Q: [f64; 5] = [
        -0.01562499995,
        0.1430488765e-3,
        -0.6911147651e-5,
        0.7621095161e-6,
        -0.934935152e-7,
    ];

    // f32 versions
    pub const SMALL_P_F32: [f32; 6] = [
        57568490574.0,
        -13362590354.0,
        651619640.7,
        -11214424.18,
        77392.33017,
        -184.9052456,
    ];

    pub const SMALL_Q_F32: [f32; 6] = [
        57568490411.0,
        1029532985.0,
        9494680.718,
        59272.64853,
        267.8532712,
        1.0,
    ];

    pub const ASYMP_P_F32: [f32; 5] = [
        1.0,
        -0.001098628627e-2,
        0.2734510407e-4,
        -0.2073370639e-5,
        0.2093887211e-6,
    ];

    pub const ASYMP_Q_F32: [f32; 5] = [
        -0.01562499995,
        0.1430488765e-3,
        -0.6911147651e-5,
        0.7621095161e-6,
        -0.934935152e-7,
    ];

    pub const FRAC_PI_4: f64 = std::f64::consts::FRAC_PI_4;
    pub const TWO_OVER_PI: f64 = std::f64::consts::FRAC_2_PI;
    pub const FRAC_PI_4_F32: f32 = std::f32::consts::FRAC_PI_4;
    pub const TWO_OVER_PI_F32: f32 = std::f32::consts::FRAC_2_PI;
}

// ============================================================================
// Bessel J1 Coefficients
// ============================================================================

pub mod bessel_j1 {
    /// Small argument polynomial (|x| < 8): numerator (multiplied by x)
    pub const SMALL_P: [f64; 6] = [
        72362614232.0,
        -7895059235.0,
        242396853.1,
        -2972611.439,
        15704.48260,
        -30.16036606,
    ];

    /// Small argument polynomial (|x| < 8): denominator
    pub const SMALL_Q: [f64; 6] = [
        144725228442.0,
        2300535178.0,
        18583304.74,
        99447.43394,
        376.9991397,
        1.0,
    ];

    /// Asymptotic expansion (|x| >= 8): P polynomial
    pub const ASYMP_P: [f64; 5] = [
        1.0,
        0.183105e-2,
        -0.3516396496e-4,
        0.2457520174e-5,
        -0.240337019e-6,
    ];

    /// Asymptotic expansion (|x| >= 8): Q polynomial
    pub const ASYMP_Q: [f64; 5] = [
        0.04687499995,
        -0.2002690873e-3,
        0.8449199096e-5,
        -0.88228987e-6,
        0.105787412e-6,
    ];

    // f32 versions
    pub const SMALL_P_F32: [f32; 6] = [
        72362614232.0,
        -7895059235.0,
        242396853.1,
        -2972611.439,
        15704.48260,
        -30.16036606,
    ];

    pub const SMALL_Q_F32: [f32; 6] = [
        144725228442.0,
        2300535178.0,
        18583304.74,
        99447.43394,
        376.9991397,
        1.0,
    ];

    pub const ASYMP_P_F32: [f32; 5] = [
        1.0,
        0.183105e-2,
        -0.3516396496e-4,
        0.2457520174e-5,
        -0.240337019e-6,
    ];

    pub const ASYMP_Q_F32: [f32; 5] = [
        0.04687499995,
        -0.2002690873e-3,
        0.8449199096e-5,
        -0.88228987e-6,
        0.105787412e-6,
    ];

    pub const FRAC_3PI_4: f64 = 3.0 * std::f64::consts::FRAC_PI_4;
    pub const TWO_OVER_PI: f64 = std::f64::consts::FRAC_2_PI;
    pub const FRAC_3PI_4_F32: f32 = 3.0 * std::f32::consts::FRAC_PI_4;
    pub const TWO_OVER_PI_F32: f32 = std::f32::consts::FRAC_2_PI;
}

// ============================================================================
// Modified Bessel I0 Coefficients
// ============================================================================

pub mod bessel_i0 {
    /// Asymptotic expansion coefficients for large arguments
    pub const ASYMP: [f64; 7] = [
        1.0,
        0.125,
        0.0703125,
        0.0732421875,
        0.1121520996,
        0.2271080017,
        0.5725014209,
    ];

    pub const ASYMP_F32: [f32; 7] = [
        1.0,
        0.125,
        0.0703125,
        0.0732421875,
        0.1121520996,
        0.2271080017,
        0.5725014209,
    ];

    /// Threshold for switching between power series and asymptotic
    pub const THRESHOLD: f64 = 15.0;
    pub const THRESHOLD_F32: f32 = 15.0;
}

// ============================================================================
// Modified Bessel I1 Coefficients
// ============================================================================

pub mod bessel_i1 {
    /// Asymptotic expansion coefficients for large arguments
    pub const ASYMP: [f64; 7] = [
        1.0,
        -0.375,
        -0.1171875,
        -0.1025390625,
        -0.1441955566,
        -0.2775764465,
        -0.6765925884,
    ];

    pub const ASYMP_F32: [f32; 7] = [
        1.0,
        -0.375,
        -0.1171875,
        -0.1025390625,
        -0.1441955566,
        -0.2775764465,
        -0.6765925884,
    ];

    /// Threshold for switching between power series and asymptotic
    pub const THRESHOLD: f64 = 15.0;
    pub const THRESHOLD_F32: f32 = 15.0;
}
