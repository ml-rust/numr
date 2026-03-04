//! Mathematical constants and Lanczos coefficients used by special functions.

// ============================================================================
// Mathematical Constants
// ============================================================================

/// Square root of pi: √π ≈ 1.7724538509055159
pub const SQRT_PI: f64 = 1.7724538509055160272981674833411451827975;

/// 2 / √π ≈ 1.1283791670955126 (used in erf)
pub const TWO_OVER_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;

/// Euler-Mascheroni constant: γ ≈ 0.5772156649015329
pub const EULER_MASCHERONI: f64 = 0.5772156649015328606065120900824024310422;

/// ln(√(2π)) ≈ 0.9189385332046727 (used in Stirling's approximation)
pub const LN_SQRT_2PI: f64 = 0.9189385332046727417803297364056176398614;

// ============================================================================
// Lanczos Coefficients for Gamma Function
// ============================================================================

/// Lanczos approximation coefficients (g=7, n=9).
pub const LANCZOS_G: f64 = 7.0;

/// Lanczos coefficients for g=7.
pub const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];
