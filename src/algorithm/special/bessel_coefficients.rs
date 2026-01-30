//! Polynomial coefficients for Bessel function implementations.
//!
//! These coefficients are shared across CPU, CUDA, and WebGPU backends
//! to ensure numerical consistency. Source: Numerical Recipes, 3rd Edition.

/// π/4 for asymptotic expansion phase
pub const FRAC_PI_4: f64 = std::f64::consts::FRAC_PI_4;

/// 3π/4 for J1/Y1 asymptotic expansion phase
pub const FRAC_3PI_4: f64 = 2.356_194_490_192_345;

/// 2/π ≈ 0.636619772 (used in Bessel asymptotic expansions)
pub const TWO_OVER_PI: f64 = std::f64::consts::FRAC_2_PI;

// ============================================================================
// J0: Bessel function of the first kind, order 0
// ============================================================================

/// J0 small argument numerator coefficients (|x| < 8)
pub const J0_SMALL_P: [f64; 6] = [
    57568490574.0,
    -13362590354.0,
    651619640.7,
    -11214424.18,
    77392.33017,
    -184.9052456,
];

/// J0 small argument denominator coefficients (|x| < 8)
pub const J0_SMALL_Q: [f64; 6] = [
    57568490411.0,
    1029532985.0,
    9494680.718,
    59272.64853,
    267.8532712,
    1.0,
];

/// J0 asymptotic P polynomial (|x| >= 8)
pub const J0_ASYMP_P: [f64; 5] = [
    1.0,
    -0.1098628627e-2,
    0.2734510407e-4,
    -0.2073370639e-5,
    0.2093887211e-6,
];

/// J0 asymptotic Q polynomial (|x| >= 8)
pub const J0_ASYMP_Q: [f64; 5] = [
    -0.1562499995e-1,
    0.1430488765e-3,
    -0.6911147651e-5,
    0.7621095161e-6,
    -0.934945152e-7,
];

// ============================================================================
// J1: Bessel function of the first kind, order 1
// ============================================================================

/// J1 small argument numerator coefficients (|x| < 8)
pub const J1_SMALL_P: [f64; 6] = [
    72362614232.0,
    -7895059235.0,
    242396853.1,
    -2972611.439,
    15704.48260,
    -30.16036606,
];

/// J1 small argument denominator coefficients (|x| < 8)
pub const J1_SMALL_Q: [f64; 6] = [
    144725228442.0,
    2300535178.0,
    18583304.74,
    99447.43394,
    376.9991397,
    1.0,
];

/// J1 asymptotic P polynomial (|x| >= 8)
pub const J1_ASYMP_P: [f64; 5] = [
    1.0,
    0.183105e-2,
    -0.3516396496e-4,
    0.2457520174e-5,
    -0.240337019e-6,
];

/// J1 asymptotic Q polynomial (|x| >= 8)
pub const J1_ASYMP_Q: [f64; 5] = [
    0.04687499995,
    -0.2002690873e-3,
    0.8449199096e-5,
    -0.88228987e-6,
    0.105787412e-6,
];

// ============================================================================
// Y0: Bessel function of the second kind, order 0
// ============================================================================

/// Y0 small argument numerator coefficients (x < 8)
pub const Y0_SMALL_P: [f64; 6] = [
    -2957821389.0,
    7062834065.0,
    -512359803.6,
    10879881.29,
    -86327.92757,
    228.4622733,
];

/// Y0 small argument denominator coefficients (x < 8)
pub const Y0_SMALL_Q: [f64; 6] = [
    40076544269.0,
    745249964.8,
    7189466.438,
    47447.26470,
    226.1030244,
    1.0,
];

// ============================================================================
// Y1: Bessel function of the second kind, order 1
// ============================================================================

/// Y1 small argument numerator coefficients (x < 8)
pub const Y1_SMALL_P: [f64; 6] = [
    -0.4900604943e13,
    0.1275274390e13,
    -0.5153438139e11,
    0.7349264551e9,
    -0.4237922726e7,
    0.8511937935e4,
];

/// Y1 small argument denominator coefficients (x < 8)
pub const Y1_SMALL_Q: [f64; 7] = [
    0.2499580570e14,
    0.4244198890e12,
    0.3733650367e10,
    0.2245904002e8,
    0.1020426050e6,
    0.3549632885e3,
    1.0,
];

// ============================================================================
// I0: Modified Bessel function of the first kind, order 0
// ============================================================================

/// I0 asymptotic expansion coefficients
pub const I0_ASYMP: [f64; 7] = [
    1.0,
    1.25e-01,
    7.03125e-02,
    7.324_218_75e-2,
    1.121_520_996_093_75e-1,
    2.271_080_017_089_844e-1,
    5.725_014_209_747_314e-1,
];

// ============================================================================
// I1: Modified Bessel function of the first kind, order 1
// ============================================================================

/// I1 asymptotic expansion coefficients
pub const I1_ASYMP: [f64; 7] = [
    1.0,
    -3.75e-01,
    -1.171875e-01,
    -1.025_390_625e-1,
    -1.441_955_566_406_25e-1,
    -2.775_764_465_332_031e-1,
    -6.765_925_884_246_826e-1,
];

// ============================================================================
// K0: Modified Bessel function of the second kind, order 0
// ============================================================================

/// K0 small argument polynomial coefficients (x <= 2)
pub const K0_SMALL: [f64; 7] = [
    -0.57721566,
    0.42278420,
    0.23069756,
    0.03488590,
    0.00262698,
    0.00010750,
    0.00000740,
];

/// K0 large argument polynomial coefficients (x > 2)
pub const K0_LARGE: [f64; 7] = [
    1.25331414,
    -0.07832358,
    0.02189568,
    -0.01062446,
    0.00587872,
    -0.00251540,
    0.00053208,
];

// ============================================================================
// K1: Modified Bessel function of the second kind, order 1
// ============================================================================

/// K1 small argument polynomial coefficients (x <= 2)
pub const K1_SMALL: [f64; 7] = [
    1.0,
    0.15443144,
    -0.67278579,
    -0.18156897,
    -0.01919402,
    -0.00110404,
    -0.00004686,
];

/// K1 large argument polynomial coefficients (x > 2)
pub const K1_LARGE: [f64; 7] = [
    1.25331414,
    0.23498619,
    -0.03655620,
    0.01504268,
    -0.00780353,
    0.00325614,
    -0.00068245,
];
