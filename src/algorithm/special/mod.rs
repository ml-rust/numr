//! Special mathematical functions for scientific computing
//!
//! This module defines traits for special functions required by probability
//! distributions, statistics, and scientific applications. These are critical
//! for solvr::stats to implement distributions like normal, gamma, beta, etc.
//!
//! # Functions Provided
//!
//! ## Error Functions (for normal distribution)
//! - [`erf`] - Error function
//! - [`erfc`] - Complementary error function (1 - erf(x))
//! - [`erfinv`] - Inverse error function
//!
//! ## Gamma Functions (for gamma, chi2, t, F distributions)
//! - [`gamma`] - Gamma function Γ(x)
//! - [`lgamma`] - Log-gamma function ln(Γ(x)) (numerically stable)
//! - [`digamma`] - Digamma function ψ(x) = Γ'(x)/Γ(x)
//!
//! ## Beta Functions (for beta distribution)
//! - [`beta`] - Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)
//! - [`betainc`] - Regularized incomplete beta function I_x(a,b)
//!
//! ## Incomplete Gamma (for gamma/chi2 CDF)
//! - [`gammainc`] - Lower regularized incomplete gamma P(a,x)
//! - [`gammaincc`] - Upper regularized incomplete gamma Q(a,x) = 1 - P(a,x)
//!
//! ## Bessel Functions
//! - [`bessel_j0`], [`bessel_j1`] - First kind J₀, J₁
//! - [`bessel_y0`], [`bessel_y1`] - Second kind Y₀, Y₁
//! - [`bessel_i0`], [`bessel_i1`] - Modified first kind I₀, I₁
//! - [`bessel_k0`], [`bessel_k1`] - Modified second kind K₀, K₁
//!
//! ## Elliptic Integrals
//! - [`ellipk`] - Complete elliptic integral of first kind K(m)
//! - [`ellipe`] - Complete elliptic integral of second kind E(m)
//!
//! ## Hypergeometric Functions
//! - [`hyp2f1`] - Gauss hypergeometric function ₂F₁(a, b; c; z)
//! - [`hyp1f1`] - Confluent hypergeometric function ₁F₁(a; b; z)
//!
//! ## Airy Functions
//! - [`airy_ai`] - Airy function of first kind Ai(x)
//! - [`airy_bi`] - Airy function of second kind Bi(x)
//!
//! ## Legendre Functions and Spherical Harmonics
//! - [`legendre_p`] - Legendre polynomial P_n(x)
//! - [`legendre_p_assoc`] - Associated Legendre function P_n^m(x)
//! - [`sph_harm`] - Real spherical harmonic Y_n^m(θ, φ)
//!
//! ## Fresnel Integrals
//! - [`fresnel_s`] - Fresnel sine integral S(x)
//! - [`fresnel_c`] - Fresnel cosine integral C(x)
//!
//! # Algorithm Sources
//!
//! Implementations follow well-established numerical algorithms:
//! - Cody's rational approximation for erf/erfc
//! - Lanczos approximation for gamma/lgamma
//! - Continued fraction expansion for incomplete gamma/beta
//! - Newton-Raphson iteration for inverse functions
//! - Numerical Recipes polynomial approximations for Bessel functions
//! - AGM method for elliptic integrals
//! - Power series with transformations for hypergeometric functions
//! - Power series and asymptotic expansions for Airy functions
//! - Three-term recurrence for Legendre polynomials

pub mod bessel_coefficients;
pub mod scalar;

pub use scalar::*;

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

// ============================================================================
// Special Functions Trait
// ============================================================================

/// Special mathematical functions for scientific computing.
///
/// All backends must implement these functions to enable solvr probability
/// distributions and statistical functions.
///
/// # Implementation Notes
///
/// - Functions operate element-wise on tensors
/// - Input validation (domain checks) should return appropriate errors
/// - Numerical stability is critical - use established algorithms
/// - GPU implementations can use the same algorithms as CPU
pub trait SpecialFunctions<R: Runtime> {
    // ========================================================================
    // Error Functions
    // ========================================================================

    /// Compute the error function element-wise.
    ///
    /// ```text
    /// erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
    /// ```
    ///
    /// # Properties
    /// - Domain: all real numbers
    /// - Range: (-1, 1)
    /// - erf(0) = 0
    /// - erf(∞) = 1, erf(-∞) = -1
    /// - erf(-x) = -erf(x) (odd function)
    fn erf(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the complementary error function element-wise.
    ///
    /// ```text
    /// erfc(x) = 1 - erf(x) = (2/√π) ∫ₓ^∞ e^(-t²) dt
    /// ```
    ///
    /// For large x, erf(x) ≈ 1 and computing 1 - erf(x) loses precision.
    /// erfc(x) computes the small tail directly, maintaining accuracy.
    fn erfc(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the inverse error function element-wise.
    ///
    /// Returns y such that erf(y) = x.
    ///
    /// # Properties
    /// - Domain: (-1, 1)
    /// - Range: all real numbers
    /// - erfinv(0) = 0
    fn erfinv(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Gamma Functions
    // ========================================================================

    /// Compute the gamma function element-wise.
    ///
    /// ```text
    /// Γ(x) = ∫₀^∞ t^(x-1) e^(-t) dt
    /// ```
    ///
    /// # Properties
    /// - Γ(n) = (n-1)! for positive integers
    /// - Γ(1) = 1, Γ(1/2) = √π
    /// - Has poles at non-positive integers (returns NaN/Inf)
    fn gamma(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the log-gamma function element-wise.
    ///
    /// ```text
    /// lgamma(x) = ln(|Γ(x)|)
    /// ```
    ///
    /// Γ(x) grows extremely fast (Γ(171) overflows F64).
    /// lgamma computes the logarithm directly without overflow.
    fn lgamma(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the digamma (psi) function element-wise.
    ///
    /// ```text
    /// ψ(x) = d/dx ln(Γ(x)) = Γ'(x)/Γ(x)
    /// ```
    fn digamma(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Beta Functions
    // ========================================================================

    /// Compute the beta function element-wise.
    ///
    /// ```text
    /// B(a, b) = Γ(a)Γ(b)/Γ(a+b)
    /// ```
    fn beta(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the regularized incomplete beta function element-wise.
    ///
    /// ```text
    /// I_x(a,b) = B(x;a,b)/B(a,b) = (1/B(a,b)) ∫₀ˣ t^(a-1)(1-t)^(b-1) dt
    /// ```
    fn betainc(&self, a: &Tensor<R>, b: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Incomplete Gamma Functions
    // ========================================================================

    /// Compute the lower regularized incomplete gamma function.
    ///
    /// ```text
    /// P(a, x) = γ(a,x)/Γ(a) = (1/Γ(a)) ∫₀ˣ t^(a-1) e^(-t) dt
    /// ```
    fn gammainc(&self, a: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the upper regularized incomplete gamma function.
    ///
    /// ```text
    /// Q(a, x) = 1 - P(a, x)
    /// ```
    fn gammaincc(&self, a: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the inverse of the lower regularized incomplete gamma function.
    ///
    /// Returns x such that P(a, x) = p.
    ///
    /// # Properties
    /// - Domain: p in [0, 1], a > 0
    /// - Range: x >= 0
    /// - gammaincinv(a, 0) = 0
    /// - gammaincinv(a, 1) = ∞
    fn gammaincinv(&self, a: &Tensor<R>, p: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the inverse of the regularized incomplete beta function.
    ///
    /// Returns x such that I_x(a, b) = p.
    ///
    /// # Properties
    /// - Domain: p in [0, 1], a > 0, b > 0
    /// - Range: x in [0, 1]
    /// - betaincinv(a, b, 0) = 0
    /// - betaincinv(a, b, 1) = 1
    fn betaincinv(&self, a: &Tensor<R>, b: &Tensor<R>, p: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Bessel Functions
    // ========================================================================

    /// Compute Bessel function of the first kind, order 0.
    ///
    /// J₀(0) = 1, even function, oscillates with decreasing amplitude.
    fn bessel_j0(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute Bessel function of the first kind, order 1.
    ///
    /// J₁(0) = 0, odd function, oscillates with decreasing amplitude.
    fn bessel_j1(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute Bessel function of the second kind, order 0 (Neumann function).
    ///
    /// Y₀(x) → -∞ as x → 0⁺. Domain: x > 0.
    fn bessel_y0(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute Bessel function of the second kind, order 1 (Neumann function).
    ///
    /// Y₁(x) → -∞ as x → 0⁺. Domain: x > 0.
    fn bessel_y1(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute modified Bessel function of the first kind, order 0.
    ///
    /// I₀(0) = 1, even function, grows exponentially.
    fn bessel_i0(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute modified Bessel function of the first kind, order 1.
    ///
    /// I₁(0) = 0, odd function, grows exponentially.
    fn bessel_i1(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute modified Bessel function of the second kind, order 0.
    ///
    /// K₀(x) → ∞ as x → 0⁺. Domain: x > 0. Decays exponentially.
    fn bessel_k0(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute modified Bessel function of the second kind, order 1.
    ///
    /// K₁(x) → ∞ as x → 0⁺. Domain: x > 0. Decays exponentially.
    fn bessel_k1(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Elliptic Integrals
    // ========================================================================

    /// Compute the complete elliptic integral of the first kind K(m).
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
    fn ellipk(&self, m: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the complete elliptic integral of the second kind E(m).
    ///
    /// ```text
    /// E(m) = ∫₀^(π/2) √(1 - m·sin²θ) dθ
    /// ```
    ///
    /// # Properties
    /// - Domain: m ∈ [0, 1]
    /// - E(0) = π/2
    /// - E(1) = 1
    fn ellipe(&self, m: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Hypergeometric Functions
    // ========================================================================

    /// Compute the Gauss hypergeometric function ₂F₁(a, b; c; z).
    ///
    /// ```text
    /// ₂F₁(a, b; c; z) = Σ_{n=0}^∞ (a)_n (b)_n / ((c)_n n!) z^n
    /// ```
    ///
    /// # Properties
    /// - Converges for |z| < 1
    /// - ₂F₁(a, b; c; 0) = 1
    ///
    /// # Arguments
    /// - a, b, c: Scalar parameters
    /// - z: Input tensor
    fn hyp2f1(&self, a: f64, b: f64, c: f64, z: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the confluent hypergeometric function ₁F₁(a; b; z) (Kummer's M).
    ///
    /// ```text
    /// ₁F₁(a; b; z) = M(a, b, z) = Σ_{n=0}^∞ (a)_n / ((b)_n n!) z^n
    /// ```
    ///
    /// # Properties
    /// - ₁F₁(a; b; 0) = 1
    /// - ₁F₁(0; b; z) = 1
    /// - Entire function in z
    fn hyp1f1(&self, a: f64, b: f64, z: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Airy Functions
    // ========================================================================

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
    fn airy_ai(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

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
    fn airy_bi(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Legendre Functions
    // ========================================================================

    /// Compute the Legendre polynomial P_n(x).
    ///
    /// # Properties
    /// - Domain: x ∈ [-1, 1]
    /// - P_n(1) = 1
    /// - P_n(-1) = (-1)^n
    /// - P_0(x) = 1, P_1(x) = x
    fn legendre_p(&self, n: i32, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the associated Legendre function P_n^m(x).
    ///
    /// Uses Condon-Shortley phase convention (factor of (-1)^m).
    ///
    /// # Properties
    /// - Domain: x ∈ [-1, 1], 0 ≤ m ≤ n
    /// - P_n^0(x) = P_n(x)
    fn legendre_p_assoc(&self, n: i32, m: i32, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the real spherical harmonic Y_n^m(θ, φ).
    ///
    /// Returns the real-valued spherical harmonic with Schmidt semi-normalization.
    /// - m > 0: Y_n^m ∝ P_n^m(cos θ) cos(mφ)
    /// - m = 0: Y_n^0 ∝ P_n(cos θ)
    /// - m < 0: Y_n^m ∝ P_n^|m|(cos θ) sin(|m|φ)
    ///
    /// # Arguments
    /// - n: degree (n ≥ 0)
    /// - m: order (-n ≤ m ≤ n)
    /// - theta: polar angle θ ∈ [0, π] (colatitude)
    /// - phi: azimuthal angle φ ∈ [0, 2π)
    fn sph_harm(&self, n: i32, m: i32, theta: &Tensor<R>, phi: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Fresnel Integrals
    // ========================================================================

    /// Compute the Fresnel sine integral S(x).
    ///
    /// ```text
    /// S(x) = ∫₀ˣ sin(π t²/2) dt
    /// ```
    ///
    /// # Properties
    /// - S(0) = 0
    /// - S(∞) = 0.5
    /// - S(-x) = -S(x) (odd function)
    fn fresnel_s(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the Fresnel cosine integral C(x).
    ///
    /// ```text
    /// C(x) = ∫₀ˣ cos(π t²/2) dt
    /// ```
    ///
    /// # Properties
    /// - C(0) = 0
    /// - C(∞) = 0.5
    /// - C(-x) = -C(x) (odd function)
    fn fresnel_c(&self, x: &Tensor<R>) -> Result<Tensor<R>>;
}

// ============================================================================
// Validation Helpers
// ============================================================================

/// Validate that dtype is suitable for special functions.
pub fn validate_special_dtype(dtype: crate::dtype::DType) -> Result<()> {
    use crate::dtype::DType;
    use crate::error::Error;

    match dtype {
        DType::F32 | DType::F64 => Ok(()),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "special function (requires F32 or F64)",
        }),
    }
}

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
