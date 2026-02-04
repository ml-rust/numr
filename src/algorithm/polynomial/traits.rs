//! Trait definition for polynomial operations

use super::types::PolynomialRoots;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Algorithmic contract for polynomial operations
///
/// All backends implementing polynomial operations MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
///
/// # Coefficient Convention
///
/// Polynomials are represented as 1D tensors in ascending power order:
/// - `coeffs[0]` = constant term (c₀)
/// - `coeffs[n]` = leading coefficient (cₙ)
/// - p(x) = c₀ + c₁x + c₂x² + ... + cₙxⁿ
pub trait PolynomialAlgorithms<R: Runtime> {
    /// Find roots of a polynomial via companion matrix eigendecomposition
    ///
    /// # Algorithm
    ///
    /// 1. Build companion matrix C from normalized coefficients
    /// 2. Compute eigenvalues of C using `eig_decompose`
    /// 3. Return eigenvalues as roots
    ///
    /// The companion matrix for monic polynomial xⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₀ is:
    ///
    /// ```text
    /// C = [ 0   0   ...  0  -a₀  ]
    ///     [ 1   0   ...  0  -a₁  ]
    ///     [ 0   1   ...  0  -a₂  ]
    ///     [ .   .   ...  .   .   ]
    ///     [ 0   0   ...  1  -aₙ₋₁]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `coeffs` - Polynomial coefficients [n+1] in ascending order
    ///
    /// # Returns
    ///
    /// [`PolynomialRoots`] containing n roots as separate real/imaginary tensors
    ///
    /// # Errors
    ///
    /// - If leading coefficient is zero (degenerate polynomial)
    /// - If tensor is not 1D or empty
    fn polyroots(&self, coeffs: &Tensor<R>) -> Result<PolynomialRoots<R>>;

    /// Evaluate polynomial at given points using Horner's method
    ///
    /// # Algorithm
    ///
    /// Horner's method for numerical stability:
    /// ```text
    /// result = cₙ
    /// for i in (n-1)..0:
    ///     result = result * x + cᵢ
    /// ```
    ///
    /// This requires only n multiplications and n additions, and is
    /// numerically stable.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - Polynomial coefficients [n+1] in ascending order
    /// * `x` - Points at which to evaluate [m] or any shape
    ///
    /// # Returns
    ///
    /// Tensor of same shape as `x` containing p(x) values
    fn polyval(&self, coeffs: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Construct polynomial coefficients from roots
    ///
    /// # Algorithm
    ///
    /// Iteratively convolve linear factors:
    /// ```text
    /// coeffs = [1]  // Start with constant polynomial
    /// for each root r:
    ///     if r is real:
    ///         coeffs = convolve(coeffs, [1, -r])
    ///     else:  // Complex conjugate pair
    ///         coeffs = convolve(coeffs, [|r|², -2*Re(r), 1])
    /// ```
    ///
    /// # Arguments
    ///
    /// * `roots_real` - Real parts of roots [n]
    /// * `roots_imag` - Imaginary parts of roots [n]
    ///
    /// # Returns
    ///
    /// Polynomial coefficients [n+1] in ascending order, normalized to monic
    /// (leading coefficient = 1)
    ///
    /// # Complex Conjugate Handling
    ///
    /// Complex roots must come in conjugate pairs. The function processes
    /// pairs together to produce real quadratic factors.
    fn polyfromroots(&self, roots_real: &Tensor<R>, roots_imag: &Tensor<R>) -> Result<Tensor<R>>;

    /// Multiply two polynomials via convolution
    ///
    /// # Algorithm
    ///
    /// Direct convolution (discrete polynomial multiplication):
    /// ```text
    /// c[k] = Σᵢ a[i] * b[k-i]  for valid indices i
    /// ```
    ///
    /// For polynomials a(x) and b(x), the product c(x) = a(x) * b(x) has:
    /// - Degree: deg(a) + deg(b)
    /// - Length: len(a) + len(b) - 1
    ///
    /// # Arguments
    ///
    /// * `a` - First polynomial coefficients [m] in ascending order
    /// * `b` - Second polynomial coefficients [n] in ascending order
    ///
    /// # Returns
    ///
    /// Product polynomial coefficients [m+n-1] in ascending order
    fn polymul(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;
}
