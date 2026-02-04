//! Result types for polynomial operations

use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Result of polynomial root finding
///
/// Complex roots are stored as separate real and imaginary tensors to support
/// backends without native complex number support (e.g., WebGPU).
///
/// # Root Storage
///
/// For a polynomial of degree n, both tensors have shape [n].
///
/// - Real roots: `roots_imag[i] = 0`
/// - Complex roots come in conjugate pairs:
///   - `roots_real[j] + i * roots_imag[j]`
///   - `roots_real[j+1] - i * roots_imag[j+1]`
///
/// # Example
///
/// ```ignore
/// // Polynomial: x² - 2x + 2 (roots: 1 ± i)
/// let roots = client.polyroots(&coeffs)?;
/// // roots_real = [1.0, 1.0]
/// // roots_imag = [1.0, -1.0]
/// ```
pub struct PolynomialRoots<R: Runtime> {
    /// Real parts of roots [n]
    pub roots_real: Tensor<R>,

    /// Imaginary parts of roots [n]
    /// Zero for real roots, non-zero for complex conjugate pairs
    pub roots_imag: Tensor<R>,
}
