//! Polynomial multiplication via convolution

use super::{DTypeSupport, convolve_impl};
use crate::algorithm::fft::FftAlgorithms;
use crate::algorithm::polynomial::helpers::{
    validate_polynomial_coeffs, validate_polynomial_dtype,
};
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ComplexOps, IndexingOps, ReduceOps, ShapeOps, UtilityOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Multiply two polynomials via convolution
///
/// # Algorithm
///
/// For polynomials a(x) and b(x), c(x) = a(x) * b(x) where:
/// c[k] = Σᵢ a[i] * b[k-i] for valid i
///
/// Result has length: len(a) + len(b) - 1
///
/// # Implementation
///
/// Uses optimized convolution that automatically selects:
/// - Direct convolution for small polynomials (n*m < 64)
/// - FFT-based convolution for large polynomials (O(n log n))
///
/// All operations stay on-device without GPU↔CPU transfers.
pub fn polymul_impl<R, C>(
    client: &C,
    a: &Tensor<R>,
    b: &Tensor<R>,
    dtype_support: DTypeSupport,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + ReduceOps<R>
        + FftAlgorithms<R>
        + ComplexOps<R>,
{
    validate_polynomial_dtype(a.dtype())?;
    validate_polynomial_dtype(b.dtype())?;
    dtype_support.check(a.dtype(), "polymul")?;

    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }

    validate_polynomial_coeffs(a.shape())?;
    validate_polynomial_coeffs(b.shape())?;

    // Delegate to optimized convolution
    convolve_impl(client, a, b, dtype_support)
}
