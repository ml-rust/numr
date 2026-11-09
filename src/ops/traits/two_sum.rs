//! Rounded addition and its represented residual.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Elementwise error-free addition under the stated arithmetic envelope.
pub trait TwoSumOps<R: Runtime> {
    /// Returns `(sum, residual)` using separate rounded additions and subtractions.
    ///
    /// Inputs broadcast and retain their F32 or F64 dtype and device.
    /// `sum` equals the backend's ordinary rounded `a + b`.
    /// In exact arithmetic, `sum + residual == a + b` for finite operands and intermediates.
    /// This identity requires round-to-nearest, ties-to-even, and gradual underflow at every operation.
    /// Flush-to-zero, other rounding modes, and reassociation invalidate the identity.
    /// The operation does not detect or change the backend arithmetic envelope.
    ///
    /// Nonfinite operands or overflowing intermediates propagate backend IEEE arithmetic.
    /// Under IEEE arithmetic, infinite or NaN sums produce a NaN residual.
    /// No exact decomposition is promised outside the finite arithmetic envelope.
    ///
    /// # Errors
    ///
    /// Rejects mismatched dtypes, incompatible shapes, or devices differing from the client.
    /// Rejects dtypes other than F32/F64 and backend-unsupported dtypes.
    fn two_sum(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<(Tensor<R>, Tensor<R>)>;
}
