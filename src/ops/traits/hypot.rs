//! Stable elementwise Euclidean magnitude.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Stable elementwise Euclidean magnitude operations.
pub trait HypotOps<R: Runtime> {
    /// Computes `sqrt(a² + b²)` with scaling to avoid intermediate square overflow.
    ///
    /// Inputs broadcast and retain their floating dtype and device.
    /// Narrow floats compute in F32 and cast the result once.
    /// Zero pairs return positive zero. Infinity takes precedence over NaN.
    /// Otherwise, NaN propagates.
    ///
    /// # Errors
    ///
    /// Rejects different dtypes, incompatible shapes, and devices differing from the client.
    /// Rejects non-floating dtypes and dtypes unsupported by the backend or disabled features.
    fn hypot(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;
}
