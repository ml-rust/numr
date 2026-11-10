//! Checked elementwise midpoint for binary floating-point values.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Elementwise midpoint operations for F32 and F64 tensors.
pub trait MidpointOps<R: Runtime> {
    /// Returns the average of corresponding input values without avoidable sum overflow.
    ///
    /// Inputs broadcast and retain their dtype and device. Small equal subnormal values
    /// remain represented under round-to-nearest ties-to-even with gradual underflow.
    /// Signed zeros follow ordinary rounded addition. NaN propagates. Opposite infinities
    /// return NaN, and equal infinities retain their sign.
    ///
    /// The runtime must declare [`crate::runtime::FloatArithmeticContract::Ieee754Basic`].
    /// Callers must preserve its required active arithmetic environment.
    /// This operation never changes the environment or substitutes another runtime.
    ///
    /// The finite range behavior follows [Rust scalar midpoint](https://doc.rust-lang.org/std/primitive.f64.html#method.midpoint).
    ///
    /// # Errors
    ///
    /// Rejects mismatched dtypes, incompatible shapes, and devices differing from the client.
    /// Rejects dtypes other than F32/F64, unsupported compute dtypes, and insufficient arithmetic contracts.
    fn midpoint(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;
}
