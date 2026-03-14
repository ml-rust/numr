//! Scalar operations trait for tensor-scalar operations.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

use super::TensorOps;

/// Scalar operations trait for tensor-scalar operations
pub trait ScalarOps<R: Runtime>: TensorOps<R> {
    /// Add scalar to tensor: a + scalar
    fn add_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;

    /// Subtract scalar from tensor: a - scalar
    fn sub_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;

    /// Multiply tensor by scalar: a * scalar
    fn mul_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;

    /// Divide tensor by scalar: a / scalar
    fn div_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;

    /// Raise tensor to scalar power: a^scalar
    fn pow_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;

    /// Reverse subtract: scalar - a
    fn rsub_scalar(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>>;

    /// Fused multiply-add scalar: a * scale + bias
    ///
    /// Applies an affine transform to each element in a single pass.
    /// Common in normalization (scale + shift) and quantization.
    fn fused_mul_add_scalar(&self, a: &Tensor<R>, scale: f64, bias: f64) -> Result<Tensor<R>>;
}
