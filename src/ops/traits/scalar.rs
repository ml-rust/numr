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
}
