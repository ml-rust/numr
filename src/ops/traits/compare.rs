//! Comparison operations trait.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

use super::TensorOps;

/// Comparison operations trait
pub trait CompareOps<R: Runtime>: TensorOps<R> {
    /// Element-wise equality: a == b
    fn eq(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise inequality: a != b
    fn ne(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise less than: a < b
    fn lt(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise less than or equal: a <= b
    fn le(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise greater than: a > b
    fn gt(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise greater than or equal: a >= b
    fn ge(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;
}
