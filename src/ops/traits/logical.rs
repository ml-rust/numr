//! Logical operations trait for boolean tensors.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

use super::TensorOps;

/// Logical operations trait for boolean tensors
///
/// All operations work on U8 tensors where 0 = false, non-zero = true.
pub trait LogicalOps<R: Runtime>: TensorOps<R> {
    /// Element-wise logical AND: a && b
    fn logical_and(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise logical OR: a || b
    fn logical_or(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise logical XOR: a ^ b
    fn logical_xor(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Element-wise logical NOT: !a
    fn logical_not(&self, a: &Tensor<R>) -> Result<Tensor<R>>;
}
