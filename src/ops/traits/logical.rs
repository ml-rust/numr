//! Logical operations trait for boolean tensors.

use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Logical operations trait for boolean tensors
///
/// All operations work on U8 tensors where 0 = false, non-zero = true.
pub trait LogicalOps<R: Runtime> {
    /// Element-wise logical AND: a && b
    fn logical_and(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (a, b);
        Err(Error::NotImplemented {
            feature: "LogicalOps::logical_and",
        })
    }

    /// Element-wise logical OR: a || b
    fn logical_or(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (a, b);
        Err(Error::NotImplemented {
            feature: "LogicalOps::logical_or",
        })
    }

    /// Element-wise logical XOR: a ^ b
    fn logical_xor(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (a, b);
        Err(Error::NotImplemented {
            feature: "LogicalOps::logical_xor",
        })
    }

    /// Element-wise logical NOT: !a
    fn logical_not(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "LogicalOps::logical_not",
        })
    }
}
