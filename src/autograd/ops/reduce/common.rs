//! Shared utilities for reduction backward implementations

use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Ensure a tensor is contiguous, making a copy if necessary.
#[inline]
pub(super) fn ensure_contiguous<R: Runtime>(tensor: Tensor<R>) -> Tensor<R> {
    if tensor.is_contiguous() {
        tensor
    } else {
        tensor.contiguous()
    }
}
