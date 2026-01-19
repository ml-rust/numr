//! Shared helper functions for runtime backends
//!
//! This module contains helper functions that are used across multiple backends
//! (CPU, CUDA, WebGPU) to reduce code duplication and maintain consistency.

use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Ensure a tensor is contiguous in memory.
///
/// If the tensor is already contiguous (elements laid out consecutively in memory),
/// returns a clone (zero-copy, just increments Arc refcount). Otherwise,
/// creates a new contiguous copy of the data by materializing the strided view.
///
/// This is typically required before passing tensors to backend kernels
/// that expect contiguous memory layout, or for operations that need
/// to work with the underlying data pointer directly.
///
/// # Parameters
///
/// * `tensor` - The input tensor which may or may not be contiguous
///
/// # Returns
///
/// A new tensor that is guaranteed to be contiguous. If the input was already
/// contiguous, this is zero-copy (just clones the Arc). Otherwise, data is copied.
///
/// # Example
///
/// ```ignore
/// use numr::runtime::helpers::ensure_contiguous;
/// use numr::tensor::Tensor;
/// use numr::runtime::cpu::CpuRuntime;
///
/// let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device);
/// let b = a.transpose()?;  // Not contiguous after transpose
/// let c = ensure_contiguous(&b);  // Makes a contiguous copy
/// assert!(c.is_contiguous());
/// ```
#[inline]
pub fn ensure_contiguous<R: Runtime>(tensor: &Tensor<R>) -> Tensor<R> {
    if tensor.is_contiguous() {
        tensor.clone()
    } else {
        tensor.contiguous()
    }
}
