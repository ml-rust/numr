//! Shared helper functions for runtime backends
//!
//! This module contains helper functions that are used across multiple backends
//! (CPU, CUDA, WebGPU) to reduce code duplication and maintain consistency.

use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

// ============================================================================
// Utility Operation Validation
// ============================================================================

/// Validate arange parameters and compute the number of elements.
///
/// # Arguments
///
/// * `start` - Start value (inclusive)
/// * `stop` - Stop value (exclusive)
/// * `step` - Step between consecutive values
///
/// # Returns
///
/// The number of elements in the resulting tensor, or an error if parameters are invalid.
///
/// # Errors
///
/// * `InvalidArgument` if step is zero
/// * `InvalidArgument` if step direction doesn't match start→stop direction
#[inline]
pub fn validate_arange(start: f64, stop: f64, step: f64) -> Result<usize> {
    // Step cannot be zero
    if step == 0.0 {
        return Err(Error::InvalidArgument {
            arg: "step",
            reason: "step cannot be zero".to_string(),
        });
    }

    // Step direction must match start→stop direction
    if (stop > start && step < 0.0) || (stop < start && step > 0.0) {
        return Err(Error::InvalidArgument {
            arg: "step",
            reason: "step sign must match direction from start to stop".to_string(),
        });
    }

    // Calculate number of elements
    let numel = if start == stop {
        0
    } else {
        ((stop - start) / step).ceil() as usize
    };

    Ok(numel)
}

/// Validate linspace parameters.
///
/// # Arguments
///
/// * `steps` - Number of points to generate
///
/// # Returns
///
/// Ok(()) if parameters are valid. Note: steps=0 is valid (produces empty tensor).
#[inline]
pub fn validate_linspace(steps: usize) -> Result<()> {
    // All values are valid for linspace; steps=0 produces empty tensor
    // steps=1 produces single-element tensor with value=start
    let _ = steps;
    Ok(())
}

/// Validate eye (identity matrix) parameters.
///
/// # Arguments
///
/// * `n` - Number of rows
/// * `m` - Number of columns (if None, defaults to n)
///
/// # Returns
///
/// The (rows, cols) dimensions of the identity matrix.
#[inline]
pub fn validate_eye(n: usize, m: Option<usize>) -> (usize, usize) {
    let cols = m.unwrap_or(n);
    (n, cols)
}

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
