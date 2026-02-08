//! Runtime backends for tensor computation
//!
//! This module defines the `Runtime` trait and provides implementations
//! for different compute backends (CPU, CUDA, WebGPU).
//!
//! # Architecture
//!
//! ```text
//! Runtime (backend identity)
//! ├── Device (identifies a specific GPU/CPU)
//! ├── Client (dispatches operations, owns stream/queue)
//! ├── Allocator (memory management with freeze support)
//! └── RawHandle (escape hatch for custom kernels)
//! ```

mod allocator;
pub mod helpers;
pub mod kernel;
pub mod shape_ops;
pub mod sparse_utils;
pub mod statistics_common;
pub mod traits;

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "wgpu")]
pub mod wgpu;

// CPU fallback utilities for GPU backends
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub mod fallback;

pub use allocator::{AllocGuard, Allocator, DefaultAllocator};
pub use helpers::{
    compute_broadcast_shape, ensure_contiguous, normalize_dim, validate_arange,
    validate_binary_dtypes, validate_eye, validate_linspace,
};
pub use traits::{Device, Runtime, RuntimeClient};

// ============================================================================
// Shared Helpers
// ============================================================================

/// Compute contiguous (row-major) strides for a given shape.
///
/// For a shape `[d0, d1, d2, ...]`, the strides are computed as:
/// - `strides[i] = product of dims[i+1..]`
/// - Last dimension always has stride 1
///
/// # Example
///
/// ```
/// use numr::runtime::compute_contiguous_strides;
///
/// assert_eq!(compute_contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
/// assert_eq!(compute_contiguous_strides(&[5]), vec![1]);
/// assert_eq!(compute_contiguous_strides(&[]), vec![]);
/// ```
#[inline]
pub fn compute_contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
