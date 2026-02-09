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
pub(crate) mod helpers;
pub(crate) mod shape_ops;
#[cfg(feature = "sparse")]
pub(crate) mod sparse_utils;
pub(crate) mod statistics_common;
pub mod traits;

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "wgpu")]
pub mod wgpu;

// CPU fallback utilities for GPU backends
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub(crate) mod fallback;

#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub(crate) use allocator::AllocGuard;
pub use allocator::Allocator;
pub(crate) use allocator::DefaultAllocator;
pub(crate) use helpers::{
    compute_broadcast_shape, ensure_contiguous, normalize_dim, validate_arange,
    validate_binary_dtypes, validate_eye,
};
pub use traits::{Device, Runtime, RuntimeClient};

// ============================================================================
// Shared Helpers
// ============================================================================

#[cfg(any(feature = "cuda", feature = "wgpu"))]
/// Compute contiguous (row-major) strides for a given shape.
///
/// For a shape `[d0, d1, d2, ...]`, the strides are computed as:
/// - `strides[i] = product of dims[i+1..]`
/// - Last dimension always has stride 1
#[inline]
pub(crate) fn compute_contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
