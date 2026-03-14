//! Trait for runtime clients that handle operation dispatch

use super::Runtime;

/// Trait for runtime clients that handle operation dispatch
pub trait RuntimeClient<R: Runtime>: Clone + Send + Sync {
    /// Get the device this client operates on
    fn device(&self) -> &R::Device;

    /// Synchronize: wait for all pending operations to complete
    fn synchronize(&self);

    /// Get the allocator for this client
    fn allocator(&self) -> &R::Allocator;

    /// Get the raw CUDA stream handle for compute-communication overlap.
    ///
    /// Returns `Some(handle)` on CUDA backends where the handle is the
    /// `CUstream` pointer cast to `u64`. Returns `None` on CPU/WebGPU.
    fn compute_stream_handle(&self) -> Option<u64> {
        None
    }
}
