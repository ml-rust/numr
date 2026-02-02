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
}
