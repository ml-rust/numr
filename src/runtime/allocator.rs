//! Memory allocator traits and default implementation
//!
//! The Allocator trait provides memory management with optional "freeze" support
//! for graph capture scenarios (e.g., CUDA Graphs).

/// Memory allocator trait for runtime backends
///
/// Allocators manage device memory with optional support for "freezing" -
/// a mode where allocations are captured for graph replay.
pub trait Allocator: Clone + Send + Sync {
    /// Allocate memory of given size
    ///
    /// Returns a device pointer (u64) that can be used for operations.
    fn allocate(&self, size_bytes: usize) -> u64;

    /// Deallocate memory
    fn deallocate(&self, ptr: u64, size_bytes: usize);

    /// Freeze the allocator for graph capture
    ///
    /// After freezing, allocations may be captured for replay.
    /// Not all allocators support this (returns false by default).
    fn freeze(&self) -> bool {
        false
    }

    /// Unfreeze the allocator
    fn unfreeze(&self) {
        // Default: no-op
    }

    /// Check if the allocator is frozen
    fn is_frozen(&self) -> bool {
        false
    }

    /// Get the total allocated bytes
    fn allocated_bytes(&self) -> usize {
        0 // Default: tracking not supported
    }
}

/// Default allocator that delegates to Runtime methods
///
/// This is a simple allocator that just calls the runtime's allocate/deallocate.
/// It doesn't support freezing or memory tracking.
#[derive(Clone, Debug)]
pub struct DefaultAllocator<D> {
    device: D,
    allocate_fn: fn(usize, &D) -> u64,
    deallocate_fn: fn(u64, usize, &D),
}

impl<D: Clone + Send + Sync> DefaultAllocator<D> {
    /// Create a new default allocator
    pub fn new(
        device: D,
        allocate_fn: fn(usize, &D) -> u64,
        deallocate_fn: fn(u64, usize, &D),
    ) -> Self {
        Self {
            device,
            allocate_fn,
            deallocate_fn,
        }
    }

    /// Get the device this allocator is associated with
    pub fn device(&self) -> &D {
        &self.device
    }
}

impl<D: Clone + Send + Sync> Allocator for DefaultAllocator<D> {
    fn allocate(&self, size_bytes: usize) -> u64 {
        (self.allocate_fn)(size_bytes, &self.device)
    }

    fn deallocate(&self, ptr: u64, size_bytes: usize) {
        (self.deallocate_fn)(ptr, size_bytes, &self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_allocator_trait_bounds() {
        fn assert_allocator<A: Allocator>() {}
        assert_allocator::<DefaultAllocator<()>>();
    }
}
