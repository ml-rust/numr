//! Memory allocator traits and default implementation
//!
//! The Allocator trait provides memory management with optional "freeze" support
//! for graph capture scenarios (e.g., CUDA Graphs).

/// Allocation statistics for debugging and profiling
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AllocationStats {
    /// Total number of allocations made (cumulative)
    pub total_allocations: usize,
    /// Total bytes allocated (cumulative)
    pub total_bytes: usize,
    /// Number of allocations currently live (not yet deallocated)
    pub active_allocations: usize,
    /// Whether the allocator is currently frozen
    pub is_frozen: bool,
    /// Peak memory usage in bytes (high-water mark)
    pub peak_usage: usize,
}

/// Memory allocator trait for runtime backends
///
/// Allocators manage device memory with optional support for "freezing" -
/// a mode where allocations are captured for graph replay.
pub trait Allocator: Clone + Send + Sync {
    /// Allocate memory of given size
    ///
    /// Returns a device pointer (u64) that can be used for operations.
    /// Returns `Err(OutOfMemory)` if allocation fails.
    fn allocate(&self, size_bytes: usize) -> crate::error::Result<u64>;

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

    /// Get allocation statistics
    ///
    /// Returns detailed allocation stats including active count, peak usage,
    /// and frozen state. Default returns zeroed stats for allocators without tracking.
    fn stats(&self) -> AllocationStats {
        AllocationStats::default()
    }

    /// Reset allocator counters and reclaim pooled memory.
    ///
    /// When `active_allocations == 0`, this zeros out stats counters
    /// (total_allocations, total_bytes, peak_usage) and releases any
    /// internally pooled/cached buffers back to the OS or driver.
    ///
    /// # Errors
    ///
    /// Returns `Err(AllocatorBusy)` if `active_allocations > 0`.
    /// Caller must drop all tensors/storage referencing this allocator's
    /// memory before calling reset — active allocations mean live
    /// Storage references exist, and reclaiming that memory would
    /// cause use-after-free.
    fn reset(&self) -> crate::error::Result<()> {
        Ok(())
    }
}

/// Default allocator that delegates to Runtime methods
///
/// This is a simple allocator that just calls the runtime's allocate/deallocate.
/// It doesn't support freezing or memory tracking.
#[derive(Clone, Debug)]
pub struct DefaultAllocator<D> {
    device: D,
    allocate_fn: fn(usize, &D) -> crate::error::Result<u64>,
    deallocate_fn: fn(u64, usize, &D),
}

impl<D: Clone + Send + Sync> DefaultAllocator<D> {
    /// Create a new default allocator
    pub fn new(
        device: D,
        allocate_fn: fn(usize, &D) -> crate::error::Result<u64>,
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
    fn allocate(&self, size_bytes: usize) -> crate::error::Result<u64> {
        (self.allocate_fn)(size_bytes, &self.device)
    }

    fn deallocate(&self, ptr: u64, size_bytes: usize) {
        (self.deallocate_fn)(ptr, size_bytes, &self.device)
    }
}

/// Tracking allocator state (behind Arc<Mutex<...>> for thread-safe sharing)
#[derive(Debug)]
struct TrackingState<A: Allocator> {
    inner: A,
    total_allocations: usize,
    total_bytes: usize,
    active_allocations: usize,
    active_bytes: usize,
    peak_usage: usize,
    frozen: bool,
}

/// Allocator wrapper that tracks allocation statistics.
///
/// Wraps any `Allocator` implementation with proper tracking of active
/// allocations, total bytes, peak usage, and freeze/reset support.
///
/// Thread-safe via `Arc<Mutex<...>>` — cloning shares the same state.
///
/// # Example
///
/// ```ignore
/// let inner = DefaultAllocator::new(device, alloc_fn, dealloc_fn);
/// let tracking = TrackingAllocator::new(inner);
///
/// let ptr = tracking.allocate(1024)?;
/// assert_eq!(tracking.stats().active_allocations, 1);
/// assert_eq!(tracking.stats().active_bytes(), 1024);
///
/// tracking.deallocate(ptr, 1024);
/// assert_eq!(tracking.stats().active_allocations, 0);
///
/// tracking.reset()?; // succeeds: no active allocations
/// ```
#[derive(Debug)]
pub struct TrackingAllocator<A: Allocator> {
    state: std::sync::Arc<std::sync::Mutex<TrackingState<A>>>,
}

impl<A: Allocator> Clone for TrackingAllocator<A> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<A: Allocator> TrackingAllocator<A> {
    /// Create a new tracking allocator wrapping `inner`.
    pub fn new(inner: A) -> Self {
        Self {
            state: std::sync::Arc::new(std::sync::Mutex::new(TrackingState {
                inner,
                total_allocations: 0,
                total_bytes: 0,
                active_allocations: 0,
                active_bytes: 0,
                peak_usage: 0,
                frozen: false,
            })),
        }
    }

    /// Get the current number of live bytes (convenience for active_bytes in stats)
    pub fn active_bytes(&self) -> usize {
        let s = self.state.lock().unwrap();
        s.active_bytes
    }
}

impl<A: Allocator> Allocator for TrackingAllocator<A> {
    fn allocate(&self, size_bytes: usize) -> crate::error::Result<u64> {
        let mut s = self.state.lock().unwrap();
        if s.frozen {
            return Err(crate::error::Error::AllocatorFrozen);
        }
        let ptr = s.inner.allocate(size_bytes)?;
        s.total_allocations += 1;
        s.total_bytes += size_bytes;
        s.active_allocations += 1;
        s.active_bytes += size_bytes;
        if s.active_bytes > s.peak_usage {
            s.peak_usage = s.active_bytes;
        }
        Ok(ptr)
    }

    fn deallocate(&self, ptr: u64, size_bytes: usize) {
        let mut s = self.state.lock().unwrap();
        s.inner.deallocate(ptr, size_bytes);
        s.active_allocations = s.active_allocations.saturating_sub(1);
        s.active_bytes = s.active_bytes.saturating_sub(size_bytes);
    }

    fn freeze(&self) -> bool {
        let mut s = self.state.lock().unwrap();
        s.frozen = true;
        true
    }

    fn unfreeze(&self) {
        let mut s = self.state.lock().unwrap();
        s.frozen = false;
    }

    fn is_frozen(&self) -> bool {
        let s = self.state.lock().unwrap();
        s.frozen
    }

    fn allocated_bytes(&self) -> usize {
        let s = self.state.lock().unwrap();
        s.active_bytes
    }

    fn stats(&self) -> AllocationStats {
        let s = self.state.lock().unwrap();
        AllocationStats {
            total_allocations: s.total_allocations,
            total_bytes: s.total_bytes,
            active_allocations: s.active_allocations,
            is_frozen: s.frozen,
            peak_usage: s.peak_usage,
        }
    }

    fn reset(&self) -> crate::error::Result<()> {
        let mut s = self.state.lock().unwrap();
        if s.active_allocations > 0 {
            return Err(crate::error::Error::AllocatorBusy {
                active_allocations: s.active_allocations,
            });
        }
        s.total_allocations = 0;
        s.total_bytes = 0;
        s.active_bytes = 0;
        s.peak_usage = 0;
        // frozen state is NOT reset — caller must explicitly unfreeze
        Ok(())
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
/// RAII guard for GPU memory allocations.
///
/// Ensures memory is deallocated when the guard is dropped, preventing leaks
/// on error paths. Call [`release`](AllocGuard::release) to take ownership of the
/// pointer (e.g., when transferring it into a `Tensor`).
pub struct AllocGuard<'a, A: Allocator> {
    allocator: &'a A,
    ptr: u64,
    size: usize,
    released: bool,
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
impl<'a, A: Allocator> AllocGuard<'a, A> {
    /// Allocate memory and wrap it in a guard.
    pub fn new(allocator: &'a A, size_bytes: usize) -> crate::error::Result<Self> {
        let ptr = allocator.allocate(size_bytes)?;
        Ok(Self {
            allocator,
            ptr,
            size: size_bytes,
            released: false,
        })
    }

    /// Get the raw pointer.
    #[inline]
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// Release ownership of the pointer, preventing deallocation on drop.
    ///
    /// Returns the raw pointer for use in tensor construction.
    #[inline]
    pub fn release(mut self) -> u64 {
        self.released = true;
        self.ptr
    }
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
impl<A: Allocator> Drop for AllocGuard<'_, A> {
    fn drop(&mut self) {
        if !self.released && self.ptr != 0 {
            self.allocator.deallocate(self.ptr, self.size);
        }
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

    /// Simple in-memory allocator for testing (uses Vec storage behind the scenes)
    #[derive(Clone)]
    struct TestAllocator;

    impl Allocator for TestAllocator {
        fn allocate(&self, size_bytes: usize) -> crate::error::Result<u64> {
            if size_bytes == 0 {
                return Ok(0);
            }
            let layout = std::alloc::Layout::from_size_align(size_bytes, 8).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(crate::error::Error::OutOfMemory { size: size_bytes });
            }
            Ok(ptr as u64)
        }

        fn deallocate(&self, ptr: u64, size_bytes: usize) {
            if ptr == 0 || size_bytes == 0 {
                return;
            }
            let layout = std::alloc::Layout::from_size_align(size_bytes, 8).unwrap();
            unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
        }
    }

    #[test]
    fn test_tracking_allocator_basic_stats() {
        let tracking = TrackingAllocator::new(TestAllocator);

        let stats = tracking.stats();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.active_allocations, 0);
        assert_eq!(stats.peak_usage, 0);
        assert!(!stats.is_frozen);

        let ptr1 = tracking.allocate(1024).unwrap();
        let stats = tracking.stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_bytes, 1024);
        assert_eq!(stats.active_allocations, 1);
        assert_eq!(stats.peak_usage, 1024);

        let ptr2 = tracking.allocate(2048).unwrap();
        let stats = tracking.stats();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_bytes, 3072);
        assert_eq!(stats.active_allocations, 2);
        assert_eq!(stats.peak_usage, 3072);

        tracking.deallocate(ptr1, 1024);
        let stats = tracking.stats();
        assert_eq!(stats.active_allocations, 1);
        assert_eq!(stats.peak_usage, 3072); // peak unchanged

        tracking.deallocate(ptr2, 2048);
        let stats = tracking.stats();
        assert_eq!(stats.active_allocations, 0);
        assert_eq!(stats.peak_usage, 3072); // peak unchanged
    }

    #[test]
    fn test_tracking_allocator_allocated_bytes() {
        let tracking = TrackingAllocator::new(TestAllocator);

        assert_eq!(tracking.allocated_bytes(), 0);

        let ptr = tracking.allocate(512).unwrap();
        assert_eq!(tracking.allocated_bytes(), 512);
        assert_eq!(tracking.active_bytes(), 512);

        tracking.deallocate(ptr, 512);
        assert_eq!(tracking.allocated_bytes(), 0);
    }

    #[test]
    fn test_tracking_allocator_freeze() {
        let tracking = TrackingAllocator::new(TestAllocator);

        assert!(!tracking.is_frozen());
        assert!(tracking.freeze());
        assert!(tracking.is_frozen());

        // Allocation must fail while frozen
        let result = tracking.allocate(128);
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::error::Error::AllocatorFrozen => {}
            other => panic!("expected AllocatorFrozen, got: {other}"),
        }

        tracking.unfreeze();
        assert!(!tracking.is_frozen());

        // Allocation succeeds after unfreeze
        let ptr = tracking.allocate(128).unwrap();
        tracking.deallocate(ptr, 128);
    }

    #[test]
    fn test_tracking_allocator_reset_success() {
        let tracking = TrackingAllocator::new(TestAllocator);

        let ptr = tracking.allocate(1024).unwrap();
        tracking.deallocate(ptr, 1024);

        // All deallocated, reset should succeed
        tracking.reset().unwrap();

        let stats = tracking.stats();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.active_allocations, 0);
        assert_eq!(stats.peak_usage, 0);
    }

    #[test]
    fn test_tracking_allocator_reset_busy() {
        let tracking = TrackingAllocator::new(TestAllocator);

        let ptr = tracking.allocate(1024).unwrap();

        // Active allocation, reset must fail
        let result = tracking.reset();
        assert!(result.is_err());
        match result.unwrap_err() {
            crate::error::Error::AllocatorBusy {
                active_allocations: 1,
            } => {}
            other => panic!("expected AllocatorBusy(1), got: {other}"),
        }

        // Clean up
        tracking.deallocate(ptr, 1024);
    }

    #[test]
    fn test_tracking_allocator_peak_across_cycles() {
        let tracking = TrackingAllocator::new(TestAllocator);

        // Cycle 1: allocate 4096 bytes total
        let p1 = tracking.allocate(2048).unwrap();
        let p2 = tracking.allocate(2048).unwrap();
        assert_eq!(tracking.stats().peak_usage, 4096);
        tracking.deallocate(p1, 2048);
        tracking.deallocate(p2, 2048);

        // Peak is still 4096 (cumulative until reset)
        assert_eq!(tracking.stats().peak_usage, 4096);

        // Reset clears peak
        tracking.reset().unwrap();
        assert_eq!(tracking.stats().peak_usage, 0);

        // Cycle 2: smaller allocation
        let p3 = tracking.allocate(512).unwrap();
        assert_eq!(tracking.stats().peak_usage, 512);
        tracking.deallocate(p3, 512);
    }

    #[test]
    fn test_tracking_allocator_clone_shares_state() {
        let tracking = TrackingAllocator::new(TestAllocator);
        let clone = tracking.clone();

        let ptr = tracking.allocate(256).unwrap();
        // Clone sees the same stats (Arc-shared state)
        assert_eq!(clone.stats().active_allocations, 1);

        clone.deallocate(ptr, 256);
        assert_eq!(tracking.stats().active_allocations, 0);
    }

    #[test]
    fn test_tracking_allocator_freeze_preserved_on_reset() {
        let tracking = TrackingAllocator::new(TestAllocator);
        tracking.freeze();
        // Reset with no active allocations succeeds but freeze is preserved
        tracking.reset().unwrap();
        assert!(tracking.is_frozen());
    }

    #[test]
    fn test_allocation_stats_default() {
        let stats = AllocationStats::default();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.active_allocations, 0);
        assert!(!stats.is_frozen);
        assert_eq!(stats.peak_usage, 0);
    }
}
