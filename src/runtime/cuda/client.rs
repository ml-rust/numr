//! CUDA Client implementation
//!
//! CudaClient owns stream and context for direct cudarc access.
//!
//! # Thread Safety
//!
//! `CudaClient` is `Clone` and can be shared across threads. The underlying
//! CUDA context and stream are reference-counted via `Arc`. However, CUDA
//! operations must be performed on the thread that owns the context or after
//! calling `context.bind_to_thread()`.

use cudarc::cublas::CudaBlas;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use super::CudaRuntime;
use super::device::{CudaDevice, CudaError};
use crate::runtime::{Allocator, RuntimeClient};

// ============================================================================
// CudaClient
// ============================================================================

/// CUDA Runtime Client
///
/// Owns CUDA context and stream for direct kernel launches.
/// All tensor operations launch on this stream.
///
/// # Stream Ownership
///
/// The key insight: All ops MUST launch on `self.stream()` for correct ordering.
/// Operations launched on different streams may execute out of order.
///
/// # Panics
///
/// Memory allocation via the allocator may panic on CUDA OOM conditions.
/// See the module-level documentation for details.
#[derive(Clone)]
pub struct CudaClient {
    /// GPU device index
    pub(crate) device: CudaDevice,

    /// CUDA context for this device (owns GPU context)
    pub(crate) context: Arc<CudaContext>,

    /// Stream on which all kernels launch (compute stream)
    pub(crate) stream: Arc<CudaStream>,

    /// Dedicated stream for D2H copies (overlaps with compute stream)
    pub(crate) copy_stream: Arc<CudaStream>,

    /// cuBLAS handle for GEMM operations
    pub(crate) cublas: Arc<CudaBlas>,

    /// Allocator for memory management
    pub(crate) allocator: CudaAllocator,

    /// Raw handle for custom kernel launching
    pub(crate) raw_handle: CudaRawHandle,
}

impl std::fmt::Debug for CudaClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaClient")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// CudaAllocator
// ============================================================================

/// Maximum number of cached buffers per exact byte-size bucket.
///
/// 64 gives enough depth to absorb a full decode loop's worth of temporaries
/// (typically 8–32 per forward pass) while keeping the worst-case cached
/// footprint to 64× the largest recurring buffer — acceptable for VRAM.
const FREE_LIST_CAP: usize = 64;

/// CUDA stream-ordered allocator with a Rust-side free list.
///
/// Combines a per-size-bucket Rust free list (fast path: no driver call) with
/// `cuMemAllocAsync`/`cuMemFreeAsync` for cold misses and pool management.
///
/// # Safety invariant — single canonical stream
///
/// Every pointer in the free list was allocated on `self.stream`. Every
/// deallocation is also issued on `self.stream`. Because all alloc/free ops
/// share the same stream, there is no cross-stream ordering hazard: a buffer
/// popped from the free list is guaranteed to have no pending use on any
/// other stream before it is returned to the caller.
///
/// This invariant holds because `CudaClient::new` always returns the cached
/// canonical client for a device (via `register_or_get_client`), and stream
/// creation only happens in `new_uncached`. The `copy_stream` is exclusively
/// for D2H copies and never used with the allocator.
///
/// # Pool Threshold
///
/// The default memory pool's release threshold is set to `u64::MAX` at context
/// creation so that freed segments stay warm for reuse in decode loops.
#[derive(Clone)]
pub struct CudaAllocator {
    stream: Arc<CudaStream>,
    /// Per-size free list: size_bytes → VecDeque of device pointers.
    ///
    /// VecDeque gives O(1) push_back / pop_front so the oldest entry is
    /// evicted first when the cap is reached (FIFO within a bucket).
    free_list: Arc<Mutex<HashMap<u64, VecDeque<u64>>>>,
    /// When frozen, alloc/free go directly to the driver to create proper
    /// CUDA graph alloc/free nodes — bypassing the Rust free list.
    frozen: Arc<std::sync::atomic::AtomicBool>,
    /// Cached handle to the default memory pool for the device. Stored as
    /// `u64` (the raw pointer value) so the struct stays `Send + Sync`.
    /// Used only by the OOM retry path to call `cuMemPoolTrimTo`.
    /// Zero when the pool could not be obtained at construction.
    pool_handle: u64,
}

impl CudaAllocator {
    /// Allocate directly from the driver (no free-list lookup).
    ///
    /// On failure, drains the Rust free list back to the driver pool so those
    /// segments become available for reuse, syncs the stream, then retries once.
    unsafe fn driver_alloc(&self, size_bytes: usize) -> crate::error::Result<u64> {
        let mut ptr: u64 = 0;
        let result =
            cudarc::driver::sys::cuMemAllocAsync(&mut ptr, size_bytes, self.stream.cu_stream());
        if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Ok(ptr);
        }

        // Drain free list: return cached segments to the driver pool so it can
        // reclaim VRAM that our Rust-side cache was holding "live" from the
        // driver's perspective.
        let drained: Vec<u64> = {
            let mut map = self.free_list.lock().unwrap();
            map.drain()
                .flat_map(|(_, bucket)| bucket.into_iter())
                .collect()
        };
        for p in drained {
            let _ = cudarc::driver::sys::cuMemFreeAsync(p, self.stream.cu_stream());
        }

        // Sync so the pool can process the frees before the retry alloc.
        let _ = self.stream.synchronize();

        // Trim the pool to release cached segments back to the OS. The
        // pool retains freed allocations (per the release threshold) which
        // is fast for tight decode loops but fragments the address space
        // when a one-shot large allocation arrives after many small frees.
        // Calling trim only on the retry path keeps the steady-state cache
        // behaviour but recovers from fragmentation when it matters.
        if self.pool_handle != 0 {
            let pool = self.pool_handle as cudarc::driver::sys::CUmemoryPool;
            let _ = cudarc::driver::sys::cuMemPoolTrimTo(pool, 0);
        }

        let result =
            cudarc::driver::sys::cuMemAllocAsync(&mut ptr, size_bytes, self.stream.cu_stream());
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(crate::error::Error::OutOfMemory { size: size_bytes });
        }
        Ok(ptr)
    }

    /// Free directly to the driver (no free-list insertion).
    unsafe fn driver_free(&self, ptr: u64) {
        let _ = cudarc::driver::sys::cuMemFreeAsync(ptr, self.stream.cu_stream());
    }
}

impl Allocator for CudaAllocator {
    fn allocate(&self, size_bytes: usize) -> crate::error::Result<u64> {
        if size_bytes == 0 {
            return Ok(0);
        }

        // Graph-capture path: bypass free list so the driver records a proper
        // alloc node in the CUDA graph.
        if self.frozen.load(std::sync::atomic::Ordering::Relaxed) {
            return unsafe { self.driver_alloc(size_bytes) };
        }

        // Fast path: pop from the free list if a cached buffer exists.
        {
            let mut map = self.free_list.lock().unwrap();
            if let Some(bucket) = map.get_mut(&(size_bytes as u64)) {
                if let Some(ptr) = bucket.pop_front() {
                    return Ok(ptr);
                }
            }
        }

        // Cold miss: ask the driver.
        unsafe { self.driver_alloc(size_bytes) }
    }

    fn deallocate(&self, ptr: u64, size_bytes: usize) {
        if ptr == 0 {
            return;
        }

        // Graph-capture path: issue a driver free node directly.
        if self.frozen.load(std::sync::atomic::Ordering::Relaxed) {
            unsafe { self.driver_free(ptr) };
            return;
        }

        let evict = {
            let mut map = self.free_list.lock().unwrap();
            let bucket = map.entry(size_bytes as u64).or_default();
            if bucket.len() < FREE_LIST_CAP {
                bucket.push_back(ptr);
                None
            } else {
                // Cap exceeded: evict the oldest buffer and keep the new one.
                let oldest = bucket.pop_front();
                bucket.push_back(ptr);
                oldest
            }
        };

        // Evict outside the lock so the lock is not held during a driver call.
        if let Some(old_ptr) = evict {
            unsafe { self.driver_free(old_ptr) };
        }
    }

    fn is_frozen(&self) -> bool {
        self.frozen.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn freeze(&self) -> bool {
        self.frozen
            .store(true, std::sync::atomic::Ordering::Relaxed);
        true
    }

    fn unfreeze(&self) {
        self.frozen
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    fn reset(&self) -> crate::error::Result<()> {
        // Drain the free list and return all cached buffers to the driver pool.
        // Callers must have dropped all live tensors (which call `deallocate`)
        // before calling `reset`, so every pointer here is idle on the stream.
        let drained: Vec<u64> = {
            let mut map = self.free_list.lock().unwrap();
            map.drain()
                .flat_map(|(_, bucket)| bucket.into_iter())
                .collect()
        };
        for ptr in drained {
            unsafe { self.driver_free(ptr) };
        }
        Ok(())
    }
}

// ============================================================================
// CudaClient Implementation
// ============================================================================

impl CudaClient {
    /// Create a CUDA client for a device.
    ///
    /// Returns the canonical client for this device index from the global cache.
    /// If no client exists for this device yet, a new one is created, registered,
    /// and returned. Subsequent calls with the same device index always return the
    /// same underlying streams, context, and allocator — even from call sites that
    /// do not go through the runtime's `get_or_create_client` (e.g.,
    /// `build_device_client` in embedding pipelines).
    ///
    /// This ensures that tensor allocations (which route through
    /// `CudaRuntime::allocate` → `get_or_create_client`) and kernel launches
    /// (which use the client supplied by the caller) always share the same CUDA
    /// stream, maintaining the stream-ordering invariant required by
    /// `cuMemAllocAsync`/`cuMemFreeAsync`.
    ///
    /// # Errors
    ///
    /// Returns an error if device initialisation fails (invalid device index,
    /// driver error, cuBLAS init failure, etc.).
    pub fn new(device: CudaDevice) -> Result<Self, CudaError> {
        // Return the cached canonical client if one already exists.
        if let Some(cached) = super::cache::try_get_cached_client(device.index) {
            return Ok(cached);
        }
        let client = Self::new_uncached(device)?;
        Ok(super::cache::register_or_get_client(
            client.device.index,
            client,
        ))
    }

    /// Construct a brand-new client without consulting the cache.
    ///
    /// Used internally by `get_or_create_client` and `new`. External callers
    /// should always use `CudaClient::new`.
    pub(super) fn new_uncached(device: CudaDevice) -> Result<Self, CudaError> {
        // Create CUDA context for this device
        let context = CudaContext::new(device.index).map_err(|e| {
            CudaError::ContextError(format!(
                "Failed to create CUDA context for device {}: {:?}",
                device.index, e
            ))
        })?;

        // Bind context to current thread for proper cuBLAS operation
        context.bind_to_thread().map_err(|e| {
            CudaError::ContextError(format!("Failed to bind CUDA context to thread: {:?}", e))
        })?;

        // Create compute stream
        let stream = context.new_stream().map_err(|e| {
            CudaError::ContextError(format!("Failed to create CUDA stream: {:?}", e))
        })?;

        // Create dedicated copy stream for overlapped D2H transfers
        let copy_stream = context.new_stream().map_err(|e| {
            CudaError::ContextError(format!("Failed to create CUDA copy stream: {:?}", e))
        })?;

        // Initialize cuBLAS handle for GEMM operations
        let cublas = CudaBlas::new(stream.clone())
            .map_err(|e| CudaError::CublasError(format!("Failed to initialize cuBLAS: {:?}", e)))?;

        // Configure the default memory pool with a bounded release threshold.
        // `u64::MAX` (cache everything forever) is great for tight decode loops
        // but causes fragmentation: after loading many small tensors then trying
        // to allocate a multi-GB contiguous block (e.g. a large model's weight),
        // the pool reports OOM despite ample free VRAM because its address space
        // is fragmented. A 512 MiB threshold lets the pool keep moderate caches
        // for hot reuse but reclaim larger freed segments back to the OS.
        // The OOM retry path additionally trims the pool to 0 to recover
        // from fragmentation when a large request would otherwise fail.
        let mut pool_handle: u64 = 0;
        unsafe {
            let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
            let result =
                cudarc::driver::sys::cuDeviceGetDefaultMemPool(&mut pool, device.index as i32);
            if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS && !pool.is_null() {
                let threshold: u64 = 512 * 1024 * 1024;
                let _ = cudarc::driver::sys::cuMemPoolSetAttribute(
                    pool,
                    cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &threshold as *const u64 as *mut std::ffi::c_void,
                );
                pool_handle = pool as u64;
            }
        }

        let allocator = CudaAllocator {
            stream: stream.clone(),
            free_list: Arc::new(Mutex::new(HashMap::new())),
            frozen: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            pool_handle,
        };

        let raw_handle = CudaRawHandle {
            context: context.clone(),
            stream: stream.clone(),
        };

        Ok(Self {
            device,
            context,
            stream,
            copy_stream,
            cublas: Arc::new(cublas),
            allocator,
            raw_handle,
        })
    }

    /// Get reference to the CUDA stream.
    ///
    /// **CRITICAL**: All kernel launches MUST use this stream for correct ordering.
    #[inline]
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Get the Arc-wrapped CUDA stream for operations that need ownership.
    #[inline]
    pub fn stream_arc(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Get reference to the CUDA context.
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Get reference to the copy stream (for overlapped D2H transfers).
    #[inline]
    pub fn copy_stream(&self) -> &CudaStream {
        &self.copy_stream
    }

    /// Get reference to the cuBLAS handle.
    #[inline]
    pub fn cublas(&self) -> &CudaBlas {
        &self.cublas
    }

    /// Record an event on the compute stream.
    ///
    /// Returns an event handle that can be passed to `copy_stream_wait_event`.
    pub fn record_event_on_compute(&self) -> Result<u64, CudaError> {
        use cudarc::driver::sys::{CUevent_flags, cuEventCreate, cuEventRecord};
        unsafe {
            let mut event = std::ptr::null_mut();
            let r = cuEventCreate(&mut event, CUevent_flags::CU_EVENT_DISABLE_TIMING as u32);
            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(CudaError::ContextError(format!(
                    "cuEventCreate failed: {:?}",
                    r
                )));
            }
            let r = cuEventRecord(event, self.stream.cu_stream());
            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                cudarc::driver::sys::cuEventDestroy_v2(event);
                return Err(CudaError::ContextError(format!(
                    "cuEventRecord failed: {:?}",
                    r
                )));
            }
            Ok(event as u64)
        }
    }

    /// Make the copy stream wait for an event recorded on the compute stream.
    pub fn copy_stream_wait_event(&self, event: u64) -> Result<(), CudaError> {
        use cudarc::driver::sys::cuStreamWaitEvent;
        unsafe {
            let r = cuStreamWaitEvent(
                self.copy_stream.cu_stream(),
                event as cudarc::driver::sys::CUevent,
                0,
            );
            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(CudaError::ContextError(format!(
                    "cuStreamWaitEvent failed: {:?}",
                    r
                )));
            }
        }
        Ok(())
    }

    /// Pre-load CUDA PTX modules to avoid JIT compilation latency on first use.
    ///
    /// Call this during warmup with the list of numr kernel module names
    /// that will be used during inference.
    pub fn preload_modules(&self, module_names: &[&'static str]) -> crate::error::Result<()> {
        crate::runtime::cuda::kernels::preload_modules(
            &self.context,
            self.device.index,
            module_names,
        )
    }

    /// Destroy a CUDA event handle returned by `record_event_on_compute`.
    ///
    /// Must be called after the copy stream has finished using the event
    /// (i.e., after `copy_stream.synchronize()`). Passing an already-destroyed
    /// or invalid handle is safe (CUDA ignores it).
    pub fn destroy_event(&self, event: u64) {
        unsafe {
            cudarc::driver::sys::cuEventDestroy_v2(event as cudarc::driver::sys::CUevent);
        }
    }
}

impl RuntimeClient<CudaRuntime> for CudaClient {
    fn device(&self) -> &CudaDevice {
        &self.device
    }

    fn synchronize(&self) {
        if let Err(e) = self.stream.synchronize() {
            eprintln!("[numr::cuda] Stream synchronization failed: {:?}", e);
        }
    }

    fn allocator(&self) -> &CudaAllocator {
        &self.allocator
    }

    fn compute_stream_handle(&self) -> Option<u64> {
        Some(self.stream.cu_stream() as u64)
    }
}

// ============================================================================
// CudaRawHandle
// ============================================================================

/// Raw handle for custom kernel launching.
///
/// Provides access to the CUDA context and stream for users who want
/// to launch their own kernels outside of numr's operation system.
///
/// # Example
///
/// ```ignore
/// let client = CudaRuntime::default_client(&device);
/// let handle = CudaRuntime::raw_handle(&client);
///
/// // Use handle.stream for custom kernel launches
/// // Use handle.context for context management
/// ```
#[derive(Clone)]
pub struct CudaRawHandle {
    /// CUDA context for device management
    pub context: Arc<CudaContext>,
    /// CUDA stream for kernel execution
    pub stream: Arc<CudaStream>,
}
