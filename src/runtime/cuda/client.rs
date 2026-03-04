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
use std::sync::Arc;

use super::CudaRuntime;
use super::device::{CudaDevice, CudaError};
use crate::runtime::{Allocator, RuntimeClient};

// ============================================================================
// Internal Helpers
// ============================================================================

/// Check if the CUDA context on the current thread is valid.
///
/// # Safety
///
/// This function calls CUDA driver API directly.
#[inline]
unsafe fn is_cuda_context_valid() -> bool {
    let mut ctx: cudarc::driver::sys::CUcontext = std::ptr::null_mut();
    // SAFETY: cuCtxGetCurrent is safe to call at any time and writes to the provided pointer.
    let result = unsafe { cudarc::driver::sys::cuCtxGetCurrent(&mut ctx) };
    result == cudarc::driver::sys::CUresult::CUDA_SUCCESS && !ctx.is_null()
}

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

/// CUDA caching allocator with Rust-side free lists.
///
/// Maintains per-size free lists of GPU buffers. On deallocation, buffers are
/// returned to the free list instead of calling `cuMemFreeAsync`. On allocation,
/// the free list is checked first, bypassing the CUDA driver entirely for repeat
/// allocations of the same size. This is critical for inference decode loops where
/// the same buffer sizes are allocated every step.
///
/// Falls through to `cuMemAllocAsync` for sizes not in the cache.
///
#[derive(Clone)]
pub struct CudaAllocator {
    stream: Arc<CudaStream>,
    /// Free list: size_bytes → Vec<device_ptr>
    cache: Arc<std::sync::Mutex<std::collections::HashMap<usize, Vec<u64>>>>,
    /// When frozen, bypass the cache entirely. Used during CUDA graph capture
    /// so that `cuMemAllocAsync`/`cuMemFreeAsync` create proper graph nodes.
    frozen: Arc<std::sync::atomic::AtomicBool>,
}

impl Allocator for CudaAllocator {
    fn allocate(&self, size_bytes: usize) -> crate::error::Result<u64> {
        if size_bytes == 0 {
            return Ok(0);
        }

        // When frozen (graph capture), bypass cache — go straight to driver
        // so cuMemAllocAsync creates a proper graph allocation node.
        if !self.frozen.load(std::sync::atomic::Ordering::Relaxed) {
            // Check free list first
            let mut cache = self.cache.lock().unwrap();
            if let Some(ptrs) = cache.get_mut(&size_bytes)
                && let Some(ptr) = ptrs.pop()
            {
                return Ok(ptr);
            }
        }

        // Allocate from CUDA driver (stream-ordered)
        unsafe {
            let mut ptr: u64 = 0;
            let result =
                cudarc::driver::sys::cuMemAllocAsync(&mut ptr, size_bytes, self.stream.cu_stream());

            if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Ok(ptr);
            }

            // Sync stream to flush pending async frees, then retry
            let _ = self.stream.synchronize();

            let result =
                cudarc::driver::sys::cuMemAllocAsync(&mut ptr, size_bytes, self.stream.cu_stream());

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(crate::error::Error::OutOfMemory { size: size_bytes });
            }

            Ok(ptr)
        }
    }

    fn deallocate(&self, ptr: u64, size_bytes: usize) {
        if ptr == 0 {
            return;
        }

        // When frozen (graph capture), bypass cache — call cuMemFreeAsync
        // so the driver creates a proper graph free node.
        if self.frozen.load(std::sync::atomic::Ordering::Relaxed) {
            unsafe {
                let _ = cudarc::driver::sys::cuMemFreeAsync(ptr, self.stream.cu_stream());
            }
            return;
        }

        // Return to free list for reuse
        let mut cache = self.cache.lock().unwrap();
        cache.entry(size_bytes).or_default().push(ptr);
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
        // Flush all cached buffers back to CUDA
        let mut cache = self.cache.lock().unwrap();
        for (_size, ptrs) in cache.drain() {
            for ptr in ptrs {
                unsafe {
                    if is_cuda_context_valid() {
                        let _ = cudarc::driver::sys::cuMemFreeAsync(ptr, self.stream.cu_stream());
                    }
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// CudaClient Implementation
// ============================================================================

impl CudaClient {
    /// Create a new CUDA client for a device.
    ///
    /// This initializes the CUDA context, creates a stream, and sets up cuBLAS.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - CUDA context creation fails (e.g., invalid device ID)
    /// - Stream creation fails
    /// - cuBLAS initialization fails
    pub fn new(device: CudaDevice) -> Result<Self, CudaError> {
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

        // Configure the default memory pool to cache freed allocations instead
        // of returning them to the OS. This dramatically reduces allocation overhead
        // for repetitive workloads (e.g. inference decode loops).
        unsafe {
            let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
            let result =
                cudarc::driver::sys::cuDeviceGetDefaultMemPool(&mut pool, device.index as i32);
            if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS && !pool.is_null() {
                let threshold: u64 = u64::MAX; // Keep all freed memory cached
                let _ = cudarc::driver::sys::cuMemPoolSetAttribute(
                    pool,
                    cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &threshold as *const u64 as *mut std::ffi::c_void,
                );
            }
        }

        let allocator = CudaAllocator {
            stream: stream.clone(),
            cache: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            frozen: Arc::new(std::sync::atomic::AtomicBool::new(false)),
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
