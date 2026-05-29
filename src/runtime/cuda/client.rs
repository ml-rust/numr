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
use super::allocator::CudaAllocator;
use super::device::{CudaDevice, CudaError};
use super::sobol_cache::SobolDvCache;
use crate::runtime::RuntimeClient;

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

    /// Persistent cache of Sobol direction-vector device buffers.
    ///
    /// Buffers are allocated once per unique dimension count and reused on
    /// every subsequent call, including inside CUDA graph capture regions.
    /// This avoids H2D memcpy nodes with stack-local source pointers.
    pub(crate) sobol_dv_cache: Arc<SobolDvCache>,
}

impl std::fmt::Debug for CudaClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaClient")
            .field("device", &self.device)
            .finish_non_exhaustive()
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

        let allocator = CudaAllocator::new(stream.clone(), pool_handle);

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
            sobol_dv_cache: SobolDvCache::new(),
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

    /// Pre-populate the Sobol direction-vector device buffer for `dimension`.
    ///
    /// Computes `dimension * 32` direction vectors on the host, uploads them to
    /// a persistent device buffer, and stores the pointer in the per-client
    /// cache. Subsequent calls to `sobol(…, dimension, …)` — including those
    /// executed inside a CUDA graph capture region — will use the cached pointer
    /// instead of performing a new H2D copy, which would embed a memcpy node
    /// with a freed host-pointer source in the captured graph.
    ///
    /// # Contract for CUDA graph capture
    ///
    /// Call `warmup_sobol(dimension)` **once**, outside any capture region,
    /// before using `sobol(…, dimension, …)` inside `capture_graph_into`. The
    /// upload stream is synchronised before this method returns, so the device
    /// buffer is fully ready when the next capture begins.
    ///
    /// Calling `warmup_sobol` with the same `dimension` multiple times is safe
    /// and cheap (the cache entry already exists; no second allocation is made).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `dimension` exceeds the maximum supported by the Joe & Kuo dataset
    ///   (21 201).
    /// - Device memory allocation fails.
    /// - The H2D copy fails.
    pub fn warmup_sobol(&self, dimension: usize) -> crate::error::Result<()> {
        use crate::ops::common::quasirandom::{SOBOL_BITS, SOBOL_MAX_DIMENSIONS};

        if dimension == 0 {
            return Err(crate::error::Error::InvalidArgument {
                arg: "dimension",
                reason: "Sobol dimension must be at least 1".into(),
            });
        }
        if dimension > SOBOL_MAX_DIMENSIONS {
            return Err(crate::error::Error::InvalidArgument {
                arg: "dimension",
                reason: format!(
                    "Sobol dimension {} exceeds maximum supported value {}",
                    dimension, SOBOL_MAX_DIMENSIONS
                ),
            });
        }

        let dim_u32 = dimension as u32;

        // Fast path: already cached.
        if self.sobol_dv_cache.get(dim_u32).is_some() {
            return Ok(());
        }

        // Compute direction vectors on the host.
        let direction_vectors =
            crate::ops::common::quasirandom::compute_all_direction_vectors(dimension);
        let num_u32s = direction_vectors.len();
        debug_assert_eq!(num_u32s, dimension * SOBOL_BITS);

        let dv_bytes = bytemuck::cast_slice::<u32, u8>(&direction_vectors);

        // Allocate a device buffer directly via the driver (bypassing the
        // caching allocator's frozen path) so the address survives across
        // graph replays.  We use `cuMemAllocAsync` on the compute stream
        // for proper pool membership and synchronise before returning.
        let dv_ptr: u64 = unsafe {
            let mut ptr: u64 = 0;
            let r = cudarc::driver::sys::cuMemAllocAsync(
                &mut ptr,
                dv_bytes.len(),
                self.stream.cu_stream(),
            );
            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(crate::error::Error::OutOfMemory {
                    size: dv_bytes.len(),
                });
            }
            ptr
        };

        // H2D copy.
        unsafe {
            let r = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                dv_ptr,
                dv_bytes.as_ptr() as *const std::ffi::c_void,
                dv_bytes.len(),
                self.stream.cu_stream(),
            );
            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                // Free the buffer we just allocated before returning the error.
                let _ = cudarc::driver::sys::cuMemFreeAsync(dv_ptr, self.stream.cu_stream());
                return Err(crate::error::Error::Backend(format!(
                    "Sobol warmup H2D copy failed: {:?}",
                    r
                )));
            }
        }

        // Synchronise: the buffer must be fully uploaded before any subsequent
        // capture region references the pointer.
        self.stream
            .synchronize()
            .map_err(|e| crate::error::Error::Internal(format!("stream sync failed: {:?}", e)))?;

        // Store in cache. Ownership of the device pointer is transferred.
        // SAFETY: ptr is a valid, fully-uploaded device buffer.
        unsafe { self.sobol_dv_cache.insert(dim_u32, dv_ptr, num_u32s) };

        Ok(())
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
