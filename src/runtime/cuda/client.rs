//! CUDA Client implementation
//!
//! CudaClient owns stream and context for direct cudarc access.
//!
//! # Thread Safety
//!
//! `CudaClient` is `Clone`, `Send` and `Sync`. Any thread can use it, and
//! the caller does not bind the CUDA context first. Every numr entry point
//! that reaches the driver makes the client's context current on the calling
//! thread before its first driver call. This holds on a thread that has
//! never made a CUDA call, and on a thread where another device's context is
//! current.
//!
//! Work from all threads shares one compute stream per device, so it runs
//! in submission order. A CUDA graph capture on one thread makes other
//! threads' enqueues on that device wait until the capture ends.

use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::CudaRuntime;
use super::allocator::CudaAllocator;
use super::capture::GuardedStream;
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

    /// Stream on which all kernels launch (compute stream).
    ///
    /// Typed as [`GuardedStream`] so a kernel launcher cannot reach the bare
    /// stream: every launch made through it takes the device's capture lock
    /// for the enqueue. Work that is not such an enqueue goes through
    /// `GuardedStream::raw`.
    pub(crate) stream: GuardedStream,

    /// Dedicated stream for D2H copies (overlaps with compute stream)
    pub(crate) copy_stream: Arc<CudaStream>,

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
    /// driver error, stream creation failure, etc.).
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

        // Bind the context to this thread before creating its streams
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

        // Configure the default memory pool with a bounded release threshold.
        // `u64::MAX` (cache everything forever) is great for tight decode loops
        // but causes fragmentation: after loading many small tensors then trying
        // to allocate a multi-GB contiguous block (e.g. a large model's weight),
        // the pool reports OOM despite ample free VRAM because its address space
        // is fragmented. A 512 MiB threshold lets the pool keep moderate caches
        // for hot reuse but reclaim larger freed segments back to the OS.
        // The OOM retry path additionally trims the pool to 0 to recover
        // from fragmentation when a large request would otherwise fail.
        //
        // The threshold is tunable per workload/hardware via the
        // `NUMR_CUDA_POOL_RELEASE_THRESHOLD_MB` env var (value in MiB); many-shape
        // workloads (e.g. variable-length embedding ingest) may prefer a smaller
        // threshold to reclaim aggressively, while tight decode loops may raise it.
        let mut pool_handle: u64 = 0;
        unsafe {
            let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
            let result =
                cudarc::driver::sys::cuDeviceGetDefaultMemPool(&mut pool, device.index as i32);
            if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS && !pool.is_null() {
                let threshold: u64 = super::env_config::env_mib_to_bytes(
                    "NUMR_CUDA_POOL_RELEASE_THRESHOLD_MB",
                    512 * 1024 * 1024,
                );
                let _ = cudarc::driver::sys::cuMemPoolSetAttribute(
                    pool,
                    cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &threshold as *const u64 as *mut std::ffi::c_void,
                );
                pool_handle = pool as u64;
            }
        }

        let raw_handle = CudaRawHandle {
            context: context.clone(),
            stream: stream.clone(),
        };

        let stream = GuardedStream::new(stream, device.index);
        let allocator = CudaAllocator::new(stream.clone(), pool_handle);
        let sobol_dv_cache = SobolDvCache::new(context.clone());

        Ok(Self {
            device,
            context,
            stream,
            copy_stream,
            allocator,
            raw_handle,
            sobol_dv_cache,
        })
    }

    /// Get reference to the CUDA compute stream.
    ///
    /// **CRITICAL**: All kernel launches MUST use this stream for correct ordering.
    #[inline]
    pub fn stream(&self) -> &GuardedStream {
        &self.stream
    }

    /// Whether THIS thread is inside a CUDA graph capture region on this
    /// client's device.
    ///
    /// Work that synchronizes, reads a result back, or records timing events
    /// cannot be recorded into a graph and asks this. It answers for the
    /// calling thread only, so a thread that merely shares the device's
    /// compute stream with a capture running elsewhere gets `false` and does
    /// its full work — that thread's enqueues wait for the capture to end.
    #[inline]
    pub fn is_capturing(&self) -> bool {
        super::capture::thread_is_capturing(self.device.index)
    }

    /// Whether the DRIVER reports the compute stream in capture mode.
    ///
    /// True on every thread once any thread starts capturing, so it cannot
    /// answer "may this call synchronize" — use [`CudaClient::is_capturing`]
    /// for that. A failed status query reads as "not capturing".
    pub fn stream_capture_active(&self) -> bool {
        use cudarc::driver::sys::CUstreamCaptureStatus;
        self.stream
            .raw()
            .capture_status()
            .map(|s| s != CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE)
            .unwrap_or(false)
    }

    /// Get the Arc-wrapped CUDA stream for operations that need ownership.
    #[inline]
    pub fn stream_arc(&self) -> &Arc<CudaStream> {
        self.stream.arc()
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

    /// Record an event on the compute stream.
    ///
    /// Returns an event handle that can be passed to `copy_stream_wait_event`.
    pub fn record_event_on_compute(&self) -> Result<u64, CudaError> {
        use cudarc::driver::sys::{CUevent_flags, cuEventCreate, cuEventRecord};
        self.bind_context()?;
        unsafe {
            let mut event = std::ptr::null_mut();
            let r = cuEventCreate(&mut event, CUevent_flags::CU_EVENT_DISABLE_TIMING as u32);
            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(CudaError::ContextError(format!(
                    "cuEventCreate failed: {:?}",
                    r
                )));
            }
            let r = cuEventRecord(event, self.stream.raw().cu_stream());
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
        self.bind_context()?;
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

        let cu_stream = self.stream.raw().cu_stream();

        // One permit covers the allocation and the copy, so no capture can
        // open between them. The wait below takes its own; nesting two read
        // sides on one thread would deadlock against a waiting writer.
        let dv_ptr: u64 = {
            let _permit = self.stream.enqueue_permit()?;

            // Allocate a device buffer directly via the driver (bypassing the
            // caching allocator's frozen path) so the address survives across
            // graph replays.  We use `cuMemAllocAsync` on the compute stream
            // for proper pool membership and synchronise before returning.
            let ptr: u64 = unsafe {
                let mut ptr: u64 = 0;
                let r = cudarc::driver::sys::cuMemAllocAsync(&mut ptr, dv_bytes.len(), cu_stream);
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(super::allocator::alloc_error(dv_bytes.len(), r));
                }
                ptr
            };

            // H2D copy.
            unsafe {
                let r = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                    ptr,
                    dv_bytes.as_ptr() as *const std::ffi::c_void,
                    dv_bytes.len(),
                    cu_stream,
                );
                if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    // Free the buffer we just allocated before returning the error.
                    let _ = cudarc::driver::sys::cuMemFreeAsync(ptr, cu_stream);
                    return Err(crate::error::Error::Backend(format!(
                        "Sobol warmup H2D copy failed: {:?}",
                        r
                    )));
                }
            }

            ptr
        };

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
    ///
    /// When the context cannot be made current the event cannot be destroyed,
    /// and it stays allocated until the context is destroyed.
    pub fn destroy_event(&self, event: u64) {
        if self.bind_context().is_err() {
            return;
        }
        unsafe {
            cudarc::driver::sys::cuEventDestroy_v2(event as cudarc::driver::sys::CUevent);
        }
    }

    /// Make this client's context current on the calling thread, for a
    /// driver call that takes no stream through a permit.
    fn bind_context(&self) -> Result<(), CudaError> {
        super::context::bind_context(&self.context).map_err(|e| {
            CudaError::ContextError(format!(
                "Failed to make the CUDA context of device {} current on this thread: {:?}",
                self.device.index, e
            ))
        })
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
        Some(self.stream.raw().cu_stream() as u64)
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
