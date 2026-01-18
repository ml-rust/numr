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

/// Log a CUDA memory operation failure.
#[cold]
#[inline(never)]
fn log_cuda_memory_error(operation: &str, ptr: u64, result: cudarc::driver::sys::CUresult) {
    eprintln!(
        "[numr::cuda] {} failed for ptr 0x{:x}: {:?}",
        operation, ptr, result
    );
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

    /// Stream on which all kernels launch
    pub(crate) stream: Arc<CudaStream>,

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

/// CUDA allocator that uses stream-ordered allocation.
///
/// This allocator uses `cuMemAllocAsync` and `cuMemFreeAsync` for efficient
/// stream-ordered memory management. Memory operations are synchronized with
/// kernel execution on the associated stream.
///
/// # Panics
///
/// The `allocate` method panics if CUDA memory allocation fails, following
/// CUDA best practices where OOM is typically unrecoverable.
#[derive(Clone)]
pub struct CudaAllocator {
    stream: Arc<CudaStream>,
}

impl Allocator for CudaAllocator {
    /// Allocate GPU memory using stream-ordered allocation.
    ///
    /// # Panics
    ///
    /// Panics if CUDA memory allocation fails (e.g., out of GPU memory).
    fn allocate(&self, size_bytes: usize) -> u64 {
        if size_bytes == 0 {
            return 0;
        }

        unsafe {
            let mut ptr: u64 = 0;
            let result =
                cudarc::driver::sys::cuMemAllocAsync(&mut ptr, size_bytes, self.stream.cu_stream());

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                panic!(
                    "[numr::cuda] Allocation failed: {} bytes ({:?})",
                    size_bytes, result
                );
            }

            ptr
        }
    }

    fn deallocate(&self, ptr: u64, _size_bytes: usize) {
        if ptr == 0 {
            return;
        }

        unsafe {
            // Check if CUDA context is still valid before attempting free
            if !is_cuda_context_valid() {
                // Context is gone - memory will be reclaimed by driver
                return;
            }

            let result = cudarc::driver::sys::cuMemFreeAsync(ptr, self.stream.cu_stream());

            // Log failures but don't panic - deallocation errors are typically benign
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS
                && result != cudarc::driver::sys::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS
            {
                log_cuda_memory_error("cuMemFreeAsync", ptr, result);
            }
        }
    }

    fn is_frozen(&self) -> bool {
        false // CUDA allocator doesn't support freeze
    }

    fn freeze(&self) -> bool {
        // No-op for CUDA - always succeeds
        true
    }

    fn unfreeze(&self) {
        // No-op for CUDA
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

        // Create a stream in this context
        let stream = context.new_stream().map_err(|e| {
            CudaError::ContextError(format!("Failed to create CUDA stream: {:?}", e))
        })?;

        // Initialize cuBLAS handle for GEMM operations
        let cublas = CudaBlas::new(stream.clone())
            .map_err(|e| CudaError::CublasError(format!("Failed to initialize cuBLAS: {:?}", e)))?;

        let allocator = CudaAllocator {
            stream: stream.clone(),
        };

        let raw_handle = CudaRawHandle {
            context: context.clone(),
            stream: stream.clone(),
        };

        Ok(Self {
            device,
            context,
            stream,
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

    /// Get reference to the cuBLAS handle.
    #[inline]
    pub fn cublas(&self) -> &CudaBlas {
        &self.cublas
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
