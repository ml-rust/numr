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
pub mod kernel;
pub mod sparse_utils;

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "wgpu")]
pub mod wgpu;

// CPU fallback utilities for GPU backends
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub mod fallback;

pub use allocator::{Allocator, DefaultAllocator};

/// Core trait for compute backends
///
/// `Runtime` abstracts over different compute devices (CPU, GPU, etc.).
/// It uses static dispatch via generics for zero-cost abstraction.
///
/// # Associated Types
///
/// - `Device`: Identifies a specific compute unit (e.g., GPU 0, GPU 1)
/// - `Client`: Handles operation dispatch and synchronization
/// - `Allocator`: Memory management with optional freeze support
/// - `RawHandle`: Escape hatch for custom kernel launching
///
/// # Example
///
/// ```ignore
/// use numr::runtime::{Runtime, CpuRuntime};
///
/// fn compute<R: Runtime>(device: &R::Device) {
///     let ptr = R::allocate(1024, DType::F32, device);
///     // ... use memory ...
///     R::deallocate(ptr, 1024, DType::F32, device);
/// }
/// ```
pub trait Runtime: Clone + Send + Sync + 'static {
    /// Device identifier type
    type Device: Device;

    /// Client for dispatching operations
    type Client: RuntimeClient<Self>;

    /// Memory allocator type
    type Allocator: Allocator;

    /// Raw handle for custom kernel launching (escape hatch)
    ///
    /// For CPU: `()` (no raw handle needed)
    /// For CUDA: Access to CudaDevice/CudaStream
    /// For WGPU: Access to wgpu::Device/Queue
    type RawHandle: Send + Sync;

    /// Human-readable name of this runtime
    fn name() -> &'static str;

    /// Does this backend support graph capture (e.g., CUDA Graphs)?
    fn supports_graph_capture() -> bool {
        false
    }

    /// Allocate device memory
    ///
    /// Returns a device pointer (u64) that can be used for operations.
    fn allocate(size_bytes: usize, device: &Self::Device) -> u64;

    /// Deallocate device memory
    fn deallocate(ptr: u64, size_bytes: usize, device: &Self::Device);

    /// Copy data from host to device
    fn copy_to_device(src: &[u8], dst: u64, device: &Self::Device);

    /// Copy data from device to host
    fn copy_from_device(src: u64, dst: &mut [u8], device: &Self::Device);

    /// Copy data within device (device to device)
    fn copy_within_device(src: u64, dst: u64, size_bytes: usize, device: &Self::Device);

    /// Copy strided data to a contiguous buffer
    ///
    /// This is the proper way to make a non-contiguous tensor contiguous.
    /// Each backend implements this according to its memory model:
    /// - CPU/CUDA: can use pointer arithmetic (src_handle + byte_offset)
    /// - WGPU: must use compute shader with buffer binding + offset uniform
    ///
    /// # Parameters
    /// - `src_handle`: Source buffer handle (pointer for CUDA, buffer ID for WGPU)
    /// - `src_byte_offset`: Byte offset into source buffer
    /// - `dst_handle`: Destination buffer handle
    /// - `shape`: Shape of the tensor
    /// - `strides`: Strides of the source tensor (in elements, not bytes)
    /// - `elem_size`: Size of each element in bytes
    fn copy_strided(
        src_handle: u64,
        src_byte_offset: usize,
        dst_handle: u64,
        shape: &[usize],
        strides: &[isize],
        elem_size: usize,
        device: &Self::Device,
    );

    /// Get the default device
    fn default_device() -> Self::Device;

    /// Get the default client for a device
    fn default_client(device: &Self::Device) -> Self::Client;

    /// Get the raw handle from a client (escape hatch for custom kernels)
    fn raw_handle(client: &Self::Client) -> &Self::RawHandle;
}

/// Trait for device identification
pub trait Device: Clone + Send + Sync + 'static {
    /// Unique identifier for this device
    fn id(&self) -> usize;

    /// Check if two devices are the same
    fn is_same(&self, other: &Self) -> bool {
        self.id() == other.id()
    }

    /// Human-readable name
    fn name(&self) -> String {
        format!("Device({})", self.id())
    }
}

/// Trait for runtime clients that handle operation dispatch
pub trait RuntimeClient<R: Runtime>: Clone + Send + Sync {
    /// Get the device this client operates on
    fn device(&self) -> &R::Device;

    /// Synchronize: wait for all pending operations to complete
    fn synchronize(&self);

    /// Get the allocator for this client
    fn allocator(&self) -> &R::Allocator;
}
