//! WebGPU Client implementation.
//!
//! `WgpuClient` owns the WebGPU device and queue for operation dispatch.
//!
//! # Thread Safety
//!
//! `WgpuClient` is `Clone` and the underlying wgpu::Device and wgpu::Queue
//! are already `Send + Sync` by design.

use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
#[cfg(feature = "wgpu-raw-interop")]
use wgpu::Adapter;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

use super::WgpuRuntime;
pub use super::allocator::{WgpuAllocator, get_buffer};
#[cfg(feature = "wgpu-raw-interop")]
use super::device::describe_installed;
#[cfg(not(target_arch = "wasm32"))]
use super::device::query_adapter_info_blocking;
use super::device::{WgpuDevice, WgpuError, query_adapter_info};
use super::shaders::PipelineCache;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

// ============================================================================
// WgpuClient
// ============================================================================

/// WebGPU Runtime Client.
///
/// Owns WebGPU device and queue for operation dispatch.
/// All tensor operations are submitted through this client's queue.
///
/// # Buffer Management
///
/// The client uses storage buffers for tensor data and staging buffers
/// for CPU<->GPU transfers. Buffers are created on-demand and can be
/// reused through the allocator.
#[derive(Clone)]
pub struct WgpuClient {
    /// GPU device identifier
    pub(crate) device_id: WgpuDevice,

    /// WebGPU device handle
    pub(crate) wgpu_device: Arc<Device>,

    /// WebGPU queue for command submission
    pub(crate) queue: Arc<Queue>,

    /// Allocator for buffer management
    pub(crate) allocator: WgpuAllocator,

    /// Raw handle for custom kernel access
    pub(crate) raw_handle: WgpuRawHandle,

    /// Pipeline cache for compute shaders
    pub(crate) pipeline_cache: Arc<PipelineCache>,
}

impl std::fmt::Debug for WgpuClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuClient")
            .field("device", &self.device_id)
            .finish_non_exhaustive()
    }
}

impl WgpuClient {
    /// Create or retrieve a cached WebGPU client for a device.
    ///
    /// This returns the cached client for the given device index, creating one
    /// if it doesn't exist yet. All clients for the same device index share
    /// the same underlying `wgpu::Device` and queue, which is required because
    /// wgpu buffers are device-specific and cannot be used across devices.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No suitable GPU adapter is found
    /// - Device creation fails
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(device: WgpuDevice) -> Result<Self, WgpuError> {
        super::cache::get_or_create_client(&device)
    }

    /// Create or retrieve a cached WebGPU client asynchronously.
    ///
    /// Awaits adapter and device discovery natively without blocking the calling thread,
    /// so it works on both native targets and browser `wasm32` WebGPU runtimes.
    ///
    /// # Errors
    ///
    /// Returns an error if no suitable GPU adapter is found or device creation fails.
    pub async fn new_async(device: WgpuDevice) -> Result<Self, WgpuError> {
        if let Some(cached) = super::cache::get_cached_client(&device) {
            return Ok(cached);
        }
        let client = Self::new_uncached_async(device).await?;
        Ok(super::cache::insert_or_get_client(client))
    }

    /// Internal: create a fresh WebGPU client asynchronously (not cached).
    async fn new_uncached_async(device: WgpuDevice) -> Result<Self, WgpuError> {
        let (adapter, info) = query_adapter_info(device.index).await?;
        let features = adapter.features();
        let required_features = if features.contains(wgpu::Features::SUBGROUP) {
            wgpu::Features::SUBGROUP
        } else {
            wgpu::Features::empty()
        };

        let (wgpu_device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("numr WebGPU Device"),
                required_features,
                // The adapter's own maxima. The WebGPU defaults cap a
                // buffer at 256 MiB, below one LLM embedding table.
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .map_err(|e| WgpuError::DeviceError(format!("{:?}", e)))?;

        let device_with_info = WgpuDevice::with_info(device.index, info);
        Ok(Self::from_parts(device_with_info, wgpu_device, queue))
    }

    /// Internal: create a fresh WebGPU client (not cached).
    ///
    /// This is called by the cache layer. External users should call `new()`
    /// which goes through the cache to ensure a single `wgpu::Device` per
    /// device index.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn new_uncached(device: WgpuDevice) -> Result<Self, WgpuError> {
        let (adapter, info) = query_adapter_info_blocking(device.index)?;

        // Request device with compute features
        let (wgpu_device, queue) = pollster::block_on(async {
            let features = adapter.features();
            let required_features = if features.contains(wgpu::Features::SUBGROUP) {
                wgpu::Features::SUBGROUP
            } else {
                wgpu::Features::empty()
            };

            adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("numr WebGPU Device"),
                    required_features,
                    // The adapter's own maxima. The WebGPU defaults cap a
                    // buffer at 256 MiB, below one LLM embedding table.
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                })
                .await
        })
        .map_err(|e| WgpuError::DeviceError(format!("{:?}", e)))?;

        let device_with_info = WgpuDevice::with_info(device.index, info);
        Ok(Self::from_parts(device_with_info, wgpu_device, queue))
    }

    /// Adopt a `wgpu::Device` that the caller opened, as the client of `index`.
    ///
    /// Every later `WgpuClient::new`, `Runtime::default_client` and tensor
    /// allocation for `index` then uses this device. Use it when the device
    /// needs more than `WgpuClient::new` requests, such as extensions or
    /// queues opened through wgpu-hal.
    ///
    /// `adapter` must be the adapter that opened `device`. It supplies the
    /// reported name and backend, and the compute capability check. The
    /// reported limits and subgroup support come from `device`, since a
    /// device can use less than its adapter offers. `WgpuClient::new` requests
    /// the adapter's maxima. A device opened with less caps buffers at its own
    /// limits.
    ///
    /// `device` and `queue` are cloneable handles, so the caller can keep
    /// clones. A refused install drops the handles it received.
    ///
    /// # Errors
    ///
    /// - `WgpuError::IndexInUse` when `index` already has a client, either
    ///   installed or created on first use.
    /// - `WgpuError::DeviceUnsupported` when the adapter has no compute
    ///   shaders, or `device` has a limit below what numr kernels need.
    #[cfg(feature = "wgpu-raw-interop")]
    pub fn install(
        index: usize,
        adapter: &Adapter,
        device: Device,
        queue: Queue,
    ) -> Result<Self, WgpuError> {
        let info = describe_installed(index, adapter, &device)?;
        let client = Self::from_parts(WgpuDevice::with_info(index, info), device, queue);
        super::cache::install_client(client)
    }

    /// Build a client over an open device and queue.
    fn from_parts(device_id: WgpuDevice, wgpu_device: Device, queue: Queue) -> Self {
        let wgpu_device = Arc::new(wgpu_device);
        let queue = Arc::new(queue);

        let allocator = WgpuAllocator {
            device: wgpu_device.clone(),
            queue: queue.clone(),
        };

        let raw_handle = WgpuRawHandle {
            device: wgpu_device.clone(),
            queue: queue.clone(),
        };

        let pipeline_cache = Arc::new(PipelineCache::new(wgpu_device.clone(), queue.clone()));

        Self {
            device_id,
            wgpu_device,
            queue,
            allocator,
            raw_handle,
            pipeline_cache,
        }
    }

    /// Get reference to the WebGPU device.
    #[cfg(feature = "wgpu-raw-interop")]
    #[inline]
    pub fn wgpu_device(&self) -> &Device {
        &self.wgpu_device
    }

    #[cfg(not(feature = "wgpu-raw-interop"))]
    #[inline]
    pub(crate) fn wgpu_device(&self) -> &Device {
        &self.wgpu_device
    }

    /// Get Arc-wrapped WebGPU device.
    #[cfg(feature = "wgpu-raw-interop")]
    #[inline]
    pub fn wgpu_device_arc(&self) -> &Arc<Device> {
        &self.wgpu_device
    }

    /// Get reference to the WebGPU queue.
    #[cfg(feature = "wgpu-raw-interop")]
    #[inline]
    pub fn wgpu_queue(&self) -> &Queue {
        &self.queue
    }

    #[cfg(not(feature = "wgpu-raw-interop"))]
    #[inline]
    pub(crate) fn wgpu_queue(&self) -> &Queue {
        &self.queue
    }

    /// Get reference to the pipeline cache.
    #[cfg(feature = "wgpu-raw-interop")]
    #[inline]
    pub fn pipeline_cache(&self) -> &PipelineCache {
        &self.pipeline_cache
    }

    #[cfg(not(feature = "wgpu-raw-interop"))]
    #[inline]
    pub(crate) fn pipeline_cache(&self) -> &PipelineCache {
        &self.pipeline_cache
    }

    /// Create a storage buffer for tensor data.
    ///
    /// Storage buffers can be read/written by compute shaders.
    #[cfg(feature = "wgpu-raw-interop")]
    pub fn create_storage_buffer(&self, label: &str, size: u64) -> Buffer {
        self.create_storage_buffer_inner(label, size)
    }

    #[cfg(all(test, not(feature = "wgpu-raw-interop")))]
    pub(crate) fn create_storage_buffer(&self, label: &str, size: u64) -> Buffer {
        self.create_storage_buffer_inner(label, size)
    }

    #[cfg(any(test, feature = "wgpu-raw-interop"))]
    fn create_storage_buffer_inner(&self, label: &str, size: u64) -> Buffer {
        self.wgpu_device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Create a staging buffer for CPU readback.
    ///
    /// Staging buffers are used to copy data from GPU to CPU.
    #[cfg(feature = "wgpu-raw-interop")]
    pub fn create_staging_buffer(&self, label: &str, size: u64) -> Buffer {
        self.create_staging_buffer_inner(label, size)
    }

    #[cfg(not(feature = "wgpu-raw-interop"))]
    pub(crate) fn create_staging_buffer(&self, label: &str, size: u64) -> Buffer {
        self.create_staging_buffer_inner(label, size)
    }

    fn create_staging_buffer_inner(&self, label: &str, size: u64) -> Buffer {
        self.wgpu_device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a uniform buffer for shader parameters.
    #[cfg(feature = "wgpu-raw-interop")]
    pub fn create_uniform_buffer(&self, label: &str, size: u64) -> Buffer {
        self.create_uniform_buffer_inner(label, size)
    }

    #[cfg(not(feature = "wgpu-raw-interop"))]
    pub(crate) fn create_uniform_buffer(&self, label: &str, size: u64) -> Buffer {
        self.create_uniform_buffer_inner(label, size)
    }

    fn create_uniform_buffer_inner(&self, label: &str, size: u64) -> Buffer {
        self.wgpu_device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Write data to a buffer.
    #[cfg(feature = "wgpu-raw-interop")]
    pub fn write_buffer<T: bytemuck::Pod>(&self, buffer: &Buffer, data: &[T]) {
        self.write_buffer_inner(buffer, data);
    }

    #[cfg(not(feature = "wgpu-raw-interop"))]
    pub(crate) fn write_buffer<T: bytemuck::Pod>(&self, buffer: &Buffer, data: &[T]) {
        self.write_buffer_inner(buffer, data);
    }

    fn write_buffer_inner<T: bytemuck::Pod>(&self, buffer: &Buffer, data: &[T]) {
        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(data));
    }

    /// Submit commands and wait for completion.
    #[cfg(feature = "wgpu-raw-interop")]
    pub fn submit_and_wait(&self, encoder: wgpu::CommandEncoder) {
        self.submit_and_wait_inner(encoder);
    }

    #[cfg(not(feature = "wgpu-raw-interop"))]
    pub(crate) fn submit_and_wait(&self, encoder: wgpu::CommandEncoder) {
        self.submit_and_wait_inner(encoder);
    }

    fn submit_and_wait_inner(&self, encoder: wgpu::CommandEncoder) {
        let submission = self.queue.submit(std::iter::once(encoder.finish()));
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.wgpu_device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(60)),
        });
        #[cfg(target_arch = "wasm32")]
        let _ = submission;
    }

    /// Read buffer data back to CPU asynchronously without blocking the calling thread.
    ///
    /// Awaits `map_async` over an async oneshot channel. Safe on browser `wasm32`
    /// where `PollType::Wait` is a no-op and blocking channels deadlock.
    #[cfg(feature = "wgpu-raw-interop")]
    pub async fn read_buffer_async<T: bytemuck::Pod>(
        &self,
        staging: &Buffer,
        output: &mut [T],
    ) -> crate::error::Result<()> {
        self.read_buffer_async_inner(staging, output).await
    }

    async fn read_buffer_async_inner<T: bytemuck::Pod>(
        &self,
        staging: &Buffer,
        output: &mut [T],
    ) -> crate::error::Result<()> {
        let slice = staging.slice(..);

        let (sender, receiver) = super::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result);
        });

        #[cfg(not(target_arch = "wasm32"))]
        self.wgpu_device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(60)),
            })
            .map_err(|e| {
                crate::error::Error::Backend(format!(
                    "GPU poll failed during async buffer read: {e}"
                ))
            })?;

        let map_result = receiver.await.ok_or_else(|| {
            crate::error::Error::Backend(
                "map_async callback was not invoked during async buffer read".into(),
            )
        })?;
        map_result.map_err(|e| {
            crate::error::Error::Backend(format!("map_async failed during async buffer read: {e}"))
        })?;

        {
            let data = slice.get_mapped_range().map_err(|e| {
                crate::error::Error::Backend(format!(
                    "get_mapped_range failed during async buffer read: {e}"
                ))
            })?;
            let src: &[T] = bytemuck::cast_slice(&data);
            output.copy_from_slice(&src[..output.len()]);
        }

        staging.unmap();
        Ok(())
    }

    /// Copy a `WgpuRuntime` tensor back to a host `Vec<T>` asynchronously.
    ///
    /// Safe on both native targets and browser `wasm32` WebGPU runtimes (no blocking poll).
    pub async fn read_tensor_to_vec_async<T: bytemuck::Pod>(
        &self,
        tensor: &Tensor<WgpuRuntime>,
    ) -> crate::error::Result<Vec<T>> {
        let dtype = tensor.dtype();
        let dtype_bytes = dtype.size_in_bytes();
        let elem_bytes = std::mem::size_of::<T>();
        if elem_bytes != dtype_bytes {
            return Err(crate::error::Error::HostReadTypeMismatch {
                dtype: dtype.short_name(),
                dtype_bytes,
                elem_bytes,
            });
        }

        let numel = tensor.numel();
        if numel == 0 {
            return Ok(Vec::new());
        }

        let contiguous = tensor.contiguous()?;
        let src_ptr = contiguous.storage().ptr();
        let byte_len = numel * dtype_bytes;
        let aligned_len = byte_len.div_ceil(4) * 4;

        let buffer = super::allocator::get_buffer(src_ptr).ok_or_else(|| {
            crate::error::Error::Backend("Buffer not found for read_tensor_to_vec_async".into())
        })?;

        let staging = self.create_staging_buffer("read_tensor_async_staging", aligned_len as u64);
        let mut encoder =
            self.wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("read_tensor_to_vec_async"),
                });
        encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, aligned_len as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let mut bytes = vec![0u8; byte_len];
        self.read_buffer_async_inner(&staging, &mut bytes).await?;

        let mut out = vec![T::zeroed(); numel];
        bytemuck::cast_slice_mut(&mut out).copy_from_slice(&bytes);
        Ok(out)
    }

    /// Read a single-element `WgpuRuntime` tensor back to a host scalar asynchronously.
    ///
    /// Safe on both native targets and browser `wasm32` WebGPU runtimes.
    pub async fn read_scalar_async<T: bytemuck::Pod + Copy>(
        &self,
        tensor: &Tensor<WgpuRuntime>,
    ) -> crate::error::Result<T> {
        if tensor.numel() != 1 {
            return Err(crate::error::Error::InvalidArgument {
                arg: "tensor",
                reason: format!(
                    "read_scalar_async requires a 1-element tensor, got {} elements",
                    tensor.numel()
                ),
            });
        }
        let values = self.read_tensor_to_vec_async::<T>(tensor).await?;
        Ok(values[0])
    }

    /// Read buffer data back to CPU (blocking).
    #[cfg(feature = "wgpu-raw-interop")]
    pub fn read_buffer<T: bytemuck::Pod>(
        &self,
        staging: &Buffer,
        output: &mut [T],
    ) -> crate::error::Result<()> {
        self.read_buffer_inner(staging, output)
    }

    #[cfg(not(feature = "wgpu-raw-interop"))]
    pub(crate) fn read_buffer<T: bytemuck::Pod>(
        &self,
        staging: &Buffer,
        output: &mut [T],
    ) -> crate::error::Result<()> {
        self.read_buffer_inner(staging, output)
    }

    fn read_buffer_inner<T: bytemuck::Pod>(
        &self,
        staging: &Buffer,
        output: &mut [T],
    ) -> crate::error::Result<()> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (staging, output);
            return Err(crate::error::Error::Backend(
                "synchronous read_buffer is not supported on wasm32; use read_buffer_async".into(),
            ));
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let slice = staging.slice(..);

            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });

            self.wgpu_device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(60)),
                })
                .map_err(|e| {
                    crate::error::Error::Backend(format!("GPU poll failed during buffer read: {e}"))
                })?;

            // Check map_async result
            let map_result = receiver.recv().map_err(|_| {
                crate::error::Error::Backend(
                    "map_async callback was not invoked during buffer read".into(),
                )
            })?;
            map_result.map_err(|e| {
                crate::error::Error::Backend(format!("map_async failed during buffer read: {e}"))
            })?;

            {
                let data = slice.get_mapped_range().map_err(|e| {
                    crate::error::Error::Backend(format!(
                        "get_mapped_range failed during buffer read: {e}"
                    ))
                })?;
                let src: &[T] = bytemuck::cast_slice(&data);
                output.copy_from_slice(&src[..output.len()]);
            }

            staging.unmap();
            Ok(())
        }
    }
}

impl RuntimeClient<WgpuRuntime> for WgpuClient {
    fn device(&self) -> &WgpuDevice {
        &self.device_id
    }

    fn synchronize(&self) {
        // Wait for all GPU work to complete (native only; browser WebGPU polls via event loop)
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.wgpu_device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(60)),
        });
    }

    fn allocator(&self) -> &WgpuAllocator {
        &self.allocator
    }
}

// ============================================================================
// WgpuRawHandle
// ============================================================================

/// Raw handle for custom kernel launching.
///
/// Provides access to the WebGPU device and queue for users who want
/// to launch their own compute shaders outside of numr's operation system.
///
/// # Example
///
/// ```ignore
/// let client = WgpuRuntime::default_client(&device);
/// let handle = WgpuRuntime::raw_handle(&client);
///
/// // Use handle.device to create pipelines
/// // Use handle.queue for command submission
/// ```
#[derive(Clone)]
pub struct WgpuRawHandle {
    /// WebGPU device for pipeline creation
    #[cfg_attr(not(feature = "wgpu-raw-interop"), allow(dead_code))]
    pub(crate) device: Arc<Device>,
    /// WebGPU queue for command submission
    #[cfg_attr(not(feature = "wgpu-raw-interop"), allow(dead_code))]
    pub(crate) queue: Arc<Queue>,
}

#[cfg(feature = "wgpu-raw-interop")]
impl WgpuRawHandle {
    /// Raw `wgpu::Device` handle.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Raw `wgpu::Queue` handle.
    #[inline]
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{Device, RuntimeClient};

    #[test]
    fn test_wgpu_client_creation() {
        let device = WgpuDevice::new(0);
        match WgpuClient::new(device) {
            Ok(client) => {
                println!("Client created for: {}", client.device().name());
                assert_eq!(client.device().id(), 0);
            }
            Err(e) => {
                println!("No GPU available, skipping test: {}", e);
            }
        }
    }

    #[test]
    fn test_wgpu_buffer_roundtrip() {
        let device = WgpuDevice::new(0);
        match WgpuClient::new(device) {
            Ok(client) => {
                // Create test data
                let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
                let size = data.len() * std::mem::size_of::<f32>();

                // Create storage buffer
                let storage = client.create_storage_buffer("test", size as u64);

                // Write data to GPU
                client.write_buffer(&storage, &data);

                // Create staging buffer for readback
                let staging = client.create_staging_buffer("staging", size as u64);

                // Copy storage -> staging
                let mut encoder =
                    client
                        .wgpu_device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("copy"),
                        });
                encoder.copy_buffer_to_buffer(&storage, 0, &staging, 0, size as u64);
                client.submit_and_wait(encoder);

                // Read back
                let mut result = vec![0.0f32; data.len()];
                client
                    .read_buffer(&staging, &mut result)
                    .expect("readback should succeed");

                assert_eq!(data, result);
                println!("Buffer roundtrip successful: {:?}", result);
            }
            Err(e) => {
                println!("No GPU available, skipping test: {}", e);
            }
        }
    }
}
