//! WebGPU Client implementation.
//!
//! `WgpuClient` owns the WebGPU device and queue for operation dispatch.
//!
//! # Thread Safety
//!
//! `WgpuClient` is `Clone` and the underlying wgpu::Device and wgpu::Queue
//! are already `Send + Sync` by design.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

use super::WgpuRuntime;
use super::device::{WgpuDevice, WgpuError, query_adapter_info_blocking};
use super::shaders::PipelineCache;
use crate::runtime::{Allocator, RuntimeClient};

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
    /// Create a new WebGPU client for a device.
    ///
    /// This initializes the WebGPU device and queue.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No suitable GPU adapter is found
    /// - Device creation fails
    pub fn new(device: WgpuDevice) -> Result<Self, WgpuError> {
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
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                })
                .await
        })
        .map_err(|e| WgpuError::DeviceError(format!("{:?}", e)))?;

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

        let device_with_info = WgpuDevice::with_info(device.index, info);

        Ok(Self {
            device_id: device_with_info,
            wgpu_device,
            queue,
            allocator,
            raw_handle,
            pipeline_cache,
        })
    }

    /// Get reference to the WebGPU device.
    #[inline]
    pub fn wgpu_device(&self) -> &Device {
        &self.wgpu_device
    }

    /// Get Arc-wrapped WebGPU device.
    #[inline]
    pub fn wgpu_device_arc(&self) -> &Arc<Device> {
        &self.wgpu_device
    }

    /// Get reference to the WebGPU queue.
    #[inline]
    pub fn wgpu_queue(&self) -> &Queue {
        &self.queue
    }

    /// Get reference to the pipeline cache.
    #[inline]
    pub fn pipeline_cache(&self) -> &PipelineCache {
        &self.pipeline_cache
    }

    /// Create a storage buffer for tensor data.
    ///
    /// Storage buffers can be read/written by compute shaders.
    pub fn create_storage_buffer(&self, label: &str, size: u64) -> Buffer {
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
    pub fn create_staging_buffer(&self, label: &str, size: u64) -> Buffer {
        self.wgpu_device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a uniform buffer for shader parameters.
    pub fn create_uniform_buffer(&self, label: &str, size: u64) -> Buffer {
        self.wgpu_device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Write data to a buffer.
    pub fn write_buffer<T: bytemuck::Pod>(&self, buffer: &Buffer, data: &[T]) {
        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(data));
    }

    /// Submit commands and wait for completion.
    pub fn submit_and_wait(&self, encoder: wgpu::CommandEncoder) {
        let submission = self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.wgpu_device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(60)),
        });
    }

    /// Read buffer data back to CPU (blocking).
    pub fn read_buffer<T: bytemuck::Pod>(
        &self,
        staging: &Buffer,
        output: &mut [T],
    ) -> crate::error::Result<()> {
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
            let data = slice.get_mapped_range();
            let src: &[T] = bytemuck::cast_slice(&data);
            output.copy_from_slice(&src[..output.len()]);
        }

        staging.unmap();
        Ok(())
    }
}

impl RuntimeClient<WgpuRuntime> for WgpuClient {
    fn device(&self) -> &WgpuDevice {
        &self.device_id
    }

    fn synchronize(&self) {
        // Wait for all GPU work to complete
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
// WgpuAllocator
// ============================================================================

/// WebGPU buffer allocator.
///
/// Allocates storage buffers for tensor data. Buffers are identified by
/// their GPU address (buffer ID cast to u64).
///
/// # Implementation Note
///
/// WebGPU doesn't expose raw pointers like CUDA. Instead, we use a buffer
/// registry that maps allocation IDs to wgpu::Buffer objects.
#[derive(Clone)]
pub struct WgpuAllocator {
    device: Arc<Device>,
    #[allow(dead_code)] // Reserved for future compute shader allocations
    queue: Arc<Queue>,
}

/// Global buffer registry for mapping allocation IDs to buffers.
///
/// This is necessary because WebGPU doesn't expose raw GPU pointers.
/// We use a global registry to map u64 IDs to Buffer objects.
static BUFFER_REGISTRY: std::sync::OnceLock<parking_lot::Mutex<HashMap<u64, Arc<Buffer>>>> =
    std::sync::OnceLock::new();

/// Counter for generating unique buffer IDs.
static BUFFER_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn get_buffer_registry() -> &'static parking_lot::Mutex<HashMap<u64, Arc<Buffer>>> {
    BUFFER_REGISTRY.get_or_init(|| parking_lot::Mutex::new(HashMap::new()))
}

/// Get a buffer by its ID.
pub(crate) fn get_buffer(id: u64) -> Option<Arc<Buffer>> {
    if id == 0 {
        return None;
    }
    let registry = get_buffer_registry();
    let guard = registry.lock();
    guard.get(&id).cloned()
}

impl Allocator for WgpuAllocator {
    /// Allocate a storage buffer and return its ID.
    ///
    /// The returned ID can be used with `get_buffer()` to retrieve the
    /// actual wgpu::Buffer for operations.
    fn allocate(&self, size_bytes: usize) -> crate::error::Result<u64> {
        if size_bytes == 0 {
            return Ok(0);
        }

        // WebGPU requires buffer sizes to be aligned to 4 bytes
        let aligned_size = size_bytes.div_ceil(4) * 4;

        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("numr tensor buffer"),
            size: aligned_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Generate unique ID and register buffer
        let id = BUFFER_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let registry = get_buffer_registry();
        let mut guard = registry.lock();
        guard.insert(id, Arc::new(buffer));

        Ok(id)
    }

    fn deallocate(&self, ptr: u64, _size_bytes: usize) {
        if ptr == 0 {
            return;
        }

        // Remove buffer from registry (drop will release GPU memory)
        let registry = get_buffer_registry();
        let mut guard = registry.lock();
        guard.remove(&ptr);
    }

    fn is_frozen(&self) -> bool {
        false // WebGPU allocator doesn't support freeze
    }

    fn freeze(&self) -> bool {
        true // No-op, always succeeds
    }

    fn unfreeze(&self) {
        // No-op
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
    pub device: Arc<Device>,
    /// WebGPU queue for command submission
    pub queue: Arc<Queue>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{Allocator, Device, RuntimeClient};

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
    fn test_wgpu_allocator() {
        let device = WgpuDevice::new(0);
        match WgpuClient::new(device) {
            Ok(client) => {
                let allocator = client.allocator();

                // Allocate buffer
                let id = allocator.allocate(1024).expect("allocation should succeed");
                assert_ne!(id, 0);

                // Verify buffer exists
                let buffer = get_buffer(id);
                assert!(buffer.is_some());

                // Deallocate
                allocator.deallocate(id, 1024);

                // Buffer should be gone
                let buffer = get_buffer(id);
                assert!(buffer.is_none());
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
