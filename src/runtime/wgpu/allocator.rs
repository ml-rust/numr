//! WebGPU buffer allocator and the registry that maps allocation IDs to buffers.

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

use crate::runtime::Allocator;

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
    pub(super) device: Arc<Device>,
    #[allow(dead_code)] // Reserved for future compute shader allocations
    pub(super) queue: Arc<Queue>,
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
pub fn get_buffer(id: u64) -> Option<Arc<Buffer>> {
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
    ///
    /// # Errors
    ///
    /// `BackendLimitation` when the size exceeds the device's `max_buffer_size`.
    fn allocate(&self, size_bytes: usize) -> crate::error::Result<u64> {
        if size_bytes == 0 {
            return Ok(0);
        }

        // WebGPU requires buffer sizes to be aligned to 4 bytes
        let aligned_size = size_bytes.div_ceil(4) * 4;
        let max_buffer_size = self.device.limits().max_buffer_size;
        if aligned_size as u64 > max_buffer_size {
            return Err(crate::error::Error::BackendLimitation {
                backend: "wgpu",
                operation: "allocate",
                reason: format!(
                    "{aligned_size} bytes exceed the device's max_buffer_size of {max_buffer_size}"
                ),
            });
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::RuntimeClient;
    use crate::runtime::wgpu::{WgpuClient, WgpuDevice};

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
    fn allocation_past_max_buffer_size_is_a_backend_limitation() {
        let device = WgpuDevice::new(0);
        let Ok(client) = WgpuClient::new(device) else {
            return;
        };
        let max = client.allocator().device.limits().max_buffer_size;
        let Ok(size) = usize::try_from(max + 4) else {
            return;
        };
        match client.allocator().allocate(size) {
            Err(crate::error::Error::BackendLimitation { operation, .. }) => {
                assert_eq!(operation, "allocate");
            }
            other => panic!("expected BackendLimitation, got {other:?}"),
        }
    }
}
