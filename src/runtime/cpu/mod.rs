//! CPU runtime implementation
//!
//! The CPU runtime uses standard heap allocation and provides a reference
//! implementation for all tensor operations.

use super::{DefaultAllocator, Device, Runtime, RuntimeClient};
use std::alloc::{alloc_zeroed, dealloc, Layout as AllocLayout};

/// CPU compute runtime
///
/// This is the default runtime that works on any platform.
/// Memory is allocated on the heap using the system allocator.
#[derive(Clone, Debug, Default)]
pub struct CpuRuntime;

/// CPU device (there's only one: the host CPU)
#[derive(Clone, Debug, Default)]
pub struct CpuDevice {
    id: usize,
}

impl CpuDevice {
    /// Create a new CPU device
    pub fn new() -> Self {
        Self { id: 0 }
    }
}

impl Device for CpuDevice {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> String {
        "cpu".to_string()
    }
}

/// CPU client for operation dispatch
#[derive(Clone, Debug)]
pub struct CpuClient {
    device: CpuDevice,
    allocator: CpuAllocator,
}

impl CpuClient {
    /// Create a new CPU client
    pub fn new(device: CpuDevice) -> Self {
        let allocator = create_cpu_allocator(device.clone());
        Self { device, allocator }
    }
}

impl RuntimeClient<CpuRuntime> for CpuClient {
    fn device(&self) -> &CpuDevice {
        &self.device
    }

    fn synchronize(&self) {
        // CPU operations are synchronous, nothing to do
    }

    fn allocator(&self) -> &CpuAllocator {
        &self.allocator
    }
}

/// CPU-specific allocator type alias
pub type CpuAllocator = DefaultAllocator<CpuDevice>;

/// Create a CPU allocator for the given device
fn create_cpu_allocator(device: CpuDevice) -> CpuAllocator {
    DefaultAllocator::new(
            device,
            |size, _dev| {
                if size == 0 {
                    return 0;
                }
                let align = 64; // AVX-512 alignment
                let layout = AllocLayout::from_size_align(size, align)
                    .expect("Invalid allocation layout");
                let ptr = unsafe { alloc_zeroed(layout) };
                if ptr.is_null() {
                    panic!("Failed to allocate {} bytes", size);
                }
                ptr as u64
            },
            |ptr, size, _dev| {
                if ptr == 0 || size == 0 {
                    return;
                }
                let align = 64;
                let layout = AllocLayout::from_size_align(size, align)
                    .expect("Invalid allocation layout");
                unsafe {
                    dealloc(ptr as *mut u8, layout);
                }
            },
        )
}

impl Runtime for CpuRuntime {
    type Device = CpuDevice;
    type Client = CpuClient;
    type Allocator = CpuAllocator;
    type RawHandle = (); // CPU has no special handle needed

    fn name() -> &'static str {
        "cpu"
    }

    fn allocate(size_bytes: usize, _device: &Self::Device) -> u64 {
        if size_bytes == 0 {
            return 0;
        }

        // Use aligned allocation for SIMD compatibility
        let align = 64; // AVX-512 alignment
        let layout = AllocLayout::from_size_align(size_bytes, align)
            .expect("Invalid allocation layout");

        let ptr = unsafe { alloc_zeroed(layout) };

        if ptr.is_null() {
            panic!("Failed to allocate {} bytes", size_bytes);
        }

        ptr as u64
    }

    fn deallocate(ptr: u64, size_bytes: usize, _device: &Self::Device) {
        if ptr == 0 || size_bytes == 0 {
            return;
        }

        let align = 64;
        let layout = AllocLayout::from_size_align(size_bytes, align)
            .expect("Invalid allocation layout");

        unsafe {
            dealloc(ptr as *mut u8, layout);
        }
    }

    fn copy_to_device(src: &[u8], dst: u64, _device: &Self::Device) {
        if src.is_empty() || dst == 0 {
            return;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst as *mut u8, src.len());
        }
    }

    fn copy_from_device(src: u64, dst: &mut [u8], _device: &Self::Device) {
        if dst.is_empty() || src == 0 {
            return;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst.as_mut_ptr(), dst.len());
        }
    }

    fn copy_within_device(src: u64, dst: u64, size_bytes: usize, _device: &Self::Device) {
        if size_bytes == 0 || src == 0 || dst == 0 {
            return;
        }

        unsafe {
            // Use copy (not copy_nonoverlapping) in case src and dst overlap
            std::ptr::copy(src as *const u8, dst as *mut u8, size_bytes);
        }
    }

    fn default_device() -> Self::Device {
        CpuDevice::new()
    }

    fn default_client(device: &Self::Device) -> Self::Client {
        CpuClient::new(device.clone())
    }

    fn raw_handle(_client: &Self::Client) -> &Self::RawHandle {
        &()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Allocator;

    #[test]
    fn test_allocate_deallocate() {
        let device = CpuDevice::new();
        let ptr = CpuRuntime::allocate(1024, &device);
        assert_ne!(ptr, 0);
        CpuRuntime::deallocate(ptr, 1024, &device);
    }

    #[test]
    fn test_copy_roundtrip() {
        let device = CpuDevice::new();
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let ptr = CpuRuntime::allocate(data.len(), &device);
        CpuRuntime::copy_to_device(&data, ptr, &device);

        let mut result = vec![0u8; data.len()];
        CpuRuntime::copy_from_device(ptr, &mut result, &device);

        assert_eq!(data, result);

        CpuRuntime::deallocate(ptr, data.len(), &device);
    }

    #[test]
    fn test_copy_within_device() {
        let device = CpuDevice::new();
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let src = CpuRuntime::allocate(data.len(), &device);
        let dst = CpuRuntime::allocate(data.len(), &device);

        CpuRuntime::copy_to_device(&data, src, &device);
        CpuRuntime::copy_within_device(src, dst, data.len(), &device);

        let mut result = vec![0u8; data.len()];
        CpuRuntime::copy_from_device(dst, &mut result, &device);

        assert_eq!(data, result);

        CpuRuntime::deallocate(src, data.len(), &device);
        CpuRuntime::deallocate(dst, data.len(), &device);
    }

    #[test]
    fn test_zero_allocation() {
        let device = CpuDevice::new();
        let ptr = CpuRuntime::allocate(0, &device);
        assert_eq!(ptr, 0);
        CpuRuntime::deallocate(ptr, 0, &device); // Should not panic
    }

    #[test]
    fn test_client_allocator() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let ptr = client.allocator().allocate(256);
        assert_ne!(ptr, 0);
        client.allocator().deallocate(ptr, 256);
    }

    #[test]
    fn test_raw_handle() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let _handle: &() = CpuRuntime::raw_handle(&client);
        // For CPU, handle is just ()
    }
}
