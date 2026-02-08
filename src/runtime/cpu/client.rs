//! CPU client and allocator implementation

use super::device::CpuDevice;
use super::runtime::CpuRuntime;
use crate::runtime::{DefaultAllocator, RuntimeClient};
use std::alloc::{Layout as AllocLayout, alloc_zeroed, dealloc};

/// CPU client for operation dispatch
#[derive(Clone, Debug)]
pub struct CpuClient {
    pub(crate) device: CpuDevice,
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
            let layout =
                AllocLayout::from_size_align(size, align).expect("Invalid allocation layout");
            let ptr = unsafe { alloc_zeroed(layout) };
            if ptr.is_null() {
                // Note: This closure returns u64, not Result, so we must panic here.
                // The proper error path is through Runtime::allocate which returns Result.
                panic!("Out of memory: failed to allocate {} bytes", size);
            }
            ptr as u64
        },
        |ptr, size, _dev| {
            if ptr == 0 || size == 0 {
                return;
            }
            let align = 64;
            let layout =
                AllocLayout::from_size_align(size, align).expect("Invalid allocation layout");
            unsafe {
                dealloc(ptr as *mut u8, layout);
            }
        },
    )
}
