//! CUDA runtime implementation (requires `cuda` feature)
//!
//! This module provides GPU acceleration via NVIDIA CUDA.
//!
//! # Status: NOT IMPLEMENTED
//!
//! **WARNING:** This module is a placeholder stub. All runtime methods will
//! panic if called. CUDA backend implementation is planned for Phase 2.
//!
//! Do not enable the `cuda` feature unless you are developing the CUDA backend.
//! Production code should use `CpuRuntime` until CUDA support is complete.

use super::{Device, Runtime, RuntimeClient};

/// CUDA compute runtime
#[derive(Clone, Debug, Default)]
pub struct CudaRuntime;

/// CUDA device (represents a specific GPU)
#[derive(Clone, Debug)]
pub struct CudaDevice {
    id: usize,
}

impl CudaDevice {
    /// Create a device for the specified GPU
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

impl Device for CudaDevice {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> String {
        format!("cuda:{}", self.id)
    }
}

/// CUDA client for operation dispatch
#[derive(Debug)]
pub struct CudaClient {
    device: CudaDevice,
    // TODO: Add cudarc context, stream, etc.
}

impl CudaClient {
    /// Create a new CUDA client
    pub fn new(device: CudaDevice) -> Self {
        Self { device }
    }
}

impl RuntimeClient<CudaRuntime> for CudaClient {
    fn device(&self) -> &CudaDevice {
        &self.device
    }

    fn synchronize(&self) {
        // TODO: Implement CUDA synchronization
    }
}

impl Runtime for CudaRuntime {
    type Device = CudaDevice;
    type Client = CudaClient;

    fn name() -> &'static str {
        "cuda"
    }

    fn allocate(size_bytes: usize, device: &Self::Device) -> u64 {
        let _ = (size_bytes, device);
        panic!(
            "CUDA backend not implemented (Phase 2). \
             Use CpuRuntime for production code. \
             See NUMR_TDD.md for implementation roadmap."
        )
    }

    fn deallocate(ptr: u64, size_bytes: usize, device: &Self::Device) {
        let _ = (ptr, size_bytes, device);
        panic!(
            "CUDA backend not implemented (Phase 2). \
             Use CpuRuntime for production code."
        )
    }

    fn copy_to_device(src: &[u8], dst: u64, device: &Self::Device) {
        let _ = (src, dst, device);
        panic!(
            "CUDA backend not implemented (Phase 2). \
             Use CpuRuntime for production code."
        )
    }

    fn copy_from_device(src: u64, dst: &mut [u8], device: &Self::Device) {
        let _ = (src, dst, device);
        panic!(
            "CUDA backend not implemented (Phase 2). \
             Use CpuRuntime for production code."
        )
    }

    fn default_device() -> Self::Device {
        CudaDevice::new(0)
    }
}
