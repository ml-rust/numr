//! WebGPU runtime implementation (requires `wgpu` feature)
//!
//! This module provides cross-platform GPU acceleration via WebGPU.
//!
//! # Status: NOT IMPLEMENTED
//!
//! **WARNING:** This module is a placeholder stub. All runtime methods will
//! panic if called. WebGPU backend implementation is planned for Phase 3.
//!
//! Do not enable the `wgpu` feature unless you are developing the WebGPU backend.
//! Production code should use `CpuRuntime` until WebGPU support is complete.

use super::{Device, Runtime, RuntimeClient};

/// WebGPU compute runtime
#[derive(Clone, Debug, Default)]
pub struct WgpuRuntime;

/// WebGPU device
#[derive(Clone, Debug)]
pub struct WgpuDevice {
    id: usize,
}

impl WgpuDevice {
    /// Create a device for the specified adapter
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

impl Device for WgpuDevice {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> String {
        format!("wgpu:{}", self.id)
    }
}

/// WebGPU client for operation dispatch
#[derive(Debug)]
pub struct WgpuClient {
    device: WgpuDevice,
    // TODO: Add wgpu::Device, wgpu::Queue, etc.
}

impl WgpuClient {
    /// Create a new WebGPU client
    pub fn new(device: WgpuDevice) -> Self {
        Self { device }
    }
}

impl RuntimeClient<WgpuRuntime> for WgpuClient {
    fn device(&self) -> &WgpuDevice {
        &self.device
    }

    fn synchronize(&self) {
        // TODO: Implement WebGPU synchronization
    }
}

impl Runtime for WgpuRuntime {
    type Device = WgpuDevice;
    type Client = WgpuClient;

    fn name() -> &'static str {
        "wgpu"
    }

    fn allocate(size_bytes: usize, device: &Self::Device) -> u64 {
        let _ = (size_bytes, device);
        panic!(
            "WebGPU backend not implemented (Phase 3). \
             Use CpuRuntime for production code. \
             See NUMR_TDD.md for implementation roadmap."
        )
    }

    fn deallocate(ptr: u64, size_bytes: usize, device: &Self::Device) {
        let _ = (ptr, size_bytes, device);
        panic!(
            "WebGPU backend not implemented (Phase 3). \
             Use CpuRuntime for production code."
        )
    }

    fn copy_to_device(src: &[u8], dst: u64, device: &Self::Device) {
        let _ = (src, dst, device);
        panic!(
            "WebGPU backend not implemented (Phase 3). \
             Use CpuRuntime for production code."
        )
    }

    fn copy_from_device(src: u64, dst: &mut [u8], device: &Self::Device) {
        let _ = (src, dst, device);
        panic!(
            "WebGPU backend not implemented (Phase 3). \
             Use CpuRuntime for production code."
        )
    }

    fn default_device() -> Self::Device {
        WgpuDevice::new(0)
    }
}
