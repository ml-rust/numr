//! WebGPU runtime implementation
//!
//! This module provides cross-platform GPU acceleration via WebGPU.
//!
//! # Features
//!
//! - `WgpuDevice` - Represents a WebGPU GPU adapter
//! - `WgpuClient` - Manages device and queue, dispatches operations
//! - `WgpuRuntime` - Implements the generic Runtime trait
//! - `TensorOps` - WebGPU-accelerated tensor operations
//!
//! # Backend Support
//!
//! WebGPU abstracts over multiple GPU APIs:
//! - Vulkan (Linux, Windows, Android)
//! - Metal (macOS, iOS)
//! - DirectX 12 (Windows)
//! - OpenGL (fallback)
//!
//! # Panics
//!
//! The following operations may panic on WebGPU errors:
//!
//! - `Runtime::allocate` - Panics if buffer creation fails
//! - `Runtime::copy_to_device` - Panics if write operation fails
//! - `Runtime::copy_from_device` - Panics if read operation fails

mod client;
mod device;
mod fft;
mod linalg;
mod ops;
pub mod shaders;
mod special;
mod statistics;

// Re-export runtime types for this backend
pub use crate::tensor::Tensor;
pub use client::{WgpuAllocator, WgpuClient, WgpuRawHandle};
pub use device::{WgpuDevice, WgpuError};

use crate::runtime::{Allocator, Runtime, RuntimeClient};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

// ============================================================================
// Internal Helpers
// ============================================================================

/// Global client cache: device index -> cached WgpuClient
///
/// This caches WgpuClient instances per device to avoid creating new
/// WebGPU devices and queues on every operation.
static CLIENT_CACHE: OnceLock<Mutex<HashMap<usize, WgpuClient>>> = OnceLock::new();

/// Get or create a cached WgpuClient for a device.
fn get_or_create_client(device: &WgpuDevice) -> WgpuClient {
    let cache = CLIENT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache_guard = cache.lock().unwrap_or_else(|e| e.into_inner());

    if let Some(client) = cache_guard.get(&device.index) {
        return client.clone();
    }

    // Create new client and cache it
    let client = WgpuClient::new(device.clone()).expect("Failed to create WGPU client");
    cache_guard.insert(device.index, client.clone());

    client
}

// ============================================================================
// Runtime Implementation
// ============================================================================

/// WebGPU Runtime adapter
///
/// Implements the generic Runtime trait for WebGPU backend.
/// Provides cross-platform GPU acceleration.
#[derive(Clone, Debug, Default)]
pub struct WgpuRuntime;

impl Runtime for WgpuRuntime {
    type Device = WgpuDevice;
    type Client = WgpuClient;
    type Allocator = WgpuAllocator;
    type RawHandle = WgpuRawHandle;

    fn name() -> &'static str {
        "wgpu"
    }

    fn supports_graph_capture() -> bool {
        false // WebGPU doesn't have CUDA-style graph capture
    }

    /// Allocate GPU memory (storage buffer).
    ///
    /// # Panics
    ///
    /// Panics if buffer creation fails.
    fn allocate(size_bytes: usize, device: &Self::Device) -> u64 {
        if size_bytes == 0 {
            return 0;
        }

        let client = get_or_create_client(device);
        client.allocator.allocate(size_bytes)
    }

    fn deallocate(ptr: u64, size_bytes: usize, device: &Self::Device) {
        if ptr == 0 {
            return;
        }

        let client = get_or_create_client(device);
        client.allocator.deallocate(ptr, size_bytes);
    }

    /// Copy data from host to device.
    ///
    /// # Panics
    ///
    /// Panics if the buffer doesn't exist or write fails.
    fn copy_to_device(src: &[u8], dst: u64, device: &Self::Device) {
        if src.is_empty() || dst == 0 {
            return;
        }

        let client = get_or_create_client(device);

        // Get the buffer from registry
        let buffer = client::get_buffer(dst).expect("Buffer not found for copy_to_device");

        // Write data to buffer
        client.queue.write_buffer(&buffer, 0, src);

        // Ensure write is complete
        client.synchronize();
    }

    /// Copy data from device to host.
    ///
    /// # Panics
    ///
    /// Panics if the buffer doesn't exist or read fails.
    fn copy_from_device(src: u64, dst: &mut [u8], device: &Self::Device) {
        if dst.is_empty() || src == 0 {
            return;
        }

        let client = get_or_create_client(device);

        // Get the source buffer from registry
        let buffer = client::get_buffer(src).expect("Buffer not found for copy_from_device");

        // Create a staging buffer for readback
        let staging = client.create_staging_buffer("copy_staging", dst.len() as u64);

        // Copy from storage to staging
        let mut encoder =
            client
                .wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("copy_from_device"),
                });
        encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, dst.len() as u64);
        client.submit_and_wait(encoder);

        // Read from staging buffer
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = client.wgpu_device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(60)),
        });

        {
            let data = slice.get_mapped_range();
            dst.copy_from_slice(&data[..dst.len()]);
        }

        staging.unmap();
    }

    /// Copy data within device memory.
    ///
    /// # Panics
    ///
    /// Panics if either buffer doesn't exist.
    fn copy_within_device(src: u64, dst: u64, size_bytes: usize, device: &Self::Device) {
        if size_bytes == 0 || src == 0 || dst == 0 {
            return;
        }

        let client = get_or_create_client(device);

        let src_buffer = client::get_buffer(src).expect("Source buffer not found");
        let dst_buffer = client::get_buffer(dst).expect("Destination buffer not found");

        let mut encoder =
            client
                .wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("copy_within_device"),
                });
        encoder.copy_buffer_to_buffer(&src_buffer, 0, &dst_buffer, 0, size_bytes as u64);
        client.submit_and_wait(encoder);
    }

    /// Copy strided data to a contiguous buffer using a compute shader.
    ///
    /// Unlike CPU/CUDA which can use pointer arithmetic, WGPU buffers are opaque
    /// handles that don't support arithmetic. This method uses a compute shader
    /// to read elements at strided offsets and write them contiguously.
    ///
    /// # Panics
    ///
    /// Panics if buffers don't exist or shader execution fails.
    fn copy_strided(
        src_handle: u64,
        src_byte_offset: usize,
        dst_handle: u64,
        shape: &[usize],
        strides: &[isize],
        elem_size: usize,
        device: &Self::Device,
    ) {
        if src_handle == 0 || dst_handle == 0 || shape.is_empty() {
            return;
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return;
        }

        let client = get_or_create_client(device);
        let src_buffer = client::get_buffer(src_handle).expect("Source buffer not found");
        let dst_buffer = client::get_buffer(dst_handle).expect("Destination buffer not found");

        // Create shader module
        let module = client
            .wgpu_device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("strided_copy"),
                source: wgpu::ShaderSource::Wgsl(shaders::copy::STRIDED_COPY_SHADER.into()),
            });

        // Create bind group layout (all storage buffers to avoid alignment issues)
        let bind_group_layout =
            client
                .wgpu_device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("strided_copy_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Create pipeline
        let pipeline_layout =
            client
                .wgpu_device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("strided_copy_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline =
            client
                .wgpu_device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("strided_copy_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("strided_copy"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Prepare parameters
        // Element size in 4-byte units (WGSL array<u32>)
        let elem_size_units = elem_size.div_ceil(4) as u32;
        // Source offset in 4-byte units
        let src_offset_units = src_byte_offset.div_ceil(4) as u32;

        // Pad shape and strides to 8 elements
        let mut shape_arr = [0u32; 8];
        let mut strides_arr = [0i32; 8];
        for (i, &s) in shape.iter().enumerate().take(8) {
            shape_arr[i] = s as u32;
        }
        for (i, &s) in strides.iter().enumerate().take(8) {
            strides_arr[i] = s as i32;
        }

        // Create params buffer (numel, ndim, elem_size_units, src_offset_units, shape[8], strides[8])
        // Total: 4 + 4 + 4 + 4 + 32 + 32 = 80 bytes
        let mut params_data = Vec::with_capacity(80);
        params_data.extend_from_slice(&(numel as u32).to_le_bytes());
        params_data.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        params_data.extend_from_slice(&elem_size_units.to_le_bytes());
        params_data.extend_from_slice(&src_offset_units.to_le_bytes());
        for s in shape_arr {
            params_data.extend_from_slice(&s.to_le_bytes());
        }
        for s in strides_arr {
            params_data.extend_from_slice(&s.to_le_bytes());
        }

        let params_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("strided_copy_params"),
            size: params_data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        client.queue.write_buffer(&params_buffer, 0, &params_data);

        // Create bind group
        let bind_group = client
            .wgpu_device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("strided_copy_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        // Dispatch compute shader
        let workgroups = shaders::workgroup_count(numel);
        let mut encoder =
            client
                .wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("strided_copy"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("strided_copy_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        client.submit_and_wait(encoder);
    }

    fn default_device() -> Self::Device {
        WgpuDevice::new(0)
    }

    fn default_client(device: &Self::Device) -> Self::Client {
        get_or_create_client(device)
    }

    fn raw_handle(client: &Self::Client) -> &Self::RawHandle {
        &client.raw_handle
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Get the default WebGPU device (first adapter)
pub fn wgpu_device() -> WgpuDevice {
    WgpuDevice::new(0)
}

/// Get a specific WebGPU device by adapter index
pub fn wgpu_device_id(index: usize) -> WgpuDevice {
    WgpuDevice::new(index)
}

/// Check if WebGPU is available on this system
pub fn is_wgpu_available() -> bool {
    device::query_adapter_info_blocking(0).is_ok()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{Allocator, Device, RuntimeClient};

    #[test]
    fn test_wgpu_device_creation() {
        let device = WgpuDevice::new(0);
        assert_eq!(device.id(), 0);
        assert_eq!(device.name(), "wgpu:0");
    }

    #[test]
    fn test_wgpu_allocate_deallocate() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = WgpuDevice::new(0);
        let ptr = WgpuRuntime::allocate(1024, &device);
        assert_ne!(ptr, 0);
        WgpuRuntime::deallocate(ptr, 1024, &device);
    }

    #[test]
    fn test_wgpu_copy_roundtrip() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = WgpuDevice::new(0);
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let ptr = WgpuRuntime::allocate(data.len(), &device);
        WgpuRuntime::copy_to_device(&data, ptr, &device);

        let mut result = vec![0u8; data.len()];
        WgpuRuntime::copy_from_device(ptr, &mut result, &device);

        assert_eq!(data, result);

        WgpuRuntime::deallocate(ptr, data.len(), &device);
    }

    #[test]
    fn test_wgpu_copy_within_device() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = WgpuDevice::new(0);
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let src = WgpuRuntime::allocate(data.len(), &device);
        let dst = WgpuRuntime::allocate(data.len(), &device);

        WgpuRuntime::copy_to_device(&data, src, &device);
        WgpuRuntime::copy_within_device(src, dst, data.len(), &device);

        let mut result = vec![0u8; data.len()];
        WgpuRuntime::copy_from_device(dst, &mut result, &device);

        assert_eq!(data, result);

        WgpuRuntime::deallocate(src, data.len(), &device);
        WgpuRuntime::deallocate(dst, data.len(), &device);
    }

    #[test]
    fn test_wgpu_client_creation() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);
        assert_eq!(client.device().id(), 0);
    }

    #[test]
    fn test_wgpu_client_allocator() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);

        let ptr = client.allocator().allocate(256);
        assert_ne!(ptr, 0);
        client.allocator().deallocate(ptr, 256);
    }
}
