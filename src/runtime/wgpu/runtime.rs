//! WebGPU runtime implementation

use super::cache::get_or_create_client;
use super::client::WgpuClient;
use super::device::WgpuDevice;
use super::shaders;
use crate::runtime::{Allocator, Runtime, RuntimeClient};
use std::time::Duration;

/// WebGPU Runtime adapter
///
/// Implements the generic Runtime trait for WebGPU backend.
/// Provides cross-platform GPU acceleration.
#[derive(Clone, Debug, Default)]
pub struct WgpuRuntime;

impl Runtime for WgpuRuntime {
    type Device = WgpuDevice;
    type Client = WgpuClient;
    type Allocator = super::WgpuAllocator;
    type RawHandle = super::WgpuRawHandle;

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
        let buffer = super::client::get_buffer(dst).expect("Buffer not found for copy_to_device");

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
        let buffer = super::client::get_buffer(src).expect("Buffer not found for copy_from_device");

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

        let src_buffer = super::client::get_buffer(src).expect("Source buffer not found");
        let dst_buffer = super::client::get_buffer(dst).expect("Destination buffer not found");

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
        let src_buffer = super::client::get_buffer(src_handle).expect("Source buffer not found");
        let dst_buffer =
            super::client::get_buffer(dst_handle).expect("Destination buffer not found");

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
    super::device::query_adapter_info_blocking(0).is_ok()
}
