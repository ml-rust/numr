//! WebGPU runtime implementation

use super::cache::get_or_create_client;

/// Convert WgpuError to numr Error for use in Runtime trait methods.
fn wgpu_err(e: super::device::WgpuError) -> crate::error::Error {
    crate::error::Error::Backend(format!("WebGPU client error: {e}"))
}
use super::client::WgpuClient;
use super::device::WgpuDevice;
use super::shaders;
use crate::runtime::{Allocator, NoOpGraph, Runtime, RuntimeClient};
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
    type Graph = NoOpGraph;
    type RawHandle = super::WgpuRawHandle;
    type DType = crate::dtype::DType;

    fn name() -> &'static str {
        "wgpu"
    }

    fn supports_graph_capture() -> bool {
        false // WebGPU doesn't have CUDA-style graph capture
    }

    /// WGSL kernels compute in F32 only. A loader decodes F16, BF16 and
    /// F64 weights to F32.
    fn computes_float_dtype(dtype: Self::DType) -> bool {
        dtype == crate::dtype::DType::F32
    }

    /// A handle is a buffer id, and a kernel binds the whole buffer.
    fn handles_take_byte_offsets() -> bool {
        false
    }

    /// Allocate GPU memory (storage buffer).
    ///
    /// Returns `Err(OutOfMemory)` if buffer creation fails.
    fn allocate(size_bytes: usize, device: &Self::Device) -> crate::error::Result<u64> {
        if size_bytes == 0 {
            return Ok(0);
        }

        let client = get_or_create_client(device).map_err(wgpu_err)?;
        client.allocator.allocate(size_bytes)
    }

    fn deallocate(ptr: u64, size_bytes: usize, device: &Self::Device) {
        if ptr == 0 {
            return;
        }

        if let Ok(client) = get_or_create_client(device) {
            client.allocator.deallocate(ptr, size_bytes);
        }
    }

    /// Copy data from host to device.
    ///
    /// Returns an error if the buffer doesn't exist.
    fn copy_to_device(src: &[u8], dst: u64, device: &Self::Device) -> crate::error::Result<()> {
        if src.is_empty() || dst == 0 {
            return Ok(());
        }

        let client = get_or_create_client(device).map_err(wgpu_err)?;

        // Get the buffer from registry
        let buffer = super::client::get_buffer(dst).ok_or_else(|| {
            crate::error::Error::Backend("Buffer not found for copy_to_device".into())
        })?;

        // WebGPU copies whole 4-byte words, and `allocate` already rounded the
        // buffer up to that. A payload whose length is not a multiple of 4 — a
        // run of 18-byte GGUF blocks, a TCF payload with an odd super-block
        // count — is written as its aligned prefix plus one zero-padded tail
        // word, so it reaches the device intact instead of tripping wgpu's
        // copy-alignment validation.
        let tail = src.len() % 4;
        if tail == 0 {
            client.queue.write_buffer(&buffer, 0, src);
        } else {
            let head = src.len() - tail;
            if head > 0 {
                client.queue.write_buffer(&buffer, 0, &src[..head]);
            }
            let mut word = [0u8; 4];
            word[..tail].copy_from_slice(&src[head..]);
            client.queue.write_buffer(&buffer, head as u64, &word);
        }

        // Ensure write is complete
        client.synchronize();
        Ok(())
    }

    /// Copy data from device to host.
    ///
    /// Returns an error if the buffer doesn't exist or read fails.
    fn copy_from_device(
        src: u64,
        dst: &mut [u8],
        device: &Self::Device,
    ) -> crate::error::Result<()> {
        if dst.is_empty() || src == 0 {
            return Ok(());
        }

        let client = get_or_create_client(device).map_err(wgpu_err)?;

        // Get the source buffer from registry
        let buffer = super::client::get_buffer(src).ok_or_else(|| {
            crate::error::Error::Backend("Buffer not found for copy_from_device".into())
        })?;

        // `copy_buffer_to_buffer` and `map_async` both reject a size that is
        // not a multiple of 4. A payload whose length is not a multiple of 4 —
        // a run of 18-byte GGUF blocks, a TCF payload with an odd super-block
        // count — is staged and mapped at its rounded-up length. `allocate`
        // rounded the source buffer up the same way, so the extra bytes are in
        // bounds, and only the first `dst.len()` bytes reach the caller.
        let aligned_len = dst.len().div_ceil(4) * 4;

        // Create a staging buffer for readback
        let staging = client.create_staging_buffer("copy_staging", aligned_len as u64);

        // Copy from storage to staging
        let mut encoder =
            client
                .wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("copy_from_device"),
                });
        encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, aligned_len as u64);
        client.submit_and_wait(encoder);

        // Read from staging buffer
        let slice = staging.slice(..);

        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        client
            .wgpu_device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(60)),
            })
            .map_err(|e| {
                crate::error::Error::Backend(format!(
                    "GPU poll failed during copy_from_device: {e}"
                ))
            })?;

        // Check map_async result
        let map_result = receiver.recv().map_err(|_| {
            crate::error::Error::Backend(
                "map_async callback was not invoked during copy_from_device".into(),
            )
        })?;
        map_result.map_err(|e| {
            crate::error::Error::Backend(format!("map_async failed during copy_from_device: {e}"))
        })?;

        {
            let data = slice.get_mapped_range().map_err(|e| {
                crate::error::Error::Backend(format!(
                    "get_mapped_range failed during copy_from_device: {e}"
                ))
            })?;
            dst.copy_from_slice(&data[..dst.len()]);
        }

        staging.unmap();
        Ok(())
    }

    /// Copy data within device memory.
    ///
    /// Returns an error if either buffer doesn't exist.
    fn copy_within_device(
        src: u64,
        dst: u64,
        size_bytes: usize,
        device: &Self::Device,
    ) -> crate::error::Result<()> {
        if size_bytes == 0 || src == 0 || dst == 0 {
            return Ok(());
        }

        let client = get_or_create_client(device).map_err(wgpu_err)?;

        let src_buffer = super::client::get_buffer(src)
            .ok_or_else(|| crate::error::Error::Backend("Source buffer not found".into()))?;
        let dst_buffer = super::client::get_buffer(dst)
            .ok_or_else(|| crate::error::Error::Backend("Destination buffer not found".into()))?;

        let mut encoder =
            client
                .wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("copy_within_device"),
                });
        encoder.copy_buffer_to_buffer(&src_buffer, 0, &dst_buffer, 0, size_bytes as u64);
        client.submit_and_wait(encoder);
        Ok(())
    }

    /// Copy strided data to a contiguous buffer using a compute shader.
    ///
    /// WGPU buffers are opaque handles with no pointer arithmetic, so a
    /// shader reads the strided source and writes the destination densely.
    /// Every element size works: sizes that are not a multiple of 4 bytes,
    /// and unaligned source offsets, take a byte-gathering kernel.
    ///
    /// Returns an error if a buffer is missing or the copy exceeds the
    /// kernel's rank or 32-bit addressing limits.
    fn copy_strided(
        src_handle: u64,
        src_byte_offset: usize,
        dst_handle: u64,
        shape: &[usize],
        strides: &[isize],
        elem_size: usize,
        device: &Self::Device,
    ) -> crate::error::Result<()> {
        // A rank-0 shape is one element. Its product is 1, so it still copies.
        if src_handle == 0 || dst_handle == 0 || shape.iter().product::<usize>() == 0 {
            return Ok(());
        }

        let client = get_or_create_client(device).map_err(wgpu_err)?;
        let src_buffer = super::client::get_buffer(src_handle).ok_or_else(|| {
            crate::error::Error::Backend("Source buffer not found for copy_strided".into())
        })?;
        let dst_buffer = super::client::get_buffer(dst_handle).ok_or_else(|| {
            crate::error::Error::Backend("Destination buffer not found for copy_strided".into())
        })?;
        shaders::copy::launch_strided_copy(
            &client,
            &src_buffer,
            &dst_buffer,
            shape,
            strides,
            elem_size,
            src_byte_offset,
        )
    }

    fn capture_graph_into<F>(
        client: &Self::Client,
        inputs: &[&crate::tensor::Tensor<Self>],
        outputs: &[&crate::tensor::Tensor<Self>],
        f: F,
    ) -> crate::error::Result<crate::runtime::CapturedGraph<Self>>
    where
        F: FnOnce(&Self::Client) -> crate::error::Result<()>,
    {
        f(client)?;
        let owned_inputs: Vec<_> = inputs.iter().map(|t| (*t).clone()).collect();
        let owned_outputs: Vec<_> = outputs.iter().map(|t| (*t).clone()).collect();
        Ok(crate::runtime::CapturedGraph::new(
            crate::runtime::NoOpGraph,
            owned_inputs,
            owned_outputs,
        ))
    }

    fn default_device() -> Self::Device {
        WgpuDevice::new(0)
    }

    fn default_client(device: &Self::Device) -> Self::Client {
        get_or_create_client(device).expect("Failed to create WebGPU client")
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A length that is not a multiple of 4 must survive a device round trip.
    ///
    /// 18 bytes is one GGUF `Q4_0` block, the realistic case.
    #[test]
    fn test_copy_round_trip_unaligned_length() {
        let device = WgpuRuntime::default_device();
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let src: Vec<u8> = (0..18u8)
            .map(|i| i.wrapping_mul(7).wrapping_add(3))
            .collect();
        let ptr = WgpuRuntime::allocate(src.len(), &device).expect("allocation should succeed");

        WgpuRuntime::copy_to_device(&src, ptr, &device).expect("host to device copy");

        let mut dst = vec![0u8; src.len()];
        WgpuRuntime::copy_from_device(ptr, &mut dst, &device).expect("device to host copy");

        WgpuRuntime::deallocate(ptr, src.len(), &device);

        assert_eq!(dst, src, "unaligned round trip must be byte-exact");
    }
}
