//! Global client cache for WebGPU runtime

use super::client::WgpuClient;
use super::device::WgpuDevice;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Global client cache: device index -> cached WgpuClient
///
/// This caches WgpuClient instances per device to avoid creating new
/// WebGPU devices and queues on every operation.
static CLIENT_CACHE: OnceLock<Mutex<HashMap<usize, WgpuClient>>> = OnceLock::new();

/// Get or create a cached WgpuClient for a device.
///
/// This ensures only one `wgpu::Device` exists per device index. All
/// `WgpuClient` instances for the same index share the same underlying
/// device, queue, and buffer storage. This is critical because wgpu
/// buffers belong to the device that created them and cannot be used
/// in bind groups on a different device.
pub(super) fn get_or_create_client(
    device: &WgpuDevice,
) -> Result<WgpuClient, super::device::WgpuError> {
    let cache = CLIENT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache_guard = cache.lock().unwrap_or_else(|e| e.into_inner());

    if let Some(client) = cache_guard.get(&device.index) {
        return Ok(client.clone());
    }

    // Create new client via uncached path and cache it
    let client = WgpuClient::new_uncached(device.clone())?;
    cache_guard.insert(device.index, client.clone());

    Ok(client)
}
