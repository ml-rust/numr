//! WebGPU device implementation.
//!
//! `WgpuDevice` wraps a WebGPU adapter and provides device properties.
//!
//! # Thread Safety
//!
//! `WgpuDevice` is `Clone` and `Send + Sync`. The underlying adapter info
//! is cached on construction.

use std::fmt;
use std::sync::Arc;
use wgpu::{Adapter, Backend, Limits};

/// Error type for WebGPU operations.
#[derive(Debug, Clone)]
pub enum WgpuError {
    /// No suitable GPU adapter found.
    NoAdapter,
    /// Device request failed.
    DeviceError(String),
    /// Buffer operation failed.
    BufferError(String),
    /// Shader compilation failed.
    ShaderError(String),
}

impl fmt::Display for WgpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WgpuError::NoAdapter => write!(f, "No suitable WebGPU adapter found"),
            WgpuError::DeviceError(msg) => write!(f, "WebGPU device error: {}", msg),
            WgpuError::BufferError(msg) => write!(f, "WebGPU buffer error: {}", msg),
            WgpuError::ShaderError(msg) => write!(f, "WebGPU shader error: {}", msg),
        }
    }
}

impl std::error::Error for WgpuError {}

/// Cached adapter information for a WebGPU device.
#[derive(Clone)]
pub(crate) struct AdapterInfo {
    /// Adapter name (e.g., "NVIDIA GeForce RTX 4090")
    name: String,
    /// Backend type (Vulkan, Metal, DX12, etc.)
    backend: Backend,
    /// Device limits
    limits: Limits,
    /// Whether subgroup operations are supported
    subgroups_supported: bool,
    /// Minimum subgroup size (0 if not supported)
    min_subgroup_size: u32,
    /// Maximum subgroup size (0 if not supported)
    max_subgroup_size: u32,
}

/// WebGPU device identifier.
///
/// Represents a specific GPU device. The device index maps to the order
/// of adapters returned by WebGPU enumeration.
///
/// # Example
///
/// ```ignore
/// let device = WgpuDevice::new(0);  // First GPU
/// println!("Device: {}", device.name());
/// ```
#[derive(Clone)]
pub struct WgpuDevice {
    /// Device index (adapter order)
    pub(crate) index: usize,
    /// Cached adapter info (lazily populated)
    info: Option<Arc<AdapterInfo>>,
}

impl WgpuDevice {
    /// Create a device identifier for the specified adapter index.
    ///
    /// This does not initialize the GPU - that happens when `WgpuClient::new`
    /// is called.
    pub fn new(index: usize) -> Self {
        Self { index, info: None }
    }

    /// Create a device with cached adapter info.
    ///
    /// This is used internally when the adapter has already been queried.
    pub(crate) fn with_info(index: usize, info: Arc<AdapterInfo>) -> Self {
        Self {
            index,
            info: Some(info),
        }
    }

    /// Get the adapter name.
    ///
    /// Returns "unknown" if the device hasn't been initialized yet.
    pub fn adapter_name(&self) -> String {
        self.info
            .as_ref()
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Get the backend type (Vulkan, Metal, DX12, etc.).
    ///
    /// Returns `None` if the device hasn't been initialized yet.
    pub fn backend(&self) -> Option<Backend> {
        self.info.as_ref().map(|i| i.backend)
    }

    /// Get device limits.
    ///
    /// Returns default limits if the device hasn't been initialized yet.
    pub fn limits(&self) -> Limits {
        self.info
            .as_ref()
            .map(|i| i.limits.clone())
            .unwrap_or_default()
    }

    /// Check if subgroup operations are supported.
    pub fn subgroups_supported(&self) -> bool {
        self.info.as_ref().is_some_and(|i| i.subgroups_supported)
    }

    /// Get subgroup size range.
    ///
    /// Returns `(min, max)` or `(0, 0)` if subgroups are not supported.
    pub fn subgroup_size(&self) -> (u32, u32) {
        self.info
            .as_ref()
            .map(|i| (i.min_subgroup_size, i.max_subgroup_size))
            .unwrap_or((0, 0))
    }

    /// Get maximum workgroup size.
    ///
    /// This is typically 256 or 1024 depending on the GPU.
    pub fn max_workgroup_size(&self) -> u32 {
        self.limits().max_compute_workgroup_size_x
    }

    /// Get maximum storage buffer binding size.
    pub fn max_storage_buffer_size(&self) -> u64 {
        self.limits().max_storage_buffer_binding_size as u64
    }
}

impl super::super::Device for WgpuDevice {
    fn id(&self) -> usize {
        self.index
    }

    fn name(&self) -> String {
        format!("wgpu:{}", self.index)
    }
}

impl fmt::Debug for WgpuDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuDevice")
            .field("index", &self.index)
            .field("adapter", &self.adapter_name())
            .field("backend", &self.backend())
            .finish()
    }
}

// ============================================================================
// Adapter Discovery
// ============================================================================

/// Query adapter information for a device index.
///
/// This is an async operation that requests a WebGPU adapter.
pub(crate) async fn query_adapter_info(
    index: usize,
) -> Result<(Adapter, Arc<AdapterInfo>), WgpuError> {
    let instance = wgpu::Instance::default();

    // Enumerate all adapters and select by index
    let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    if adapters.is_empty() {
        return Err(WgpuError::NoAdapter);
    }

    // If index is out of range, use the first adapter
    let adapter = if index < adapters.len() {
        // SAFETY: We just verified index < adapters.len()
        let mut adapters = adapters;
        adapters.swap_remove(index)
    } else {
        // Request high-performance adapter as fallback
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| WgpuError::NoAdapter)?
    };

    let wgpu_info = adapter.get_info();
    let limits = adapter.limits();
    let features = adapter.features();

    let subgroups_supported = features.contains(wgpu::Features::SUBGROUP);
    // In wgpu 28+, subgroup sizes are not directly available via limits,
    // so we use typical GPU subgroup sizes as fallback (most GPUs use 32)
    let (min_subgroup_size, max_subgroup_size) = if subgroups_supported {
        (4, 64) // Common range across NVIDIA (32), AMD (32-64), Intel (8-32)
    } else {
        (0, 0)
    };

    let info = Arc::new(AdapterInfo {
        name: wgpu_info.name,
        backend: wgpu_info.backend,
        limits,
        subgroups_supported,
        min_subgroup_size,
        max_subgroup_size,
    });

    Ok((adapter, info))
}

/// Query adapter information synchronously using pollster.
pub(crate) fn query_adapter_info_blocking(
    index: usize,
) -> Result<(Adapter, Arc<AdapterInfo>), WgpuError> {
    pollster::block_on(query_adapter_info(index))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Device;

    #[test]
    fn test_wgpu_device_creation() {
        let device = WgpuDevice::new(0);
        assert_eq!(device.id(), 0);
        assert_eq!(device.name(), "wgpu:0");
    }

    #[test]
    fn test_wgpu_device_with_adapter() {
        // Try to query actual adapter info
        match query_adapter_info_blocking(0) {
            Ok((_, info)) => {
                let device = WgpuDevice::with_info(0, info);
                println!("Adapter: {}", device.adapter_name());
                println!("Backend: {:?}", device.backend());
                println!("Max workgroup size: {}", device.max_workgroup_size());
                println!("Subgroups: {}", device.subgroups_supported());
                assert!(!device.adapter_name().is_empty());
            }
            Err(e) => {
                println!("No GPU available, skipping test: {}", e);
            }
        }
    }
}
