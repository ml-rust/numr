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
use wgpu::{Adapter, Backend, DownlevelFlags, Features, Limits};

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
    /// `WgpuClient::install` received an index that already has a client.
    IndexInUse {
        /// The device index the cache already holds.
        index: usize,
    },
    /// `WgpuClient::install` received a device that cannot run numr kernels.
    DeviceUnsupported {
        /// The device index of the refused install.
        index: usize,
        /// The unmet requirement.
        reason: String,
    },
}

impl fmt::Display for WgpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WgpuError::NoAdapter => write!(f, "No suitable WebGPU adapter found"),
            WgpuError::DeviceError(msg) => write!(f, "WebGPU device error: {}", msg),
            WgpuError::BufferError(msg) => write!(f, "WebGPU buffer error: {}", msg),
            WgpuError::ShaderError(msg) => write!(f, "WebGPU shader error: {}", msg),
            WgpuError::IndexInUse { index } => write!(
                f,
                "WebGPU device index {index} already has a client. Call WgpuClient::install \
                 before the first use of the index, or pass an unused index"
            ),
            WgpuError::DeviceUnsupported { index, reason } => write!(
                f,
                "WebGPU device for index {index} cannot run numr kernels: {reason}. Open the \
                 device on a compute-capable adapter with at least these limits, for example \
                 adapter.limits()"
            ),
        }
    }
}

impl std::error::Error for WgpuError {}

/// Cached adapter information for a WebGPU device.
#[derive(Clone)]
pub(crate) struct AdapterInfo {
    /// Adapter name, as the driver reports it
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

impl AdapterInfo {
    /// Describe a device on an adapter from what the device can use.
    fn new(adapter: wgpu::AdapterInfo, limits: Limits, features: Features) -> Arc<Self> {
        let subgroups_supported = features.contains(Features::SUBGROUP);
        // In wgpu 28+, subgroup sizes are not directly available via limits,
        // so we use typical GPU subgroup sizes as fallback (most GPUs use 32)
        let (min_subgroup_size, max_subgroup_size) = if subgroups_supported {
            (4, 64) // Common range across NVIDIA (32), AMD (32-64), Intel (8-32)
        } else {
            (0, 0)
        };
        Arc::new(Self {
            name: adapter.name,
            backend: adapter.backend,
            limits,
            subgroups_supported,
            min_subgroup_size,
            max_subgroup_size,
        })
    }
}

/// Storage buffers the largest kernel binds. The sparse merge kernels bind 9.
const MIN_STORAGE_BUFFERS: u32 = if cfg!(feature = "sparse") { 9 } else { 8 };

/// Check that a device can run numr kernels, and describe it.
///
/// numr needs no optional wgpu feature: no kernel enables a WGSL extension,
/// and subgroups are unused. It needs compute shaders and the limits its
/// kernels bind and dispatch at. A device past these floors still caps buffer
/// sizes at its own `max_buffer_size` and `max_storage_buffer_binding_size`.
///
/// # Errors
///
/// `WgpuError::DeviceUnsupported` names the unmet requirement.
pub(crate) fn describe_installed(
    index: usize,
    adapter: &Adapter,
    device: &wgpu::Device,
) -> Result<Arc<AdapterInfo>, WgpuError> {
    let limits = device.limits();
    check_installable(index, adapter.get_downlevel_capabilities().flags, &limits)?;
    Ok(AdapterInfo::new(
        adapter.get_info(),
        limits,
        device.features(),
    ))
}

/// The requirement check of [`describe_installed`], on values alone.
fn check_installable(
    index: usize,
    flags: DownlevelFlags,
    limits: &Limits,
) -> Result<(), WgpuError> {
    if !flags.contains(DownlevelFlags::COMPUTE_SHADERS) {
        return Err(WgpuError::DeviceUnsupported {
            index,
            reason: "the adapter does not run compute shaders".to_string(),
        });
    }
    let floors = [
        (
            "max_storage_buffers_per_shader_stage",
            limits.max_storage_buffers_per_shader_stage,
            MIN_STORAGE_BUFFERS,
        ),
        (
            "max_uniform_buffers_per_shader_stage",
            limits.max_uniform_buffers_per_shader_stage,
            1,
        ),
        ("max_bind_groups", limits.max_bind_groups, 1),
        (
            "max_bindings_per_bind_group",
            limits.max_bindings_per_bind_group,
            MIN_STORAGE_BUFFERS + 1,
        ),
        (
            "max_compute_invocations_per_workgroup",
            limits.max_compute_invocations_per_workgroup,
            256,
        ),
        (
            "max_compute_workgroup_size_x",
            limits.max_compute_workgroup_size_x,
            256,
        ),
        (
            "max_compute_workgroup_size_y",
            limits.max_compute_workgroup_size_y,
            16,
        ),
        (
            "max_compute_workgroups_per_dimension",
            limits.max_compute_workgroups_per_dimension,
            65535,
        ),
    ];
    for (name, have, need) in floors {
        if have < need {
            return Err(WgpuError::DeviceUnsupported {
                index,
                reason: format!("{name} is {have}, and numr kernels need {need}"),
            });
        }
    }
    Ok(())
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
#[derive(Clone, Default)]
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

    // Enumerate all adapters and select by index.
    let mut adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    let adapter = if index < adapters.len() {
        adapters.swap_remove(index)
    } else if adapters.is_empty() && index == 0 {
        // A browser host lists no adapters. It still hands out its default.
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|_| WgpuError::NoAdapter)?
    } else if adapters.is_empty() {
        return Err(WgpuError::NoAdapter);
    } else {
        // An index past the list is an error, never a silent pick of another
        // adapter.
        return Err(WgpuError::DeviceError(format!(
            "adapter {index} does not exist: {} adapter(s) found",
            adapters.len()
        )));
    };

    let info = AdapterInfo::new(adapter.get_info(), adapter.limits(), adapter.features());

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

    #[test]
    fn test_install_floors_name_the_unmet_limit() {
        let compute = DownlevelFlags::COMPUTE_SHADERS;
        let at_floor = Limits {
            max_storage_buffers_per_shader_stage: MIN_STORAGE_BUFFERS,
            ..Limits::default()
        };
        assert!(check_installable(7, compute, &at_floor).is_ok());

        let below = Limits {
            max_storage_buffers_per_shader_stage: MIN_STORAGE_BUFFERS - 1,
            ..Limits::default()
        };
        match check_installable(7, compute, &below) {
            Err(WgpuError::DeviceUnsupported { index: 7, reason }) => {
                assert!(reason.contains("max_storage_buffers_per_shader_stage"));
            }
            other => panic!("expected DeviceUnsupported, got {other:?}"),
        }

        match check_installable(7, DownlevelFlags::empty(), &at_floor) {
            Err(WgpuError::DeviceUnsupported { index: 7, reason }) => {
                assert!(reason.contains("compute shaders"));
            }
            other => panic!("expected DeviceUnsupported, got {other:?}"),
        }
    }

    #[test]
    fn test_out_of_range_adapter_is_an_error() {
        if query_adapter_info_blocking(0).is_err() {
            println!("No GPU available, skipping test");
            return;
        }
        match query_adapter_info_blocking(10_000) {
            Err(WgpuError::DeviceError(msg)) => assert!(msg.contains("10000"), "{msg}"),
            Err(other) => panic!("expected DeviceError, got {other}"),
            Ok(_) => panic!("adapter 10000 opened another adapter"),
        }
    }
}
