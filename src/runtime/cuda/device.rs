//! CUDA Device implementation
//!
//! Provides CUDA device abstraction using cudarc for direct GPU control.

use crate::runtime::Device;

/// CUDA Device using cudarc
///
/// Represents a single GPU device and manages context for kernel launches.
/// Used by CudaClient for stream management.
#[derive(Clone, Debug)]
pub struct CudaDevice {
    /// Index of the GPU device (0, 1, 2, ...)
    pub(crate) index: usize,
}

impl CudaDevice {
    /// Create a new CUDA device
    pub fn new(index: usize) -> Self {
        Self { index }
    }

    /// Get the compute capability of this CUDA device
    ///
    /// Returns (major, minor) version numbers (e.g., (8, 6) for sm_86 / RTX 3090)
    ///
    /// # Examples
    /// - (7, 5): Turing (RTX 20xx, T4)
    /// - (8, 0): Ampere (A100)
    /// - (8, 6): Ampere (RTX 30xx, A6000)
    /// - (8, 9): Ada Lovelace (RTX 40xx, L4)
    /// - (9, 0): Hopper (H100)
    pub fn compute_capability(&self) -> Result<(u32, u32), CudaError> {
        let device = cudarc::driver::result::device::get(self.index as i32).map_err(|e| {
            CudaError::DeviceError(format!("Failed to get CUDA device {}: {:?}", self.index, e))
        })?;

        let major = unsafe {
            cudarc::driver::result::device::get_attribute(
                device,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
        }
        .map_err(|e| CudaError::DeviceError(format!("Failed to get compute capability major: {:?}", e)))? as u32;

        let minor = unsafe {
            cudarc::driver::result::device::get_attribute(
                device,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )
        }
        .map_err(|e| CudaError::DeviceError(format!("Failed to get compute capability minor: {:?}", e)))? as u32;

        Ok((major, minor))
    }

    /// Synchronize all operations on this device
    ///
    /// This synchronizes the current CUDA context on this thread.
    /// For stream-specific synchronization, use `CudaClient::synchronize()` instead.
    pub fn sync(&self) -> Result<(), CudaError> {
        cudarc::driver::result::ctx::synchronize().map_err(|e| {
            CudaError::SyncError(format!(
                "Failed to synchronize CUDA context for device {}: {:?}",
                self.index, e
            ))
        })
    }

    /// Get memory information for this device
    ///
    /// Returns (free_bytes, total_bytes) for the device's global memory.
    pub fn memory_info(&self) -> Result<(u64, u64), CudaError> {
        let (free, total) = cudarc::driver::result::mem_get_info().map_err(|e| {
            CudaError::DeviceError(format!(
                "Failed to get memory info for device {}: {:?}",
                self.index, e
            ))
        })?;
        Ok((free as u64, total as u64))
    }

    /// Get available (free) GPU memory in bytes
    pub fn available_memory(&self) -> Result<u64, CudaError> {
        let (free, _) = self.memory_info()?;
        Ok(free)
    }

    /// Get total GPU memory in bytes
    pub fn total_memory(&self) -> Result<u64, CudaError> {
        let (_, total) = self.memory_info()?;
        Ok(total)
    }
}

impl Device for CudaDevice {
    fn id(&self) -> usize {
        self.index
    }

    fn name(&self) -> String {
        format!("cuda:{}", self.index)
    }
}

impl Default for CudaDevice {
    fn default() -> Self {
        Self::new(0)
    }
}

/// CUDA-specific errors
#[derive(Debug, Clone)]
pub enum CudaError {
    /// Device initialization or query error
    DeviceError(String),
    /// Memory allocation error
    AllocationError(String),
    /// Memory copy error
    CopyError(String),
    /// Kernel launch error
    KernelError(String),
    /// Synchronization error
    SyncError(String),
    /// cuBLAS error
    CublasError(String),
    /// cusparse error
    CusparseError(String),
    /// Context error
    ContextError(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::DeviceError(msg) => write!(f, "CUDA device error: {}", msg),
            CudaError::AllocationError(msg) => write!(f, "CUDA allocation error: {}", msg),
            CudaError::CopyError(msg) => write!(f, "CUDA copy error: {}", msg),
            CudaError::KernelError(msg) => write!(f, "CUDA kernel error: {}", msg),
            CudaError::SyncError(msg) => write!(f, "CUDA sync error: {}", msg),
            CudaError::CublasError(msg) => write!(f, "cuBLAS error: {}", msg),
            CudaError::CusparseError(msg) => write!(f, "cusparse error: {}", msg),
            CudaError::ContextError(msg) => write!(f, "CUDA context error: {}", msg),
        }
    }
}

impl std::error::Error for CudaError {}
