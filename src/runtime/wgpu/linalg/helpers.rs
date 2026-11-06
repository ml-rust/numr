//! Helper functions and macros for WebGPU linear algebra operations.

use super::super::{WgpuClient, WgpuRuntime};
use crate::dtype::DType;
use crate::tensor::{Layout, Storage, Tensor};

/// Helper macro to get a GPU buffer from a pointer with proper error context.
///
/// # Usage
///
/// The call site must have `get_buffer` and `Error` in scope:
/// ```ignore
/// use super::super::client::get_buffer;
/// use crate::error::Error;
///
/// let buffer = get_buffer_or_err!(ptr, "buffer name");
/// ```
macro_rules! get_buffer_or_err {
    ($ptr:expr, $name:expr) => {
        get_buffer($ptr).ok_or_else(|| {
            Error::Internal(format!(
                "Failed to get {} buffer from GPU allocation",
                $name
            ))
        })?
    };
}

pub(crate) use get_buffer_or_err;

impl WgpuClient {
    /// Create a tensor that owns a raw WebGPU buffer pointer.
    ///
    /// The tensor frees the buffer on drop, so the caller gives up ownership.
    ///
    /// # Safety
    ///
    /// The caller MUST ensure:
    /// 1. `ptr` is a WebGPU buffer ID from this device's allocator
    /// 2. The allocation holds exactly `shape.iter().product() * dtype.size_in_bytes()` bytes
    /// 3. Nothing else frees `ptr` or holds ownership of it
    pub(crate) unsafe fn tensor_from_raw(
        ptr: u64,
        shape: &[usize],
        dtype: DType,
        device: &super::super::WgpuDevice,
    ) -> Tensor<WgpuRuntime> {
        let len = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        let storage = unsafe { Storage::<WgpuRuntime>::from_ptr_owned(ptr, len, dtype, device) };
        let layout = Layout::contiguous(shape);
        Tensor::from_parts(storage, layout)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::WgpuDevice;
    use super::*;

    /// Runs on any device list: pointer 0 of an empty shape touches no GPU.
    #[test]
    fn a_tensor_from_a_raw_pointer_owns_it() {
        let device = WgpuDevice::new(0);
        // SAFETY: pointer 0 of an empty shape holds nothing, and its drop frees nothing.
        let t = unsafe { WgpuClient::tensor_from_raw(0, &[0], DType::F32, &device) };
        assert!(t.owns_memory());
    }
}
