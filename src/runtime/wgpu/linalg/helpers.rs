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
    /// Create a tensor from a raw WebGPU buffer pointer.
    ///
    /// # Safety
    ///
    /// The caller MUST ensure:
    /// 1. `ptr` is a valid WebGPU buffer ID
    /// 2. The allocation contains sufficient bytes
    /// 3. The buffer remains valid for tensor lifetime
    /// 4. No aliasing (single owner)
    /// 5. Buffer was allocated on the same device
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
        let storage = unsafe { Storage::<WgpuRuntime>::from_ptr(ptr, len, dtype, device) };
        let layout = Layout::contiguous(shape);
        Tensor::from_parts(storage, layout)
    }
}
