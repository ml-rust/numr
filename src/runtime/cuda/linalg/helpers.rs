//! Helper utilities for CUDA linear algebra

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use crate::dtype::DType;
use crate::tensor::{Layout, Storage, Tensor};

impl CudaClient {
    /// Create a tensor that owns a raw CUDA GPU pointer.
    ///
    /// The tensor frees the pointer on drop, so the caller gives up ownership.
    ///
    /// # Safety
    ///
    /// The caller MUST ensure:
    /// 1. `ptr` came from this device's allocator, or is 0 for an empty tensor
    /// 2. The allocation holds exactly `shape.iter().product() * dtype.size_in_bytes()` bytes
    /// 3. Nothing else frees `ptr` or holds ownership of it
    pub(crate) unsafe fn tensor_from_raw(
        ptr: u64,
        shape: &[usize],
        dtype: DType,
        device: &super::super::CudaDevice,
    ) -> Tensor<CudaRuntime> {
        let len = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        let storage = unsafe { Storage::<CudaRuntime>::from_ptr_owned(ptr, len, dtype, device) };
        let layout = Layout::contiguous(shape);
        Tensor::from_parts(storage, layout)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::CudaDevice;
    use super::*;

    #[test]
    fn a_tensor_from_a_raw_pointer_owns_it() {
        let device = CudaDevice::new(0);
        // SAFETY: pointer 0 of an empty shape holds nothing, and its drop frees nothing.
        let t = unsafe { CudaClient::tensor_from_raw(0, &[0], DType::F32, &device) };
        assert!(t.owns_memory());
    }
}
