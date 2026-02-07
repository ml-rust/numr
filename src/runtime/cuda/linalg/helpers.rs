//! Helper utilities for CUDA linear algebra

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use crate::dtype::DType;
use crate::tensor::{Layout, Storage, Tensor};

impl CudaClient {
    /// Create a tensor from a raw CUDA GPU pointer.
    ///
    /// # Safety
    ///
    /// The caller MUST ensure:
    /// 1. `ptr` points to valid CUDA device memory
    /// 2. The allocation contains at least `shape.iter().product() * dtype.size_in_bytes()` bytes
    /// 3. The GPU memory remains valid for the tensor's lifetime
    /// 4. No other tensor holds ownership of the same memory
    /// 5. `ptr` was allocated on the same device as `device`
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
        let storage = unsafe { Storage::<CudaRuntime>::from_ptr(ptr, len, dtype, device) };
        let layout = Layout::contiguous(shape);
        Tensor::from_parts(storage, layout)
    }
}
