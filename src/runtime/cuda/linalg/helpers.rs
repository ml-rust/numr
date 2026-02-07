//! Helper utilities for CUDA linear algebra

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use crate::dtype::{DType, Element};
use crate::tensor::{Layout, Storage, Tensor};

/// Trait for elements that support linear algebra operations.
///
/// This trait extends `Element` with operations needed for numerical
/// linear algebra algorithms.
pub trait LinalgElement: Element + Sized {
    /// Returns machine epsilon for this type
    fn epsilon_val() -> f64;
    /// Returns absolute value
    fn abs_val(&self) -> Self;
    /// Returns square root
    fn sqrt_val(&self) -> Self;
    /// Returns negation
    fn neg_val(&self) -> Self;
}

impl LinalgElement for f32 {
    #[inline]
    fn epsilon_val() -> f64 {
        f32::EPSILON as f64
    }
    #[inline]
    fn abs_val(&self) -> Self {
        self.abs()
    }
    #[inline]
    fn sqrt_val(&self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn neg_val(&self) -> Self {
        -*self
    }
}

impl LinalgElement for f64 {
    #[inline]
    fn epsilon_val() -> f64 {
        f64::EPSILON
    }
    #[inline]
    fn abs_val(&self) -> Self {
        self.abs()
    }
    #[inline]
    fn sqrt_val(&self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn neg_val(&self) -> Self {
        -*self
    }
}

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
