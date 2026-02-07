//! Helper functions and macros for WebGPU linear algebra operations.

use super::super::{WgpuClient, WgpuRuntime};
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
