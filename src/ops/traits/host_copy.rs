//! Host-to-device copy into a pre-allocated tensor.

use crate::dtype::Element;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Writing host data straight into a tensor the caller already owns.
///
/// This is the host-side counterpart of
/// [`BinaryOps::copy_into`](crate::ops::BinaryOps::copy_into): it fills an
/// existing device tensor from a host slice with no intermediate tensor and
/// no allocation.
pub trait HostCopyOps<R: Runtime> {
    /// Write `src` into the pre-allocated `out`: `out = src`.
    ///
    /// `out` keeps its device address, so a buffer a captured graph points
    /// at stays valid across the write. On CUDA the copy is a
    /// stream-ordered host-to-device memcpy on the client's compute stream,
    /// so it records as a memcpy node under graph capture.
    ///
    /// `out` must be contiguous, must have `T`'s dtype, and must hold
    /// exactly `src.len()` elements.
    ///
    /// # Arguments
    /// * `out` - Pre-allocated, contiguous destination tensor (overwritten)
    /// * `src` - Host elements, one per element of `out`
    ///
    /// # Errors
    /// Returns `Error::DTypeMismatch` when `T::DTYPE` differs from `out`'s
    /// dtype, `Error::ShapeMismatch` when the lengths differ, and
    /// `Error::Backend` when `out` is not contiguous.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let out = Tensor::<CpuRuntime>::zeros(&[2, 2], DType::I32, &device)?;
    /// client.write_host_slice(&out, &[1i32, 2, 3, 4])?;
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn write_host_slice<T: Element>(&self, out: &Tensor<R>, src: &[T]) -> Result<()>;
}
