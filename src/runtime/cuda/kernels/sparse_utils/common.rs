//! Shared pieces of the sparse utility launchers: module name, dtype suffix
//! and the I32 to I64 cast.

#![allow(unsafe_op_in_unsafe_fn)]

use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::runtime::cuda::CudaRuntime;
use crate::runtime::cuda::GuardedStream;
use crate::tensor::Tensor;

// ============================================================================
// Module name
// ============================================================================

/// CUDA module name for sparse utility kernels (filtering, sums, NNZ counting, conversions).
pub const SPARSE_UTILS_MODULE: &str = "sparse_utils";

// ============================================================================
// Helper: dtype suffix for kernel names
// ============================================================================

pub(super) fn dtype_suffix<T: Element>() -> Result<&'static str> {
    match T::DTYPE {
        DType::F32 => Ok("f32"),
        DType::F64 => Ok("f64"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        dtype => Err(Error::UnsupportedDType {
            dtype,
            op: "sparse utility kernel (CUDA supports F32, F64, F16, BF16)",
        }),
    }
}

// ============================================================================
// Helper: Cast I32 to I64
// ============================================================================

/// Cast I32 tensor to I64 (for row_ptrs after scan)
///
/// # Safety
///
/// - `input` must be a valid `CudaRuntime` tensor with `DType::I32` residing on the device
///   associated with `context`.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub(super) unsafe fn cast_i32_to_i64_gpu(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    input: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let n = input.numel();
    let output = Tensor::<CudaRuntime>::zeros(&[n], DType::I64, device)?;

    // Use cast kernel from cast.rs
    use super::super::cast::launch_cast;

    let input_ptr = input.ptr();
    let output_ptr = output.ptr();

    unsafe {
        launch_cast(
            context,
            stream,
            device_index,
            DType::I32,
            DType::I64,
            input_ptr,
            output_ptr,
            n,
        )?;
    }

    Ok(output)
}
