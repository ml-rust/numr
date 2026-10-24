//! Cast operation implementation for WebGPU.

use super::helpers::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::{cast_i64, elementwise};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// The dtypes WGSL holds natively.
const NATIVE_32: [DType; 3] = [DType::F32, DType::I32, DType::U32];

/// True when `native_cast_op` runs `src_dtype -> dst_dtype` on the GPU:
/// any pair of F32, I32 and U32, or I64 to or from one of them.
pub(crate) fn is_native_cast(src_dtype: DType, dst_dtype: DType) -> bool {
    (NATIVE_32.contains(&src_dtype) && NATIVE_32.contains(&dst_dtype))
        || cast_i64::is_i64_cast_pair(src_dtype, dst_dtype)
}

/// Native cast operation using WGSL compute shaders.
///
/// Covers every pair [`is_native_cast`] accepts. The data never leaves the GPU.
///
/// # Errors
///
/// `UnsupportedDType` for any other pair. `BackendLimitation` when the
/// element count does not fit the kernels' u32 indexing.
pub(crate) fn native_cast_op(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dst_dtype: DType,
) -> Result<Tensor<WgpuRuntime>> {
    let src_dtype = a.dtype();
    let numel = a.numel();

    // Allocate output with target dtype
    let out = alloc_output(client, a.shape(), dst_dtype)?;
    if numel == 0 {
        return Ok(out);
    }

    let a_contig = ensure_contiguous(a)?;
    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = CastParams {
        numel: u32::try_from(numel).map_err(|_| Error::BackendLimitation {
            backend: "wgpu",
            operation: "cast",
            reason: format!(
                "{numel} elements of shape {:?} exceed the u32 indexing of the cast kernels",
                a.shape()
            ),
        })?,
    };
    let params_buf = create_params_buffer(client, &params);

    let launch = if cast_i64::is_i64_cast_pair(src_dtype, dst_dtype) {
        cast_i64::launch_cast_i64_op
    } else {
        elementwise::launch_cast_op
    };
    launch(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &out_buf,
        &params_buf,
        numel,
        src_dtype,
        dst_dtype,
    )?;

    Ok(out)
}
