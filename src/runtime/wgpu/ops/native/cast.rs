//! Cast operation implementation for WebGPU.

use super::helpers::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::{cast_32, cast_i64};
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
/// The cast runs in windows of at most [`window_elements`] elements, so a
/// tensor larger than the device's storage binding limit still casts.
///
/// # Errors
///
/// `UnsupportedDType` for any other pair. `BackendLimitation` when the
/// device's binding limit cannot hold one aligned window.
pub(crate) fn native_cast_op(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dst_dtype: DType,
) -> Result<Tensor<WgpuRuntime>> {
    let src_dtype = a.dtype();
    let numel = a.numel();
    if src_dtype == dst_dtype {
        return Ok(a.clone());
    }

    // Allocate output with target dtype
    let out = alloc_output(client, a.shape(), dst_dtype)?;
    if numel == 0 {
        return Ok(out);
    }

    let a_contig = ensure_contiguous(a)?;
    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let launch = if cast_i64::is_i64_cast_pair(src_dtype, dst_dtype) {
        cast_i64::launch_cast_i64_op
    } else {
        cast_32::launch_cast_op
    };

    let limits = client.wgpu_device().limits();
    let src_size = src_dtype.size_in_bytes() as u64;
    let dst_size = dst_dtype.size_in_bytes() as u64;
    let window = window_elements(
        limits.max_storage_buffer_binding_size,
        u64::from(limits.min_storage_buffer_offset_alignment),
        src_size.max(dst_size),
    )
    .ok_or_else(|| Error::BackendLimitation {
        backend: "wgpu",
        operation: "cast",
        reason: format!(
            "storage binding limit {} bytes holds no aligned window of {src_dtype:?} -> \
             {dst_dtype:?} elements",
            limits.max_storage_buffer_binding_size
        ),
    })?;

    let mut start = 0usize;
    while start < numel {
        let len = window.min(numel - start);
        let binding = |buffer, elem_size: u64| wgpu::BufferBinding {
            buffer,
            offset: start as u64 * elem_size,
            size: std::num::NonZeroU64::new(len as u64 * elem_size),
        };
        // `len` is at most `window`, which `window_elements` keeps below u32.
        let params = CastParams { numel: len as u32 };
        let params_buf = create_params_buffer(client, &params);
        launch(
            client.pipeline_cache(),
            client.wgpu_queue(),
            binding(&a_buf, src_size),
            binding(&out_buf, dst_size),
            &params_buf,
            len,
            src_dtype,
            dst_dtype,
        )?;
        start += len;
    }

    Ok(out)
}

/// Elements in one cast window, or `None` when no aligned window fits.
///
/// The window is the most elements of `elem_size` bytes that fit
/// `binding_limit`, rounded down to a multiple of `alignment`. Each window
/// then starts at a byte offset that is a multiple of `alignment` for any
/// element size. The count also fits u32, which the kernels index in.
fn window_elements(binding_limit: u64, alignment: u64, elem_size: u64) -> Option<usize> {
    let alignment = alignment.max(1);
    let fit = (binding_limit / elem_size).min(u64::from(u32::MAX));
    let window = fit / alignment * alignment;
    if window == 0 {
        return None;
    }
    usize::try_from(window).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_fills_the_binding_limit_at_the_widest_element() {
        assert_eq!(window_elements(1 << 27, 256, 8), Some(1 << 24));
        assert_eq!(window_elements(1 << 27, 256, 4), Some(1 << 25));
    }

    #[test]
    fn window_rounds_down_to_the_offset_alignment() {
        assert_eq!(window_elements(1000, 32, 4), Some(224));
    }

    #[test]
    fn window_stays_within_u32() {
        assert_eq!(
            window_elements(u64::MAX, 256, 4),
            Some(u32::MAX as usize / 256 * 256)
        );
    }

    #[test]
    fn window_is_none_below_one_aligned_run() {
        assert_eq!(window_elements(1000, 256, 8), None);
    }
}
