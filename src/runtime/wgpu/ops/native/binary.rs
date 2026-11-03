//! Binary and scalar operation implementations for WebGPU.

use super::helpers::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::wgpu::client::get_buffer;
use crate::runtime::wgpu::shaders::linear_grid::element_count_u32;
use crate::runtime::wgpu::shaders::{self, elementwise};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{
    CopyIntoPlan, compute_broadcast_shape, ensure_contiguous, validate_binary_dtypes,
    validate_copy_into,
};
use crate::tensor::Tensor;

pub(crate) fn native_binary_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    // Use shared helpers for validation (same as CPU and CUDA backends)
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    let out = alloc_output(client, &out_shape, dtype)?;
    run_binary(client, op, a, b, &out, &out_shape, dtype)?;

    Ok(out)
}

/// Destination-passing binary op: writes `op(a, b)` into the caller-owned `out`
/// tensor instead of allocating. Required for destination-passing workflows
/// where the output buffer must have a stable identity. `out` must be
/// contiguous, share the inputs' dtype, and have shape `broadcast(a, b)`.
pub(crate) fn native_binary_op_into(
    client: &WgpuClient,
    op: &'static str,
    out: &Tensor<WgpuRuntime>,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<()> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    if out.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: out.dtype(),
        });
    }
    if out.shape() != out_shape.as_slice() {
        return Err(Error::ShapeMismatch {
            expected: out_shape,
            got: out.shape().to_vec(),
        });
    }
    if !out.is_contiguous() {
        return Err(Error::Backend(
            "native_binary_op_into: destination tensor must be contiguous".into(),
        ));
    }

    run_binary(client, op, a, b, out, &out_shape, dtype)
}

/// Destination-passing copy: writes `src` into the caller-owned `out` tensor.
///
/// `out` must share `src`'s shape and dtype and can be a strided view. A
/// contiguous `out` at offset 0 takes one buffer-to-buffer copy on the queue,
/// and a strided `src` is made contiguous first. Any other `out` takes the
/// strided-to-strided kernel, which supports elements of 4-byte multiples
/// (F32, I32, U32) and returns `BackendLimitation` for other sizes.
pub(crate) fn native_copy_into(
    client: &WgpuClient,
    out: &Tensor<WgpuRuntime>,
    src: &Tensor<WgpuRuntime>,
) -> Result<()> {
    if validate_copy_into(out, src, "native_copy_into")? == CopyIntoPlan::Skip {
        return Ok(());
    }
    let elem_size = src.dtype().size_in_bytes();
    if !out.is_contiguous() || out.offset() != 0 {
        let src_buf = get_buffer(src.storage().ptr()).ok_or_else(|| {
            Error::Internal("native_copy_into: source buffer is not in the registry".into())
        })?;
        let out_buf = get_buffer(out.storage().ptr()).ok_or_else(|| {
            Error::Internal("native_copy_into: destination buffer is not in the registry".into())
        })?;
        return shaders::copy_into::launch_strided_copy_into(
            client,
            &src_buf,
            &out_buf,
            out.shares_storage_with(src),
            src.shape(),
            src.strides(),
            out.strides(),
            elem_size,
            src.offset() * elem_size,
            out.offset() * elem_size,
        );
    }
    let src_contig = ensure_contiguous(src)?;
    let src_buf = get_tensor_buffer(&src_contig)?;
    let out_buf = get_tensor_buffer(out)?;

    // `copy_buffer_to_buffer` rejects a size that is not a multiple of 4.
    // `allocate` rounds every buffer up the same way, so the padded tail is
    // in bounds on both sides. Rounding up writes past `out`, so it is only
    // allowed when `out` covers its whole buffer.
    let size_bytes = src.numel() * elem_size;
    let aligned_len = size_bytes.div_ceil(4) * 4;
    if aligned_len != size_bytes && size_bytes != out.storage().size_in_bytes() {
        return Err(Error::BackendLimitation {
            backend: "wgpu",
            operation: "copy_into",
            reason: format!(
                "a destination of {size_bytes} bytes inside a larger buffer needs a multiple \
                 of 4 bytes"
            ),
        });
    }

    let mut encoder =
        client
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_into"),
            });
    encoder.copy_buffer_to_buffer(&src_buf, 0, &out_buf, 0, aligned_len as u64);
    client
        .wgpu_queue()
        .submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Shared dispatch for both the allocating and destination-passing paths.
/// `out` must already have shape `out_shape` and the validated dtype.
fn run_binary(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    out: &Tensor<WgpuRuntime>,
    out_shape: &[usize],
    dtype: DType,
) -> Result<()> {
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;

    let numel: usize = out_shape.iter().product();
    let numel_u32 = element_count_u32(numel, op)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let b_buf = get_tensor_buffer(&b_contig)?;
    let out_buf = get_tensor_buffer(out)?;

    // Use broadcast kernel if shapes differ, element-wise kernel otherwise
    if a.shape() != b.shape() {
        launch_broadcast(
            client,
            op,
            &a_buf,
            &b_buf,
            &out_buf,
            a.shape(),
            b.shape(),
            out_shape,
            numel,
            dtype,
        )?;
    } else {
        let params = BinaryParams { numel: numel_u32 };
        let params_buf = create_params_buffer(client, &params);

        elementwise::launch_binary_op(
            client.pipeline_cache(),
            client.wgpu_queue(),
            op,
            &a_buf,
            &b_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;
    }

    Ok(())
}

pub(crate) fn native_scalar_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    scalar: f64,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let numel = a.numel();

    let out = alloc_output(client, a.shape(), dtype)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = ScalarParams::new(element_count_u32(numel, op)?, scalar, dtype);
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_scalar_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        op,
        &a_buf,
        &out_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}
