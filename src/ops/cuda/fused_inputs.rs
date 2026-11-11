//! Input admission and supplied-stream copies for CUDA fused operations.

use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::{MAX_DIMS, launch_strided_copy};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

pub(super) fn check_ternary(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    c: &Tensor<CudaRuntime>,
    operation: &'static str,
) -> Result<usize> {
    if b.dtype() != a.dtype() || c.dtype() != a.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: if b.dtype() != a.dtype() {
                b.dtype()
            } else {
                c.dtype()
            },
        });
    }
    if b.shape() != a.shape() || c.shape() != a.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: if b.shape() != a.shape() {
                b.shape().to_vec()
            } else {
                c.shape().to_vec()
            },
        });
    }
    let count = check_input(client, a, operation)?;
    check_input(client, b, operation)?;
    check_input(client, c, operation)?;
    Ok(count)
}

pub(super) fn check_input(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    operation: &'static str,
) -> Result<usize> {
    if input.device().index != client.device.index {
        return Err(Error::DeviceMismatch);
    }
    let limited = |reason: String| Error::BackendLimitation {
        backend: "cuda",
        operation,
        reason,
    };
    let count = if input.shape().contains(&0) {
        0
    } else {
        input.shape().iter().try_fold(1usize, |count, &dimension| {
            count.checked_mul(dimension).ok_or_else(|| {
                limited(format!(
                    "Shape {:?} overflows usize. Reduce the tensor size.",
                    input.shape()
                ))
            })
        })?
    };
    u32::try_from(count).map_err(|_| {
        limited(format!(
            "Count {count} exceeds u32::MAX. Reduce the tensor size."
        ))
    })?;
    let element_bytes = input.dtype().size_in_bytes();
    count.checked_mul(element_bytes).ok_or_else(|| {
        limited(format!(
            "Count {count} overflows allocation bytes. Reduce the tensor size."
        ))
    })?;
    if count == 0 {
        return Ok(count);
    }
    if !input.is_contiguous() && input.ndim() > MAX_DIMS {
        return Err(limited(format!(
            "Strided rank {} exceeds {MAX_DIMS}. Collapse compatible dimensions.",
            input.ndim()
        )));
    }
    let offset_bytes = input.offset().checked_mul(element_bytes).ok_or_else(|| {
        limited("Input offset overflows byte addressing. Reduce the view offset.".into())
    })?;
    let mut minimum_delta = 0i128;
    let mut maximum_delta = 0i128;
    for (&dimension, &stride) in input.shape().iter().zip(input.strides()) {
        let delta = (dimension - 1) as i128 * stride as i128;
        minimum_delta += delta.min(0);
        maximum_delta += delta.max(0);
    }
    let lowest = input.offset() as i128 + minimum_delta;
    let highest = input.offset() as i128 + maximum_delta;
    if lowest < 0 || highest >= input.storage().len() as i128 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: format!(
                "{operation} layout addresses {lowest}..={highest} outside storage length {}. Use an in-bounds view.",
                input.storage().len()
            ),
        });
    }
    if minimum_delta * (element_bytes as i128) < i64::MIN as i128
        || maximum_delta * (element_bytes as i128) > i64::MAX as i128
    {
        return Err(limited(
            "Stride byte range exceeds i64. Reduce the view strides.".into(),
        ));
    }
    let last_byte = (highest + 1) * element_bytes as i128 - 1;
    let last_byte = u64::try_from(last_byte)
        .map_err(|_| limited("Input byte range exceeds u64. Reduce the view extent.".into()))?;
    if input.storage().ptr().checked_add(last_byte).is_none()
        || input
            .storage()
            .ptr()
            .checked_add(offset_bytes as u64)
            .is_none()
    {
        return Err(limited(
            "Input device address overflows u64. Use an in-bounds allocation.".into(),
        ));
    }
    Ok(count)
}

/// Materialize an admitted view through the supplied client's stream.
pub(super) fn materialize(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    count: usize,
) -> Result<Option<Tensor<CudaRuntime>>> {
    if input.is_contiguous() || count == 0 {
        return Ok(None);
    }
    let output = Tensor::<CudaRuntime>::empty(input.shape(), input.dtype(), &client.device)?;
    // SAFETY: admission checks the count, rank, byte ranges, and client's device.
    // The fresh destination owns count elements and cannot overlap the source.
    unsafe {
        launch_strided_copy(
            &client.context,
            &client.stream,
            client.device.index,
            input.storage().ptr(),
            output.ptr(),
            input.shape(),
            input.strides(),
            count,
            input.ndim(),
            input.dtype().size_in_bytes(),
            input.offset() * input.dtype().size_in_bytes(),
        )?;
    }
    Ok(Some(output))
}
