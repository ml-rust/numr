//! CUDA adjacent encodings through integer kernels.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::NextRepresentableOps;
use crate::runtime::cuda::kernels::{MAX_DIMS, launch_next_representable, launch_strided_copy};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

impl NextRepresentableOps<CudaRuntime> for CudaClient {
    fn next_up(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        next_representable(self, a, true)
    }

    fn next_down(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        next_representable(self, a, false)
    }
}

fn next_representable(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    upward: bool,
) -> Result<Tensor<CudaRuntime>> {
    let op = if upward { "next_up" } else { "next_down" };
    let dtype = a.dtype();
    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    if a.device().index != client.device.index {
        return Err(Error::DeviceMismatch);
    }
    let numel = checked_layout(a)?;
    let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &client.device)?;
    if numel == 0 {
        return Ok(out);
    }

    // Copy view bytes on this client's stream without creating another client.
    let materialized = if a.is_contiguous() {
        None
    } else {
        let contiguous = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &client.device)?;
        unsafe {
            launch_strided_copy(
                &client.context,
                &client.stream,
                client.device.index,
                a.storage().ptr(),
                contiguous.ptr(),
                a.shape(),
                a.strides(),
                numel,
                a.ndim(),
                dtype.size_in_bytes(),
                a.offset() * dtype.size_in_bytes(),
            )?;
        }
        Some(contiguous)
    };
    let input = materialized.as_ref().unwrap_or(a);
    unsafe {
        launch_next_representable(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            upward,
            input.ptr(),
            out.ptr(),
            numel,
        )?;
    }
    Ok(out)
}

fn checked_layout(a: &Tensor<CudaRuntime>) -> Result<usize> {
    let invalid = |reason: String| Error::InvalidArgument {
        arg: "input",
        reason,
    };
    let numel = if a.shape().contains(&0) {
        0
    } else {
        a.shape().iter().try_fold(1usize, |count, &dim| {
            count.checked_mul(dim).ok_or_else(|| {
                invalid(format!(
                    "CUDA neighbor shape {:?} overflows usize. Reduce the tensor size.",
                    a.shape()
                ))
            })
        })?
    };
    u32::try_from(numel).map_err(|_| {
        invalid(format!(
            "CUDA neighbor count {numel} exceeds u32::MAX. Reduce the tensor size."
        ))
    })?;
    if numel == 0 {
        return Ok(0);
    }
    if !a.is_contiguous() && a.ndim() > MAX_DIMS {
        return Err(invalid(format!(
            "CUDA neighbor strided rank {} exceeds {MAX_DIMS}. Collapse compatible dimensions.",
            a.ndim()
        )));
    }

    let mut lowest = a.offset() as i128;
    let mut highest = lowest;
    let mut minimum_delta = 0i128;
    let mut maximum_delta = 0i128;
    for (&dim, &stride) in a.shape().iter().zip(a.strides()) {
        let delta = (dim - 1) as i128 * stride as i128;
        minimum_delta += delta.min(0);
        maximum_delta += delta.max(0);
    }
    lowest += minimum_delta;
    highest += maximum_delta;
    let elem_size = a.dtype().size_in_bytes();
    numel.checked_mul(elem_size).ok_or_else(|| {
        invalid(format!(
            "CUDA neighbor count {numel} overflows allocation bytes. Reduce the tensor size."
        ))
    })?;
    if lowest < 0 || highest >= a.storage().len() as i128 {
        return Err(invalid(format!(
            "CUDA neighbor layout addresses {lowest}..={highest} outside storage length {}. Use an in-bounds view.",
            a.storage().len()
        )));
    }
    // Native strided-copy address arithmetic uses signed 64-bit byte deltas.
    if minimum_delta * (elem_size as i128) < i64::MIN as i128
        || maximum_delta * (elem_size as i128) > i64::MAX as i128
    {
        return Err(invalid(
            "CUDA neighbor stride byte range exceeds i64. Reduce the view strides.".into(),
        ));
    }
    let offset_bytes = a.offset().checked_mul(elem_size).ok_or_else(|| {
        invalid(format!(
            "CUDA neighbor offset {} overflows byte addressing. Reduce the view offset.",
            a.offset()
        ))
    })?;
    let last_byte = (highest + 1) * elem_size as i128 - 1;
    let last_byte = u64::try_from(last_byte).map_err(|_| {
        invalid("CUDA neighbor byte range exceeds u64. Reduce the view extent.".into())
    })?;
    let offset_bytes = u64::try_from(offset_bytes).map_err(|_| {
        invalid("CUDA neighbor byte offset exceeds u64. Reduce the view offset.".into())
    })?;
    if a.storage().ptr().checked_add(last_byte).is_none()
        || a.storage().ptr().checked_add(offset_bytes).is_none()
    {
        return Err(invalid(
            "CUDA neighbor device address overflows u64. Use an in-bounds allocation.".into(),
        ));
    }
    Ok(numel)
}
