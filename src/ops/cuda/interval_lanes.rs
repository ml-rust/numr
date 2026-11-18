//! CUDA lane operands and launches shared by the interval, expansion and correctly
//! rounded kernels.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::IntervalOperand;
use crate::ops::impl_generic::interval::{IntervalBinary, LaneOperand, lane_operand};
use crate::runtime::cuda::kernels::{IntervalArg, MAX_DIMS, launch_interval, launch_strided_copy};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

type CudaTensor = Tensor<CudaRuntime>;

/// Kernel name suffix of a float dtype.
pub(super) fn suffix(dtype: DType) -> &'static str {
    if dtype == DType::F32 { "f32" } else { "f64" }
}

/// Lane count as a kernel argument.
pub(super) fn lanes(count: usize, op: &'static str) -> Result<u32> {
    u32::try_from(count).map_err(|_| Error::InvalidArgument {
        arg: "shape",
        reason: format!("CUDA {op} lane count {count} exceeds u32::MAX. Split the batch."),
    })
}

/// Copy a broadcast or strided view into contiguous storage on the client's stream.
fn materialize(client: &CudaClient, view: &CudaTensor) -> Result<CudaTensor> {
    if view.ndim() > MAX_DIMS {
        return Err(Error::InvalidArgument {
            arg: "operand",
            reason: format!(
                "CUDA lane operand rank {} exceeds {MAX_DIMS}. Collapse compatible dimensions.",
                view.ndim()
            ),
        });
    }
    let out = CudaTensor::empty(view.shape(), view.dtype(), &client.device)?;
    let elem_size = view.dtype().size_in_bytes();
    // Both tensors live on this client's device, and `out` holds every element of `view`.
    unsafe {
        launch_strided_copy(
            &client.context,
            &client.stream,
            client.device.index,
            view.storage().ptr(),
            out.ptr(),
            view.shape(),
            view.strides(),
            view.numel(),
            view.ndim(),
            elem_size,
            view.offset() * elem_size,
        )?;
    }
    Ok(out)
}

/// Prepare a CUDA lane operand, copying general broadcast views.
pub(super) fn cuda_operand<'a>(
    client: &CudaClient,
    value: &'a CudaTensor,
    shape: &[usize],
) -> Result<LaneOperand<'a, CudaRuntime>> {
    lane_operand(value, shape, |view| materialize(client, view))
}

/// Address and step arguments of one operand.
pub(super) fn operand_args(operand: &LaneOperand<'_, CudaRuntime>) -> [IntervalArg; 2] {
    [
        IntervalArg::Ptr(operand.addr()),
        IntervalArg::U32(operand.step),
    ]
}

/// Launch `name` on this client's stream.
pub(super) fn launch(
    client: &CudaClient,
    name: &str,
    args: &[IntervalArg],
    count: usize,
) -> Result<()> {
    // Callers pass device addresses on this client sized for `count` lanes.
    unsafe {
        launch_interval(
            &client.context,
            &client.stream,
            client.device.index,
            name,
            args,
            count,
        )
    }
}

/// Flag invalid bounds of one interval on its own shape; with `positive`, also nonpositive lowers.
pub(super) fn flag_interval(
    client: &CudaClient,
    dtype: DType,
    value: IntervalOperand<'_, CudaRuntime>,
    positive: bool,
    status: &CudaTensor,
) -> Result<()> {
    let count = value.lower.numel();
    if count == 0 {
        return Ok(());
    }
    let n = lanes(count, "interval_flag")?;
    let lower = cuda_operand(client, value.lower, value.lower.shape())?;
    let upper = cuda_operand(client, value.upper, value.upper.shape())?;
    let [lp, ls] = operand_args(&lower);
    let [up, us] = operand_args(&upper);
    let args = [
        lp,
        ls,
        up,
        us,
        IntervalArg::I32(i32::from(positive)),
        IntervalArg::Ptr(status.ptr()),
        IntervalArg::U32(n),
    ];
    launch(
        client,
        &format!("interval_flag_{}", suffix(dtype)),
        &args,
        count,
    )
}

pub(super) fn input_flags(
    client: &CudaClient,
    op: IntervalBinary,
    dtype: DType,
    a: IntervalOperand<'_, CudaRuntime>,
    b: IntervalOperand<'_, CudaRuntime>,
    status: &CudaTensor,
) -> Result<()> {
    match op {
        IntervalBinary::DivPositive => flag_interval(client, dtype, b, true, status),
        IntervalBinary::DivPositiveInterval => {
            flag_interval(client, dtype, a, false, status)?;
            flag_interval(client, dtype, b, true, status)
        }
        _ => Ok(()),
    }
}
