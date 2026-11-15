//! Shared argument checks for interval and expansion kernels.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{ExpansionOperand, IntervalOperand, broadcast_shape};
use crate::runtime::{Device, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Check one floating dtype across `values`, on the client's device.
pub fn float_dtype<R, C>(client: &C, op: &'static str, values: &[&Tensor<R>]) -> Result<DType>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    let Some(first) = values.first() else {
        return Err(Error::InvalidArgument {
            arg: "operands",
            reason: format!("{op} received no operands. Supply at least one tensor."),
        });
    };
    let dtype = first.dtype();
    for value in values {
        if value.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: value.dtype(),
            });
        }
        if !client.device().is_same(value.device()) {
            return Err(Error::DeviceMismatch);
        }
    }
    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(dtype)
}

/// Check that `status` is a one-element U32 flag on the client's device.
pub fn status<R, C>(client: &C, status: &Tensor<R>) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    if !client.device().is_same(status.device()) {
        return Err(Error::DeviceMismatch);
    }
    if status.dtype() != DType::U32 || status.numel() != 1 {
        return Err(Error::InvalidArgument {
            arg: "status",
            reason: format!(
                "Interval status has dtype {:?} and shape {:?}. Create it with interval_status.",
                status.dtype(),
                status.shape()
            ),
        });
    }
    Ok(())
}

/// Check matching endpoint shapes within one interval.
pub fn interval<R: Runtime<DType = DType>>(value: IntervalOperand<'_, R>) -> Result<()> {
    if value.lower.shape() != value.upper.shape() {
        return Err(Error::ShapeMismatch {
            expected: value.lower.shape().to_vec(),
            got: value.upper.shape().to_vec(),
        });
    }
    Ok(())
}

/// Check identical component shapes across expansions.
pub fn expansions<R: Runtime<DType = DType>>(values: &[ExpansionOperand<'_, R>]) -> Result<()> {
    let Some(first) = values.first() else {
        return Ok(());
    };
    let shape = first.high.shape();
    for value in values {
        for component in [value.high, value.low, value.abs_error] {
            if component.shape() != shape {
                return Err(Error::ShapeMismatch {
                    expected: shape.to_vec(),
                    got: component.shape().to_vec(),
                });
            }
        }
    }
    Ok(())
}

/// Broadcast shape with the same error a binary operation on `lhs` and `rhs` reports.
pub fn broadcast<R: Runtime<DType = DType>>(
    lhs: &Tensor<R>,
    rhs: &Tensor<R>,
) -> Result<Vec<usize>> {
    broadcast_shape(lhs.shape(), rhs.shape()).ok_or_else(|| Error::BroadcastError {
        lhs: lhs.shape().to_vec(),
        rhs: rhs.shape().to_vec(),
    })
}

/// Whether two tensors are the same view of the same storage.
pub fn same_view<R: Runtime<DType = DType>>(a: &Tensor<R>, b: &Tensor<R>) -> bool {
    std::ptr::eq(a, b)
        || (a.storage().ptr() == b.storage().ptr()
            && a.offset() == b.offset()
            && a.shape() == b.shape()
            && a.strides() == b.strides())
}
