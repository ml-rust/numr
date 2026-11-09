//! Backend-independent error-free addition through rounded tensor primitives.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::BinaryOps;
use crate::runtime::{
    Device, Runtime, RuntimeClient, compute_broadcast_shape, validate_binary_dtypes,
};
use crate::tensor::Tensor;

/// Retains the low component of addition without host reads or fused arithmetic.
pub fn two_sum_impl<R, C>(
    client: &C,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R>,
{
    let dtype = validate_binary_dtypes(a, b)?;
    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "two_sum",
        });
    }
    let shape = compute_broadcast_shape(a, b)?;
    if !client.device().is_same(a.device()) || !client.device().is_same(b.device()) {
        return Err(Error::DeviceMismatch);
    }
    if shape.contains(&0) {
        return Ok((
            Tensor::empty(&shape, dtype, client.device())?,
            Tensor::empty(&shape, dtype, client.device())?,
        ));
    }
    let sum = client.add(a, b)?;
    let b_virtual = client.sub(&sum, a)?;
    let a_virtual = client.sub(&sum, &b_virtual)?;
    let b_roundoff = client.sub(b, &b_virtual)?;
    let a_roundoff = client.sub(a, &a_virtual)?;
    let residual = client.add(&a_roundoff, &b_roundoff)?;
    Ok((sum, residual))
}
