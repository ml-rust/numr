//! Backend-independent midpoint through checked floating-point tensor operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::impl_generic::interval::require_basic;
use crate::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps, UnaryOps};
use crate::runtime::{
    Device, Runtime, RuntimeClient, compute_broadcast_shape, validate_binary_dtypes,
};
use crate::tensor::Tensor;

/// Computes paired midpoints without host numerical reads.
pub fn midpoint_impl<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ScalarOps<R>
        + UnaryOps<R>,
{
    let dtype = validate_binary_dtypes(a, b)?;
    let half_maximum = match dtype {
        DType::F32 => f64::from(f32::MAX * 0.5),
        DType::F64 => f64::MAX * 0.5,
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "midpoint",
            });
        }
    };
    let shape = compute_broadcast_shape(a, b)?;
    if !client.device().is_same(a.device()) || !client.device().is_same(b.device()) {
        return Err(Error::DeviceMismatch);
    }
    if !R::computes_float_dtype(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "midpoint",
        });
    }
    require_basic::<R>(dtype, "midpoint")?;
    if shape.contains(&0) {
        return Tensor::empty(&shape, dtype, client.device());
    }

    let limit = Tensor::full_scalar(&[], dtype, half_maximum, client.device())?;
    let zero = Tensor::zeros(&[], dtype, client.device())?;
    let small_a = client.le(&client.abs(a)?, &limit)?;
    let small_b = client.le(&client.abs(b)?, &limit)?;
    let safe_sum = client.mul(&small_a, &small_b)?;

    // Mask before adding: an unused full sum must not overflow on finite inputs.
    let a_for_sum = client.where_cond(&safe_sum, a, &zero)?;
    let b_for_sum = client.where_cond(&safe_sum, b, &zero)?;
    let sum_midpoint = client.mul_scalar(&client.add(&a_for_sum, &b_for_sum)?, 0.5)?;
    let half_midpoint = client.add(&client.mul_scalar(a, 0.5)?, &client.mul_scalar(b, 0.5)?)?;
    client.where_cond(&safe_sum, &sum_midpoint, &half_midpoint)
}
