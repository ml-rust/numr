//! Backend-independent scaled Euclidean magnitude.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::distance_common::validate_float_dtype;
use crate::ops::{BinaryOps, CompareOps, ConditionalOps, TypeConversionOps, UnaryOps};
use crate::runtime::{
    Device, Runtime, RuntimeClient, compute_broadcast_shape, validate_binary_dtypes,
};
use crate::tensor::Tensor;

/// Computes magnitude through backend primitives without host reads.
pub fn hypot_impl<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>
        + TypeConversionOps<R>,
{
    let dtype = validate_binary_dtypes(a, b)?;
    validate_float_dtype(dtype, "hypot")?;
    let shape = compute_broadcast_shape(a, b)?;
    if !client.device().is_same(a.device()) || !client.device().is_same(b.device()) {
        return Err(Error::DeviceMismatch);
    }
    match dtype {
        DType::F32 | DType::F64 => {}
        #[cfg(feature = "f16")]
        DType::F16 | DType::BF16 => {}
        #[cfg(feature = "fp8")]
        DType::FP8E4M3 | DType::FP8E5M2 => {}
        _ => return Err(Error::UnsupportedDType { dtype, op: "hypot" }),
    }
    if dtype.is_narrow_float() {
        let a = client.cast(a, DType::F32)?;
        let b = client.cast(b, DType::F32)?;
        return client.cast(&hypot_impl(client, &a, &b)?, dtype);
    }
    if shape.contains(&0) {
        return Tensor::empty(&shape, dtype, client.device());
    }

    let a = client.abs(a)?;
    let b = client.abs(b)?;
    let high = client.maximum(&a, &b)?;
    let low = client.minimum(&a, &b)?;
    let zero = Tensor::zeros(&[], dtype, client.device())?;
    let one = Tensor::ones(&[], dtype, client.device())?;
    let both_zero = client.eq(&high, &zero)?;
    let denominator = client.where_cond(&both_zero, &one, &high)?;
    let ratio = client.div(&low, &denominator)?;
    // The squared ratio lies in [0, 1]. Tiny ratios cannot affect the rounded sum.
    let square = client.mul(&ratio, &ratio)?;
    let norm = client.sqrt(&client.add(&one, &square)?)?;
    let magnitude = client.mul(&high, &norm)?;
    let magnitude = client.where_cond(&both_zero, &zero, &magnitude)?;

    // Select explicitly because backend min/max operations differ on NaNs.
    let nan = Tensor::full_scalar(&[], dtype, f64::NAN, client.device())?;
    let magnitude = client.where_cond(&client.isnan(&a)?, &nan, &magnitude)?;
    let magnitude = client.where_cond(&client.isnan(&b)?, &nan, &magnitude)?;
    let infinity = Tensor::full_scalar(&[], dtype, f64::INFINITY, client.device())?;
    let magnitude = client.where_cond(&client.isinf(&a)?, &infinity, &magnitude)?;
    client.where_cond(&client.isinf(&b)?, &infinity, &magnitude)
}
