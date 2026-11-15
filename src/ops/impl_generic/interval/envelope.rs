//! Arithmetic envelope probes: constants shared by every backend, and the composed reference.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Probe constants of one dtype, as F64 values exactly representable in that dtype.
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeConstants {
    /// Unit roundoff spacing at one.
    pub epsilon: f64,
    /// Smallest positive subnormal.
    pub minimum_subnormal: f64,
    /// Smallest positive normal.
    pub minimum_normal: f64,
}

impl EnvelopeConstants {
    /// Constants of F32 or F64.
    pub fn for_dtype(dtype: DType) -> Result<Self> {
        match dtype {
            DType::F32 => Ok(Self {
                epsilon: f64::from(f32::EPSILON),
                minimum_subnormal: f64::from(f32::from_bits(1)),
                minimum_normal: f64::from(f32::MIN_POSITIVE),
            }),
            DType::F64 => Ok(Self {
                epsilon: f64::EPSILON,
                minimum_subnormal: f64::from_bits(1),
                minimum_normal: f64::MIN_POSITIVE,
            }),
            _ => Err(Error::UnsupportedDType {
                dtype,
                op: "interval_envelope",
            }),
        }
    }
}

#[cfg(any(feature = "wgpu", test))]
pub use composed::interval_envelope_impl;

#[cfg(any(feature = "wgpu", test))]
mod composed {
    use super::EnvelopeConstants;
    use crate::dtype::DType;
    use crate::error::Result;
    use crate::ops::ScalarOps;
    use crate::ops::impl_generic::interval::composed::ComposedClient;
    use crate::runtime::Runtime;
    use crate::tensor::Tensor;

    /// Composed probes over ordinary rounded tensor operations.
    pub fn interval_envelope_impl<R, C>(client: &C, dtype: DType) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: ComposedClient<R> + ScalarOps<R>,
    {
        let k = EnvelopeConstants::for_dtype(dtype)?;
        let constant = |value| Tensor::<R>::full_scalar(&[1], dtype, value, client.device());
        let c = client;
        let zero = constant(0.0)?;
        let one = constant(1.0)?;
        let two = constant(2.0)?;
        let half = constant(0.5)?;
        let epsilon = constant(k.epsilon)?;
        let half_epsilon = c.mul_scalar(&epsilon, 0.5)?;
        let odd = c.add(&one, &epsilon)?;
        let even_next = constant(1.0 + 2.0 * k.epsilon)?;
        let positive_ties = c.maximum(
            &c.ne(&c.add(&one, &half_epsilon)?, &one)?,
            &c.ne(&c.add(&odd, &half_epsilon)?, &even_next)?,
        )?;
        let negative_ties = c.maximum(
            &c.ne(
                &c.add(&c.neg(&one)?, &c.neg(&half_epsilon)?)?,
                &c.neg(&one)?,
            )?,
            &c.ne(
                &c.add(&c.neg(&odd)?, &c.neg(&half_epsilon)?)?,
                &c.neg(&even_next)?,
            )?,
        )?;
        let eta = constant(k.minimum_subnormal)?;
        let minimum_normal = constant(k.minimum_normal)?;
        let subnormal_inputs = c.ne(
            &c.div(&c.add(&eta, &eta)?, &minimum_normal)?,
            &constant(2.0 * k.epsilon)?,
        )?;
        let multiplication_underflow = c.ne(
            &c.div(&c.mul(&minimum_normal, &half)?, &minimum_normal)?,
            &half,
        )?;
        let division_underflow = c.ne(
            &c.div(&c.div(&minimum_normal, &two)?, &minimum_normal)?,
            &half,
        )?;
        let (positive_high, positive_low) = c.two_sum(&one, &half_epsilon)?;
        let positive_witness = c.maximum(
            &c.ne(&positive_high, &one)?,
            &c.ne(&positive_low, &half_epsilon)?,
        )?;
        let (negative_high, negative_low) = c.two_sum(&c.neg(&one)?, &c.neg(&half_epsilon)?)?;
        let negative_witness = c.maximum(
            &c.ne(&negative_high, &c.neg(&one)?)?,
            &c.ne(&negative_low, &c.neg(&half_epsilon)?)?,
        )?;
        let (residual_high, residual_low) = c.two_sum(&one, &c.neg(&eta)?)?;
        let retained_subnormal = c.maximum(
            &c.ne(&residual_high, &one)?,
            &c.ne(&c.div(&residual_low, &minimum_normal)?, &c.neg(&epsilon)?)?,
        )?;
        let (cancel_high, cancel_low) = c.two_sum(&odd, &c.neg(&one)?)?;
        let cancellation_witness = c.maximum(
            &retained_subnormal,
            &c.maximum(&c.ne(&cancel_high, &epsilon)?, &c.ne(&cancel_low, &zero)?)?,
        )?;
        let failures = [
            positive_ties,
            negative_ties,
            subnormal_inputs,
            multiplication_underflow,
            division_underflow,
            positive_witness,
            negative_witness,
            cancellation_witness,
        ];
        let mut encoded = Tensor::<R>::zeros(&[], DType::U32, client.device())?;
        for (index, mask) in failures.iter().enumerate() {
            let bit = c.cast(&c.mul_scalar(mask, f64::from(1u32 << index))?, DType::U32)?;
            encoded = c.add(&encoded, &bit.reshape(&[])?)?;
        }
        Ok(encoded)
    }
}
