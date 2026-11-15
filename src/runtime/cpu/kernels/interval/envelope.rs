//! Arithmetic envelope probes evaluated in the lane kernels' own arithmetic.

use super::lane_float::{LaneFloat, two_sum};
use crate::ops::impl_generic::interval::envelope::EnvelopeConstants;
use std::hint::black_box;

/// Failure bits of the eight probes, least significant first.
///
/// Opaque operands keep the compiler from folding probes into host constants.
pub fn probe<T: LaneFloat>(constants: EnvelopeConstants) -> u32 {
    let k = |value: f64| black_box(T::of(value));
    let (zero, one, two, half) = (k(0.0), k(1.0), k(2.0), k(0.5));
    let epsilon = k(constants.epsilon);
    let half_epsilon = epsilon * T::of(0.5);
    let odd = one + epsilon;
    let even_next = k(1.0 + 2.0 * constants.epsilon);
    let eta = k(constants.minimum_subnormal);
    let minimum_normal = k(constants.minimum_normal);
    let (positive_high, positive_low) = two_sum(one, half_epsilon);
    let (negative_high, negative_low) = two_sum(-one, -half_epsilon);
    let (residual_high, residual_low) = two_sum(one, -eta);
    let (cancel_high, cancel_low) = two_sum(odd, -one);
    let failures = [
        one + half_epsilon != one || odd + half_epsilon != even_next,
        -one + -half_epsilon != -one || -odd + -half_epsilon != -even_next,
        (eta + eta) / minimum_normal != k(2.0 * constants.epsilon),
        (minimum_normal * half) / minimum_normal != half,
        (minimum_normal / two) / minimum_normal != half,
        positive_high != one || positive_low != half_epsilon,
        negative_high != -one || negative_low != -half_epsilon,
        residual_high != one
            || residual_low / minimum_normal != -epsilon
            || cancel_high != epsilon
            || cancel_low != zero,
    ];
    failures
        .iter()
        .enumerate()
        .fold(0, |bits, (index, &failed)| {
            bits | (u32::from(failed) << index)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    #[test]
    fn ieee_cpu_arithmetic_passes_every_probe() -> crate::error::Result<()> {
        assert_eq!(probe::<f64>(EnvelopeConstants::for_dtype(DType::F64)?), 0);
        assert_eq!(probe::<f32>(EnvelopeConstants::for_dtype(DType::F32)?), 0);
        Ok(())
    }
}
