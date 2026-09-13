//! Shared validation for the parametric activations every backend implements.
//!
//! `snake_beta` has per-channel parameters, so it needs more checking than a
//! plain elementwise activation: the channel axis, the parameter shapes, the
//! dtype agreement, and the epsilon. Every backend calls
//! [`validate_snake_beta`] and then runs its kernel over the `[outer, C, inner]`
//! view it returns, so CPU, CUDA, WebGPU and the autograd backward reject the
//! same inputs with the same error.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::activation::normalize_softmax_dim;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// The kernel geometry of a `snake_beta` call.
///
/// A contiguous `x` viewed as `[outer, channels, inner]`, where `channels` is
/// `x.shape[dim]`, `outer` is the product of the dims before `dim` and `inner`
/// the product of the dims after it. The parameter index of a flat element `i`
/// is `(i / inner) % channels`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnakeGeometry {
    /// Product of the dims before the channel axis.
    pub outer: usize,
    /// Length of the channel axis; also the length of `alpha` and `beta`.
    pub channels: usize,
    /// Product of the dims after the channel axis.
    pub inner: usize,
}

/// Check the arguments of `snake_beta` / `snake_beta_bwd` and return the
/// `[outer, C, inner]` geometry.
///
/// Rules:
/// - `x` has rank >= 1 and `dim` indexes one of its axes (negative counts from
///   the end).
/// - `alpha` and `beta` are rank 1 with length `x.shape[dim]`.
/// - `alpha`, `beta` and `x` share one float dtype.
/// - `eps` is finite and >= 0.
pub fn validate_snake_beta<R: Runtime<DType = DType>>(
    x: &Tensor<R>,
    alpha: &Tensor<R>,
    beta: &Tensor<R>,
    dim: isize,
    eps: f64,
) -> Result<SnakeGeometry> {
    let shape = x.shape();
    let ndim = shape.len();
    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "snake_beta requires rank >= 1, got a scalar".to_string(),
        });
    }
    let dim_idx = normalize_softmax_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;
    let channels = shape[dim_idx];

    let dtype = x.dtype();
    if !dtype.is_float() {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "snake_beta",
        });
    }
    for (name, param) in [("alpha", alpha), ("beta", beta)] {
        if param.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: param.dtype(),
            });
        }
        let pshape = param.shape();
        if pshape.len() != 1 || pshape[0] != channels {
            return Err(Error::InvalidArgument {
                arg: name,
                reason: format!(
                    "snake_beta expects {name} of shape [{channels}] to match x.shape[{dim_idx}], \
                     got {pshape:?}"
                ),
            });
        }
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(Error::InvalidArgument {
            arg: "eps",
            reason: format!("snake_beta requires a finite eps >= 0, got {eps}"),
        });
    }

    Ok(SnakeGeometry {
        outer: shape[..dim_idx].iter().product(),
        channels,
        inner: shape[dim_idx + 1..].iter().product(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    fn tensor(shape: &[usize], dtype: DType) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::zeros(shape, dtype, &CpuDevice::new()).unwrap()
    }

    #[test]
    fn geometry_splits_around_the_channel_axis() {
        let x = tensor(&[2, 5, 7, 3], DType::F32);
        let p = tensor(&[7], DType::F32);
        let g = validate_snake_beta(&x, &p, &p, 2, 1e-9).unwrap();
        assert_eq!(
            g,
            SnakeGeometry {
                outer: 10,
                channels: 7,
                inner: 3
            }
        );
        assert_eq!(g.outer * g.channels * g.inner, x.numel());
    }

    #[test]
    fn negative_dim_counts_from_the_end() {
        let x = tensor(&[3, 4], DType::F32);
        let p = tensor(&[4], DType::F32);
        let g = validate_snake_beta(&x, &p, &p, -1, 0.0).unwrap();
        assert_eq!(g.outer, 3);
        assert_eq!(g.channels, 4);
        assert_eq!(g.inner, 1);
    }

    #[test]
    fn rank_one_input_is_its_own_channel_axis() {
        let x = tensor(&[6], DType::F64);
        let p = tensor(&[6], DType::F64);
        let g = validate_snake_beta(&x, &p, &p, 0, 1e-9).unwrap();
        assert_eq!((g.outer, g.channels, g.inner), (1, 6, 1));
    }

    #[test]
    fn scalar_input_is_rejected() {
        let x = tensor(&[], DType::F32);
        let p = tensor(&[1], DType::F32);
        assert!(matches!(
            validate_snake_beta(&x, &p, &p, 0, 1e-9),
            Err(Error::InvalidArgument { arg: "x", .. })
        ));
    }

    #[test]
    fn out_of_range_dim_is_rejected() {
        let x = tensor(&[2, 3], DType::F32);
        let p = tensor(&[3], DType::F32);
        assert!(matches!(
            validate_snake_beta(&x, &p, &p, 2, 1e-9),
            Err(Error::InvalidDimension { dim: 2, ndim: 2 })
        ));
        assert!(matches!(
            validate_snake_beta(&x, &p, &p, -3, 1e-9),
            Err(Error::InvalidDimension { dim: -3, ndim: 2 })
        ));
    }

    #[test]
    fn parameter_rank_and_length_are_checked() {
        let x = tensor(&[2, 3, 4], DType::F32);
        let good = tensor(&[3], DType::F32);
        let wrong_len = tensor(&[4], DType::F32);
        let wrong_rank = tensor(&[1, 3, 1], DType::F32);
        assert!(matches!(
            validate_snake_beta(&x, &wrong_len, &good, 1, 1e-9),
            Err(Error::InvalidArgument { arg: "alpha", .. })
        ));
        assert!(matches!(
            validate_snake_beta(&x, &good, &wrong_rank, 1, 1e-9),
            Err(Error::InvalidArgument { arg: "beta", .. })
        ));
    }

    #[test]
    fn dtype_mismatch_is_rejected() {
        let x = tensor(&[2, 3], DType::F32);
        let p32 = tensor(&[3], DType::F32);
        let p64 = tensor(&[3], DType::F64);
        assert!(matches!(
            validate_snake_beta(&x, &p64, &p32, 1, 1e-9),
            Err(Error::DTypeMismatch {
                lhs: DType::F32,
                rhs: DType::F64
            })
        ));
        assert!(matches!(
            validate_snake_beta(&x, &p32, &p64, 1, 1e-9),
            Err(Error::DTypeMismatch { .. })
        ));
    }

    #[test]
    fn integer_input_is_rejected() {
        let x = tensor(&[2, 3], DType::I32);
        let p = tensor(&[3], DType::I32);
        assert!(matches!(
            validate_snake_beta(&x, &p, &p, 1, 1e-9),
            Err(Error::UnsupportedDType {
                dtype: DType::I32,
                ..
            })
        ));
    }

    #[test]
    fn bad_eps_is_rejected() {
        let x = tensor(&[2, 3], DType::F32);
        let p = tensor(&[3], DType::F32);
        for eps in [-1e-9, f64::NAN, f64::INFINITY] {
            assert!(
                matches!(
                    validate_snake_beta(&x, &p, &p, 1, eps),
                    Err(Error::InvalidArgument { arg: "eps", .. })
                ),
                "eps {eps} must be rejected"
            );
        }
    }
}
