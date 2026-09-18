//! Shared `fwht` argument validation.
//!
//! Every backend checks the same things before it touches a kernel: the block
//! size is a nonzero power of two that divides the last dim, the input is a
//! floating-point tensor with at least one dim, and `signs` (when given) is a
//! 1-D tensor of the same dtype whose width equals the last dim. The checks
//! live here so CPU and CUDA reject the same inputs with the same error
//! variants instead of each keeping its own copy.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Checks `fwht` arguments and returns the last-dim width.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] when `block_size` is zero, not a power of
/// two, does not divide the last dim, `x` has zero dimensions, or `signs` is not
/// 1-D with width equal to the last dim. Returns [`Error::DTypeMismatch`] when
/// `signs` has a different dtype than `x`. Returns [`Error::UnsupportedDType`]
/// for a non-floating-point `x`.
pub fn validate_fwht_args<R: Runtime<DType = DType>>(
    x: &Tensor<R>,
    block_size: usize,
    signs: Option<&Tensor<R>>,
) -> Result<usize> {
    if block_size == 0 {
        return Err(Error::InvalidArgument {
            arg: "block_size",
            reason: "must not be zero".to_string(),
        });
    }
    if !block_size.is_power_of_two() {
        return Err(Error::InvalidArgument {
            arg: "block_size",
            reason: format!("{block_size} is not a power of two"),
        });
    }

    let dtype = x.dtype();
    if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
        return Err(Error::UnsupportedDType { dtype, op: "fwht" });
    }

    let shape = x.shape();
    if shape.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "must have at least 1 dimension".to_string(),
        });
    }

    let last_dim = shape[shape.len() - 1];
    if !last_dim.is_multiple_of(block_size) {
        return Err(Error::InvalidArgument {
            arg: "block_size",
            reason: format!("{block_size} does not divide the last dim {last_dim}"),
        });
    }

    if let Some(s) = signs {
        let s_shape = s.shape();
        if s_shape.len() != 1 {
            return Err(Error::InvalidArgument {
                arg: "signs",
                reason: format!("must be 1-D, got {} dims", s_shape.len()),
            });
        }
        if s_shape[0] != last_dim {
            return Err(Error::InvalidArgument {
                arg: "signs",
                reason: format!("width {} does not match last dim {}", s_shape[0], last_dim),
            });
        }
        if s.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: s.dtype(),
            });
        }
    }

    Ok(last_dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    fn tensor(data: &[f32], shape: &[usize]) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(data, shape, &CpuDevice::new()).unwrap()
    }

    #[test]
    fn returns_last_dim() {
        let x = tensor(&[0.0; 16], &[2, 8]);
        assert_eq!(validate_fwht_args(&x, 4, None).unwrap(), 8);
    }

    #[test]
    fn zero_block_size_is_rejected() {
        let x = tensor(&[0.0; 8], &[8]);
        assert!(matches!(
            validate_fwht_args(&x, 0, None),
            Err(Error::InvalidArgument {
                arg: "block_size",
                ..
            })
        ));
    }

    #[test]
    fn non_power_of_two_block_size_is_rejected() {
        let x = tensor(&[0.0; 6], &[6]);
        assert!(matches!(
            validate_fwht_args(&x, 3, None),
            Err(Error::InvalidArgument {
                arg: "block_size",
                ..
            })
        ));
    }

    #[test]
    fn block_size_not_dividing_last_dim_is_rejected() {
        let x = tensor(&[0.0; 16], &[16]);
        assert!(matches!(
            validate_fwht_args(&x, 32, None),
            Err(Error::InvalidArgument {
                arg: "block_size",
                ..
            })
        ));
    }

    #[test]
    fn integer_dtype_is_rejected() {
        let x = Tensor::<CpuRuntime>::from_slice(&[0i32; 8], &[8], &CpuDevice::new()).unwrap();
        assert!(matches!(
            validate_fwht_args(&x, 8, None),
            Err(Error::UnsupportedDType { op: "fwht", .. })
        ));
    }

    #[test]
    fn signs_width_mismatch_is_rejected() {
        let x = tensor(&[0.0; 8], &[8]);
        let signs = tensor(&[1.0; 4], &[4]);
        assert!(matches!(
            validate_fwht_args(&x, 8, Some(&signs)),
            Err(Error::InvalidArgument { arg: "signs", .. })
        ));
    }

    #[test]
    fn signs_rank_mismatch_is_rejected() {
        let x = tensor(&[0.0; 8], &[8]);
        let signs = tensor(&[1.0; 8], &[2, 4]);
        assert!(matches!(
            validate_fwht_args(&x, 8, Some(&signs)),
            Err(Error::InvalidArgument { arg: "signs", .. })
        ));
    }

    #[test]
    fn signs_dtype_mismatch_is_rejected() {
        let x = tensor(&[0.0; 8], &[8]);
        let signs =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64; 8], &[8], &CpuDevice::new()).unwrap();
        assert!(matches!(
            validate_fwht_args(&x, 8, Some(&signs)),
            Err(Error::DTypeMismatch { .. })
        ));
    }
}
