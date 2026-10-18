//! Relative singular-value cutoff shared by `lstsq` and `pinverse`.
//!
//! A singular value `s` counts toward the rank when `s > rcond * max(s)`.
//! Every other singular value is zeroed, never inverted. The cutoff runs on
//! the device that holds `s`.

use super::super::helpers::{linalg_working_dtype, machine_epsilon};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{CompareOps, ConditionalOps, ReduceOps, ScalarOps, UnaryOps};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Resolve the relative cutoff for an `[m, n]` matrix of `dtype`.
///
/// `None` gives `max(m, n) * eps`, where `eps` is the machine epsilon of the
/// working dtype: F64 for F64 input, F32 for every other float dtype.
///
/// # Errors
///
/// `InvalidArgument` when `rcond` is negative or not finite.
pub fn resolve_rcond(rcond: Option<f64>, m: usize, n: usize, dtype: DType) -> Result<f64> {
    match rcond {
        None => Ok(m.max(n) as f64 * machine_epsilon(linalg_working_dtype(dtype))),
        Some(r) if r.is_finite() && r >= 0.0 => Ok(r),
        Some(r) => Err(Error::InvalidArgument {
            arg: "rcond",
            reason: format!("rcond must be finite and >= 0, got {r}"),
        }),
    }
}

/// Invert the singular values above `rcond * max(s)` and count them.
///
/// `s` is `[.., p]` with `p > 0`, one row of singular values per matrix.
/// Returns `(s_inv, rank)`:
/// - `s_inv` has the shape of `s`. It holds `1 / s` above the cutoff and `0`
///   elsewhere.
/// - `rank` is `[..]`, the count above the cutoff, in the dtype of `s`.
///
/// A zero `max(s)` gives a zero cutoff, so every `s_inv` is `0` and the rank
/// is `0`.
pub fn svd_cutoff<R, C>(client: &C, s: &Tensor<R>, rcond: f64) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: CompareOps<R> + ConditionalOps<R> + ReduceOps<R> + ScalarOps<R> + UnaryOps<R>,
{
    let last = s
        .ndim()
        .checked_sub(1)
        .ok_or_else(|| Error::InvalidArgument {
            arg: "s",
            reason: "singular values must have at least 1 dim, got a scalar".to_string(),
        })?;
    let s_max = ReduceOps::max(client, s, &[last], true)?;
    let cutoff = ScalarOps::mul_scalar(client, &s_max, rcond)?;
    let keep = CompareOps::gt(client, s, &cutoff)?;
    let s_recip = UnaryOps::recip(client, s)?;
    let zero = Tensor::<R>::zeros(&[], s.dtype(), s.device())?;
    let s_inv = ConditionalOps::where_cond(client, &keep, &s_recip, &zero)?;
    let rank = ReduceOps::sum(client, &keep, &[last], false)?;
    Ok((s_inv, rank))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn client() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        (CpuClient::new(device.clone()), device)
    }

    #[test]
    fn default_rcond_uses_working_epsilon() {
        let r = resolve_rcond(None, 3, 5, DType::F64).unwrap();
        assert_eq!(r, 5.0 * f64::EPSILON);
        let r = resolve_rcond(None, 4, 2, DType::F32).unwrap();
        assert_eq!(r, 4.0 * f32::EPSILON as f64);
        let r = resolve_rcond(None, 4, 2, DType::F16).unwrap();
        assert_eq!(r, 4.0 * f32::EPSILON as f64);
    }

    #[test]
    fn explicit_rcond_checked() {
        assert_eq!(resolve_rcond(Some(0.0), 2, 2, DType::F64).unwrap(), 0.0);
        assert_eq!(resolve_rcond(Some(1e-3), 2, 2, DType::F64).unwrap(), 1e-3);
        for bad in [-1e-3, f64::NAN, f64::INFINITY] {
            let err = resolve_rcond(Some(bad), 2, 2, DType::F64).unwrap_err();
            assert!(matches!(err, Error::InvalidArgument { arg: "rcond", .. }));
        }
    }

    #[test]
    fn cutoff_zeroes_small_values() {
        let (c, d) = client();
        let s = Tensor::<CpuRuntime>::from_slice(&[4.0f64, 2.0, 1e-20, 0.0], &[4], &d).unwrap();
        let (s_inv, rank) = svd_cutoff(&c, &s, 1e-10).unwrap();
        assert_eq!(s_inv.to_vec::<f64>(), vec![0.25, 0.5, 0.0, 0.0]);
        assert_eq!(rank.shape(), &[] as &[usize]);
        assert_eq!(rank.to_vec::<f64>(), vec![2.0]);
    }

    #[test]
    fn zero_values_give_rank_zero() {
        let (c, d) = client();
        let s = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &d).unwrap();
        let (s_inv, rank) = svd_cutoff(&c, &s, 0.0).unwrap();
        assert_eq!(s_inv.to_vec::<f32>(), vec![0.0, 0.0]);
        assert_eq!(rank.to_vec::<f32>(), vec![0.0]);
    }

    #[test]
    fn rows_cut_independently() {
        let (c, d) = client();
        let s = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.5, 100.0, 1e-3], &[2, 2], &d).unwrap();
        let (s_inv, rank) = svd_cutoff(&c, &s, 1e-2).unwrap();
        assert_eq!(s_inv.to_vec::<f64>(), vec![1.0, 2.0, 0.01, 0.0]);
        assert_eq!(rank.shape(), &[2]);
        assert_eq!(rank.to_vec::<f64>(), vec![2.0, 1.0]);
    }
}
