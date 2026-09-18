//! Fast Walsh-Hadamard transform helper for CPU tensors.

use super::super::kernels;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Normalized Walsh-Hadamard transform on each `block_size` segment of the last axis.
pub fn fwht_impl(
    client: &CpuClient,
    x: &Tensor<CpuRuntime>,
    block_size: usize,
    signs: Option<&Tensor<CpuRuntime>>,
) -> Result<Tensor<CpuRuntime>> {
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

    if x.numel() == 0 {
        return Tensor::<CpuRuntime>::empty(shape, dtype, &client.device);
    }

    let x_contig = ensure_contiguous(x)?;
    let signs_contig = signs.map(ensure_contiguous).transpose()?;
    let rows = x.numel() / last_dim;

    let out = Tensor::<CpuRuntime>::empty(shape, dtype, &client.device)?;

    let x_ptr = x_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        let signs_ptr = signs_contig.as_ref().map(|s| s.ptr() as *const T);
        unsafe {
            kernels::fwht_kernel::<T>(
                x_ptr as *const T,
                out_ptr as *mut T,
                signs_ptr,
                last_dim,
                rows,
                block_size,
            );
        }
    }, "fwht");

    Ok(out)
}
