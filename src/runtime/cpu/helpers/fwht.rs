//! Fast Walsh-Hadamard transform helper for CPU tensors.

use super::super::kernels;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::error::Result;
use crate::ops::common::validate_fwht_args;
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Normalized Walsh-Hadamard transform on each `block_size` segment of the last axis.
pub fn fwht_impl(
    client: &CpuClient,
    x: &Tensor<CpuRuntime>,
    block_size: usize,
    signs: Option<&Tensor<CpuRuntime>>,
) -> Result<Tensor<CpuRuntime>> {
    let last_dim = validate_fwht_args(x, block_size, signs)?;
    let shape = x.shape();
    let dtype = x.dtype();

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
