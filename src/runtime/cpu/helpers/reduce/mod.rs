//! Reduction operation helpers for CPU tensors

mod common;
mod multi_dim;
mod precision;
mod single_dim;

pub use precision::reduce_impl_with_precision;

use common::should_fuse_multi_dim_reduction;
use multi_dim::reduce_multi_dim_fused;
use single_dim::reduce_single_dim;

use crate::dispatch_dtype;
use crate::error::{Error, Result};
use crate::ops::{AccumulationPrecision, Kernel, ReduceOp, reduce_output_shape};
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Reduce implementation with native precision
pub fn reduce_impl(
    client: &CpuClient,
    op: ReduceOp,
    a: &Tensor<CpuRuntime>,
    dims: &[usize],
    keepdim: bool,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    for &d in dims {
        if d >= ndim {
            return Err(Error::InvalidDimension {
                dim: d as isize,
                ndim,
            });
        }
    }

    // Fast path: reduce last dimension when contiguous (uses SIMD kernel)
    if dims.len() == 1 && dims[0] == ndim - 1 && a.is_contiguous() {
        let reduce_size = shape[ndim - 1];
        let outer_size: usize = shape[..ndim - 1].iter().product();
        let outer_size = outer_size.max(1);

        let out_shape = reduce_output_shape(shape, dims, keepdim);
        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);

        let a_ptr = a.storage().ptr();
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                <CpuClient as Kernel<CpuRuntime>>::reduce::<T>(
                    client,
                    op,
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                );
            }
        }, op_name);

        Ok(out)
    } else if dims.is_empty() {
        Ok(a.clone())
    } else if should_fuse_multi_dim_reduction(a, dims) {
        reduce_multi_dim_fused(
            client,
            op,
            a,
            dims,
            keepdim,
            AccumulationPrecision::Native,
            op_name,
        )
    } else {
        let a_contig = ensure_contiguous(a);

        let mut sorted_dims: Vec<usize> = dims.to_vec();
        sorted_dims.sort_unstable();
        sorted_dims.reverse();

        let mut current = a_contig;
        for &dim in &sorted_dims {
            current = reduce_single_dim(client, op, &current, dim, keepdim, op_name)?;
        }

        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::{AccumulationPrecision, ReduceOps};
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_fused_multi_dim_sum_matches_expected() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let data: Vec<f32> = (1..=24).map(|v| v as f32).collect();
        let a = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3, 4], &device);

        let out = client.sum(&a, &[1, 2], false).unwrap();
        let got: Vec<f32> = out.to_vec();
        assert_eq!(got, vec![78.0, 222.0]);
    }

    #[test]
    fn test_fused_multi_dim_mean_keepdim_matches_expected() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let data: Vec<f32> = (1..=24).map(|v| v as f32).collect();
        let a = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3, 4], &device);

        let out = client.mean(&a, &[0, 2], true).unwrap();
        assert_eq!(out.shape(), &[1, 3, 1]);
        let got: Vec<f32> = out.to_vec();
        assert_eq!(got, vec![8.5, 12.5, 16.5]);
    }

    #[test]
    fn test_fused_multi_dim_max_and_precision_sum() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let data: Vec<f32> = (1..=24).map(|v| v as f32).collect();
        let a = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3, 4], &device);

        let max_out = client.max(&a, &[0, 1], false).unwrap();
        let max_vals: Vec<f32> = max_out.to_vec();
        assert_eq!(max_vals, vec![21.0, 22.0, 23.0, 24.0]);

        let sum_prec = client
            .sum_with_precision(&a, &[0, 2], false, AccumulationPrecision::FP64)
            .unwrap();
        let sum_vals: Vec<f32> = sum_prec.to_vec();
        assert_eq!(sum_vals, vec![68.0, 100.0, 132.0]);
    }
}
