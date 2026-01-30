//! Quantile, percentile, and median operations for WebGPU runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{TensorOps, compute_reduce_strides, reduce_dim_output_shape};
use crate::runtime::statistics_common::{Interpolation, compute_quantile_interpolation};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{RuntimeClient, ensure_contiguous, normalize_dim};
use crate::tensor::Tensor;

/// Compute quantile along a dimension using composition.
///
/// WebGPU quantile uses GPU-based sorting followed by CPU-side interpolation.
/// For non-F32 types, data is cast to F32, computed, then cast back.
pub fn quantile_impl(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    q: f64,
    dim: Option<isize>,
    keepdim: bool,
    interpolation: &str,
) -> Result<Tensor<WgpuRuntime>> {
    // Validate q is in [0, 1]
    if !(0.0..=1.0).contains(&q) {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("Quantile q must be in [0, 1], got {}", q),
        });
    }

    let interp = Interpolation::parse(interpolation)?;
    let dtype = a.dtype();

    // Handle None dim: flatten to 1D first
    if dim.is_none() {
        let numel = a.numel();
        if numel == 0 {
            let out_shape = if keepdim { vec![1; a.ndim()] } else { vec![] };
            return Ok(Tensor::<WgpuRuntime>::empty(
                &out_shape,
                dtype,
                client.device(),
            ));
        }

        let flat = a.reshape(&[numel])?;
        return quantile_impl(client, &flat, q, Some(0), keepdim, interpolation);
    }

    let dim_val = dim.unwrap();
    let shape = a.shape();
    let ndim = shape.len();

    if ndim == 0 {
        return Ok(a.clone());
    }

    let dim_idx = normalize_dim(dim_val, ndim)?;
    let dim_size = shape[dim_idx];

    if dim_size == 0 {
        let out_shape = reduce_dim_output_shape(shape, dim_idx, keepdim);
        return Ok(Tensor::<WgpuRuntime>::empty(
            &out_shape,
            dtype,
            client.device(),
        ));
    }

    // Sort along dimension using WebGPU sort
    let sorted = client.sort(a, dim_val, false)?;

    // Compute output shape and strides
    let out_shape = reduce_dim_output_shape(shape, dim_idx, keepdim);
    let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim_idx);

    // Calculate quantile indices using shared logic
    let (floor_idx, ceil_idx, frac) =
        crate::runtime::statistics_common::compute_quantile_indices(q, reduce_size);

    // Ensure sorted is contiguous for data access
    let sorted_contig = ensure_contiguous(&sorted);

    // Check for empty output
    let out_numel = out_shape.iter().product::<usize>();
    if out_numel == 0 {
        return Ok(Tensor::<WgpuRuntime>::empty(
            &out_shape,
            dtype,
            client.device(),
        ));
    }

    // Compute on CPU and copy back
    // WebGPU only supports F32 and I32/U32 natively
    match dtype {
        DType::F32 => {
            let sorted_data: Vec<f32> = sorted_contig.to_vec();
            let result = compute_quantile_interpolation(
                &sorted_data,
                outer_size,
                reduce_size,
                inner_size,
                floor_idx,
                ceil_idx,
                frac,
                interp,
            );
            Ok(Tensor::<WgpuRuntime>::from_slice(
                &result,
                &out_shape,
                client.device(),
            ))
        }
        _ => {
            // For other dtypes, cast to f32, compute, cast back
            let sorted_f32 = client.cast(&sorted, DType::F32)?;
            let result_f32 = quantile_impl(
                client,
                &sorted_f32,
                q,
                Some(dim_val),
                keepdim,
                interpolation,
            )?;
            client.cast(&result_f32, dtype)
        }
    }
}

/// Compute percentile (quantile * 100) along a dimension.
pub fn percentile_impl(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    p: f64,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<Tensor<WgpuRuntime>> {
    if !(0.0..=100.0).contains(&p) {
        return Err(Error::InvalidArgument {
            arg: "p",
            reason: format!("Percentile p must be in [0, 100], got {}", p),
        });
    }

    quantile_impl(client, a, p / 100.0, dim, keepdim, "linear")
}

/// Compute median (50th percentile) along a dimension.
pub fn median_impl(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<Tensor<WgpuRuntime>> {
    quantile_impl(client, a, 0.5, dim, keepdim, "linear")
}
