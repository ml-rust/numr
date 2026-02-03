//! Common utilities for CUDA sparse linear algebra.

use super::super::{CudaClient, CudaRuntime};
use crate::algorithm::sparse_linalg::{IcDecomposition, IluDecomposition};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// Validate dtype for CUDA sparse linear algebra (F32 and F64 only).
pub fn validate_cuda_dtype(dtype: DType, op: &'static str) -> Result<()> {
    if dtype != DType::F32 && dtype != DType::F64 {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

/// Split factored LU matrix into L and U components.
pub fn split_lu_cuda(
    client: &CudaClient,
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
    values_gpu: &Tensor<CudaRuntime>,
    dtype: DType,
) -> Result<IluDecomposition<CudaRuntime>> {
    // Extract values to CPU for splitting (TODO: could be GPU kernel)
    let values: Vec<f64> = match dtype {
        DType::F32 => values_gpu
            .to_vec::<f32>()
            .iter()
            .map(|&x| x as f64)
            .collect(),
        DType::F64 => values_gpu.to_vec(),
        _ => unreachable!(),
    };

    // Split into L and U
    let mut l_row_ptrs = vec![0i64; n + 1];
    let mut l_col_indices = Vec::new();
    let mut l_values = Vec::new();
    let mut u_row_ptrs = vec![0i64; n + 1];
    let mut u_col_indices = Vec::new();
    let mut u_values = Vec::new();

    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;

        for idx in start..end {
            let j = col_indices[idx] as usize;
            let val = values[idx];

            if j < i {
                l_col_indices.push(j as i64);
                l_values.push(val);
            } else {
                u_col_indices.push(j as i64);
                u_values.push(val);
            }
        }

        l_row_ptrs[i + 1] = l_col_indices.len() as i64;
        u_row_ptrs[i + 1] = u_col_indices.len() as i64;
    }

    // Create output tensors
    let l_row_ptrs_t = Tensor::<CudaRuntime>::from_slice(&l_row_ptrs, &[n + 1], &client.device);
    let l_col_indices_t =
        Tensor::<CudaRuntime>::from_slice(&l_col_indices, &[l_col_indices.len()], &client.device);
    let u_row_ptrs_t = Tensor::<CudaRuntime>::from_slice(&u_row_ptrs, &[n + 1], &client.device);
    let u_col_indices_t =
        Tensor::<CudaRuntime>::from_slice(&u_col_indices, &[u_col_indices.len()], &client.device);

    let (l_values_t, u_values_t) = match dtype {
        DType::F32 => {
            let l_f32: Vec<f32> = l_values.iter().map(|&x| x as f32).collect();
            let u_f32: Vec<f32> = u_values.iter().map(|&x| x as f32).collect();
            (
                Tensor::<CudaRuntime>::from_slice(&l_f32, &[l_f32.len()], &client.device),
                Tensor::<CudaRuntime>::from_slice(&u_f32, &[u_f32.len()], &client.device),
            )
        }
        DType::F64 => (
            Tensor::<CudaRuntime>::from_slice(&l_values, &[l_values.len()], &client.device),
            Tensor::<CudaRuntime>::from_slice(&u_values, &[u_values.len()], &client.device),
        ),
        _ => unreachable!(),
    };

    let l = CsrData::new(l_row_ptrs_t, l_col_indices_t, l_values_t, [n, n])?;
    let u = CsrData::new(u_row_ptrs_t, u_col_indices_t, u_values_t, [n, n])?;

    Ok(IluDecomposition { l, u })
}

/// Extract lower triangular matrix after IC factorization.
pub fn extract_lower_cuda(
    client: &CudaClient,
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
    values_gpu: &Tensor<CudaRuntime>,
    dtype: DType,
) -> Result<IcDecomposition<CudaRuntime>> {
    // Extract values to CPU (TODO: could be GPU kernel)
    let values: Vec<f64> = match dtype {
        DType::F32 => values_gpu
            .to_vec::<f32>()
            .iter()
            .map(|&x| x as f64)
            .collect(),
        DType::F64 => values_gpu.to_vec(),
        _ => unreachable!(),
    };

    // Filter to lower triangle
    let mut new_row_ptrs = vec![0i64; n + 1];
    let mut new_col_indices = Vec::new();
    let mut new_values = Vec::new();

    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;

        for idx in start..end {
            let j = col_indices[idx] as usize;
            if j <= i {
                new_col_indices.push(j as i64);
                new_values.push(values[idx]);
            }
        }

        new_row_ptrs[i + 1] = new_col_indices.len() as i64;
    }

    // Create output tensors
    let l_row_ptrs_t = Tensor::<CudaRuntime>::from_slice(&new_row_ptrs, &[n + 1], &client.device);
    let l_col_indices_t = Tensor::<CudaRuntime>::from_slice(
        &new_col_indices,
        &[new_col_indices.len()],
        &client.device,
    );

    let l_values_t = match dtype {
        DType::F32 => {
            let f32_vals: Vec<f32> = new_values.iter().map(|&x| x as f32).collect();
            Tensor::<CudaRuntime>::from_slice(&f32_vals, &[f32_vals.len()], &client.device)
        }
        DType::F64 => {
            Tensor::<CudaRuntime>::from_slice(&new_values, &[new_values.len()], &client.device)
        }
        _ => unreachable!(),
    };

    let l = CsrData::new(l_row_ptrs_t, l_col_indices_t, l_values_t, [n, n])?;

    Ok(IcDecomposition { l })
}
