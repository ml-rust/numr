//! Common utilities for CUDA sparse linear algebra.

use super::super::{CudaClient, CudaRuntime};
use crate::algorithm::sparse_linalg::{IcDecomposition, IluDecomposition};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels;
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
///
/// Uses GPU kernel to scatter values - no GPU→CPU transfer of values.
pub fn split_lu_cuda(
    client: &CudaClient,
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
    values_gpu: &Tensor<CudaRuntime>,
    dtype: DType,
) -> Result<IluDecomposition<CudaRuntime>> {
    let nnz = values_gpu.numel();

    // Build L and U structure on CPU (row_ptrs and col_indices are already on CPU)
    // Also build index mapping arrays for the GPU scatter kernel
    let mut l_row_ptrs = vec![0i64; n + 1];
    let mut l_col_indices = Vec::new();
    let mut u_row_ptrs = vec![0i64; n + 1];
    let mut u_col_indices = Vec::new();

    // l_map[i] = destination index in L values, or -1 if not in L
    // u_map[i] = destination index in U values, or -1 if not in U
    let mut l_map = vec![-1i32; nnz];
    let mut u_map = vec![-1i32; nnz];

    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;

        for idx in start..end {
            let j = col_indices[idx] as usize;

            if j < i {
                // Goes to L (strictly lower triangular)
                l_map[idx] = l_col_indices.len() as i32;
                l_col_indices.push(j as i64);
            } else {
                // Goes to U (upper triangular including diagonal)
                u_map[idx] = u_col_indices.len() as i32;
                u_col_indices.push(j as i64);
            }
        }

        l_row_ptrs[i + 1] = l_col_indices.len() as i64;
        u_row_ptrs[i + 1] = u_col_indices.len() as i64;
    }

    let l_nnz = l_col_indices.len();
    let u_nnz = u_col_indices.len();

    // Create structure tensors on GPU
    let l_row_ptrs_t = Tensor::<CudaRuntime>::from_slice(&l_row_ptrs, &[n + 1], &client.device);
    let l_col_indices_t =
        Tensor::<CudaRuntime>::from_slice(&l_col_indices, &[l_nnz], &client.device);
    let u_row_ptrs_t = Tensor::<CudaRuntime>::from_slice(&u_row_ptrs, &[n + 1], &client.device);
    let u_col_indices_t =
        Tensor::<CudaRuntime>::from_slice(&u_col_indices, &[u_nnz], &client.device);

    // Upload mapping arrays to GPU
    let l_map_gpu = Tensor::<CudaRuntime>::from_slice(&l_map, &[nnz], &client.device);
    let u_map_gpu = Tensor::<CudaRuntime>::from_slice(&u_map, &[nnz], &client.device);

    // Allocate output value tensors on GPU
    let l_values_t = Tensor::<CudaRuntime>::empty(&[l_nnz], dtype, &client.device);
    let u_values_t = Tensor::<CudaRuntime>::empty(&[u_nnz], dtype, &client.device);

    // Launch GPU kernel to scatter values
    unsafe {
        match dtype {
            DType::F32 => {
                kernels::launch_split_lu_scatter_f32(
                    &client.context,
                    &client.stream,
                    client.device.index,
                    values_gpu.storage().ptr(),
                    l_values_t.storage().ptr(),
                    u_values_t.storage().ptr(),
                    l_map_gpu.storage().ptr(),
                    u_map_gpu.storage().ptr(),
                    nnz as i32,
                )?;
            }
            DType::F64 => {
                kernels::launch_split_lu_scatter_f64(
                    &client.context,
                    &client.stream,
                    client.device.index,
                    values_gpu.storage().ptr(),
                    l_values_t.storage().ptr(),
                    u_values_t.storage().ptr(),
                    l_map_gpu.storage().ptr(),
                    u_map_gpu.storage().ptr(),
                    nnz as i32,
                )?;
            }
            _ => unreachable!(),
        }
    }

    let l = CsrData::new(l_row_ptrs_t, l_col_indices_t, l_values_t, [n, n])?;
    let u = CsrData::new(u_row_ptrs_t, u_col_indices_t, u_values_t, [n, n])?;

    Ok(IluDecomposition { l, u })
}

/// Extract lower triangular matrix after IC factorization.
///
/// Uses GPU kernel to scatter values - no GPU→CPU transfer of values.
pub fn extract_lower_cuda(
    client: &CudaClient,
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
    values_gpu: &Tensor<CudaRuntime>,
    dtype: DType,
) -> Result<IcDecomposition<CudaRuntime>> {
    let nnz = values_gpu.numel();

    // Build lower triangular structure on CPU (row_ptrs and col_indices are already on CPU)
    // Also build index mapping array for the GPU scatter kernel
    let mut new_row_ptrs = vec![0i64; n + 1];
    let mut new_col_indices = Vec::new();

    // lower_map[i] = destination index in output values, or -1 if not in lower
    let mut lower_map = vec![-1i32; nnz];

    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;

        for idx in start..end {
            let j = col_indices[idx] as usize;
            if j <= i {
                // Include lower triangle (j <= i includes diagonal)
                lower_map[idx] = new_col_indices.len() as i32;
                new_col_indices.push(j as i64);
            }
        }

        new_row_ptrs[i + 1] = new_col_indices.len() as i64;
    }

    let lower_nnz = new_col_indices.len();

    // Create structure tensors on GPU
    let l_row_ptrs_t = Tensor::<CudaRuntime>::from_slice(&new_row_ptrs, &[n + 1], &client.device);
    let l_col_indices_t =
        Tensor::<CudaRuntime>::from_slice(&new_col_indices, &[lower_nnz], &client.device);

    // Upload mapping array to GPU
    let lower_map_gpu = Tensor::<CudaRuntime>::from_slice(&lower_map, &[nnz], &client.device);

    // Allocate output value tensor on GPU
    let l_values_t = Tensor::<CudaRuntime>::empty(&[lower_nnz], dtype, &client.device);

    // Launch GPU kernel to scatter values
    unsafe {
        match dtype {
            DType::F32 => {
                kernels::launch_extract_lower_scatter_f32(
                    &client.context,
                    &client.stream,
                    client.device.index,
                    values_gpu.storage().ptr(),
                    l_values_t.storage().ptr(),
                    lower_map_gpu.storage().ptr(),
                    nnz as i32,
                )?;
            }
            DType::F64 => {
                kernels::launch_extract_lower_scatter_f64(
                    &client.context,
                    &client.stream,
                    client.device.index,
                    values_gpu.storage().ptr(),
                    l_values_t.storage().ptr(),
                    lower_map_gpu.storage().ptr(),
                    nnz as i32,
                )?;
            }
            _ => unreachable!(),
        }
    }

    let l = CsrData::new(l_row_ptrs_t, l_col_indices_t, l_values_t, [n, n])?;

    Ok(IcDecomposition { l })
}
