//! CUDA ILU(0) factorization implementation.

use super::super::{CudaClient, CudaRuntime};
use super::common::{split_lu_cuda, validate_cuda_dtype};
use crate::algorithm::sparse_linalg::{IluDecomposition, IluOptions, validate_square_sparse};
use crate::algorithm::sparse_linalg::{compute_levels_ilu, flatten_levels};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// ILU(0) factorization for CUDA.
pub fn ilu0_cuda(
    client: &CudaClient,
    a: &CsrData<CudaRuntime>,
    options: IluOptions,
) -> Result<IluDecomposition<CudaRuntime>> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();
    validate_cuda_dtype(dtype, "ilu0")?;

    // Extract CSR data for level analysis (this transfer is unavoidable for analysis)
    let row_ptrs: Vec<i64> = a.row_ptrs().to_vec();
    let col_indices: Vec<i64> = a.col_indices().to_vec();

    // Compute level schedule
    let schedule = compute_levels_ilu(n, &row_ptrs, &col_indices)?;
    let (level_ptrs, level_rows) = flatten_levels(&schedule);

    // Device reference
    let device = &client.device;

    // Allocate GPU buffers for level data
    let level_rows_gpu =
        Tensor::<CudaRuntime>::from_slice(&level_rows, &[level_rows.len()], device);

    // Convert row_ptrs and col_indices to i32 for GPU
    let row_ptrs_i32: Vec<i32> = row_ptrs.iter().map(|&x| x as i32).collect();
    let col_indices_i32: Vec<i32> = col_indices.iter().map(|&x| x as i32).collect();
    let row_ptrs_gpu =
        Tensor::<CudaRuntime>::from_slice(&row_ptrs_i32, &[row_ptrs_i32.len()], device);
    let col_indices_gpu =
        Tensor::<CudaRuntime>::from_slice(&col_indices_i32, &[col_indices_i32.len()], device);

    // Clone values for in-place factorization
    let values_gpu = a.values().clone();

    // Allocate diagonal indices buffer
    let diag_indices_gpu = Tensor::<CudaRuntime>::zeros(&[n], DType::I32, device);

    // Find diagonal indices on GPU
    unsafe {
        kernels::launch_find_diag_indices(
            &client.context,
            &client.stream,
            client.device.index,
            row_ptrs_gpu.storage().ptr(),
            col_indices_gpu.storage().ptr(),
            diag_indices_gpu.storage().ptr(),
            n as i32,
        )?;
    }

    // Process each level
    for level in 0..schedule.num_levels {
        let level_start = level_ptrs[level] as usize;
        let level_end = level_ptrs[level + 1] as usize;
        let level_size = (level_end - level_start) as i32;

        if level_size == 0 {
            continue;
        }

        // Get pointer to this level's rows
        let level_rows_ptr =
            level_rows_gpu.storage().ptr() + (level_start * std::mem::size_of::<i32>()) as u64;

        match dtype {
            DType::F32 => unsafe {
                kernels::launch_ilu0_level_f32(
                    &client.context,
                    &client.stream,
                    client.device.index,
                    level_rows_ptr,
                    level_size,
                    row_ptrs_gpu.storage().ptr(),
                    col_indices_gpu.storage().ptr(),
                    values_gpu.storage().ptr(),
                    diag_indices_gpu.storage().ptr(),
                    n as i32,
                    options.diagonal_shift as f32,
                )?;
            },
            DType::F64 => unsafe {
                kernels::launch_ilu0_level_f64(
                    &client.context,
                    &client.stream,
                    client.device.index,
                    level_rows_ptr,
                    level_size,
                    row_ptrs_gpu.storage().ptr(),
                    col_indices_gpu.storage().ptr(),
                    values_gpu.storage().ptr(),
                    diag_indices_gpu.storage().ptr(),
                    n as i32,
                    options.diagonal_shift,
                )?;
            },
            _ => unreachable!(),
        }
    }

    // Synchronize to ensure factorization is complete
    client
        .stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("CUDA stream sync failed: {:?}", e)))?;

    // Split into L and U
    split_lu_cuda(client, n, &row_ptrs, &col_indices, &values_gpu, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::sparse_linalg::SparseLinAlgAlgorithms;
    use crate::runtime::Runtime;

    fn get_client() -> CudaClient {
        let device = CudaRuntime::default_device();
        CudaRuntime::default_client(&device)
    }

    #[test]
    fn test_ilu0_basic() {
        let client = get_client();
        let device = &client.device;

        // Create a simple 3x3 sparse matrix in CSR format:
        // A = [ 4 -1  0]
        //     [-1  4 -1]
        //     [ 0 -1  4]
        let row_ptrs = Tensor::<CudaRuntime>::from_slice(&[0i64, 2, 5, 7], &[4], device);
        let col_indices =
            Tensor::<CudaRuntime>::from_slice(&[0i64, 1, 0, 1, 2, 1, 2], &[7], device);
        let values = Tensor::<CudaRuntime>::from_slice(
            &[4.0f32, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
            &[7],
            device,
        );

        let a = CsrData::new(row_ptrs, col_indices, values, [3, 3])
            .expect("CSR creation should succeed");

        let decomp = client
            .ilu0(&a, IluOptions::default())
            .expect("ILU0 should succeed");

        // L should be lower triangular (unit diagonal stored implicitly)
        // U should be upper triangular
        assert_eq!(decomp.l.shape, [3, 3]);
        assert_eq!(decomp.u.shape, [3, 3]);
    }
}
