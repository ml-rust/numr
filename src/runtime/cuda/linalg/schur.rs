//! Schur decomposition for CUDA

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use super::super::kernels;
use crate::algorithm::linalg::{SchurDecomposition, validate_linalg_dtype, validate_square_matrix};
use crate::error::Result;
use crate::runtime::{Allocator, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Schur decomposition: A = Z @ T @ Z^T
pub fn schur_decompose_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
) -> Result<SchurDecomposition<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Handle trivial cases that don't require iteration
    if n == 0 {
        let z_ptr = client.allocator().allocate(0);
        let t_ptr = client.allocator().allocate(0);
        let z = unsafe { CudaClient::tensor_from_raw(z_ptr, &[0, 0], dtype, device) };
        let t = unsafe { CudaClient::tensor_from_raw(t_ptr, &[0, 0], dtype, device) };
        return Ok(SchurDecomposition { z, t });
    }

    // Allocate GPU buffers for T (working copy of A) and Z (orthogonal matrix)
    let matrix_size = n * n * dtype.size_in_bytes();
    let t_ptr = client.allocator().allocate(matrix_size);
    let z_ptr = client.allocator().allocate(matrix_size);
    let flag_ptr = client.allocator().allocate(std::mem::size_of::<i32>());

    // Copy A to T (will be modified in-place)
    CudaRuntime::copy_within_device(a.storage().ptr(), t_ptr, matrix_size, device)?;

    // Initialize converged flag to 0
    let zero_flag = [0i32];
    CudaRuntime::copy_to_device(bytemuck::cast_slice(&zero_flag), flag_ptr, device)?;

    // Launch native CUDA Schur decomposition kernel
    let result = unsafe {
        kernels::launch_schur_decompose(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            t_ptr,
            z_ptr,
            flag_ptr,
            n,
        )
    };

    // Handle kernel error
    if let Err(e) = result {
        client.allocator().deallocate(t_ptr, matrix_size);
        client.allocator().deallocate(z_ptr, matrix_size);
        client
            .allocator()
            .deallocate(flag_ptr, std::mem::size_of::<i32>());
        return Err(e);
    }

    client.synchronize();

    // Clean up flag buffer
    client
        .allocator()
        .deallocate(flag_ptr, std::mem::size_of::<i32>());

    let z = unsafe { CudaClient::tensor_from_raw(z_ptr, &[n, n], dtype, device) };
    let t = unsafe { CudaClient::tensor_from_raw(t_ptr, &[n, n], dtype, device) };
    Ok(SchurDecomposition { z, t })
}
