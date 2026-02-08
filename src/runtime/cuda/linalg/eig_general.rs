//! General eigendecomposition for CUDA (non-symmetric matrices)

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use super::super::kernels;
use crate::algorithm::linalg::{
    GeneralEigenDecomposition, validate_linalg_dtype, validate_square_matrix,
};
use crate::error::Result;
use crate::runtime::{Allocator, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// General eigendecomposition for non-symmetric matrices
pub fn eig_decompose_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
) -> Result<GeneralEigenDecomposition<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(GeneralEigenDecomposition {
            eigenvalues_real: Tensor::<CudaRuntime>::from_slice(&[] as &[f64], &[0], device),
            eigenvalues_imag: Tensor::<CudaRuntime>::from_slice(&[] as &[f64], &[0], device),
            eigenvectors_real: Tensor::<CudaRuntime>::from_slice(&[] as &[f64], &[0, 0], device),
            eigenvectors_imag: Tensor::<CudaRuntime>::from_slice(&[] as &[f64], &[0, 0], device),
        });
    }

    // Allocate GPU buffers
    let matrix_size = n * n * dtype.size_in_bytes();
    let vector_size = n * dtype.size_in_bytes();

    let t_ptr = client.allocator().allocate(matrix_size); // Working buffer (Schur form)
    let z_ptr = client.allocator().allocate(matrix_size); // Schur vectors
    let eval_real_ptr = client.allocator().allocate(vector_size); // Real part of eigenvalues
    let eval_imag_ptr = client.allocator().allocate(vector_size); // Imag part of eigenvalues
    let evec_real_ptr = client.allocator().allocate(matrix_size); // Real part of eigenvectors
    let evec_imag_ptr = client.allocator().allocate(matrix_size); // Imag part of eigenvectors
    let flag_ptr = client.allocator().allocate(std::mem::size_of::<i32>());

    // Copy A to T (working buffer)
    CudaRuntime::copy_within_device(a.storage().ptr(), t_ptr, matrix_size, device)?;

    // Initialize converged flag to 0
    let zero_flag = [0i32];
    CudaRuntime::copy_to_device(bytemuck::cast_slice(&zero_flag), flag_ptr, device)?;

    // Launch native CUDA general eigenvalue decomposition kernel
    let result = unsafe {
        kernels::launch_eig_general(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            t_ptr,
            z_ptr,
            eval_real_ptr,
            eval_imag_ptr,
            evec_real_ptr,
            evec_imag_ptr,
            flag_ptr,
            n,
        )
    };

    // Handle kernel error - clean up all buffers
    if let Err(e) = result {
        client.allocator().deallocate(t_ptr, matrix_size);
        client.allocator().deallocate(z_ptr, matrix_size);
        client.allocator().deallocate(eval_real_ptr, vector_size);
        client.allocator().deallocate(eval_imag_ptr, vector_size);
        client.allocator().deallocate(evec_real_ptr, matrix_size);
        client.allocator().deallocate(evec_imag_ptr, matrix_size);
        client
            .allocator()
            .deallocate(flag_ptr, std::mem::size_of::<i32>());
        return Err(e);
    }

    client.synchronize();

    // Clean up working buffers (keep result buffers)
    client.allocator().deallocate(t_ptr, matrix_size);
    client.allocator().deallocate(z_ptr, matrix_size);
    client
        .allocator()
        .deallocate(flag_ptr, std::mem::size_of::<i32>());

    // Create result tensors from GPU pointers
    let eigenvalues_real =
        unsafe { CudaClient::tensor_from_raw(eval_real_ptr, &[n], dtype, device) };
    let eigenvalues_imag =
        unsafe { CudaClient::tensor_from_raw(eval_imag_ptr, &[n], dtype, device) };
    let eigenvectors_real =
        unsafe { CudaClient::tensor_from_raw(evec_real_ptr, &[n, n], dtype, device) };
    let eigenvectors_imag =
        unsafe { CudaClient::tensor_from_raw(evec_imag_ptr, &[n, n], dtype, device) };

    Ok(GeneralEigenDecomposition {
        eigenvalues_real,
        eigenvalues_imag,
        eigenvectors_real,
        eigenvectors_imag,
    })
}
