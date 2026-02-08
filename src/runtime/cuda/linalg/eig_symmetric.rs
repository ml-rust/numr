//! Symmetric eigendecomposition for CUDA
//!
//! All operations run entirely on GPU with zero CPU transfers.

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use super::super::kernels;
use crate::algorithm::linalg::{EigenDecomposition, validate_linalg_dtype, validate_square_matrix};
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::{AllocGuard, Allocator, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Symmetric eigendecomposition via Jacobi method - runs entirely on GPU.
pub fn eig_decompose_symmetric_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
) -> Result<EigenDecomposition<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Handle empty matrix
    if n == 0 {
        let eigenvalues_ptr = client.allocator().allocate(0)?;
        let eigenvectors_ptr = client.allocator().allocate(0)?;
        let eigenvalues =
            unsafe { CudaClient::tensor_from_raw(eigenvalues_ptr, &[0], dtype, device) };
        let eigenvectors =
            unsafe { CudaClient::tensor_from_raw(eigenvectors_ptr, &[0, 0], dtype, device) };
        return Ok(EigenDecomposition {
            eigenvalues,
            eigenvectors,
        });
    }

    // Handle 1x1 matrix
    if n == 1 {
        let eigenvalues_size = dtype.size_in_bytes();
        let eigenvectors_size = dtype.size_in_bytes();
        let eigenvalues_ptr = client.allocator().allocate(eigenvalues_size)?;
        let eigenvectors_ptr = client.allocator().allocate(eigenvectors_size)?;

        // Copy the single element as eigenvalue
        CudaRuntime::copy_within_device(
            a.storage().ptr(),
            eigenvalues_ptr,
            eigenvalues_size,
            device,
        )?;

        // Eigenvector is [1.0]
        match dtype {
            DType::F32 => {
                let one: [u8; 4] = 1.0f32.to_ne_bytes();
                CudaRuntime::copy_to_device(&one, eigenvectors_ptr, device)?;
            }
            DType::F64 => {
                let one: [u8; 8] = 1.0f64.to_ne_bytes();
                CudaRuntime::copy_to_device(&one, eigenvectors_ptr, device)?;
            }
            _ => unreachable!(),
        }

        let eigenvalues =
            unsafe { CudaClient::tensor_from_raw(eigenvalues_ptr, &[1], dtype, device) };
        let eigenvectors =
            unsafe { CudaClient::tensor_from_raw(eigenvectors_ptr, &[1, 1], dtype, device) };
        return Ok(EigenDecomposition {
            eigenvalues,
            eigenvectors,
        });
    }

    let elem_size = dtype.size_in_bytes();

    // Allocate working buffers on GPU
    let work_size = n * n * elem_size;
    let eigenvectors_size = n * n * elem_size;
    let eigenvalues_size = n * elem_size;
    let flag_size = std::mem::size_of::<i32>();

    let work_guard = AllocGuard::new(client.allocator(), work_size)?;
    let eigenvectors_guard = AllocGuard::new(client.allocator(), eigenvectors_size)?;
    let eigenvalues_guard = AllocGuard::new(client.allocator(), eigenvalues_size)?;
    let converged_flag_guard = AllocGuard::new(client.allocator(), flag_size)?;

    let work_ptr = work_guard.ptr();
    let eigenvectors_ptr = eigenvectors_guard.ptr();
    let eigenvalues_ptr = eigenvalues_guard.ptr();
    let converged_flag_ptr = converged_flag_guard.ptr();

    // Copy input to working buffer
    CudaRuntime::copy_within_device(a.storage().ptr(), work_ptr, work_size, device)?;

    // Zero-initialize converged flag
    let zero_i32: [u8; 4] = [0; 4];
    CudaRuntime::copy_to_device(&zero_i32, converged_flag_ptr, device)?;

    // Launch eigendecomposition kernel
    let result = unsafe {
        kernels::launch_eig_jacobi_symmetric(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            work_ptr,
            eigenvectors_ptr,
            eigenvalues_ptr,
            converged_flag_ptr,
            n,
        )
    };

    result?;

    client.synchronize();

    // Compute absolute values of eigenvalues for sorting by magnitude (all on GPU)
    let abs_eigenvalues_size = n * elem_size;
    let abs_eigenvalues_guard = AllocGuard::new(client.allocator(), abs_eigenvalues_size)?;
    let abs_eigenvalues_ptr = abs_eigenvalues_guard.ptr();

    let abs_result = unsafe {
        kernels::launch_unary_op(
            client.context(),
            client.stream(),
            device.index,
            "abs",
            dtype,
            eigenvalues_ptr,
            abs_eigenvalues_ptr,
            n,
        )
    };

    abs_result?;

    // GPU argsort to get sorted indices (descending by magnitude)
    let indices_size = n * std::mem::size_of::<i64>();
    let indices_guard = AllocGuard::new(client.allocator(), indices_size)?;
    let indices_ptr = indices_guard.ptr();

    let argsort_result = unsafe {
        kernels::launch_argsort(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            abs_eigenvalues_ptr, // sort by absolute values
            indices_ptr,         // output: sorted indices
            1,                   // outer_size
            n,                   // sort_size
            1,                   // inner_size
            true,                // descending (largest magnitude first)
        )
    };

    argsort_result?;

    // Reorder eigenvalues using GPU index_select
    let eigenvalues_sorted_size = n * elem_size;
    let eigenvalues_sorted_guard = AllocGuard::new(client.allocator(), eigenvalues_sorted_size)?;
    let eigenvalues_sorted_ptr = eigenvalues_sorted_guard.ptr();

    let eigenvalues_select_result = unsafe {
        kernels::launch_index_select(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            eigenvalues_ptr,        // input
            indices_ptr,            // indices
            eigenvalues_sorted_ptr, // output
            1,                      // outer_size
            n,                      // dim_size
            1,                      // inner_size
            n,                      // index_len (all n eigenvalues)
        )
    };

    eigenvalues_select_result?;

    // Reorder eigenvector columns using GPU index_select
    // eigenvectors is [n, n], select n columns -> [n, n]
    let eigenvectors_sorted_size = n * n * elem_size;
    let eigenvectors_sorted_guard = AllocGuard::new(client.allocator(), eigenvectors_sorted_size)?;
    let eigenvectors_sorted_ptr = eigenvectors_sorted_guard.ptr();

    let eigenvectors_select_result = unsafe {
        kernels::launch_index_select(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            eigenvectors_ptr,        // input [n, n]
            indices_ptr,             // indices
            eigenvectors_sorted_ptr, // output [n, n]
            n,                       // outer_size (rows)
            n,                       // dim_size (columns to select from)
            1,                       // inner_size
            n,                       // index_len (all n columns)
        )
    };

    eigenvectors_select_result?;

    // Create final tensors
    let eigenvalues = unsafe {
        CudaClient::tensor_from_raw(eigenvalues_sorted_guard.release(), &[n], dtype, device)
    };
    let eigenvectors = unsafe {
        CudaClient::tensor_from_raw(eigenvectors_sorted_guard.release(), &[n, n], dtype, device)
    };

    Ok(EigenDecomposition {
        eigenvalues,
        eigenvectors,
    })
}
