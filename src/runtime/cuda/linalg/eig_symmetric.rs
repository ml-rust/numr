//! Symmetric eigendecomposition for CUDA

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use super::super::kernels;
use crate::algorithm::linalg::{EigenDecomposition, validate_linalg_dtype, validate_square_matrix};
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::{Allocator, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Symmetric eigendecomposition via Jacobi method
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
        let eigenvalues_ptr = client.allocator().allocate(0);
        let eigenvectors_ptr = client.allocator().allocate(0);
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
        let eigenvalues_ptr = client.allocator().allocate(eigenvalues_size);
        let eigenvectors_ptr = client.allocator().allocate(eigenvectors_size);

        // Copy the single element as eigenvalue
        CudaRuntime::copy_within_device(
            a.storage().ptr(),
            eigenvalues_ptr,
            eigenvalues_size,
            device,
        );

        // Eigenvector is [1.0]
        match dtype {
            DType::F32 => {
                let one: [u8; 4] = 1.0f32.to_ne_bytes();
                CudaRuntime::copy_to_device(&one, eigenvectors_ptr, device);
            }
            DType::F64 => {
                let one: [u8; 8] = 1.0f64.to_ne_bytes();
                CudaRuntime::copy_to_device(&one, eigenvectors_ptr, device);
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

    // Allocate working buffers on GPU
    let work_size = n * n * dtype.size_in_bytes();
    let work_ptr = client.allocator().allocate(work_size);

    let eigenvectors_size = n * n * dtype.size_in_bytes();
    let eigenvectors_ptr = client.allocator().allocate(eigenvectors_size);

    let eigenvalues_size = n * dtype.size_in_bytes();
    let eigenvalues_ptr = client.allocator().allocate(eigenvalues_size);

    let flag_size = std::mem::size_of::<i32>();
    let converged_flag_ptr = client.allocator().allocate(flag_size);

    // Helper for cleanup on error
    let cleanup = |allocator: &super::super::CudaAllocator| {
        allocator.deallocate(work_ptr, work_size);
        allocator.deallocate(eigenvectors_ptr, eigenvectors_size);
        allocator.deallocate(eigenvalues_ptr, eigenvalues_size);
        allocator.deallocate(converged_flag_ptr, flag_size);
    };

    // Copy input to working buffer
    CudaRuntime::copy_within_device(a.storage().ptr(), work_ptr, work_size, device);

    // Zero-initialize converged flag
    let zero_i32: [u8; 4] = [0; 4];
    CudaRuntime::copy_to_device(&zero_i32, converged_flag_ptr, device);

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

    if let Err(e) = result {
        cleanup(client.allocator());
        return Err(e);
    }

    client.synchronize();

    // Clean up converged flag
    client.allocator().deallocate(converged_flag_ptr, flag_size);

    // Read back eigenvalues for sorting
    let eigenvalues_data: Vec<f64> = match dtype {
        DType::F32 => {
            let mut bytes = vec![0u8; n * 4];
            CudaRuntime::copy_from_device(eigenvalues_ptr, &mut bytes, device);
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]) as f64)
                .collect()
        }
        DType::F64 => {
            let mut bytes = vec![0u8; n * 8];
            CudaRuntime::copy_from_device(eigenvalues_ptr, &mut bytes, device);
            bytes
                .chunks_exact(8)
                .map(|c| f64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect()
        }
        _ => unreachable!(),
    };

    // Sort indices by descending magnitude
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| {
        eigenvalues_data[j]
            .abs()
            .partial_cmp(&eigenvalues_data[i].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Reorder and create final tensors
    let (eigenvalues_final, eigenvectors_final) = match dtype {
        DType::F32 => reorder_eig_f32(
            client,
            device,
            eigenvectors_ptr,
            &eigenvalues_data,
            &indices,
            n,
        ),
        DType::F64 => reorder_eig_f64(
            client,
            device,
            eigenvectors_ptr,
            &eigenvalues_data,
            &indices,
            n,
        ),
        _ => unreachable!(),
    };

    // Clean up working buffers
    client.allocator().deallocate(work_ptr, work_size);
    client
        .allocator()
        .deallocate(eigenvectors_ptr, eigenvectors_size);
    client
        .allocator()
        .deallocate(eigenvalues_ptr, eigenvalues_size);

    Ok(EigenDecomposition {
        eigenvalues: eigenvalues_final,
        eigenvectors: eigenvectors_final,
    })
}

fn reorder_eig_f32(
    client: &CudaClient,
    device: &super::super::CudaDevice,
    eigenvectors_ptr: u64,
    eigenvalues_data: &[f64],
    indices: &[usize],
    n: usize,
) -> (Tensor<CudaRuntime>, Tensor<CudaRuntime>) {
    // Read eigenvectors
    let mut v_bytes = vec![0u8; n * n * 4];
    CudaRuntime::copy_from_device(eigenvectors_ptr, &mut v_bytes, device);
    let v_data: Vec<f32> = v_bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Sorted eigenvalues
    let eigenvalues_sorted: Vec<f32> = indices
        .iter()
        .map(|&idx| eigenvalues_data[idx] as f32)
        .collect();

    // Sorted eigenvector columns
    let mut v_sorted = vec![0.0f32; n * n];
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        for i in 0..n {
            v_sorted[i * n + new_idx] = v_data[i * n + old_idx];
        }
    }

    // Allocate final output
    let eigenvalues_size = n * 4;
    let eigenvectors_size = n * n * 4;
    let eigenvalues_ptr_final = client.allocator().allocate(eigenvalues_size);
    let eigenvectors_ptr_final = client.allocator().allocate(eigenvectors_size);

    // Copy sorted results to GPU
    let eigenvalues_bytes: Vec<u8> = eigenvalues_sorted
        .iter()
        .flat_map(|&v| v.to_ne_bytes())
        .collect();
    CudaRuntime::copy_to_device(&eigenvalues_bytes, eigenvalues_ptr_final, device);

    let eigenvectors_bytes: Vec<u8> = v_sorted.iter().flat_map(|&v| v.to_ne_bytes()).collect();
    CudaRuntime::copy_to_device(&eigenvectors_bytes, eigenvectors_ptr_final, device);

    let eigenvalues_final =
        unsafe { CudaClient::tensor_from_raw(eigenvalues_ptr_final, &[n], DType::F32, device) };
    let eigenvectors_final =
        unsafe { CudaClient::tensor_from_raw(eigenvectors_ptr_final, &[n, n], DType::F32, device) };

    (eigenvalues_final, eigenvectors_final)
}

fn reorder_eig_f64(
    client: &CudaClient,
    device: &super::super::CudaDevice,
    eigenvectors_ptr: u64,
    eigenvalues_data: &[f64],
    indices: &[usize],
    n: usize,
) -> (Tensor<CudaRuntime>, Tensor<CudaRuntime>) {
    // Read eigenvectors
    let mut v_bytes = vec![0u8; n * n * 8];
    CudaRuntime::copy_from_device(eigenvectors_ptr, &mut v_bytes, device);
    let v_data: Vec<f64> = v_bytes
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();

    // Sorted eigenvalues
    let eigenvalues_sorted: Vec<f64> = indices.iter().map(|&idx| eigenvalues_data[idx]).collect();

    // Sorted eigenvector columns
    let mut v_sorted = vec![0.0f64; n * n];
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        for i in 0..n {
            v_sorted[i * n + new_idx] = v_data[i * n + old_idx];
        }
    }

    // Allocate final output
    let eigenvalues_size = n * 8;
    let eigenvectors_size = n * n * 8;
    let eigenvalues_ptr_final = client.allocator().allocate(eigenvalues_size);
    let eigenvectors_ptr_final = client.allocator().allocate(eigenvectors_size);

    // Copy sorted results to GPU
    let eigenvalues_bytes: Vec<u8> = eigenvalues_sorted
        .iter()
        .flat_map(|&v| v.to_ne_bytes())
        .collect();
    CudaRuntime::copy_to_device(&eigenvalues_bytes, eigenvalues_ptr_final, device);

    let eigenvectors_bytes: Vec<u8> = v_sorted.iter().flat_map(|&v| v.to_ne_bytes()).collect();
    CudaRuntime::copy_to_device(&eigenvectors_bytes, eigenvectors_ptr_final, device);

    let eigenvalues_final =
        unsafe { CudaClient::tensor_from_raw(eigenvalues_ptr_final, &[n], DType::F64, device) };
    let eigenvectors_final =
        unsafe { CudaClient::tensor_from_raw(eigenvectors_ptr_final, &[n, n], DType::F64, device) };

    (eigenvalues_final, eigenvectors_final)
}
