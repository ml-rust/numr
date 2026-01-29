//! SVD decomposition for CUDA

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use super::super::kernels;
use crate::algorithm::linalg::{SvdDecomposition, validate_linalg_dtype, validate_matrix_2d};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::{Allocator, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// SVD decomposition via Jacobi method
pub fn svd_decompose_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
) -> Result<SvdDecomposition<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let k = m.min(n);
    let dtype = a.dtype();
    let device = client.device();

    // Handle empty matrix
    if m == 0 || n == 0 {
        let u_ptr = client.allocator().allocate(0);
        let s_ptr = client.allocator().allocate(0);
        let vt_ptr = client.allocator().allocate(0);
        let u = unsafe { CudaClient::tensor_from_raw(u_ptr, &[m, k], dtype, device) };
        let s = unsafe { CudaClient::tensor_from_raw(s_ptr, &[k], dtype, device) };
        let vt = unsafe { CudaClient::tensor_from_raw(vt_ptr, &[k, n], dtype, device) };
        return Ok(SvdDecomposition { u, s, vt });
    }

    // If m < n, transpose and swap U/V at the end
    let transpose = m < n;
    let (work_m, work_n) = if transpose { (n, m) } else { (m, n) };
    let work_k = work_m.min(work_n);

    // Allocate working buffers on GPU
    let b_size = work_m * work_n * dtype.size_in_bytes();
    let b_ptr = client.allocator().allocate(b_size);

    let v_size = work_n * work_n * dtype.size_in_bytes();
    let v_ptr = client.allocator().allocate(v_size);

    let s_size = work_n * dtype.size_in_bytes();
    let s_ptr = client.allocator().allocate(s_size);

    let flag_size = std::mem::size_of::<i32>();
    let converged_flag_ptr = client.allocator().allocate(flag_size);

    // Helper for cleanup on error
    let cleanup = |allocator: &super::super::CudaAllocator| {
        allocator.deallocate(b_ptr, b_size);
        allocator.deallocate(v_ptr, v_size);
        allocator.deallocate(s_ptr, s_size);
        allocator.deallocate(converged_flag_ptr, flag_size);
    };

    // Copy input to B, transposing if needed using GPU transpose kernel
    if transpose {
        // Use optimized GPU transpose: A[m,n] -> B[n,m]
        let result = unsafe {
            kernels::launch_transpose(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                a.storage().ptr(),
                b_ptr,
                m, // rows of input
                n, // cols of input
            )
        };
        if let Err(e) = result {
            cleanup(client.allocator());
            return Err(e);
        }
    } else {
        CudaRuntime::copy_within_device(a.storage().ptr(), b_ptr, b_size, device);
    }

    // Zero-initialize converged flag
    let zero_i32: [u8; 4] = [0; 4];
    CudaRuntime::copy_to_device(&zero_i32, converged_flag_ptr, device);

    // Launch SVD Jacobi kernel
    let result = unsafe {
        kernels::launch_svd_jacobi(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            b_ptr,
            v_ptr,
            s_ptr,
            converged_flag_ptr,
            work_m,
            work_n,
        )
    };

    if let Err(e) = result {
        cleanup(client.allocator());
        return Err(e);
    }

    client.synchronize();

    // Clean up converged flag
    client.allocator().deallocate(converged_flag_ptr, flag_size);

    // Read back singular values and indices for sorting
    let s_data: Vec<f64> = match dtype {
        DType::F32 => {
            let mut bytes = vec![0u8; work_n * 4];
            CudaRuntime::copy_from_device(s_ptr, &mut bytes, device);
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]) as f64)
                .collect()
        }
        DType::F64 => {
            let mut bytes = vec![0u8; work_n * 8];
            CudaRuntime::copy_from_device(s_ptr, &mut bytes, device);
            bytes
                .chunks_exact(8)
                .map(|c| f64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect()
        }
        _ => unreachable!(),
    };

    // Sort indices by descending singular value
    let mut indices: Vec<usize> = (0..work_n).collect();
    indices.sort_by(|&i, &j| {
        s_data[j]
            .partial_cmp(&s_data[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Reorder and create final tensors
    let (u_final, s_final, vt_final) = match dtype {
        DType::F32 => reorder_svd_f32(
            client, device, b_ptr, v_ptr, &s_data, &indices, m, n, k, work_m, work_n, work_k,
            transpose,
        ),
        DType::F64 => reorder_svd_f64(
            client, device, b_ptr, v_ptr, &s_data, &indices, m, n, k, work_m, work_n, work_k,
            transpose,
        ),
        _ => {
            client.allocator().deallocate(b_ptr, b_size);
            client.allocator().deallocate(v_ptr, v_size);
            client.allocator().deallocate(s_ptr, s_size);
            return Err(Error::UnsupportedDType {
                dtype,
                op: "svd_decompose",
            });
        }
    };

    // Clean up working buffers
    client.allocator().deallocate(b_ptr, b_size);
    client.allocator().deallocate(v_ptr, v_size);
    client.allocator().deallocate(s_ptr, s_size);

    Ok(SvdDecomposition {
        u: u_final,
        s: s_final,
        vt: vt_final,
    })
}

fn reorder_svd_f32(
    _client: &CudaClient,
    device: &super::super::CudaDevice,
    b_ptr: u64,
    v_ptr: u64,
    s_data: &[f64],
    indices: &[usize],
    m: usize,
    n: usize,
    k: usize,
    work_m: usize,
    work_n: usize,
    work_k: usize,
    transpose: bool,
) -> (
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
) {
    // Read B (normalized U columns)
    let mut u_bytes = vec![0u8; work_m * work_n * 4];
    CudaRuntime::copy_from_device(b_ptr, &mut u_bytes, device);
    let u_data: Vec<f32> = u_bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Read V
    let mut v_bytes = vec![0u8; work_n * work_n * 4];
    CudaRuntime::copy_from_device(v_ptr, &mut v_bytes, device);
    let v_data: Vec<f32> = v_bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Sorted singular values (take first work_k)
    let s_sorted: Vec<f32> = indices
        .iter()
        .take(work_k)
        .map(|&idx| s_data[idx] as f32)
        .collect();

    // Sorted U columns
    let mut u_sorted = vec![0.0f32; work_m * work_k];
    for (new_idx, &old_idx) in indices.iter().take(work_k).enumerate() {
        for i in 0..work_m {
            u_sorted[i * work_k + new_idx] = u_data[i * work_n + old_idx];
        }
    }

    // Sorted V^T rows (V columns transposed)
    let mut vt_sorted = vec![0.0f32; work_k * work_n];
    for (new_idx, &old_idx) in indices.iter().take(work_k).enumerate() {
        for j in 0..work_n {
            vt_sorted[new_idx * work_n + j] = v_data[j * work_n + old_idx];
        }
    }

    if transpose {
        let mut u_final_data = vec![0.0f32; m * k];
        for i in 0..k {
            for j in 0..m {
                u_final_data[j * k + i] = vt_sorted[i * work_n + j];
            }
        }

        let mut vt_final_data = vec![0.0f32; k * n];
        for i in 0..work_m {
            for j in 0..work_k {
                vt_final_data[j * n + i] = u_sorted[i * work_k + j];
            }
        }

        (
            Tensor::<CudaRuntime>::from_slice(&u_final_data, &[m, k], device),
            Tensor::<CudaRuntime>::from_slice(&s_sorted, &[k], device),
            Tensor::<CudaRuntime>::from_slice(&vt_final_data, &[k, n], device),
        )
    } else {
        (
            Tensor::<CudaRuntime>::from_slice(&u_sorted, &[m, k], device),
            Tensor::<CudaRuntime>::from_slice(&s_sorted, &[k], device),
            Tensor::<CudaRuntime>::from_slice(&vt_sorted, &[k, n], device),
        )
    }
}

fn reorder_svd_f64(
    _client: &CudaClient,
    device: &super::super::CudaDevice,
    b_ptr: u64,
    v_ptr: u64,
    s_data: &[f64],
    indices: &[usize],
    m: usize,
    n: usize,
    k: usize,
    work_m: usize,
    work_n: usize,
    work_k: usize,
    transpose: bool,
) -> (
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
) {
    // Read B (normalized U columns)
    let mut u_bytes = vec![0u8; work_m * work_n * 8];
    CudaRuntime::copy_from_device(b_ptr, &mut u_bytes, device);
    let u_data: Vec<f64> = u_bytes
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();

    // Read V
    let mut v_bytes = vec![0u8; work_n * work_n * 8];
    CudaRuntime::copy_from_device(v_ptr, &mut v_bytes, device);
    let v_data: Vec<f64> = v_bytes
        .chunks_exact(8)
        .map(|c| f64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();

    // Sorted singular values
    let s_sorted: Vec<f64> = indices
        .iter()
        .take(work_k)
        .map(|&idx| s_data[idx])
        .collect();

    // Sorted U columns
    let mut u_sorted = vec![0.0f64; work_m * work_k];
    for (new_idx, &old_idx) in indices.iter().take(work_k).enumerate() {
        for i in 0..work_m {
            u_sorted[i * work_k + new_idx] = u_data[i * work_n + old_idx];
        }
    }

    // Sorted V^T rows
    let mut vt_sorted = vec![0.0f64; work_k * work_n];
    for (new_idx, &old_idx) in indices.iter().take(work_k).enumerate() {
        for j in 0..work_n {
            vt_sorted[new_idx * work_n + j] = v_data[j * work_n + old_idx];
        }
    }

    if transpose {
        let mut u_final_data = vec![0.0f64; m * k];
        for i in 0..k {
            for j in 0..m {
                u_final_data[j * k + i] = vt_sorted[i * work_n + j];
            }
        }

        let mut vt_final_data = vec![0.0f64; k * n];
        for i in 0..work_m {
            for j in 0..work_k {
                vt_final_data[j * n + i] = u_sorted[i * work_k + j];
            }
        }

        (
            Tensor::<CudaRuntime>::from_slice(&u_final_data, &[m, k], device),
            Tensor::<CudaRuntime>::from_slice(&s_sorted, &[k], device),
            Tensor::<CudaRuntime>::from_slice(&vt_final_data, &[k, n], device),
        )
    } else {
        (
            Tensor::<CudaRuntime>::from_slice(&u_sorted, &[m, k], device),
            Tensor::<CudaRuntime>::from_slice(&s_sorted, &[k], device),
            Tensor::<CudaRuntime>::from_slice(&vt_sorted, &[k, n], device),
        )
    }
}
