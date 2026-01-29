//! Matrix decomposition implementations (LU, Cholesky, QR) for CUDA

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use super::super::kernels;
use crate::algorithm::linalg::{
    CholeskyDecomposition, LuDecomposition, QrDecomposition, validate_linalg_dtype,
    validate_matrix_2d, validate_square_matrix,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::{Allocator, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// LU decomposition with partial pivoting
pub fn lu_decompose_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
) -> Result<LuDecomposition<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let k = m.min(n);
    let dtype = a.dtype();
    let device = client.device();

    // Allocate output tensors on GPU
    let lu_size = m * n * dtype.size_in_bytes();
    let lu_ptr = client.allocator().allocate(lu_size);

    let pivots_size = k * std::mem::size_of::<i64>();
    let pivots_ptr = client.allocator().allocate(pivots_size);

    let num_swaps_size = std::mem::size_of::<i32>();
    let num_swaps_ptr = client.allocator().allocate(num_swaps_size);

    let singular_flag_size = std::mem::size_of::<i32>();
    let singular_flag_ptr = client.allocator().allocate(singular_flag_size);

    // Copy input to LU buffer
    CudaRuntime::copy_within_device(a.storage().ptr(), lu_ptr, lu_size, device);

    // Zero-initialize flags
    let zero_i32: [u8; 4] = [0; 4];
    CudaRuntime::copy_to_device(&zero_i32, num_swaps_ptr, device);
    CudaRuntime::copy_to_device(&zero_i32, singular_flag_ptr, device);

    // Launch kernel
    unsafe {
        kernels::launch_lu_decompose(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            lu_ptr,
            pivots_ptr,
            num_swaps_ptr,
            singular_flag_ptr,
            m,
            n,
        )?;
    }

    client.synchronize();

    // Read back flags
    let mut num_swaps_bytes = [0u8; 4];
    let mut singular_flag_bytes = [0u8; 4];
    CudaRuntime::copy_from_device(num_swaps_ptr, &mut num_swaps_bytes, device);
    CudaRuntime::copy_from_device(singular_flag_ptr, &mut singular_flag_bytes, device);

    let num_swaps = i32::from_ne_bytes(num_swaps_bytes) as usize;
    let singular = i32::from_ne_bytes(singular_flag_bytes) != 0;

    // Clean up flag allocations
    client.allocator().deallocate(num_swaps_ptr, num_swaps_size);
    client
        .allocator()
        .deallocate(singular_flag_ptr, singular_flag_size);

    if singular {
        client.allocator().deallocate(lu_ptr, lu_size);
        client.allocator().deallocate(pivots_ptr, pivots_size);
        return Err(Error::Internal(format!(
            "LU decomposition failed: {}x{} matrix is singular (zero pivot encountered)",
            m, n
        )));
    }

    // Create tensors from GPU memory
    let lu = unsafe { CudaClient::tensor_from_raw(lu_ptr, &[m, n], dtype, device) };
    let pivots = unsafe { CudaClient::tensor_from_raw(pivots_ptr, &[k], DType::I64, device) };

    Ok(LuDecomposition {
        lu,
        pivots,
        num_swaps,
    })
}

/// Cholesky decomposition
pub fn cholesky_decompose_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
) -> Result<CholeskyDecomposition<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Allocate output on GPU
    let l_size = n * n * dtype.size_in_bytes();
    let l_ptr = client.allocator().allocate(l_size);

    let not_pd_flag_size = std::mem::size_of::<i32>();
    let not_pd_flag_ptr = client.allocator().allocate(not_pd_flag_size);

    // Copy input to L buffer
    CudaRuntime::copy_within_device(a.storage().ptr(), l_ptr, l_size, device);

    // Zero-initialize flag
    let zero_i32: [u8; 4] = [0; 4];
    CudaRuntime::copy_to_device(&zero_i32, not_pd_flag_ptr, device);

    // Launch kernel
    unsafe {
        kernels::launch_cholesky_decompose(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            l_ptr,
            not_pd_flag_ptr,
            n,
        )?;
    }

    client.synchronize();

    // Read back flag
    let mut not_pd_bytes = [0u8; 4];
    CudaRuntime::copy_from_device(not_pd_flag_ptr, &mut not_pd_bytes, device);
    let not_pd = i32::from_ne_bytes(not_pd_bytes) != 0;

    client
        .allocator()
        .deallocate(not_pd_flag_ptr, not_pd_flag_size);

    if not_pd {
        client.allocator().deallocate(l_ptr, l_size);
        return Err(Error::Internal(format!(
            "Cholesky decomposition failed: {}x{} matrix is not positive definite",
            n, n
        )));
    }

    let l = unsafe { CudaClient::tensor_from_raw(l_ptr, &[n, n], dtype, device) };

    Ok(CholeskyDecomposition { l })
}

/// QR decomposition (internal implementation)
pub fn qr_decompose_internal(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    thin: bool,
) -> Result<QrDecomposition<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let k = m.min(n);
    let dtype = a.dtype();
    let device = client.device();

    // Q dimensions: [m, m] for full, [m, k] for thin
    let q_cols = if thin { k } else { m };
    let q_size = m * q_cols * dtype.size_in_bytes();
    let q_ptr = client.allocator().allocate(q_size);

    // R is [m, n] but only upper triangular part is meaningful
    let r_size = m * n * dtype.size_in_bytes();
    let r_ptr = client.allocator().allocate(r_size);

    // Workspace for Householder vector (size m elements)
    let workspace_size = m * dtype.size_in_bytes();
    let workspace_ptr = client.allocator().allocate(workspace_size);

    // Copy A to R (will be modified in place)
    CudaRuntime::copy_within_device(a.storage().ptr(), r_ptr, r_size, device);

    let result = unsafe {
        kernels::launch_qr_decompose(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            q_ptr,
            r_ptr,
            workspace_ptr,
            m,
            n,
            thin,
        )
    };

    // Clean up workspace (always, regardless of success/failure)
    client.allocator().deallocate(workspace_ptr, workspace_size);

    // Handle kernel error after cleanup
    if let Err(e) = result {
        client.allocator().deallocate(q_ptr, q_size);
        client.allocator().deallocate(r_ptr, r_size);
        return Err(e);
    }

    client.synchronize();

    let q = unsafe { CudaClient::tensor_from_raw(q_ptr, &[m, q_cols], dtype, device) };

    // For thin QR, R should be [k, n]
    let r = if thin && m > n {
        // For thin QR with m > n, R is k x n
        unsafe { CudaClient::tensor_from_raw(r_ptr, &[k, n], dtype, device) }
    } else if thin {
        // For thin QR with m <= n, R is m x n
        unsafe { CudaClient::tensor_from_raw(r_ptr, &[m, n], dtype, device) }
    } else {
        // Full QR, R is m x n
        unsafe { CudaClient::tensor_from_raw(r_ptr, &[m, n], dtype, device) }
    };

    Ok(QrDecomposition { q, r })
}
