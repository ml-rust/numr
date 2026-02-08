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
use crate::runtime::{AllocGuard, Runtime, RuntimeClient};
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
    let pivots_size = k * std::mem::size_of::<i64>();
    let num_swaps_size = std::mem::size_of::<i32>();
    let singular_flag_size = std::mem::size_of::<i32>();

    let lu_guard = AllocGuard::new(client.allocator(), lu_size)?;
    let pivots_guard = AllocGuard::new(client.allocator(), pivots_size)?;
    let num_swaps_guard = AllocGuard::new(client.allocator(), num_swaps_size)?;
    let singular_flag_guard = AllocGuard::new(client.allocator(), singular_flag_size)?;

    let lu_ptr = lu_guard.ptr();
    let pivots_ptr = pivots_guard.ptr();
    let num_swaps_ptr = num_swaps_guard.ptr();
    let singular_flag_ptr = singular_flag_guard.ptr();

    // Copy input to LU buffer
    CudaRuntime::copy_within_device(a.storage().ptr(), lu_ptr, lu_size, device)?;

    // Zero-initialize flags
    let zero_i32: [u8; 4] = [0; 4];
    CudaRuntime::copy_to_device(&zero_i32, num_swaps_ptr, device)?;
    CudaRuntime::copy_to_device(&zero_i32, singular_flag_ptr, device)?;

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
    CudaRuntime::copy_from_device(num_swaps_ptr, &mut num_swaps_bytes, device)?;
    CudaRuntime::copy_from_device(singular_flag_ptr, &mut singular_flag_bytes, device)?;

    let num_swaps = i32::from_ne_bytes(num_swaps_bytes) as usize;
    let singular = i32::from_ne_bytes(singular_flag_bytes) != 0;

    if singular {
        return Err(Error::Internal(format!(
            "LU decomposition failed: {}x{} matrix is singular (zero pivot encountered)",
            m, n
        )));
    }

    // Create tensors from GPU memory
    let lu = unsafe { CudaClient::tensor_from_raw(lu_guard.release(), &[m, n], dtype, device) };
    let pivots =
        unsafe { CudaClient::tensor_from_raw(pivots_guard.release(), &[k], DType::I64, device) };

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
    let not_pd_flag_size = std::mem::size_of::<i32>();

    let l_guard = AllocGuard::new(client.allocator(), l_size)?;
    let not_pd_flag_guard = AllocGuard::new(client.allocator(), not_pd_flag_size)?;

    let l_ptr = l_guard.ptr();
    let not_pd_flag_ptr = not_pd_flag_guard.ptr();

    // Copy input to L buffer
    CudaRuntime::copy_within_device(a.storage().ptr(), l_ptr, l_size, device)?;

    // Zero-initialize flag
    let zero_i32: [u8; 4] = [0; 4];
    CudaRuntime::copy_to_device(&zero_i32, not_pd_flag_ptr, device)?;

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
    CudaRuntime::copy_from_device(not_pd_flag_ptr, &mut not_pd_bytes, device)?;
    let not_pd = i32::from_ne_bytes(not_pd_bytes) != 0;

    if not_pd {
        return Err(Error::Internal(format!(
            "Cholesky decomposition failed: {}x{} matrix is not positive definite",
            n, n
        )));
    }

    let l = unsafe { CudaClient::tensor_from_raw(l_guard.release(), &[n, n], dtype, device) };

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
    let r_size = m * n * dtype.size_in_bytes();
    let workspace_size = m * dtype.size_in_bytes();

    let q_guard = AllocGuard::new(client.allocator(), q_size)?;
    let r_guard = AllocGuard::new(client.allocator(), r_size)?;
    let workspace_guard = AllocGuard::new(client.allocator(), workspace_size)?;

    let q_ptr = q_guard.ptr();
    let r_ptr = r_guard.ptr();
    let workspace_ptr = workspace_guard.ptr();

    // Copy A to R (will be modified in place)
    CudaRuntime::copy_within_device(a.storage().ptr(), r_ptr, r_size, device)?;

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

    // Handle kernel error
    result?;

    client.synchronize();

    let q = unsafe { CudaClient::tensor_from_raw(q_guard.release(), &[m, q_cols], dtype, device) };

    // For thin QR, R should be [k, n]
    let r = if thin && m > n {
        // For thin QR with m > n, R is k x n
        unsafe { CudaClient::tensor_from_raw(r_guard.release(), &[k, n], dtype, device) }
    } else if thin {
        // For thin QR with m <= n, R is m x n
        unsafe { CudaClient::tensor_from_raw(r_guard.release(), &[m, n], dtype, device) }
    } else {
        // Full QR, R is m x n
        unsafe { CudaClient::tensor_from_raw(r_guard.release(), &[m, n], dtype, device) }
    };

    Ok(QrDecomposition { q, r })
}
