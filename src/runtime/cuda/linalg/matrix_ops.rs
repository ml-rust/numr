//! Matrix operations for CUDA (inverse, det, trace, diag, diagflat, rank, norm)

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use super::super::kernels;
use crate::algorithm::linalg::{
    LinearAlgebraAlgorithms, MatrixNormOrder, validate_linalg_dtype, validate_matrix_2d,
    validate_square_matrix,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ReduceOps, TensorOps, UnaryOps};
use crate::runtime::{Allocator, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Matrix inverse via LU decomposition
pub fn inverse_impl(client: &CudaClient, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Compute LU decomposition
    let lu_result = client.lu_decompose(a)?;

    // Allocate output and temporary buffers
    let inv_size = n * n * dtype.size_in_bytes();
    let inv_ptr = client.allocator().allocate(inv_size);

    let col_size = n * dtype.size_in_bytes();
    let identity_ptr = client.allocator().allocate(inv_size); // Full identity matrix
    let pb_ptr = client.allocator().allocate(col_size);
    let y_ptr = client.allocator().allocate(col_size);
    let x_ptr = client.allocator().allocate(col_size);
    let e_ptr = client.allocator().allocate(col_size);

    // Helper closure for cleanup on error
    let cleanup = |allocator: &super::super::CudaAllocator| {
        allocator.deallocate(inv_ptr, inv_size);
        allocator.deallocate(identity_ptr, inv_size);
        allocator.deallocate(pb_ptr, col_size);
        allocator.deallocate(y_ptr, col_size);
        allocator.deallocate(x_ptr, col_size);
        allocator.deallocate(e_ptr, col_size);
    };

    // Create identity matrix on GPU (no CPU transfer)
    let result = unsafe {
        kernels::launch_create_identity(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            identity_ptr,
            n,
        )
    };
    if let Err(e) = result {
        cleanup(client.allocator());
        return Err(e);
    }

    // Solve for each column of the identity matrix
    for col in 0..n {
        // Extract column from identity matrix (GPU-only)
        let result = unsafe {
            kernels::launch_extract_column(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                identity_ptr,
                e_ptr,
                n,
                n,
                col,
            )
        };
        if let Err(e) = result {
            cleanup(client.allocator());
            return Err(e);
        }

        // Apply permutation: pb = P @ e
        let result = unsafe {
            kernels::launch_apply_lu_permutation(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                e_ptr,
                pb_ptr,
                lu_result.pivots.storage().ptr(),
                n,
            )
        };
        if let Err(e) = result {
            cleanup(client.allocator());
            return Err(e);
        }

        // Forward substitution: Ly = pb (L has unit diagonal)
        let result = unsafe {
            kernels::launch_forward_sub(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                lu_result.lu.storage().ptr(),
                pb_ptr,
                y_ptr,
                n,
                true, // unit diagonal
            )
        };
        if let Err(e) = result {
            cleanup(client.allocator());
            return Err(e);
        }

        // Backward substitution: Ux = y
        let result = unsafe {
            kernels::launch_backward_sub(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                lu_result.lu.storage().ptr(),
                y_ptr,
                x_ptr,
                n,
            )
        };
        if let Err(e) = result {
            cleanup(client.allocator());
            return Err(e);
        }

        // Scatter x into column of inverse matrix (GPU-only, no CPU transfer)
        let result = unsafe {
            kernels::launch_scatter_column(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                x_ptr,
                inv_ptr,
                n,
                col,
            )
        };
        if let Err(e) = result {
            cleanup(client.allocator());
            return Err(e);
        }
    }

    // Clean up temporary buffers (keep inv_ptr for result)
    client.allocator().deallocate(identity_ptr, inv_size);
    client.allocator().deallocate(pb_ptr, col_size);
    client.allocator().deallocate(y_ptr, col_size);
    client.allocator().deallocate(x_ptr, col_size);
    client.allocator().deallocate(e_ptr, col_size);

    client.synchronize();

    let inv = unsafe { CudaClient::tensor_from_raw(inv_ptr, &[n, n], dtype, device) };

    Ok(inv)
}

/// Determinant via LU decomposition
pub fn det_impl(client: &CudaClient, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Compute LU decomposition
    let lu_result = client.lu_decompose(a)?;

    // Allocate output
    let det_size = dtype.size_in_bytes();
    let det_ptr = client.allocator().allocate(det_size);

    // Compute determinant from LU diagonal
    unsafe {
        kernels::launch_det_from_lu(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            lu_result.lu.storage().ptr(),
            det_ptr,
            n,
            lu_result.num_swaps as i32,
        )?;
    }

    client.synchronize();

    let det = unsafe { CudaClient::tensor_from_raw(det_ptr, &[], dtype, device) };

    Ok(det)
}

/// Matrix trace (sum of diagonal elements)
pub fn trace_impl(client: &CudaClient, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let min_dim = m.min(n);
    let dtype = a.dtype();
    let device = client.device();

    // Allocate output (zero-initialized for atomic add)
    let trace_size = dtype.size_in_bytes();
    let trace_ptr = client.allocator().allocate(trace_size);

    let zero_bytes = vec![0u8; trace_size];
    CudaRuntime::copy_to_device(&zero_bytes, trace_ptr, device);

    unsafe {
        kernels::launch_trace(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            a.storage().ptr(),
            trace_ptr,
            min_dim,
            n, // stride (number of columns)
        )?;
    }

    client.synchronize();

    let trace = unsafe { CudaClient::tensor_from_raw(trace_ptr, &[], dtype, device) };

    Ok(trace)
}

/// Extract diagonal elements
pub fn diag_impl(client: &CudaClient, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let min_dim = m.min(n);
    let dtype = a.dtype();
    let device = client.device();

    let diag_size = min_dim * dtype.size_in_bytes();
    let diag_ptr = client.allocator().allocate(diag_size);

    unsafe {
        kernels::launch_diag(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            a.storage().ptr(),
            diag_ptr,
            min_dim,
            n,
        )?;
    }

    client.synchronize();

    let diag = unsafe { CudaClient::tensor_from_raw(diag_ptr, &[min_dim], dtype, device) };

    Ok(diag)
}

/// Create diagonal matrix from vector
pub fn diagflat_impl(client: &CudaClient, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;

    // Input must be 1D
    if a.shape().len() != 1 {
        return Err(Error::Internal(format!(
            "diagflat requires 1D input tensor, got {}D tensor with shape {:?}",
            a.shape().len(),
            a.shape()
        )));
    }

    let n = a.shape()[0];
    let dtype = a.dtype();
    let device = client.device();

    let out_size = n * n * dtype.size_in_bytes();
    let out_ptr = client.allocator().allocate(out_size);

    unsafe {
        kernels::launch_diagflat(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            a.storage().ptr(),
            out_ptr,
            n,
        )?;
    }

    client.synchronize();

    let out = unsafe { CudaClient::tensor_from_raw(out_ptr, &[n, n], dtype, device) };

    Ok(out)
}

/// Matrix rank via QR decomposition
pub fn matrix_rank_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    tol: Option<f64>,
) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();
    let k = m.min(n);

    // Use QR decomposition to estimate rank
    let qr = client.qr_decompose(a)?;

    // Get diagonal of R
    let r_diag = LinearAlgebraAlgorithms::diag(client, &qr.r)?;

    // Allocate GPU buffers for max abs and count
    let max_size = dtype.size_in_bytes();
    let max_ptr = client.allocator().allocate(max_size);
    let count_size = std::mem::size_of::<u32>();
    let count_ptr = client.allocator().allocate(count_size);

    // Zero-initialize max and count on GPU
    let zero_bytes = vec![0u8; max_size.max(count_size)];
    CudaRuntime::copy_to_device(&zero_bytes[..max_size], max_ptr, device);
    CudaRuntime::copy_to_device(&zero_bytes[..count_size], count_ptr, device);

    // Compute max absolute value on GPU (no CPU transfer of full diagonal)
    let result = unsafe {
        kernels::launch_max_abs(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            r_diag.storage().ptr(),
            max_ptr,
            k,
        )
    };
    if let Err(e) = result {
        client.allocator().deallocate(max_ptr, max_size);
        client.allocator().deallocate(count_ptr, count_size);
        return Err(e);
    }

    client.synchronize();

    // Read max value (single element transfer, not entire diagonal)
    let max_diag: f64 = match dtype {
        DType::F32 => {
            let mut max_bytes = [0u8; 4];
            CudaRuntime::copy_from_device(max_ptr, &mut max_bytes, device);
            f32::from_ne_bytes(max_bytes) as f64
        }
        DType::F64 => {
            let mut max_bytes = [0u8; 8];
            CudaRuntime::copy_from_device(max_ptr, &mut max_bytes, device);
            f64::from_ne_bytes(max_bytes)
        }
        _ => unreachable!(),
    };

    // Compute tolerance
    let base_tol = tol.unwrap_or_else(|| {
        let eps = match dtype {
            DType::F32 => f32::EPSILON as f64,
            DType::F64 => f64::EPSILON,
            _ => f32::EPSILON as f64,
        };
        (m.max(n) as f64) * eps
    });
    let threshold = base_tol * max_diag;

    // Count elements above threshold on GPU
    let result = unsafe {
        kernels::launch_count_above_threshold(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            r_diag.storage().ptr(),
            count_ptr,
            k,
            threshold,
        )
    };
    if let Err(e) = result {
        client.allocator().deallocate(max_ptr, max_size);
        client.allocator().deallocate(count_ptr, count_size);
        return Err(e);
    }

    client.synchronize();

    // Read count (single u32 transfer)
    let mut count_bytes = [0u8; 4];
    CudaRuntime::copy_from_device(count_ptr, &mut count_bytes, device);
    let rank = u32::from_ne_bytes(count_bytes) as i64;

    // Clean up
    client.allocator().deallocate(max_ptr, max_size);
    client.allocator().deallocate(count_ptr, count_size);

    // Create rank tensor on GPU
    let rank_data: [i64; 1] = [rank];
    let rank_tensor = Tensor::<CudaRuntime>::from_slice(&rank_data, &[], device);

    Ok(rank_tensor)
}

/// Matrix norm (Frobenius only for now)
pub fn matrix_norm_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    ord: MatrixNormOrder,
) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (_m, _n) = validate_matrix_2d(a.shape())?;

    match ord {
        MatrixNormOrder::Frobenius => {
            // Frobenius norm: ||A||_F = sqrt(sum(A²))
            // Use existing tensor ops to keep data on GPU
            let squared = client.square(a)?;
            let sum_sq = client.sum(&squared, &[], false)?;
            client.sqrt(&sum_sq)
        }
        MatrixNormOrder::Spectral | MatrixNormOrder::Nuclear => Err(Error::Internal(
            "Spectral and nuclear norms require SVD (not yet implemented)".to_string(),
        )),
    }
}

/// Kronecker product: A ⊗ B
pub fn kron_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }

    let (m_a, n_a) = validate_matrix_2d(a.shape())?;
    let (m_b, n_b) = validate_matrix_2d(b.shape())?;

    let dtype = a.dtype();
    let device = client.device();

    let m_out = m_a * m_b;
    let n_out = n_a * n_b;
    let out_size = m_out * n_out * dtype.size_in_bytes();
    let out_ptr = client.allocator().allocate(out_size);

    unsafe {
        kernels::launch_kron(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            a.storage().ptr(),
            b.storage().ptr(),
            out_ptr,
            m_a,
            n_a,
            m_b,
            n_b,
        )?;
    }

    client.synchronize();

    let out = unsafe { CudaClient::tensor_from_raw(out_ptr, &[m_out, n_out], dtype, device) };

    Ok(out)
}
