//! Matrix inverse, determinant, trace, diagonal, and diagflat for CUDA

use super::super::super::CudaRuntime;
use super::super::super::client::CudaClient;
use super::super::super::kernels;
use crate::algorithm::linalg::{
    LinearAlgebraAlgorithms, validate_linalg_dtype, validate_matrix_2d, validate_square_matrix,
};
use crate::error::{Error, Result};
use crate::runtime::{AllocGuard, Runtime, RuntimeClient};
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
    let col_size = n * dtype.size_in_bytes();

    let inv_guard = AllocGuard::new(client.allocator(), inv_size)?;
    let identity_guard = AllocGuard::new(client.allocator(), inv_size)?;
    let pb_guard = AllocGuard::new(client.allocator(), col_size)?;
    let y_guard = AllocGuard::new(client.allocator(), col_size)?;
    let x_guard = AllocGuard::new(client.allocator(), col_size)?;
    let e_guard = AllocGuard::new(client.allocator(), col_size)?;

    let inv_ptr = inv_guard.ptr();
    let identity_ptr = identity_guard.ptr();
    let pb_ptr = pb_guard.ptr();
    let y_ptr = y_guard.ptr();
    let x_ptr = x_guard.ptr();
    let e_ptr = e_guard.ptr();

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
    result?;

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
        result?;

        // Apply permutation: pb = P @ e
        let result = unsafe {
            kernels::launch_apply_lu_permutation(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                e_ptr,
                pb_ptr,
                lu_result.pivots.ptr(),
                n,
            )
        };
        result?;

        // Forward substitution: Ly = pb (L has unit diagonal)
        let result = unsafe {
            kernels::launch_forward_sub(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                lu_result.lu.ptr(),
                pb_ptr,
                y_ptr,
                n,
                true, // unit diagonal
            )
        };
        result?;

        // Backward substitution: Ux = y
        let result = unsafe {
            kernels::launch_backward_sub(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                lu_result.lu.ptr(),
                y_ptr,
                x_ptr,
                n,
            )
        };
        result?;

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
        result?
    }

    client.synchronize();

    let inv = unsafe { CudaClient::tensor_from_raw(inv_guard.release(), &[n, n], dtype, device) };

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
    let det_guard = AllocGuard::new(client.allocator(), det_size)?;
    let det_ptr = det_guard.ptr();

    // Compute determinant from LU diagonal
    unsafe {
        kernels::launch_det_from_lu(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            lu_result.lu.ptr(),
            det_ptr,
            n,
            lu_result.num_swaps as i32,
        )?;
    }

    client.synchronize();

    let det = unsafe { CudaClient::tensor_from_raw(det_guard.release(), &[], dtype, device) };

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
    let trace_guard = AllocGuard::new(client.allocator(), trace_size)?;
    let trace_ptr = trace_guard.ptr();

    let zero_bytes = vec![0u8; trace_size];
    CudaRuntime::copy_to_device(&zero_bytes, trace_ptr, device)?;

    unsafe {
        kernels::launch_trace(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            a.ptr(),
            trace_ptr,
            min_dim,
            n, // stride (number of columns)
        )?;
    }

    client.synchronize();

    let trace = unsafe { CudaClient::tensor_from_raw(trace_guard.release(), &[], dtype, device) };

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
    let diag_guard = AllocGuard::new(client.allocator(), diag_size)?;
    let diag_ptr = diag_guard.ptr();

    unsafe {
        kernels::launch_diag(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            a.ptr(),
            diag_ptr,
            min_dim,
            n,
        )?;
    }

    client.synchronize();

    let diag =
        unsafe { CudaClient::tensor_from_raw(diag_guard.release(), &[min_dim], dtype, device) };

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
    let out_guard = AllocGuard::new(client.allocator(), out_size)?;
    let out_ptr = out_guard.ptr();

    unsafe {
        kernels::launch_diagflat(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            a.ptr(),
            out_ptr,
            n,
        )?;
    }

    client.synchronize();

    let out = unsafe { CudaClient::tensor_from_raw(out_guard.release(), &[n, n], dtype, device) };

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::*;
    use super::*;
    use crate::ops::MatmulOps;

    #[test]
    fn test_trace() {
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();

        // 2x2 matrix: [[1, 2], [3, 4]]
        // trace = 1 + 4 = 5
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

        let t = LinearAlgebraAlgorithms::trace(&client, &a).unwrap();
        let result: Vec<f32> = t.to_vec();

        assert!((result[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_diag() {
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();

        // 2x3 matrix
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device)
                .unwrap();

        let d = LinearAlgebraAlgorithms::diag(&client, &a).unwrap();
        let result: Vec<f32> = d.to_vec();

        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_diagflat() {
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();

        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], device).unwrap();

        let m = LinearAlgebraAlgorithms::diagflat(&client, &a).unwrap();
        let result: Vec<f32> = m.to_vec();

        assert_eq!(m.shape(), &[3, 3]);
        // Expected: [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        assert!((result[0] - 1.0).abs() < 1e-5); // [0,0]
        assert!((result[1]).abs() < 1e-5); // [0,1]
        assert!((result[4] - 2.0).abs() < 1e-5); // [1,1]
        assert!((result[8] - 3.0).abs() < 1e-5); // [2,2]
    }

    #[test]
    fn test_det() {
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();

        // 2x2 matrix: [[1, 2], [3, 4]]
        // det = 1*4 - 2*3 = -2
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

        let d = LinearAlgebraAlgorithms::det(&client, &a).unwrap();
        let result: Vec<f32> = d.to_vec();

        assert!((result[0] - (-2.0)).abs() < 1e-4);
    }

    #[test]
    fn test_inverse() {
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();

        // Test 2x2 matrix: [[4, 7], [2, 6]]
        // Inverse: [[0.6, -0.7], [-0.2, 0.4]]
        let a =
            Tensor::<CudaRuntime>::from_slice(&[4.0f32, 7.0, 2.0, 6.0], &[2, 2], device).unwrap();

        let inv = LinearAlgebraAlgorithms::inverse(&client, &a).unwrap();
        let result: Vec<f32> = inv.to_vec();

        // Check inverse values (det = 4*6 - 7*2 = 10)
        // inv = (1/10) * [[6, -7], [-2, 4]]
        assert!((result[0] - 0.6).abs() < 1e-4); // [0,0]
        assert!((result[1] - (-0.7)).abs() < 1e-4); // [0,1]
        assert!((result[2] - (-0.2)).abs() < 1e-4); // [1,0]
        assert!((result[3] - 0.4).abs() < 1e-4); // [1,1]
    }

    #[test]
    fn test_inverse_identity() {
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();

        // A @ A^-1 should equal I
        let a =
            Tensor::<CudaRuntime>::from_slice(&[4.0f32, 7.0, 2.0, 6.0], &[2, 2], device).unwrap();

        let inv = LinearAlgebraAlgorithms::inverse(&client, &a).unwrap();
        let product = client.matmul(&a, &inv).unwrap();
        let result: Vec<f32> = product.to_vec();

        // Should be identity matrix
        assert!((result[0] - 1.0).abs() < 1e-4); // [0,0]
        assert!((result[1]).abs() < 1e-4); // [0,1]
        assert!((result[2]).abs() < 1e-4); // [1,0]
        assert!((result[3] - 1.0).abs() < 1e-4); // [1,1]
    }
}
