//! Least squares solver for CUDA

use super::super::super::CudaRuntime;
use super::super::super::client::CudaClient;
use super::super::super::kernels;
use crate::algorithm::linalg::{
    LinearAlgebraAlgorithms, validate_linalg_dtype, validate_matrix_2d,
};
use crate::error::{Error, Result};
use crate::ops::MatmulOps;
use crate::runtime::{AllocGuard, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Least squares solution
pub fn lstsq_impl(
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

    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Get b dimensions
    let b_shape = b.shape();
    let (num_rhs, b_is_vector) = if b_shape.len() == 1 {
        (1, true)
    } else if b_shape.len() == 2 {
        (b_shape[1], false)
    } else {
        return Err(Error::Internal(format!(
            "lstsq requires b to be 1D or 2D tensor, got {}D tensor with shape {:?}",
            b_shape.len(),
            b_shape
        )));
    };

    // Underdetermined systems not supported yet
    if m < n {
        return Err(Error::Internal(format!(
            "lstsq: underdetermined system not yet implemented for CUDA (A is {}x{}, requires m >= n)",
            m, n
        )));
    }

    // QR decomposition
    let qr = client.qr_decompose(a)?;

    // Compute Q^T @ b using TensorOps::matmul
    // Q^T is [m, m], need to transpose Q
    let q_t = qr.q.transpose(0, 1)?;
    let b_mat = if b_is_vector {
        b.reshape(&[m, 1])?
    } else {
        b.clone()
    };

    // Q^T @ B gives [m, num_rhs]
    let qtb = client.matmul(&q_t, &b_mat)?;

    // Allocate output X [n, num_rhs] or [n] for vector
    let x_size = n * num_rhs * dtype.size_in_bytes();
    let col_size = n * dtype.size_in_bytes();

    let x_guard = AllocGuard::new(client.allocator(), x_size)?;
    let qtb_col_guard = AllocGuard::new(client.allocator(), col_size)?;
    let x_col_guard = AllocGuard::new(client.allocator(), col_size)?;

    let x_ptr = x_guard.ptr();
    let qtb_col_ptr = qtb_col_guard.ptr();
    let x_col_ptr = x_col_guard.ptr();

    // Zero initialize X
    let zero_bytes = vec![0u8; x_size];
    CudaRuntime::copy_to_device(&zero_bytes, x_ptr, device)?;

    // Solve R @ X[:, col] = (Q^T @ B)[:n, col] for each column
    // R is [m, n], upper triangular - we use top n×n block
    for rhs in 0..num_rhs {
        // Extract column from Q^T @ B (need first n elements of column rhs)
        // qtb is [m, num_rhs], we extract column rhs, first n elements
        if num_rhs == 1 {
            // Single RHS: qtb is already [m, 1], just use first n elements
            // Copy first n elements to qtb_col_ptr
            CudaRuntime::copy_within_device(qtb.ptr(), qtb_col_ptr, col_size, device)?;
        } else {
            // Multi-RHS: extract column rhs from qtb [m, num_rhs]
            // But we only need first n elements
            let result = unsafe {
                kernels::launch_extract_column(
                    client.context(),
                    client.stream(),
                    device.index,
                    dtype,
                    qtb.ptr(),
                    qtb_col_ptr,
                    n, // only extract first n elements
                    num_rhs,
                    rhs,
                )
            };
            result?
        }

        // Backward substitution: R @ x = qtb_col
        // R is [m, n] but stored as full matrix, n×n upper triangular part is valid
        let result = unsafe {
            kernels::launch_backward_sub(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                qr.r.ptr(),
                qtb_col_ptr,
                x_col_ptr,
                n,
            )
        };
        result?;

        // Scatter solution into X
        if num_rhs == 1 {
            // Single RHS: copy directly
            CudaRuntime::copy_within_device(x_col_ptr, x_ptr, col_size, device)?;
        } else {
            // Multi-RHS: scatter into column rhs of X [n, num_rhs]
            let result = unsafe {
                kernels::launch_scatter_column(
                    client.context(),
                    client.stream(),
                    device.index,
                    dtype,
                    x_col_ptr,
                    x_ptr,
                    n,
                    rhs,
                )
            };
            result?
        }
    }

    client.synchronize();

    let released_ptr = x_guard.release();
    let x = if b_is_vector {
        unsafe { CudaClient::tensor_from_raw(released_ptr, &[n], dtype, device) }
    } else {
        unsafe { CudaClient::tensor_from_raw(released_ptr, &[n, num_rhs], dtype, device) }
    };

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support::*;
    use super::*;

    #[test]
    fn test_lstsq_overdetermined() {
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();

        // Overdetermined system: A is 3x2, b is 3x1
        // A = [[1, 1], [1, 2], [1, 3]], b = [1, 2, 3]
        // Least squares solution minimizes ||Ax - b||^2
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 2.0, 1.0, 3.0], &[3, 2], device)
                .unwrap();
        let b = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], device).unwrap();

        let x = LinearAlgebraAlgorithms::lstsq(&client, &a, &b).unwrap();
        assert_eq!(x.shape(), &[2]);
        let result: Vec<f32> = x.to_vec();

        // For this system, the solution is approximately x = [0, 1]
        assert!((result[0]).abs() < 0.1, "x[0] = {} expected ~0", result[0]);
        assert!(
            (result[1] - 1.0).abs() < 0.1,
            "x[1] = {} expected ~1",
            result[1]
        );
    }

    #[test]
    fn test_lstsq_multi_rhs() {
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();

        // Overdetermined system with multiple RHS
        // A is 3x2, B is 3x2
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 2.0, 1.0, 3.0], &[3, 2], device)
                .unwrap();
        // B = [[1, 2], [2, 4], [3, 6]] (second column is 2x first)
        let b =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 4.0, 3.0, 6.0], &[3, 2], device)
                .unwrap();

        let x = LinearAlgebraAlgorithms::lstsq(&client, &a, &b).unwrap();
        assert_eq!(x.shape(), &[2, 2]);
        let result: Vec<f32> = x.to_vec();

        // Second solution should be 2x the first
        // X[:, 0] ≈ [0, 1], X[:, 1] ≈ [0, 2]
        assert!(
            (result[0]).abs() < 0.1,
            "X[0,0] = {} expected ~0",
            result[0]
        );
        assert!(
            (result[1]).abs() < 0.1,
            "X[0,1] = {} expected ~0",
            result[1]
        );
        assert!(
            (result[2] - 1.0).abs() < 0.1,
            "X[1,0] = {} expected ~1",
            result[2]
        );
        assert!(
            (result[3] - 2.0).abs() < 0.1,
            "X[1,1] = {} expected ~2",
            result[3]
        );
    }
}
