//! Solve linear system Ax = b for CUDA

use super::super::super::CudaRuntime;
use super::super::super::client::CudaClient;
use super::super::super::kernels;
use crate::algorithm::linalg::{
    LinearAlgebraAlgorithms, validate_linalg_dtype, validate_square_matrix,
};
use crate::error::{Error, Result};
use crate::runtime::{AllocGuard, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Solve linear system Ax = b
pub fn solve_impl(
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
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Determine if b is vector or matrix
    let b_shape = b.shape();
    let (num_rhs, b_is_vector) = if b_shape.len() == 1 {
        if b_shape[0] != n {
            return Err(Error::ShapeMismatch {
                expected: vec![n],
                got: b_shape.to_vec(),
            });
        }
        (1, true)
    } else if b_shape.len() == 2 {
        if b_shape[0] != n {
            return Err(Error::ShapeMismatch {
                expected: vec![n, b_shape[1]],
                got: b_shape.to_vec(),
            });
        }
        (b_shape[1], false)
    } else {
        return Err(Error::Internal(format!(
            "solve requires b to be 1D or 2D tensor, got {}D tensor with shape {:?}",
            b_shape.len(),
            b_shape
        )));
    };

    // Compute LU decomposition
    let lu_result = client.lu_decompose(a)?;

    // Allocate output and temporary buffers
    let x_size = n * num_rhs * dtype.size_in_bytes();
    let col_size = n * dtype.size_in_bytes();

    let x_guard = AllocGuard::new(client.allocator(), x_size)?;
    let b_col_guard = AllocGuard::new(client.allocator(), col_size)?;
    let pb_guard = AllocGuard::new(client.allocator(), col_size)?;
    let y_guard = AllocGuard::new(client.allocator(), col_size)?;
    let x_col_guard = AllocGuard::new(client.allocator(), col_size)?;

    let x_ptr = x_guard.ptr();
    let b_col_ptr = b_col_guard.ptr();
    let pb_ptr = pb_guard.ptr();
    let y_ptr = y_guard.ptr();
    let x_col_ptr = x_col_guard.ptr();

    // Solve for each right-hand side
    for rhs in 0..num_rhs {
        // Extract column from b (or use b directly if 1D)
        let b_ptr_for_solve = if b_is_vector {
            b.ptr()
        } else {
            // Extract column rhs from B [n, num_rhs]
            let result = unsafe {
                kernels::launch_extract_column(
                    client.context(),
                    client.stream(),
                    device.index,
                    dtype,
                    b.ptr(),
                    b_col_ptr,
                    n,
                    num_rhs,
                    rhs,
                )
            };
            result?;
            b_col_ptr
        };

        // Apply permutation: pb = P @ b_col
        let result = unsafe {
            kernels::launch_apply_lu_permutation(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                b_ptr_for_solve,
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
                x_col_ptr,
                n,
            )
        };
        result?;

        // Scatter solution into X
        if b_is_vector {
            // Single RHS: copy directly to x_ptr
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
    fn test_solve() {
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();

        // Solve [[2, 1], [1, 2]] @ x = [3, 3]
        // Solution: x = [1, 1]
        let a =
            Tensor::<CudaRuntime>::from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2], device).unwrap();
        let b = Tensor::<CudaRuntime>::from_slice(&[3.0f32, 3.0], &[2], device).unwrap();

        let x = LinearAlgebraAlgorithms::solve(&client, &a, &b).unwrap();
        let result: Vec<f32> = x.to_vec();

        assert!((result[0] - 1.0).abs() < 1e-4);
        assert!((result[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_solve_multi_rhs() {
        let Some(client) = create_client() else {
            return;
        };
        let device = client.device();

        // Solve A @ X = B where B has multiple columns
        // A = [[2, 1], [1, 2]], B = [[3, 4], [3, 5]]
        // Solutions: X[:, 0] = [1, 1], X[:, 1] = [1, 2]
        let a =
            Tensor::<CudaRuntime>::from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2], device).unwrap();
        let b =
            Tensor::<CudaRuntime>::from_slice(&[3.0f32, 4.0, 3.0, 5.0], &[2, 2], device).unwrap();

        let x = LinearAlgebraAlgorithms::solve(&client, &a, &b).unwrap();
        assert_eq!(x.shape(), &[2, 2]);
        let result: Vec<f32> = x.to_vec();

        // X[:, 0] = [1, 1] -> result[0], result[2]
        // X[:, 1] = [1, 2] -> result[1], result[3]
        assert!(
            (result[0] - 1.0).abs() < 1e-4,
            "X[0,0] = {} expected 1",
            result[0]
        );
        assert!(
            (result[1] - 1.0).abs() < 1e-4,
            "X[0,1] = {} expected 1",
            result[1]
        );
        assert!(
            (result[2] - 1.0).abs() < 1e-4,
            "X[1,0] = {} expected 1",
            result[2]
        );
        assert!(
            (result[3] - 2.0).abs() < 1e-4,
            "X[1,1] = {} expected 2",
            result[3]
        );
    }
}
