//! Linear system solvers for CUDA

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use super::super::kernels;
use crate::algorithm::linalg::{
    LinearAlgebraAlgorithms, validate_linalg_dtype, validate_matrix_2d, validate_square_matrix,
};
use crate::error::{Error, Result};
use crate::ops::MatmulOps;
use crate::runtime::{Allocator, Runtime, RuntimeClient};
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
    let x_ptr = client.allocator().allocate(x_size);

    let col_size = n * dtype.size_in_bytes();
    let b_col_ptr = client.allocator().allocate(col_size);
    let pb_ptr = client.allocator().allocate(col_size);
    let y_ptr = client.allocator().allocate(col_size);
    let x_col_ptr = client.allocator().allocate(col_size);

    // Helper for cleanup on error
    let cleanup = |allocator: &super::super::CudaAllocator| {
        allocator.deallocate(x_ptr, x_size);
        allocator.deallocate(b_col_ptr, col_size);
        allocator.deallocate(pb_ptr, col_size);
        allocator.deallocate(y_ptr, col_size);
        allocator.deallocate(x_col_ptr, col_size);
    };

    // Solve for each right-hand side
    for rhs in 0..num_rhs {
        // Extract column from b (or use b directly if 1D)
        let b_ptr_for_solve = if b_is_vector {
            b.storage().ptr()
        } else {
            // Extract column rhs from B [n, num_rhs]
            let result = unsafe {
                kernels::launch_extract_column(
                    client.context(),
                    client.stream(),
                    device.index,
                    dtype,
                    b.storage().ptr(),
                    b_col_ptr,
                    n,
                    num_rhs,
                    rhs,
                )
            };
            if let Err(e) = result {
                cleanup(client.allocator());
                return Err(e);
            }
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
                x_col_ptr,
                n,
            )
        };
        if let Err(e) = result {
            cleanup(client.allocator());
            return Err(e);
        }

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
            if let Err(e) = result {
                cleanup(client.allocator());
                return Err(e);
            }
        }
    }

    // Clean up temporary buffers (keep x_ptr for result)
    client.allocator().deallocate(b_col_ptr, col_size);
    client.allocator().deallocate(pb_ptr, col_size);
    client.allocator().deallocate(y_ptr, col_size);
    client.allocator().deallocate(x_col_ptr, col_size);

    client.synchronize();

    let x = if b_is_vector {
        unsafe { CudaClient::tensor_from_raw(x_ptr, &[n], dtype, device) }
    } else {
        unsafe { CudaClient::tensor_from_raw(x_ptr, &[n, num_rhs], dtype, device) }
    };

    Ok(x)
}

/// Solve lower triangular system Lx = b
/// Supports both single RHS (b is 1D vector) and multi-RHS (b is 2D matrix [n, nrhs])
pub fn solve_triangular_lower_impl(
    client: &CudaClient,
    l: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    unit_diagonal: bool,
) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(l.dtype())?;
    if l.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: l.dtype(),
            rhs: b.dtype(),
        });
    }
    let n = validate_square_matrix(l.shape())?;
    let dtype = l.dtype();
    let device = client.device();

    // Handle 1D or 2D b
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
            "solve_triangular_lower requires b to be 1D or 2D, got {}D with shape {:?}",
            b_shape.len(),
            b_shape
        )));
    };

    // Allocate output
    let x_size = n * num_rhs * dtype.size_in_bytes();
    let x_ptr = client.allocator().allocate(x_size);

    if b_is_vector {
        // Single RHS: direct solve
        unsafe {
            kernels::launch_forward_sub(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                l.storage().ptr(),
                b.storage().ptr(),
                x_ptr,
                n,
                unit_diagonal,
            )?;
        }
    } else {
        // Multi-RHS: solve for each column
        let col_size = n * dtype.size_in_bytes();
        let b_col_ptr = client.allocator().allocate(col_size);
        let x_col_ptr = client.allocator().allocate(col_size);

        for rhs in 0..num_rhs {
            // Extract column from B
            unsafe {
                kernels::launch_extract_column(
                    client.context(),
                    client.stream(),
                    device.index,
                    dtype,
                    b.storage().ptr(),
                    b_col_ptr,
                    n,
                    num_rhs,
                    rhs,
                )?;
            }

            // Solve L @ x_col = b_col
            unsafe {
                kernels::launch_forward_sub(
                    client.context(),
                    client.stream(),
                    device.index,
                    dtype,
                    l.storage().ptr(),
                    b_col_ptr,
                    x_col_ptr,
                    n,
                    unit_diagonal,
                )?;
            }

            // Scatter solution into X
            unsafe {
                kernels::launch_scatter_column(
                    client.context(),
                    client.stream(),
                    device.index,
                    dtype,
                    x_col_ptr,
                    x_ptr,
                    n,
                    rhs,
                )?;
            }
        }

        client.allocator().deallocate(b_col_ptr, col_size);
        client.allocator().deallocate(x_col_ptr, col_size);
    }

    client.synchronize();

    let x = if b_is_vector {
        unsafe { CudaClient::tensor_from_raw(x_ptr, &[n], dtype, device) }
    } else {
        unsafe { CudaClient::tensor_from_raw(x_ptr, &[n, num_rhs], dtype, device) }
    };

    Ok(x)
}

/// Solve upper triangular system Ux = b
/// Supports both single RHS (b is 1D vector) and multi-RHS (b is 2D matrix [n, nrhs])
pub fn solve_triangular_upper_impl(
    client: &CudaClient,
    u: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(u.dtype())?;
    if u.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: u.dtype(),
            rhs: b.dtype(),
        });
    }
    let n = validate_square_matrix(u.shape())?;
    let dtype = u.dtype();
    let device = client.device();

    // Handle 1D or 2D b
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
            "solve_triangular_upper requires b to be 1D or 2D, got {}D with shape {:?}",
            b_shape.len(),
            b_shape
        )));
    };

    // Allocate output
    let x_size = n * num_rhs * dtype.size_in_bytes();
    let x_ptr = client.allocator().allocate(x_size);

    if b_is_vector {
        // Single RHS: direct solve
        unsafe {
            kernels::launch_backward_sub(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                u.storage().ptr(),
                b.storage().ptr(),
                x_ptr,
                n,
            )?;
        }
    } else {
        // Multi-RHS: solve for each column
        let col_size = n * dtype.size_in_bytes();
        let b_col_ptr = client.allocator().allocate(col_size);
        let x_col_ptr = client.allocator().allocate(col_size);

        for rhs in 0..num_rhs {
            // Extract column from B
            unsafe {
                kernels::launch_extract_column(
                    client.context(),
                    client.stream(),
                    device.index,
                    dtype,
                    b.storage().ptr(),
                    b_col_ptr,
                    n,
                    num_rhs,
                    rhs,
                )?;
            }

            // Solve U @ x_col = b_col
            unsafe {
                kernels::launch_backward_sub(
                    client.context(),
                    client.stream(),
                    device.index,
                    dtype,
                    u.storage().ptr(),
                    b_col_ptr,
                    x_col_ptr,
                    n,
                )?;
            }

            // Scatter solution into X
            unsafe {
                kernels::launch_scatter_column(
                    client.context(),
                    client.stream(),
                    device.index,
                    dtype,
                    x_col_ptr,
                    x_ptr,
                    n,
                    rhs,
                )?;
            }
        }

        client.allocator().deallocate(b_col_ptr, col_size);
        client.allocator().deallocate(x_col_ptr, col_size);
    }

    client.synchronize();

    let x = if b_is_vector {
        unsafe { CudaClient::tensor_from_raw(x_ptr, &[n], dtype, device) }
    } else {
        unsafe { CudaClient::tensor_from_raw(x_ptr, &[n, num_rhs], dtype, device) }
    };

    Ok(x)
}

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
    let x_ptr = client.allocator().allocate(x_size);

    // Zero initialize X
    let zero_bytes = vec![0u8; x_size];
    CudaRuntime::copy_to_device(&zero_bytes, x_ptr, device)?;

    // Temporary buffers for column-wise solve
    let col_size = n * dtype.size_in_bytes();
    let qtb_col_ptr = client.allocator().allocate(col_size);
    let x_col_ptr = client.allocator().allocate(col_size);

    // Helper for cleanup on error
    let cleanup = |allocator: &super::super::CudaAllocator| {
        allocator.deallocate(x_ptr, x_size);
        allocator.deallocate(qtb_col_ptr, col_size);
        allocator.deallocate(x_col_ptr, col_size);
    };

    // Solve R @ X[:, col] = (Q^T @ B)[:n, col] for each column
    // R is [m, n], upper triangular - we use top n×n block
    for rhs in 0..num_rhs {
        // Extract column from Q^T @ B (need first n elements of column rhs)
        // qtb is [m, num_rhs], we extract column rhs, first n elements
        if num_rhs == 1 {
            // Single RHS: qtb is already [m, 1], just use first n elements
            // Copy first n elements to qtb_col_ptr
            CudaRuntime::copy_within_device(qtb.storage().ptr(), qtb_col_ptr, col_size, device)?;
        } else {
            // Multi-RHS: extract column rhs from qtb [m, num_rhs]
            // But we only need first n elements
            let result = unsafe {
                kernels::launch_extract_column(
                    client.context(),
                    client.stream(),
                    device.index,
                    dtype,
                    qtb.storage().ptr(),
                    qtb_col_ptr,
                    n, // only extract first n elements
                    num_rhs,
                    rhs,
                )
            };
            if let Err(e) = result {
                cleanup(client.allocator());
                return Err(e);
            }
        }

        // Backward substitution: R @ x = qtb_col
        // R is [m, n] but stored as full matrix, n×n upper triangular part is valid
        let result = unsafe {
            kernels::launch_backward_sub(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                qr.r.storage().ptr(),
                qtb_col_ptr,
                x_col_ptr,
                n,
            )
        };
        if let Err(e) = result {
            cleanup(client.allocator());
            return Err(e);
        }

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
            if let Err(e) = result {
                cleanup(client.allocator());
                return Err(e);
            }
        }
    }

    // Clean up temporary buffers
    client.allocator().deallocate(qtb_col_ptr, col_size);
    client.allocator().deallocate(x_col_ptr, col_size);

    client.synchronize();

    let x = if b_is_vector {
        unsafe { CudaClient::tensor_from_raw(x_ptr, &[n], dtype, device) }
    } else {
        unsafe { CudaClient::tensor_from_raw(x_ptr, &[n, num_rhs], dtype, device) }
    };

    Ok(x)
}
