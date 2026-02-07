//! Banded linear system solver for CUDA runtime
//!
//! Uses native CUDA PTX kernels - NO CPU fallback, NO GPU↔CPU transfers.

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use super::super::kernels;
use crate::algorithm::linalg::{validate_linalg_dtype, validate_matrix_2d};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::{Allocator, RuntimeClient};
use crate::tensor::Tensor;

/// Validate banded system inputs and return (n, nrhs).
fn validate_banded(
    ab_shape: &[usize],
    b_shape: &[usize],
    kl: usize,
    ku: usize,
) -> Result<(usize, usize)> {
    let (ab_rows, n) = validate_matrix_2d(ab_shape)?;
    let expected_rows = kl + ku + 1;
    if ab_rows != expected_rows {
        return Err(Error::ShapeMismatch {
            expected: vec![expected_rows, n],
            got: ab_shape.to_vec(),
        });
    }
    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "ab",
            reason: "banded system size n must be > 0".to_string(),
        });
    }
    let nrhs = match b_shape.len() {
        1 => {
            if b_shape[0] != n {
                return Err(Error::ShapeMismatch {
                    expected: vec![n],
                    got: b_shape.to_vec(),
                });
            }
            1
        }
        2 => {
            if b_shape[0] != n {
                return Err(Error::ShapeMismatch {
                    expected: vec![n, b_shape[1]],
                    got: b_shape.to_vec(),
                });
            }
            b_shape[1]
        }
        _ => {
            return Err(Error::InvalidArgument {
                arg: "b",
                reason: format!("expected 1D or 2D tensor, got {}D", b_shape.len()),
            });
        }
    };
    Ok((n, nrhs))
}

pub fn solve_banded_impl(
    client: &CudaClient,
    ab: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    kl: usize,
    ku: usize,
) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(ab.dtype())?;
    if ab.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: ab.dtype(),
            rhs: b.dtype(),
        });
    }
    let dtype = ab.dtype();
    match dtype {
        DType::F32 | DType::F64 => {}
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "solve_banded",
            });
        }
    }

    let (n, nrhs) = validate_banded(ab.shape(), b.shape(), kl, ku)?;
    let device = client.device();
    let elem_size = dtype.size_in_bytes();

    // Allocate workspace for banded LU (needed even for Thomas - kernel ignores it)
    let work_rows = 2 * kl + ku + 1;
    let work_size = work_rows * n * elem_size;
    let work_ptr = client.allocator().allocate(work_size);

    // Allocate output buffer
    let x_total_size = n * nrhs * elem_size;
    let x_ptr = client.allocator().allocate(x_total_size);

    let col_size = n * elem_size;

    // Make inputs contiguous
    let ab_contig = ab.contiguous();
    let b_contig = b.contiguous();

    let b_is_1d = b.ndim() == 1;

    if nrhs == 1 {
        // Single RHS: solve directly
        let result = unsafe {
            kernels::launch_banded_solve(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                ab_contig.storage().ptr(),
                b_contig.storage().ptr(),
                x_ptr,
                work_ptr,
                n,
                kl,
                ku,
            )
        };
        if let Err(e) = result {
            client.allocator().deallocate(work_ptr, work_size);
            client.allocator().deallocate(x_ptr, x_total_size);
            return Err(e);
        }
    } else {
        // Multi-RHS: extract each column, solve, scatter back
        let b_col_ptr = client.allocator().allocate(col_size);
        let x_col_ptr = client.allocator().allocate(col_size);

        for rhs in 0..nrhs {
            // Extract column rhs from B [n, nrhs]
            let result = unsafe {
                kernels::launch_extract_column(
                    client.context(),
                    client.stream(),
                    device.index,
                    dtype,
                    b_contig.storage().ptr(),
                    b_col_ptr,
                    n,
                    nrhs,
                    rhs,
                )
            };
            if let Err(e) = result {
                client.allocator().deallocate(b_col_ptr, col_size);
                client.allocator().deallocate(x_col_ptr, col_size);
                client.allocator().deallocate(work_ptr, work_size);
                client.allocator().deallocate(x_ptr, x_total_size);
                return Err(e);
            }

            // Solve for this column
            let result = unsafe {
                kernels::launch_banded_solve(
                    client.context(),
                    client.stream(),
                    device.index,
                    dtype,
                    ab_contig.storage().ptr(),
                    b_col_ptr,
                    x_col_ptr,
                    work_ptr,
                    n,
                    kl,
                    ku,
                )
            };
            if let Err(e) = result {
                client.allocator().deallocate(b_col_ptr, col_size);
                client.allocator().deallocate(x_col_ptr, col_size);
                client.allocator().deallocate(work_ptr, work_size);
                client.allocator().deallocate(x_ptr, x_total_size);
                return Err(e);
            }

            // Scatter solution into X [n, nrhs]
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
                client.allocator().deallocate(b_col_ptr, col_size);
                client.allocator().deallocate(x_col_ptr, col_size);
                client.allocator().deallocate(work_ptr, work_size);
                client.allocator().deallocate(x_ptr, x_total_size);
                return Err(e);
            }
        }

        client.allocator().deallocate(b_col_ptr, col_size);
        client.allocator().deallocate(x_col_ptr, col_size);
    }

    client.allocator().deallocate(work_ptr, work_size);
    client.synchronize();

    let x = if b_is_1d {
        unsafe { CudaClient::tensor_from_raw(x_ptr, &[n], dtype, device) }
    } else {
        unsafe { CudaClient::tensor_from_raw(x_ptr, &[n, nrhs], dtype, device) }
    };

    Ok(x)
}
