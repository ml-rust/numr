//! CUDA implementation of linear algebra algorithms
//!
//! This module implements the [`LinearAlgebraAlgorithms`] trait for CUDA.
//! All algorithms follow the exact specification in the trait documentation
//! to ensure backend parity with CPU/WebGPU implementations.
//!
//! Native CUDA kernels are used - NO cuSOLVER dependency.

use super::CudaRuntime;
use super::client::CudaClient;
use super::kernels;
use crate::algorithm::linalg::{
    CholeskyDecomposition, EigenDecomposition, LinearAlgebraAlgorithms, LuDecomposition,
    MatrixNormOrder, QrDecomposition, SvdDecomposition, validate_linalg_dtype, validate_matrix_2d,
    validate_square_matrix,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::TensorOps;
use crate::runtime::{Allocator, Runtime, RuntimeClient};
use crate::tensor::{Layout, Storage, Tensor};

impl LinearAlgebraAlgorithms<CudaRuntime> for CudaClient {
    fn lu_decompose(&self, a: &Tensor<CudaRuntime>) -> Result<LuDecomposition<CudaRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;
        let k = m.min(n);
        let dtype = a.dtype();
        let device = self.device();

        // Allocate output tensors on GPU
        let lu_size = m * n * dtype.size_in_bytes();
        let lu_ptr = self.allocator().allocate(lu_size);

        let pivots_size = k * std::mem::size_of::<i64>();
        let pivots_ptr = self.allocator().allocate(pivots_size);

        let num_swaps_size = std::mem::size_of::<i32>();
        let num_swaps_ptr = self.allocator().allocate(num_swaps_size);

        let singular_flag_size = std::mem::size_of::<i32>();
        let singular_flag_ptr = self.allocator().allocate(singular_flag_size);

        // Copy input to LU buffer
        CudaRuntime::copy_within_device(a.storage().ptr(), lu_ptr, lu_size, device);

        // Zero-initialize flags
        let zero_i32: [u8; 4] = [0; 4];
        CudaRuntime::copy_to_device(&zero_i32, num_swaps_ptr, device);
        CudaRuntime::copy_to_device(&zero_i32, singular_flag_ptr, device);

        // Launch kernel
        unsafe {
            kernels::launch_lu_decompose(
                self.context(),
                self.stream(),
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

        self.synchronize();

        // Read back flags
        let mut num_swaps_bytes = [0u8; 4];
        let mut singular_flag_bytes = [0u8; 4];
        CudaRuntime::copy_from_device(num_swaps_ptr, &mut num_swaps_bytes, device);
        CudaRuntime::copy_from_device(singular_flag_ptr, &mut singular_flag_bytes, device);

        let num_swaps = i32::from_ne_bytes(num_swaps_bytes) as usize;
        let singular = i32::from_ne_bytes(singular_flag_bytes) != 0;

        // Clean up flag allocations
        self.allocator().deallocate(num_swaps_ptr, num_swaps_size);
        self.allocator()
            .deallocate(singular_flag_ptr, singular_flag_size);

        if singular {
            self.allocator().deallocate(lu_ptr, lu_size);
            self.allocator().deallocate(pivots_ptr, pivots_size);
            return Err(Error::Internal(format!(
                "LU decomposition failed: {}x{} matrix is singular (zero pivot encountered)",
                m, n
            )));
        }

        // Create tensors from GPU memory
        let lu = unsafe { Self::tensor_from_raw(lu_ptr, &[m, n], dtype, device) };
        let pivots = unsafe { Self::tensor_from_raw(pivots_ptr, &[k], DType::I64, device) };

        Ok(LuDecomposition {
            lu,
            pivots,
            num_swaps,
        })
    }

    fn cholesky_decompose(
        &self,
        a: &Tensor<CudaRuntime>,
    ) -> Result<CholeskyDecomposition<CudaRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;
        let dtype = a.dtype();
        let device = self.device();

        // Allocate output on GPU
        let l_size = n * n * dtype.size_in_bytes();
        let l_ptr = self.allocator().allocate(l_size);

        let not_pd_flag_size = std::mem::size_of::<i32>();
        let not_pd_flag_ptr = self.allocator().allocate(not_pd_flag_size);

        // Copy input to L buffer
        CudaRuntime::copy_within_device(a.storage().ptr(), l_ptr, l_size, device);

        // Zero-initialize flag
        let zero_i32: [u8; 4] = [0; 4];
        CudaRuntime::copy_to_device(&zero_i32, not_pd_flag_ptr, device);

        // Launch kernel
        unsafe {
            kernels::launch_cholesky_decompose(
                self.context(),
                self.stream(),
                device.index,
                dtype,
                l_ptr,
                not_pd_flag_ptr,
                n,
            )?;
        }

        self.synchronize();

        // Read back flag
        let mut not_pd_bytes = [0u8; 4];
        CudaRuntime::copy_from_device(not_pd_flag_ptr, &mut not_pd_bytes, device);
        let not_pd = i32::from_ne_bytes(not_pd_bytes) != 0;

        self.allocator()
            .deallocate(not_pd_flag_ptr, not_pd_flag_size);

        if not_pd {
            self.allocator().deallocate(l_ptr, l_size);
            return Err(Error::Internal(format!(
                "Cholesky decomposition failed: {}x{} matrix is not positive definite",
                n, n
            )));
        }

        let l = unsafe { Self::tensor_from_raw(l_ptr, &[n, n], dtype, device) };

        Ok(CholeskyDecomposition { l })
    }

    fn qr_decompose(&self, a: &Tensor<CudaRuntime>) -> Result<QrDecomposition<CudaRuntime>> {
        self.qr_decompose_internal(a, false)
    }

    fn qr_decompose_thin(&self, a: &Tensor<CudaRuntime>) -> Result<QrDecomposition<CudaRuntime>> {
        self.qr_decompose_internal(a, true)
    }

    fn solve(
        &self,
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
        let device = self.device();

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
        let lu_result = self.lu_decompose(a)?;

        // Allocate output and temporary buffers
        let x_size = n * num_rhs * dtype.size_in_bytes();
        let x_ptr = self.allocator().allocate(x_size);

        let col_size = n * dtype.size_in_bytes();
        let b_col_ptr = self.allocator().allocate(col_size);
        let pb_ptr = self.allocator().allocate(col_size);
        let y_ptr = self.allocator().allocate(col_size);
        let x_col_ptr = self.allocator().allocate(col_size);

        // Helper for cleanup on error
        let cleanup = |allocator: &super::CudaAllocator| {
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
                        self.context(),
                        self.stream(),
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
                    cleanup(self.allocator());
                    return Err(e);
                }
                b_col_ptr
            };

            // Apply permutation: pb = P @ b_col
            let result = unsafe {
                kernels::launch_apply_lu_permutation(
                    self.context(),
                    self.stream(),
                    device.index,
                    dtype,
                    b_ptr_for_solve,
                    pb_ptr,
                    lu_result.pivots.storage().ptr(),
                    n,
                )
            };
            if let Err(e) = result {
                cleanup(self.allocator());
                return Err(e);
            }

            // Forward substitution: Ly = pb (L has unit diagonal)
            let result = unsafe {
                kernels::launch_forward_sub(
                    self.context(),
                    self.stream(),
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
                cleanup(self.allocator());
                return Err(e);
            }

            // Backward substitution: Ux = y
            let result = unsafe {
                kernels::launch_backward_sub(
                    self.context(),
                    self.stream(),
                    device.index,
                    dtype,
                    lu_result.lu.storage().ptr(),
                    y_ptr,
                    x_col_ptr,
                    n,
                )
            };
            if let Err(e) = result {
                cleanup(self.allocator());
                return Err(e);
            }

            // Scatter solution into X
            if b_is_vector {
                // Single RHS: copy directly to x_ptr
                CudaRuntime::copy_within_device(x_col_ptr, x_ptr, col_size, device);
            } else {
                // Multi-RHS: scatter into column rhs of X [n, num_rhs]
                let result = unsafe {
                    kernels::launch_scatter_column(
                        self.context(),
                        self.stream(),
                        device.index,
                        dtype,
                        x_col_ptr,
                        x_ptr,
                        n,
                        rhs,
                    )
                };
                if let Err(e) = result {
                    cleanup(self.allocator());
                    return Err(e);
                }
            }
        }

        // Clean up temporary buffers (keep x_ptr for result)
        self.allocator().deallocate(b_col_ptr, col_size);
        self.allocator().deallocate(pb_ptr, col_size);
        self.allocator().deallocate(y_ptr, col_size);
        self.allocator().deallocate(x_col_ptr, col_size);

        self.synchronize();

        let x = if b_is_vector {
            unsafe { Self::tensor_from_raw(x_ptr, &[n], dtype, device) }
        } else {
            unsafe { Self::tensor_from_raw(x_ptr, &[n, num_rhs], dtype, device) }
        };

        Ok(x)
    }

    fn solve_triangular_lower(
        &self,
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
        let device = self.device();

        // Only handle vector b for now
        if b.shape().len() != 1 || b.shape()[0] != n {
            return Err(Error::ShapeMismatch {
                expected: vec![n],
                got: b.shape().to_vec(),
            });
        }

        let x_size = n * dtype.size_in_bytes();
        let x_ptr = self.allocator().allocate(x_size);

        unsafe {
            kernels::launch_forward_sub(
                self.context(),
                self.stream(),
                device.index,
                dtype,
                l.storage().ptr(),
                b.storage().ptr(),
                x_ptr,
                n,
                unit_diagonal,
            )?;
        }

        self.synchronize();

        let x = unsafe { Self::tensor_from_raw(x_ptr, &[n], dtype, device) };

        Ok(x)
    }

    fn solve_triangular_upper(
        &self,
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
        let device = self.device();

        // Only handle vector b for now
        if b.shape().len() != 1 || b.shape()[0] != n {
            return Err(Error::ShapeMismatch {
                expected: vec![n],
                got: b.shape().to_vec(),
            });
        }

        let x_size = n * dtype.size_in_bytes();
        let x_ptr = self.allocator().allocate(x_size);

        unsafe {
            kernels::launch_backward_sub(
                self.context(),
                self.stream(),
                device.index,
                dtype,
                u.storage().ptr(),
                b.storage().ptr(),
                x_ptr,
                n,
            )?;
        }

        self.synchronize();

        let x = unsafe { Self::tensor_from_raw(x_ptr, &[n], dtype, device) };

        Ok(x)
    }

    fn lstsq(
        &self,
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
        let device = self.device();

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
        let qr = self.qr_decompose(a)?;

        // Compute Q^T @ b using TensorOps::matmul
        use crate::ops::TensorOps;

        // Q^T is [m, m], need to transpose Q
        let q_t = qr.q.transpose(0, 1)?;
        let b_mat = if b_is_vector {
            b.reshape(&[m, 1])?
        } else {
            b.clone()
        };

        // Q^T @ B gives [m, num_rhs]
        let qtb = TensorOps::matmul(self, &q_t, &b_mat)?;

        // Allocate output X [n, num_rhs] or [n] for vector
        let x_size = n * num_rhs * dtype.size_in_bytes();
        let x_ptr = self.allocator().allocate(x_size);

        // Zero initialize X
        let zero_bytes = vec![0u8; x_size];
        CudaRuntime::copy_to_device(&zero_bytes, x_ptr, device);

        // Temporary buffers for column-wise solve
        let col_size = n * dtype.size_in_bytes();
        let qtb_col_ptr = self.allocator().allocate(col_size);
        let x_col_ptr = self.allocator().allocate(col_size);

        // Helper for cleanup on error
        let cleanup = |allocator: &super::CudaAllocator| {
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
                CudaRuntime::copy_within_device(qtb.storage().ptr(), qtb_col_ptr, col_size, device);
            } else {
                // Multi-RHS: extract column rhs from qtb [m, num_rhs]
                // But we only need first n elements
                let result = unsafe {
                    kernels::launch_extract_column(
                        self.context(),
                        self.stream(),
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
                    cleanup(self.allocator());
                    return Err(e);
                }
            }

            // Backward substitution: R @ x = qtb_col
            // R is [m, n] but stored as full matrix, n×n upper triangular part is valid
            let result = unsafe {
                kernels::launch_backward_sub(
                    self.context(),
                    self.stream(),
                    device.index,
                    dtype,
                    qr.r.storage().ptr(),
                    qtb_col_ptr,
                    x_col_ptr,
                    n,
                )
            };
            if let Err(e) = result {
                cleanup(self.allocator());
                return Err(e);
            }

            // Scatter solution into X
            if num_rhs == 1 {
                // Single RHS: copy directly
                CudaRuntime::copy_within_device(x_col_ptr, x_ptr, col_size, device);
            } else {
                // Multi-RHS: scatter into column rhs of X [n, num_rhs]
                let result = unsafe {
                    kernels::launch_scatter_column(
                        self.context(),
                        self.stream(),
                        device.index,
                        dtype,
                        x_col_ptr,
                        x_ptr,
                        n,
                        rhs,
                    )
                };
                if let Err(e) = result {
                    cleanup(self.allocator());
                    return Err(e);
                }
            }
        }

        // Clean up temporary buffers
        self.allocator().deallocate(qtb_col_ptr, col_size);
        self.allocator().deallocate(x_col_ptr, col_size);

        self.synchronize();

        let x = if b_is_vector {
            unsafe { Self::tensor_from_raw(x_ptr, &[n], dtype, device) }
        } else {
            unsafe { Self::tensor_from_raw(x_ptr, &[n, num_rhs], dtype, device) }
        };

        Ok(x)
    }

    fn inverse(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;
        let dtype = a.dtype();
        let device = self.device();

        // Compute LU decomposition
        let lu_result = self.lu_decompose(a)?;

        // Allocate output and temporary buffers
        let inv_size = n * n * dtype.size_in_bytes();
        let inv_ptr = self.allocator().allocate(inv_size);

        let col_size = n * dtype.size_in_bytes();
        let identity_ptr = self.allocator().allocate(inv_size); // Full identity matrix
        let pb_ptr = self.allocator().allocate(col_size);
        let y_ptr = self.allocator().allocate(col_size);
        let x_ptr = self.allocator().allocate(col_size);
        let e_ptr = self.allocator().allocate(col_size);

        // Helper closure for cleanup on error
        let cleanup = |allocator: &super::CudaAllocator| {
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
                self.context(),
                self.stream(),
                device.index,
                dtype,
                identity_ptr,
                n,
            )
        };
        if let Err(e) = result {
            cleanup(self.allocator());
            return Err(e);
        }

        // Solve for each column of the identity matrix
        for col in 0..n {
            // Extract column from identity matrix (GPU-only)
            let result = unsafe {
                kernels::launch_extract_column(
                    self.context(),
                    self.stream(),
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
                cleanup(self.allocator());
                return Err(e);
            }

            // Apply permutation: pb = P @ e
            let result = unsafe {
                kernels::launch_apply_lu_permutation(
                    self.context(),
                    self.stream(),
                    device.index,
                    dtype,
                    e_ptr,
                    pb_ptr,
                    lu_result.pivots.storage().ptr(),
                    n,
                )
            };
            if let Err(e) = result {
                cleanup(self.allocator());
                return Err(e);
            }

            // Forward substitution: Ly = pb (L has unit diagonal)
            let result = unsafe {
                kernels::launch_forward_sub(
                    self.context(),
                    self.stream(),
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
                cleanup(self.allocator());
                return Err(e);
            }

            // Backward substitution: Ux = y
            let result = unsafe {
                kernels::launch_backward_sub(
                    self.context(),
                    self.stream(),
                    device.index,
                    dtype,
                    lu_result.lu.storage().ptr(),
                    y_ptr,
                    x_ptr,
                    n,
                )
            };
            if let Err(e) = result {
                cleanup(self.allocator());
                return Err(e);
            }

            // Scatter x into column of inverse matrix (GPU-only, no CPU transfer)
            let result = unsafe {
                kernels::launch_scatter_column(
                    self.context(),
                    self.stream(),
                    device.index,
                    dtype,
                    x_ptr,
                    inv_ptr,
                    n,
                    col,
                )
            };
            if let Err(e) = result {
                cleanup(self.allocator());
                return Err(e);
            }
        }

        // Clean up temporary buffers (keep inv_ptr for result)
        self.allocator().deallocate(identity_ptr, inv_size);
        self.allocator().deallocate(pb_ptr, col_size);
        self.allocator().deallocate(y_ptr, col_size);
        self.allocator().deallocate(x_ptr, col_size);
        self.allocator().deallocate(e_ptr, col_size);

        self.synchronize();

        let inv = unsafe { Self::tensor_from_raw(inv_ptr, &[n, n], dtype, device) };

        Ok(inv)
    }

    fn det(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;
        let dtype = a.dtype();
        let device = self.device();

        // Compute LU decomposition
        let lu_result = self.lu_decompose(a)?;

        // Allocate output
        let det_size = dtype.size_in_bytes();
        let det_ptr = self.allocator().allocate(det_size);

        // Compute determinant from LU diagonal
        unsafe {
            kernels::launch_det_from_lu(
                self.context(),
                self.stream(),
                device.index,
                dtype,
                lu_result.lu.storage().ptr(),
                det_ptr,
                n,
                lu_result.num_swaps as i32,
            )?;
        }

        self.synchronize();

        let det = unsafe { Self::tensor_from_raw(det_ptr, &[], dtype, device) };

        Ok(det)
    }

    fn trace(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;
        let min_dim = m.min(n);
        let dtype = a.dtype();
        let device = self.device();

        // Allocate output (zero-initialized for atomic add)
        let trace_size = dtype.size_in_bytes();
        let trace_ptr = self.allocator().allocate(trace_size);

        let zero_bytes = vec![0u8; trace_size];
        CudaRuntime::copy_to_device(&zero_bytes, trace_ptr, device);

        unsafe {
            kernels::launch_trace(
                self.context(),
                self.stream(),
                device.index,
                dtype,
                a.storage().ptr(),
                trace_ptr,
                min_dim,
                n, // stride (number of columns)
            )?;
        }

        self.synchronize();

        let trace = unsafe { Self::tensor_from_raw(trace_ptr, &[], dtype, device) };

        Ok(trace)
    }

    fn diag(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;
        let min_dim = m.min(n);
        let dtype = a.dtype();
        let device = self.device();

        let diag_size = min_dim * dtype.size_in_bytes();
        let diag_ptr = self.allocator().allocate(diag_size);

        unsafe {
            kernels::launch_diag(
                self.context(),
                self.stream(),
                device.index,
                dtype,
                a.storage().ptr(),
                diag_ptr,
                min_dim,
                n,
            )?;
        }

        self.synchronize();

        let diag = unsafe { Self::tensor_from_raw(diag_ptr, &[min_dim], dtype, device) };

        Ok(diag)
    }

    fn diagflat(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
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
        let device = self.device();

        let out_size = n * n * dtype.size_in_bytes();
        let out_ptr = self.allocator().allocate(out_size);

        unsafe {
            kernels::launch_diagflat(
                self.context(),
                self.stream(),
                device.index,
                dtype,
                a.storage().ptr(),
                out_ptr,
                n,
            )?;
        }

        self.synchronize();

        let out = unsafe { Self::tensor_from_raw(out_ptr, &[n, n], dtype, device) };

        Ok(out)
    }

    fn matrix_rank(
        &self,
        a: &Tensor<CudaRuntime>,
        tol: Option<f64>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;
        let dtype = a.dtype();
        let device = self.device();
        let k = m.min(n);

        // Use QR decomposition to estimate rank
        let qr = self.qr_decompose(a)?;

        // Get diagonal of R
        let r_diag = LinearAlgebraAlgorithms::diag(self, &qr.r)?;

        // Allocate GPU buffers for max abs and count
        let max_size = dtype.size_in_bytes();
        let max_ptr = self.allocator().allocate(max_size);
        let count_size = std::mem::size_of::<u32>();
        let count_ptr = self.allocator().allocate(count_size);

        // Zero-initialize max and count on GPU
        let zero_bytes = vec![0u8; max_size.max(count_size)];
        CudaRuntime::copy_to_device(&zero_bytes[..max_size], max_ptr, device);
        CudaRuntime::copy_to_device(&zero_bytes[..count_size], count_ptr, device);

        // Compute max absolute value on GPU (no CPU transfer of full diagonal)
        let result = unsafe {
            kernels::launch_max_abs(
                self.context(),
                self.stream(),
                device.index,
                dtype,
                r_diag.storage().ptr(),
                max_ptr,
                k,
            )
        };
        if let Err(e) = result {
            self.allocator().deallocate(max_ptr, max_size);
            self.allocator().deallocate(count_ptr, count_size);
            return Err(e);
        }

        self.synchronize();

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
                self.context(),
                self.stream(),
                device.index,
                dtype,
                r_diag.storage().ptr(),
                count_ptr,
                k,
                threshold,
            )
        };
        if let Err(e) = result {
            self.allocator().deallocate(max_ptr, max_size);
            self.allocator().deallocate(count_ptr, count_size);
            return Err(e);
        }

        self.synchronize();

        // Read count (single u32 transfer)
        let mut count_bytes = [0u8; 4];
        CudaRuntime::copy_from_device(count_ptr, &mut count_bytes, device);
        let rank = u32::from_ne_bytes(count_bytes) as i64;

        // Clean up
        self.allocator().deallocate(max_ptr, max_size);
        self.allocator().deallocate(count_ptr, count_size);

        // Create rank tensor on GPU
        let rank_data: [i64; 1] = [rank];
        let rank_tensor = Tensor::<CudaRuntime>::from_slice(&rank_data, &[], device);

        Ok(rank_tensor)
    }

    fn matrix_norm(
        &self,
        a: &Tensor<CudaRuntime>,
        ord: MatrixNormOrder,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (_m, _n) = validate_matrix_2d(a.shape())?;

        match ord {
            MatrixNormOrder::Frobenius => {
                // Frobenius norm: ||A||_F = sqrt(sum(A²))
                // Use existing tensor ops to keep data on GPU
                let squared = self.square(a)?;
                let sum_sq = self.sum(&squared, &[], false)?;
                self.sqrt(&sum_sq)
            }
            MatrixNormOrder::Spectral | MatrixNormOrder::Nuclear => Err(Error::Internal(
                "Spectral and nuclear norms require SVD (not yet implemented)".to_string(),
            )),
        }
    }

    fn svd_decompose(&self, a: &Tensor<CudaRuntime>) -> Result<SvdDecomposition<CudaRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;
        let k = m.min(n);
        let dtype = a.dtype();
        let device = self.device();

        // Handle empty matrix
        if m == 0 || n == 0 {
            let u_ptr = self.allocator().allocate(0);
            let s_ptr = self.allocator().allocate(0);
            let vt_ptr = self.allocator().allocate(0);
            let u = unsafe { Self::tensor_from_raw(u_ptr, &[m, k], dtype, device) };
            let s = unsafe { Self::tensor_from_raw(s_ptr, &[k], dtype, device) };
            let vt = unsafe { Self::tensor_from_raw(vt_ptr, &[k, n], dtype, device) };
            return Ok(SvdDecomposition { u, s, vt });
        }

        // If m < n, transpose and swap U/V at the end
        let transpose = m < n;
        let (work_m, work_n) = if transpose { (n, m) } else { (m, n) };
        let work_k = work_m.min(work_n);

        // Allocate working buffers on GPU
        let b_size = work_m * work_n * dtype.size_in_bytes();
        let b_ptr = self.allocator().allocate(b_size);

        let v_size = work_n * work_n * dtype.size_in_bytes();
        let v_ptr = self.allocator().allocate(v_size);

        let s_size = work_n * dtype.size_in_bytes();
        let s_ptr = self.allocator().allocate(s_size);

        let flag_size = std::mem::size_of::<i32>();
        let converged_flag_ptr = self.allocator().allocate(flag_size);

        // Helper for cleanup on error
        let cleanup = |allocator: &super::CudaAllocator| {
            allocator.deallocate(b_ptr, b_size);
            allocator.deallocate(v_ptr, v_size);
            allocator.deallocate(s_ptr, s_size);
            allocator.deallocate(converged_flag_ptr, flag_size);
        };

        // Copy input to B, transposing if needed using GPU transpose kernel
        if transpose {
            // Use optimized GPU transpose: A[m,n] -> B[n,m]
            // No CPU fallback - full GPU acceleration
            let result = unsafe {
                kernels::launch_transpose(
                    self.context(),
                    self.stream(),
                    device.index,
                    dtype,
                    a.storage().ptr(),
                    b_ptr,
                    m, // rows of input
                    n, // cols of input
                )
            };
            if let Err(e) = result {
                cleanup(self.allocator());
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
                self.context(),
                self.stream(),
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
            cleanup(self.allocator());
            return Err(e);
        }

        self.synchronize();

        // Clean up converged flag
        self.allocator().deallocate(converged_flag_ptr, flag_size);

        // Now we need to:
        // 1. Sort singular values in descending order
        // 2. Reorder U and V columns accordingly
        // 3. Extract thin U [work_m x work_k] and V^T [work_k x work_n]
        // 4. If transposed, swap U and V^T

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

        // Read U (b_ptr after normalization) and V for reordering
        // Then write sorted results back

        // This is complex, so for initial implementation do it on CPU
        // (Can be optimized with GPU permutation kernels later)

        let u_final;
        let s_final;
        let vt_final;

        match dtype {
            DType::F32 => {
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
                        // V^T[new_idx, j] = V[j, old_idx]
                        vt_sorted[new_idx * work_n + j] = v_data[j * work_n + old_idx];
                    }
                }

                if transpose {
                    // A = V' @ S @ U'^T
                    // For original A [m x n] with m < n:
                    // - U_final should be [m x k] where k = min(m,n) = m
                    // - V^T_final should be [k x n] = [m x n]
                    //
                    // U_final = V' = (vt_sorted)^T, shape [m x k]
                    let mut u_final_data = vec![0.0f32; m * k];
                    for i in 0..k {
                        for j in 0..m {
                            u_final_data[j * k + i] = vt_sorted[i * work_n + j];
                        }
                    }

                    // V^T_final = U'^T = u_sorted^T, shape [k x n]
                    let mut vt_final_data = vec![0.0f32; k * n];
                    for i in 0..work_m {
                        for j in 0..work_k {
                            vt_final_data[j * n + i] = u_sorted[i * work_k + j];
                        }
                    }

                    u_final = Tensor::<CudaRuntime>::from_slice(&u_final_data, &[m, k], device);
                    s_final = Tensor::<CudaRuntime>::from_slice(&s_sorted, &[k], device);
                    vt_final = Tensor::<CudaRuntime>::from_slice(&vt_final_data, &[k, n], device);
                } else {
                    u_final = Tensor::<CudaRuntime>::from_slice(&u_sorted, &[m, k], device);
                    s_final = Tensor::<CudaRuntime>::from_slice(&s_sorted, &[k], device);
                    vt_final = Tensor::<CudaRuntime>::from_slice(&vt_sorted, &[k, n], device);
                }
            }
            DType::F64 => {
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
                    // For original A [m x n] with m < n:
                    // - U_final should be [m x k] where k = min(m,n) = m
                    // - V^T_final should be [k x n] = [m x n]
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

                    u_final = Tensor::<CudaRuntime>::from_slice(&u_final_data, &[m, k], device);
                    s_final = Tensor::<CudaRuntime>::from_slice(&s_sorted, &[k], device);
                    vt_final = Tensor::<CudaRuntime>::from_slice(&vt_final_data, &[k, n], device);
                } else {
                    u_final = Tensor::<CudaRuntime>::from_slice(&u_sorted, &[m, k], device);
                    s_final = Tensor::<CudaRuntime>::from_slice(&s_sorted, &[k], device);
                    vt_final = Tensor::<CudaRuntime>::from_slice(&vt_sorted, &[k, n], device);
                }
            }
            _ => {
                self.allocator().deallocate(b_ptr, b_size);
                self.allocator().deallocate(v_ptr, v_size);
                self.allocator().deallocate(s_ptr, s_size);
                return Err(Error::UnsupportedDType {
                    dtype,
                    op: "svd_decompose",
                });
            }
        }

        // Clean up working buffers
        self.allocator().deallocate(b_ptr, b_size);
        self.allocator().deallocate(v_ptr, v_size);
        self.allocator().deallocate(s_ptr, s_size);

        Ok(SvdDecomposition {
            u: u_final,
            s: s_final,
            vt: vt_final,
        })
    }

    fn pinverse(&self, a: &Tensor<CudaRuntime>, rcond: Option<f64>) -> Result<Tensor<CudaRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;
        let dtype = a.dtype();
        let device = self.device();

        // Handle empty matrix
        if m == 0 || n == 0 {
            let out_ptr = self.allocator().allocate(0);
            return Ok(unsafe { Self::tensor_from_raw(out_ptr, &[n, m], dtype, device) });
        }

        // Compute SVD: A = U @ diag(S) @ V^T
        let svd = self.svd_decompose(a)?;

        // Get singular values to determine cutoff
        let k = m.min(n);
        let s_data: Vec<f64> = match dtype {
            DType::F32 => svd
                .s
                .to_vec::<f32>()
                .into_iter()
                .map(|x| x as f64)
                .collect(),
            DType::F64 => svd.s.to_vec::<f64>(),
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype,
                    op: "pinverse",
                });
            }
        };

        // Determine cutoff threshold
        let max_sv = s_data.iter().cloned().fold(0.0_f64, f64::max);
        let default_rcond = (m.max(n) as f64)
            * match dtype {
                DType::F32 => f32::EPSILON as f64,
                DType::F64 => f64::EPSILON,
                _ => f32::EPSILON as f64,
            };
        let rcond_val = rcond.unwrap_or(default_rcond);
        let cutoff = rcond_val * max_sv;

        // Compute S_inv: invert singular values above cutoff, zero otherwise
        let s_inv_data: Vec<f64> = s_data
            .iter()
            .map(|&s| if s > cutoff { 1.0 / s } else { 0.0 })
            .collect();

        // Create S_inv diagonal matrix [k x k] on GPU
        let s_inv_diag = match dtype {
            DType::F32 => {
                let s_inv_f32: Vec<f32> = s_inv_data.iter().map(|&x| x as f32).collect();
                Tensor::<CudaRuntime>::from_slice(&s_inv_f32, &[k], device)
            }
            DType::F64 => Tensor::<CudaRuntime>::from_slice(&s_inv_data, &[k], device),
            _ => unreachable!(),
        };

        // Create diagonal matrix from vector
        let s_inv_mat = LinearAlgebraAlgorithms::diagflat(self, &s_inv_diag)?;

        // Compute A^+ = V @ S_inv @ U^T
        // V^T is [k x n], so V is [n x k]
        // U is [m x k], so U^T is [k x m]
        // A^+ = V @ S_inv @ U^T = [n x k] @ [k x k] @ [k x m] = [n x m]

        // V = (V^T)^T
        let v = svd.vt.transpose(0, 1)?;
        // U^T
        let ut = svd.u.transpose(0, 1)?;

        // V @ S_inv
        let v_sinv = TensorOps::matmul(self, &v, &s_inv_mat)?;
        // (V @ S_inv) @ U^T
        let pinv = TensorOps::matmul(self, &v_sinv, &ut)?;

        Ok(pinv)
    }

    fn cond(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;
        let dtype = a.dtype();
        let device = self.device();

        // Handle empty matrix
        if m == 0 || n == 0 {
            let inf_val = match dtype {
                DType::F32 => Tensor::<CudaRuntime>::from_slice(&[f32::INFINITY], &[], device),
                DType::F64 => Tensor::<CudaRuntime>::from_slice(&[f64::INFINITY], &[], device),
                _ => return Err(Error::UnsupportedDType { dtype, op: "cond" }),
            };
            return Ok(inf_val);
        }

        // Compute SVD to get singular values
        let svd = self.svd_decompose(a)?;

        // Get singular values
        let s_data: Vec<f64> = match dtype {
            DType::F32 => svd
                .s
                .to_vec::<f32>()
                .into_iter()
                .map(|x| x as f64)
                .collect(),
            DType::F64 => svd.s.to_vec::<f64>(),
            _ => return Err(Error::UnsupportedDType { dtype, op: "cond" }),
        };

        // Condition number = max(S) / min(S)
        let max_sv = s_data.iter().cloned().fold(0.0_f64, f64::max);
        let min_sv = s_data.iter().cloned().fold(f64::INFINITY, f64::min);

        let cond_val = if min_sv == 0.0 || !min_sv.is_finite() {
            f64::INFINITY
        } else {
            max_sv / min_sv
        };

        // Create result tensor
        let result = match dtype {
            DType::F32 => Tensor::<CudaRuntime>::from_slice(&[cond_val as f32], &[], device),
            DType::F64 => Tensor::<CudaRuntime>::from_slice(&[cond_val], &[], device),
            _ => unreachable!(),
        };

        Ok(result)
    }

    fn cov(&self, a: &Tensor<CudaRuntime>, ddof: Option<usize>) -> Result<Tensor<CudaRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (n_samples, _n_features) = validate_matrix_2d(a.shape())?;
        let dtype = a.dtype();
        let device = self.device();
        let ddof_val = ddof.unwrap_or(1);

        // Need at least ddof + 1 samples
        if n_samples <= ddof_val {
            return Err(Error::Internal(format!(
                "cov: need at least {} samples for ddof={}, got {}",
                ddof_val + 1,
                ddof_val,
                n_samples
            )));
        }

        // Compute mean along axis 0 (mean of each column/feature)
        let sum = TensorOps::sum(self, a, &[0], true)?; // [1, n_features]
        let n_samples_tensor = match dtype {
            DType::F32 => Tensor::<CudaRuntime>::from_slice(&[n_samples as f32], &[], device),
            DType::F64 => Tensor::<CudaRuntime>::from_slice(&[n_samples as f64], &[], device),
            _ => return Err(Error::UnsupportedDType { dtype, op: "cov" }),
        };
        let mean = TensorOps::div(self, &sum, &n_samples_tensor)?; // [1, n_features]

        // Center the data: X_centered = X - mean (broadcast subtraction)
        let centered = TensorOps::sub(self, a, &mean)?; // [n_samples, n_features]

        // Compute covariance: C = X_centered^T @ X_centered / (n - ddof)
        let centered_t = centered.transpose(0, 1)?; // [n_features, n_samples]
        let cov_unnorm = TensorOps::matmul(self, &centered_t, &centered)?; // [n_features, n_features]

        // Normalize by (n - ddof)
        let divisor = (n_samples - ddof_val) as f64;
        let divisor_tensor = match dtype {
            DType::F32 => Tensor::<CudaRuntime>::from_slice(&[divisor as f32], &[], device),
            DType::F64 => Tensor::<CudaRuntime>::from_slice(&[divisor], &[], device),
            _ => unreachable!(),
        };
        let cov_mat = TensorOps::div(self, &cov_unnorm, &divisor_tensor)?;

        Ok(cov_mat)
    }

    fn corrcoef(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (n_samples, n_features) = validate_matrix_2d(a.shape())?;
        let dtype = a.dtype();
        let device = self.device();

        // Need at least 2 samples
        if n_samples < 2 {
            return Err(Error::Internal(format!(
                "corrcoef: need at least 2 samples, got {}",
                n_samples
            )));
        }

        // Handle edge case: no features
        if n_features == 0 {
            return Ok(Tensor::<CudaRuntime>::from_slice::<f32>(
                &[],
                &[0, 0],
                device,
            ));
        }

        // Compute covariance matrix (use LinearAlgebraAlgorithms explicitly to avoid ambiguity)
        let cov_mat = LinearAlgebraAlgorithms::cov(self, a, Some(1))?; // [n_features, n_features]

        // Extract diagonal (variances) and compute standard deviations
        let variances = LinearAlgebraAlgorithms::diag(self, &cov_mat)?; // [n_features]
        let std_devs = self.sqrt(&variances)?; // [n_features]

        // Pull std_devs to CPU for zero-variance detection
        // This ensures exact parity with CPU implementation
        let std_vec: Vec<f64> = match dtype {
            DType::F32 => std_devs
                .to_vec::<f32>()
                .into_iter()
                .map(|x| x as f64)
                .collect(),
            DType::F64 => std_devs.to_vec::<f64>(),
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype,
                    op: "corrcoef",
                });
            }
        };

        // Build correlation matrix with proper zero-variance handling
        // This matches CPU semantics exactly:
        // - diagonal: 1.0 if std > 0, else 0.0
        // - off-diagonal: cov[i,j]/(std[i]*std[j]) if both > 0, else 0.0
        let cov_vec: Vec<f64> = match dtype {
            DType::F32 => cov_mat
                .to_vec::<f32>()
                .into_iter()
                .map(|x| x as f64)
                .collect(),
            DType::F64 => cov_mat.to_vec::<f64>(),
            _ => unreachable!(),
        };

        let mut corr_data: Vec<f64> = vec![0.0; n_features * n_features];
        for i in 0..n_features {
            for j in 0..n_features {
                if i == j {
                    // Diagonal: 1.0 if std > 0, else 0.0
                    corr_data[i * n_features + j] = if std_vec[i] > 0.0 { 1.0 } else { 0.0 };
                } else {
                    // Off-diagonal: correlation if both stds > 0, else 0.0
                    let std_prod = std_vec[i] * std_vec[j];
                    corr_data[i * n_features + j] = if std_prod > 0.0 {
                        // Clamp to [-1, 1] to handle numerical errors
                        (cov_vec[i * n_features + j] / std_prod).clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };
                }
            }
        }

        // Convert back to appropriate dtype and send to GPU
        let result = match dtype {
            DType::F32 => {
                let corr_f32: Vec<f32> = corr_data.iter().map(|&x| x as f32).collect();
                Tensor::<CudaRuntime>::from_slice(&corr_f32, &[n_features, n_features], device)
            }
            DType::F64 => {
                Tensor::<CudaRuntime>::from_slice(&corr_data, &[n_features, n_features], device)
            }
            _ => unreachable!(),
        };

        Ok(result)
    }

    fn eig_decompose_symmetric(
        &self,
        a: &Tensor<CudaRuntime>,
    ) -> Result<EigenDecomposition<CudaRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let n = validate_square_matrix(a.shape())?;
        let dtype = a.dtype();
        let device = self.device();

        // Handle empty matrix
        if n == 0 {
            let eigenvalues_ptr = self.allocator().allocate(0);
            let eigenvectors_ptr = self.allocator().allocate(0);
            let eigenvalues =
                unsafe { Self::tensor_from_raw(eigenvalues_ptr, &[0], dtype, device) };
            let eigenvectors =
                unsafe { Self::tensor_from_raw(eigenvectors_ptr, &[0, 0], dtype, device) };
            return Ok(EigenDecomposition {
                eigenvalues,
                eigenvectors,
            });
        }

        // Handle 1x1 matrix
        if n == 1 {
            let eigenvalues_size = dtype.size_in_bytes();
            let eigenvectors_size = dtype.size_in_bytes();
            let eigenvalues_ptr = self.allocator().allocate(eigenvalues_size);
            let eigenvectors_ptr = self.allocator().allocate(eigenvectors_size);

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
                unsafe { Self::tensor_from_raw(eigenvalues_ptr, &[1], dtype, device) };
            let eigenvectors =
                unsafe { Self::tensor_from_raw(eigenvectors_ptr, &[1, 1], dtype, device) };
            return Ok(EigenDecomposition {
                eigenvalues,
                eigenvectors,
            });
        }

        // Allocate working buffers on GPU
        let work_size = n * n * dtype.size_in_bytes();
        let work_ptr = self.allocator().allocate(work_size);

        let eigenvectors_size = n * n * dtype.size_in_bytes();
        let eigenvectors_ptr = self.allocator().allocate(eigenvectors_size);

        let eigenvalues_size = n * dtype.size_in_bytes();
        let eigenvalues_ptr = self.allocator().allocate(eigenvalues_size);

        let flag_size = std::mem::size_of::<i32>();
        let converged_flag_ptr = self.allocator().allocate(flag_size);

        // Helper for cleanup on error
        let cleanup = |allocator: &super::CudaAllocator| {
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
                self.context(),
                self.stream(),
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
            cleanup(self.allocator());
            return Err(e);
        }

        self.synchronize();

        // Clean up converged flag
        self.allocator().deallocate(converged_flag_ptr, flag_size);

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

        // Read eigenvectors and reorder
        let eigenvalues_final;
        let eigenvectors_final;

        match dtype {
            DType::F32 => {
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
                let eigenvalues_ptr_final = self.allocator().allocate(eigenvalues_size);
                let eigenvectors_ptr_final = self.allocator().allocate(eigenvectors_size);

                // Copy sorted results to GPU
                let eigenvalues_bytes: Vec<u8> = eigenvalues_sorted
                    .iter()
                    .flat_map(|&v| v.to_ne_bytes())
                    .collect();
                CudaRuntime::copy_to_device(&eigenvalues_bytes, eigenvalues_ptr_final, device);

                let eigenvectors_bytes: Vec<u8> =
                    v_sorted.iter().flat_map(|&v| v.to_ne_bytes()).collect();
                CudaRuntime::copy_to_device(&eigenvectors_bytes, eigenvectors_ptr_final, device);

                eigenvalues_final =
                    unsafe { Self::tensor_from_raw(eigenvalues_ptr_final, &[n], dtype, device) };
                eigenvectors_final = unsafe {
                    Self::tensor_from_raw(eigenvectors_ptr_final, &[n, n], dtype, device)
                };
            }
            DType::F64 => {
                // Read eigenvectors
                let mut v_bytes = vec![0u8; n * n * 8];
                CudaRuntime::copy_from_device(eigenvectors_ptr, &mut v_bytes, device);
                let v_data: Vec<f64> = v_bytes
                    .chunks_exact(8)
                    .map(|c| f64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect();

                // Sorted eigenvalues
                let eigenvalues_sorted: Vec<f64> =
                    indices.iter().map(|&idx| eigenvalues_data[idx]).collect();

                // Sorted eigenvector columns
                let mut v_sorted = vec![0.0f64; n * n];
                for (new_idx, &old_idx) in indices.iter().enumerate() {
                    for i in 0..n {
                        v_sorted[i * n + new_idx] = v_data[i * n + old_idx];
                    }
                }

                // Allocate final output
                let eigenvalues_ptr_final = self.allocator().allocate(eigenvalues_size);
                let eigenvectors_ptr_final = self.allocator().allocate(eigenvectors_size);

                // Copy sorted results to GPU
                let eigenvalues_bytes: Vec<u8> = eigenvalues_sorted
                    .iter()
                    .flat_map(|&v| v.to_ne_bytes())
                    .collect();
                CudaRuntime::copy_to_device(&eigenvalues_bytes, eigenvalues_ptr_final, device);

                let eigenvectors_bytes: Vec<u8> =
                    v_sorted.iter().flat_map(|&v| v.to_ne_bytes()).collect();
                CudaRuntime::copy_to_device(&eigenvectors_bytes, eigenvectors_ptr_final, device);

                eigenvalues_final =
                    unsafe { Self::tensor_from_raw(eigenvalues_ptr_final, &[n], dtype, device) };
                eigenvectors_final = unsafe {
                    Self::tensor_from_raw(eigenvectors_ptr_final, &[n, n], dtype, device)
                };
            }
            _ => unreachable!(),
        }

        // Clean up working buffers
        self.allocator().deallocate(work_ptr, work_size);
        self.allocator()
            .deallocate(eigenvectors_ptr, eigenvectors_size);
        self.allocator()
            .deallocate(eigenvalues_ptr, eigenvalues_size);

        Ok(EigenDecomposition {
            eigenvalues: eigenvalues_final,
            eigenvectors: eigenvectors_final,
        })
    }
}

// Helper methods
impl CudaClient {
    /// Create a tensor from a raw CUDA GPU pointer.
    ///
    /// This is a low-level helper function used internally to wrap GPU memory
    /// allocations as tensors. The caller is responsible for ensuring all safety
    /// invariants are met.
    ///
    /// # Safety
    ///
    /// The caller MUST ensure all of the following:
    ///
    /// 1. **Valid GPU Memory**: `ptr` must point to valid CUDA device memory
    ///    allocated via `CudaRuntime::allocate()` or equivalent CUDA allocation.
    ///
    /// 2. **Sufficient Size**: The allocation must contain at least
    ///    `shape.iter().product() * dtype.size_in_bytes()` bytes. For empty
    ///    shapes (scalars), this is `1 * dtype.size_in_bytes()` bytes.
    ///
    /// 3. **Lifetime**: The GPU memory must remain valid for the entire lifetime
    ///    of the returned tensor. The tensor takes ownership of the memory and
    ///    will deallocate it when dropped.
    ///
    /// 4. **No Aliasing**: No other tensor or reference may hold ownership of
    ///    the same memory region. Creating multiple tensors from the same `ptr`
    ///    causes undefined behavior (double-free on drop).
    ///
    /// 5. **Device Affinity**: `ptr` must have been allocated on the same CUDA
    ///    device as specified by `device`. Cross-device pointers are UB.
    ///
    /// # Undefined Behavior
    ///
    /// Violating any of the above safety requirements results in undefined
    /// behavior, which may include:
    /// - Segmentation faults / access violations
    /// - Double-free memory corruption
    /// - Use-after-free bugs
    /// - Silent data corruption
    pub(crate) unsafe fn tensor_from_raw(
        ptr: u64,
        shape: &[usize],
        dtype: DType,
        device: &super::CudaDevice,
    ) -> Tensor<CudaRuntime> {
        let len = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        let storage = unsafe { Storage::<CudaRuntime>::from_ptr(ptr, len, dtype, device) };
        let layout = Layout::contiguous(shape);
        Tensor::from_parts(storage, layout)
    }

    fn qr_decompose_internal(
        &self,
        a: &Tensor<CudaRuntime>,
        thin: bool,
    ) -> Result<QrDecomposition<CudaRuntime>> {
        validate_linalg_dtype(a.dtype())?;
        let (m, n) = validate_matrix_2d(a.shape())?;
        let k = m.min(n);
        let dtype = a.dtype();
        let device = self.device();

        // Q dimensions: [m, m] for full, [m, k] for thin
        let q_cols = if thin { k } else { m };
        let q_size = m * q_cols * dtype.size_in_bytes();
        let q_ptr = self.allocator().allocate(q_size);

        // R is [m, n] but only upper triangular part is meaningful
        let r_size = m * n * dtype.size_in_bytes();
        let r_ptr = self.allocator().allocate(r_size);

        // Workspace for Householder vector (size m elements)
        let workspace_size = m * dtype.size_in_bytes();
        let workspace_ptr = self.allocator().allocate(workspace_size);

        // Copy A to R (will be modified in place)
        CudaRuntime::copy_within_device(a.storage().ptr(), r_ptr, r_size, device);

        let result = unsafe {
            kernels::launch_qr_decompose(
                self.context(),
                self.stream(),
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
        self.allocator().deallocate(workspace_ptr, workspace_size);

        // Handle kernel error after cleanup
        if let Err(e) = result {
            self.allocator().deallocate(q_ptr, q_size);
            self.allocator().deallocate(r_ptr, r_size);
            return Err(e);
        }

        self.synchronize();

        let q = unsafe { Self::tensor_from_raw(q_ptr, &[m, q_cols], dtype, device) };

        // For thin QR, R should be [k, n]
        let r = if thin && m > n {
            // For thin QR with m > n, R is k x n
            unsafe { Self::tensor_from_raw(r_ptr, &[k, n], dtype, device) }
        } else if thin {
            // For thin QR with m <= n, R is m x n
            unsafe { Self::tensor_from_raw(r_ptr, &[m, n], dtype, device) }
        } else {
            // Full QR, R is m x n
            unsafe { Self::tensor_from_raw(r_ptr, &[m, n], dtype, device) }
        };

        Ok(QrDecomposition { q, r })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::runtime::cuda::{CudaDevice, CudaRuntime};

    fn create_client() -> CudaClient {
        let device = CudaDevice::new(0);
        CudaRuntime::default_client(&device)
    }

    #[test]
    fn test_trace() {
        let client = create_client();
        let device = client.device();

        // 2x2 matrix: [[1, 2], [3, 4]]
        // trace = 1 + 4 = 5
        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device);

        let t = LinearAlgebraAlgorithms::trace(&client, &a).unwrap();
        let result: Vec<f32> = t.to_vec();

        assert!((result[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_diag() {
        let client = create_client();
        let device = client.device();

        // 2x3 matrix
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device);

        let d = LinearAlgebraAlgorithms::diag(&client, &a).unwrap();
        let result: Vec<f32> = d.to_vec();

        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_diagflat() {
        let client = create_client();
        let device = client.device();

        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], device);

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
    fn test_lu_decomposition() {
        let client = create_client();
        let device = client.device();

        // 2x2 matrix: [[4, 3], [6, 3]]
        let a = Tensor::<CudaRuntime>::from_slice(&[4.0f32, 3.0, 6.0, 3.0], &[2, 2], device);

        let lu = client.lu_decompose(&a).unwrap();

        assert_eq!(lu.lu.shape(), &[2, 2]);
        assert_eq!(lu.pivots.shape(), &[2]);
    }

    #[test]
    fn test_cholesky() {
        let client = create_client();
        let device = client.device();

        // Symmetric positive definite: [[4, 2], [2, 5]]
        let a = Tensor::<CudaRuntime>::from_slice(&[4.0f32, 2.0, 2.0, 5.0], &[2, 2], device);

        let chol = client.cholesky_decompose(&a).unwrap();

        assert_eq!(chol.l.shape(), &[2, 2]);

        // L should be lower triangular
        let l_data: Vec<f32> = chol.l.to_vec();
        assert!((l_data[1]).abs() < 1e-5); // Upper triangle should be 0
    }

    #[test]
    fn test_det() {
        let client = create_client();
        let device = client.device();

        // 2x2 matrix: [[1, 2], [3, 4]]
        // det = 1*4 - 2*3 = -2
        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device);

        let d = LinearAlgebraAlgorithms::det(&client, &a).unwrap();
        let result: Vec<f32> = d.to_vec();

        assert!((result[0] - (-2.0)).abs() < 1e-4);
    }

    #[test]
    fn test_solve() {
        let client = create_client();
        let device = client.device();

        // Solve [[2, 1], [1, 2]] @ x = [3, 3]
        // Solution: x = [1, 1]
        let a = Tensor::<CudaRuntime>::from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2], device);
        let b = Tensor::<CudaRuntime>::from_slice(&[3.0f32, 3.0], &[2], device);

        let x = LinearAlgebraAlgorithms::solve(&client, &a, &b).unwrap();
        let result: Vec<f32> = x.to_vec();

        assert!((result[0] - 1.0).abs() < 1e-4);
        assert!((result[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_inverse() {
        let client = create_client();
        let device = client.device();

        // Test 2x2 matrix: [[4, 7], [2, 6]]
        // Inverse: [[0.6, -0.7], [-0.2, 0.4]]
        let a = Tensor::<CudaRuntime>::from_slice(&[4.0f32, 7.0, 2.0, 6.0], &[2, 2], device);

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
        use crate::ops::TensorOps;
        let client = create_client();
        let device = client.device();

        // A @ A^-1 should equal I
        let a = Tensor::<CudaRuntime>::from_slice(&[4.0f32, 7.0, 2.0, 6.0], &[2, 2], device);

        let inv = LinearAlgebraAlgorithms::inverse(&client, &a).unwrap();
        let product = TensorOps::matmul(&client, &a, &inv).unwrap();
        let result: Vec<f32> = product.to_vec();

        // Should be identity matrix
        assert!((result[0] - 1.0).abs() < 1e-4); // [0,0]
        assert!((result[1]).abs() < 1e-4); // [0,1]
        assert!((result[2]).abs() < 1e-4); // [1,0]
        assert!((result[3] - 1.0).abs() < 1e-4); // [1,1]
    }

    #[test]
    fn test_matrix_rank_full() {
        let client = create_client();
        let device = client.device();

        // Full rank 2x2 matrix
        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device);

        let rank = LinearAlgebraAlgorithms::matrix_rank(&client, &a, None).unwrap();
        let result: Vec<i64> = rank.to_vec();

        assert_eq!(result[0], 2);
    }

    #[test]
    fn test_matrix_rank_deficient() {
        let client = create_client();
        let device = client.device();

        // Rank-deficient 2x2 matrix (rows are linearly dependent)
        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 4.0], &[2, 2], device);

        let rank = LinearAlgebraAlgorithms::matrix_rank(&client, &a, None).unwrap();
        let result: Vec<i64> = rank.to_vec();

        assert_eq!(result[0], 1);
    }

    #[test]
    fn test_qr_decomposition() {
        use crate::ops::TensorOps;
        let client = create_client();
        let device = client.device();

        // Test QR: A = Q @ R
        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device);

        let qr = client.qr_decompose(&a).unwrap();

        // Verify Q @ R == A
        let reconstructed = TensorOps::matmul(&client, &qr.q, &qr.r).unwrap();
        let a_data: Vec<f32> = a.to_vec();
        let reconstructed_data: Vec<f32> = reconstructed.to_vec();

        for i in 0..4 {
            assert!(
                (a_data[i] - reconstructed_data[i]).abs() < 1e-4,
                "Mismatch at {}: {} vs {}",
                i,
                a_data[i],
                reconstructed_data[i]
            );
        }
    }

    #[test]
    fn test_solve_multi_rhs() {
        let client = create_client();
        let device = client.device();

        // Solve A @ X = B where B has multiple columns
        // A = [[2, 1], [1, 2]], B = [[3, 4], [3, 5]]
        // Solutions: X[:, 0] = [1, 1], X[:, 1] = [1, 2]
        let a = Tensor::<CudaRuntime>::from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2], device);
        let b = Tensor::<CudaRuntime>::from_slice(&[3.0f32, 4.0, 3.0, 5.0], &[2, 2], device);

        let x = LinearAlgebraAlgorithms::solve(&client, &a, &b).unwrap();
        assert_eq!(x.shape(), &[2, 2]);
        let result: Vec<f32> = x.to_vec();

        // X[:, 0] = [1, 1] -> result[0], result[2]
        // X[:, 1] = [1, 2] -> result[1], result[3]
        // Row-major: X[0,0]=result[0], X[0,1]=result[1], X[1,0]=result[2], X[1,1]=result[3]
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

    #[test]
    fn test_lstsq_overdetermined() {
        let client = create_client();
        let device = client.device();

        // Overdetermined system: A is 3x2, b is 3x1
        // A = [[1, 1], [1, 2], [1, 3]], b = [1, 2, 3]
        // Least squares solution minimizes ||Ax - b||^2
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 2.0, 1.0, 3.0], &[3, 2], device);
        let b = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], device);

        let x = LinearAlgebraAlgorithms::lstsq(&client, &a, &b).unwrap();
        assert_eq!(x.shape(), &[2]);
        let result: Vec<f32> = x.to_vec();

        // For this system, the solution is approximately x = [0, 1]
        // (regression line through points (1,1), (2,2), (3,3))
        assert!((result[0]).abs() < 0.1, "x[0] = {} expected ~0", result[0]);
        assert!(
            (result[1] - 1.0).abs() < 0.1,
            "x[1] = {} expected ~1",
            result[1]
        );
    }

    #[test]
    fn test_lstsq_multi_rhs() {
        let client = create_client();
        let device = client.device();

        // Overdetermined system with multiple RHS
        // A is 3x2, B is 3x2
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 2.0, 1.0, 3.0], &[3, 2], device);
        // B = [[1, 2], [2, 4], [3, 6]] (second column is 2x first)
        let b =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 4.0, 3.0, 6.0], &[3, 2], device);

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
