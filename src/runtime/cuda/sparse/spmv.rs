//! CSR SpMV, SpMM and diagonal extraction for CUDA.
//!
//! The SpMV and SpMM kernels read `int` indices: I64 indices are cast to I32
//! on the GPU first. The diagonal kernel reads I64 indices directly.

use super::super::{CudaClient, CudaRuntime};
use super::indices::{check_float_dtype, indices_to_i32};
use crate::algorithm::sparse::validate_sparse_operands;
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::{
    launch_csr_extract_diagonal, launch_csr_spmm, launch_csr_spmv, launch_csr_spmv_warp,
    should_use_warp_kernel,
};
use crate::tensor::Tensor;

impl CudaClient {
    /// `y = A * x` for CSR `A` of `shape` and dense `x` of length `shape[1]`.
    pub(crate) fn spmv_csr_impl<T: Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CudaRuntime>> {
        let op = "spmv_csr (CUDA)";
        let [nrows, ncols] = shape;
        let dtype = values.dtype();
        check_float_dtype(dtype, op)?;
        // The check reads dtype metadata on the host. No data moves.
        validate_sparse_operands::<T>(&[], &[dtype, x.dtype()])?;
        if x.numel() != ncols {
            return Err(Error::ShapeMismatch {
                expected: vec![ncols],
                got: vec![x.numel()],
            });
        }

        let y = Tensor::<CudaRuntime>::zeros(&[nrows], dtype, values.device())?;
        if nrows == 0 {
            return Ok(y);
        }
        let row_ptrs = indices_to_i32(self, row_ptrs, op)?;
        let col_indices = indices_to_i32(self, col_indices, op)?;
        let values = values.contiguous()?;
        let x = x.contiguous()?;

        let avg_nnz_per_row = values.numel() as f32 / nrows as f32;
        let launch = if should_use_warp_kernel(avg_nnz_per_row) {
            launch_csr_spmv_warp
        } else {
            launch_csr_spmv
        };
        // SAFETY: every pointer is a live device buffer of the length the
        // kernel reads: nrows + 1 pointers, nnz indices and values, ncols x.
        unsafe {
            launch(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                row_ptrs.ptr(),
                col_indices.ptr(),
                values.ptr(),
                x.ptr(),
                y.ptr(),
                nrows,
            )?;
        }
        Ok(y)
    }

    /// `C = A * B` for CSR `A` of `shape` and dense row-major `B` of
    /// `[shape[1], n]`.
    pub(crate) fn spmm_csr_impl<T: Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CudaRuntime>> {
        let op = "spmm_csr (CUDA)";
        let [m, k] = shape;
        let dtype = values.dtype();
        check_float_dtype(dtype, op)?;
        // The check reads dtype metadata on the host. No data moves.
        validate_sparse_operands::<T>(&[], &[dtype, b.dtype()])?;
        if b.ndim() != 2 {
            return Err(Error::Internal(format!(
                "{op}: B must be 2D, got shape {:?}",
                b.shape()
            )));
        }
        let (b_k, n) = (b.shape()[0], b.shape()[1]);
        if b_k != k {
            return Err(Error::ShapeMismatch {
                expected: vec![k, n],
                got: b.shape().to_vec(),
            });
        }

        let c = Tensor::<CudaRuntime>::zeros(&[m, n], dtype, values.device())?;
        if m == 0 || n == 0 {
            return Ok(c);
        }
        let row_ptrs = indices_to_i32(self, row_ptrs, op)?;
        let col_indices = indices_to_i32(self, col_indices, op)?;
        let values = values.contiguous()?;
        let b = b.contiguous()?;
        // SAFETY: every pointer is a live device buffer of the length the
        // kernel reads: m + 1 pointers, nnz indices and values, k x n B, m x n C.
        unsafe {
            launch_csr_spmm(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                row_ptrs.ptr(),
                col_indices.ptr(),
                values.ptr(),
                b.ptr(),
                c.ptr(),
                m,
                n,
            )?;
        }
        Ok(c)
    }

    /// Diagonal of a CSR matrix: `diag[i] = A[i, i]` for `i < min(nrows, ncols)`.
    /// A missing diagonal entry reads as zero.
    pub(crate) fn extract_diagonal_csr_impl<T: Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CudaRuntime>> {
        let op = "extract_diagonal_csr (CUDA)";
        let n = shape[0].min(shape[1]);
        let dtype = values.dtype();
        check_float_dtype(dtype, op)?;
        validate_sparse_operands::<T>(&[row_ptrs.dtype(), col_indices.dtype()], &[dtype])?;

        let diag = Tensor::<CudaRuntime>::zeros(&[n], dtype, values.device())?;
        if n == 0 || values.numel() == 0 {
            return Ok(diag);
        }
        let row_ptrs = row_ptrs.contiguous()?;
        let col_indices = col_indices.contiguous()?;
        let values = values.contiguous()?;
        // SAFETY: I64 pointers and indices, nnz values and an n-element output,
        // all live device buffers.
        unsafe {
            crate::dispatch_dtype!(dtype, V => {
                launch_csr_extract_diagonal::<V>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    row_ptrs.ptr(),
                    col_indices.ptr(),
                    values.ptr(),
                    diag.ptr(),
                    n,
                )
            }, op)?;
        }
        Ok(diag)
    }
}
