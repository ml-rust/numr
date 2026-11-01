//! Sparse element-wise merges (add, sub, mul, div) for CUDA.
//!
//! CSR and CSC run the same three GPU passes:
//! 1. Count output entries per row (CSR) or column (CSC).
//! 2. Exclusive scan of the counts into output pointers.
//! 3. Merge the sorted index lists and apply the operation.
//!
//! COO converts both operands to CSR on the GPU, merges, and expands the
//! result back to COO. Inputs and outputs carry I64 indices. The merge
//! kernels read `int`, so they see I32 copies made on the GPU.

use super::super::{CudaClient, CudaRuntime};
use super::indices::{check_merge_dtype, indices_to_i32, indices_to_i64};
use crate::algorithm::sparse::validate_sparse_operands;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::{
    CompressedPtrs, SparseFormat, SparseMergeOp, exclusive_scan_i32_gpu,
    launch_compressed_merge_compute, launch_compressed_merge_count,
};
use crate::tensor::Tensor;

/// Borrowed `(pointers or row indices, indices, values)` of one sparse operand.
pub(super) type SparseParts<'a> = (
    &'a Tensor<CudaRuntime>,
    &'a Tensor<CudaRuntime>,
    &'a Tensor<CudaRuntime>,
);

/// Owned `(pointers or row indices, indices, values)` of a sparse result.
pub(super) type SparseTriple = (
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
);

impl CudaClient {
    /// Merge two CSR (`format = Csr`) or CSC (`format = Csc`) matrices with
    /// `n` rows or columns. Returns I64 pointers, I64 indices and the values.
    ///
    /// Add and sub keep the union of both patterns. Mul and div keep the
    /// intersection. Explicit zeros in the result stay stored.
    pub(super) fn compressed_merge<T: Element>(
        &self,
        format: SparseFormat,
        op: SparseMergeOp,
        a: SparseParts<'_>,
        b: SparseParts<'_>,
        n: usize,
    ) -> Result<SparseTriple> {
        let name = merge_name(format, op);
        let dtype = a.2.dtype();
        check_merge_dtype(dtype, name)?;
        // The check reads dtype metadata on the host. No data moves.
        validate_sparse_operands::<T>(&[], &[dtype, b.2.dtype()])?;
        if op == SparseMergeOp::Div && b.2.numel() == 0 && a.2.numel() > 0 {
            return Err(Error::Internal(format!(
                "{name}: B stores no entries, so each of the {} stored A entries divides by zero",
                a.2.numel()
            )));
        }
        let device = a.2.device();

        let a_ptrs = indices_to_i32(self, a.0, name)?;
        let a_indices = indices_to_i32(self, a.1, name)?;
        let a_values = a.2.contiguous()?;
        let b_ptrs = indices_to_i32(self, b.0, name)?;
        let b_indices = indices_to_i32(self, b.1, name)?;
        let b_values = b.2.contiguous()?;
        let a_dev = CompressedPtrs {
            ptrs: a_ptrs.ptr(),
            indices: a_indices.ptr(),
            values: a_values.ptr(),
        };
        let b_dev = CompressedPtrs {
            ptrs: b_ptrs.ptr(),
            indices: b_indices.ptr(),
            values: b_values.ptr(),
        };

        let counts = Tensor::<CudaRuntime>::zeros(&[n], DType::I32, device)?;
        if n > 0 {
            // SAFETY: both operands hold n + 1 pointers and their indices,
            // and `counts` holds n entries, all live on this device.
            unsafe {
                launch_compressed_merge_count(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    format,
                    op,
                    &a_dev,
                    &b_dev,
                    counts.ptr(),
                    n,
                )?;
            }
        }
        let (out_ptrs, total) = if n == 0 {
            (Tensor::<CudaRuntime>::zeros(&[1], DType::I32, device)?, 0)
        } else {
            // SAFETY: `counts` is a live I32 device tensor of n entries.
            unsafe {
                exclusive_scan_i32_gpu(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    &counts,
                )?
            }
        };

        let out_indices = Tensor::<CudaRuntime>::zeros(&[total], DType::I32, device)?;
        let out_values = Tensor::<CudaRuntime>::zeros(&[total], dtype, device)?;
        if total > 0 {
            // SAFETY: `out_ptrs` is the scan of the counts, and both outputs
            // hold its total.
            unsafe {
                launch_compressed_merge_compute(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    format,
                    op,
                    dtype,
                    &a_dev,
                    &b_dev,
                    out_ptrs.ptr(),
                    out_indices.ptr(),
                    out_values.ptr(),
                    n,
                )?;
            }
        }

        Ok((
            indices_to_i64(self, &out_ptrs)?,
            indices_to_i64(self, &out_indices)?,
            out_values,
        ))
    }

    /// Merge two COO matrices of `shape`. Both operands convert to CSR on the
    /// GPU. The result is COO in row-major order with I64 indices.
    pub(super) fn coo_merge<T: Element>(
        &self,
        op: SparseMergeOp,
        a: SparseParts<'_>,
        b: SparseParts<'_>,
        shape: [usize; 2],
    ) -> Result<SparseTriple> {
        let (a_ptrs, a_cols, a_values) = self.coo_to_csr_impl::<T>(a.0, a.1, a.2, shape)?;
        let (b_ptrs, b_cols, b_values) = self.coo_to_csr_impl::<T>(b.0, b.1, b.2, shape)?;
        let (ptrs, cols, values) = self.compressed_merge::<T>(
            SparseFormat::Csr,
            op,
            (&a_ptrs, &a_cols, &a_values),
            (&b_ptrs, &b_cols, &b_values),
            shape[0],
        )?;
        self.csr_to_coo_impl::<T>(&ptrs, &cols, &values, shape)
    }
}

fn merge_name(format: SparseFormat, op: SparseMergeOp) -> &'static str {
    match (format, op) {
        (SparseFormat::Csr, SparseMergeOp::Add) => "add_csr (CUDA)",
        (SparseFormat::Csr, SparseMergeOp::Sub) => "sub_csr (CUDA)",
        (SparseFormat::Csr, SparseMergeOp::Mul) => "mul_csr (CUDA)",
        (SparseFormat::Csr, SparseMergeOp::Div) => "div_csr (CUDA)",
        (SparseFormat::Csc, SparseMergeOp::Add) => "add_csc (CUDA)",
        (SparseFormat::Csc, SparseMergeOp::Sub) => "sub_csc (CUDA)",
        (SparseFormat::Csc, SparseMergeOp::Mul) => "mul_csc (CUDA)",
        (SparseFormat::Csc, SparseMergeOp::Div) => "div_csc (CUDA)",
        (SparseFormat::Coo, _) => "sparse merge (CUDA)",
    }
}
