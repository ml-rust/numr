//! Sparse element-wise merges (add, sub, mul, div) for WebGPU.
//!
//! CSR and CSC run the same three GPU passes:
//! 1. Count output nonzeros per row (CSR) or column (CSC).
//! 2. Exclusive scan of the counts into output pointers.
//! 3. Merge the sorted index lists and apply the operation.
//!
//! COO converts both operands to CSR on the GPU, merges, and expands the
//! result back to COO. Inputs and outputs carry I64 indices. The kernels see
//! I32 copies made on the GPU.

use super::super::ops::helpers::get_tensor_buffer;
use super::super::shaders::{
    CompressedBuffers, CompressedFormat, SparseMergeOp, launch_sparse_merge_compute,
    launch_sparse_merge_count,
};
use super::super::{WgpuClient, WgpuRuntime};
use super::indices::{
    bindable, check_value_dtype, indices_to_i32, indices_to_i64, scan_counts, uniform_words,
};
use crate::algorithm::sparse::validate_sparse_operands;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Borrowed `(pointers or row indices, indices, values)` of one sparse operand.
pub(super) type SparseParts<'a> = (
    &'a Tensor<WgpuRuntime>,
    &'a Tensor<WgpuRuntime>,
    &'a Tensor<WgpuRuntime>,
);

/// Owned `(pointers or row indices, indices, values)` of a sparse result.
pub(super) type SparseTriple = (
    Tensor<WgpuRuntime>,
    Tensor<WgpuRuntime>,
    Tensor<WgpuRuntime>,
);

impl WgpuClient {
    /// Merge two CSR (`format = Csr`) or CSC (`format = Csc`) matrices with
    /// `n` rows or columns. Returns I64 pointers, I64 indices and the values.
    ///
    /// Add and sub keep the union of both patterns. Mul and div keep the
    /// intersection. Explicit zeros in the result stay stored.
    pub(super) fn compressed_merge<T: Element>(
        &self,
        format: CompressedFormat,
        op: SparseMergeOp,
        a: SparseParts<'_>,
        b: SparseParts<'_>,
        n: usize,
    ) -> Result<SparseTriple> {
        let (a_ptrs, a_indices, a_values) = a;
        let (b_ptrs, b_indices, b_values) = b;
        let dtype = a_values.dtype();
        let name = merge_name(format, op);
        check_value_dtype(dtype, name)?;
        // The check reads dtype metadata on the host. No data moves.
        validate_sparse_operands::<T>(&[], &[dtype, b_values.dtype()])?;
        if op == SparseMergeOp::Div && b_values.numel() == 0 && a_values.numel() > 0 {
            return Err(Error::Internal(format!(
                "{name}: B stores no entries, so each of the {} stored A entries divides by zero",
                a_values.numel()
            )));
        }

        let a_ptrs = indices_to_i32(self, a_ptrs, name)?;
        let a_indices = bindable(self, &indices_to_i32(self, a_indices, name)?)?;
        let a_values = bindable(self, a_values)?;
        let b_ptrs = indices_to_i32(self, b_ptrs, name)?;
        let b_indices = bindable(self, &indices_to_i32(self, b_indices, name)?)?;
        let b_values = bindable(self, b_values)?;

        let a_ptrs_buf = get_tensor_buffer(&a_ptrs)?;
        let a_indices_buf = get_tensor_buffer(&a_indices)?;
        let a_values_buf = get_tensor_buffer(&a_values)?;
        let b_ptrs_buf = get_tensor_buffer(&b_ptrs)?;
        let b_indices_buf = get_tensor_buffer(&b_indices)?;
        let b_values_buf = get_tensor_buffer(&b_values)?;
        let params = uniform_words(self, "sparse_merge_params", [n as u32, 0, 0, 0]);

        let counts = bindable(
            self,
            &Tensor::<WgpuRuntime>::zeros(&[n], DType::I32, &self.device_id)?,
        )?;
        if n > 0 {
            let counts_buf = get_tensor_buffer(&counts)?;
            launch_sparse_merge_count(
                self.pipeline_cache(),
                self.wgpu_queue(),
                format,
                op,
                &a_ptrs_buf,
                &a_indices_buf,
                &b_ptrs_buf,
                &b_indices_buf,
                &counts_buf,
                &params,
                n,
            )?;
        }
        let (out_ptrs, total) = scan_counts(self, &counts, n)?;

        let out_indices = Tensor::<WgpuRuntime>::zeros(&[total], DType::I32, &self.device_id)?;
        let out_values = Tensor::<WgpuRuntime>::zeros(&[total], dtype, &self.device_id)?;
        if total > 0 {
            let out_ptrs_buf = get_tensor_buffer(&out_ptrs)?;
            let out_indices_buf = get_tensor_buffer(&out_indices)?;
            let out_values_buf = get_tensor_buffer(&out_values)?;
            launch_sparse_merge_compute(
                self.pipeline_cache(),
                self.wgpu_queue(),
                format,
                op,
                &CompressedBuffers {
                    ptrs: &a_ptrs_buf,
                    indices: &a_indices_buf,
                    values: &a_values_buf,
                },
                &CompressedBuffers {
                    ptrs: &b_ptrs_buf,
                    indices: &b_indices_buf,
                    values: &b_values_buf,
                },
                &out_ptrs_buf,
                &out_indices_buf,
                &out_values_buf,
                &params,
                n,
                dtype,
            )?;
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
            CompressedFormat::Csr,
            op,
            (&a_ptrs, &a_cols, &a_values),
            (&b_ptrs, &b_cols, &b_values),
            shape[0],
        )?;
        self.csr_to_coo_impl::<T>(&ptrs, &cols, &values, shape)
    }
}

fn merge_name(format: CompressedFormat, op: SparseMergeOp) -> &'static str {
    match (format, op) {
        (CompressedFormat::Csr, SparseMergeOp::Add) => "add_csr (WebGPU)",
        (CompressedFormat::Csr, SparseMergeOp::Sub) => "sub_csr (WebGPU)",
        (CompressedFormat::Csr, SparseMergeOp::Mul) => "mul_csr (WebGPU)",
        (CompressedFormat::Csr, SparseMergeOp::Div) => "div_csr (WebGPU)",
        (CompressedFormat::Csc, SparseMergeOp::Add) => "add_csc (WebGPU)",
        (CompressedFormat::Csc, SparseMergeOp::Sub) => "sub_csc (WebGPU)",
        (CompressedFormat::Csc, SparseMergeOp::Mul) => "mul_csc (WebGPU)",
        (CompressedFormat::Csc, SparseMergeOp::Div) => "div_csc (WebGPU)",
    }
}
