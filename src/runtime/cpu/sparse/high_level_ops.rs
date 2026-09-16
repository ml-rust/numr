//! SparseOps trait implementation for CPU runtime
//!
//! Thin delegation layer — actual implementations live in:
//! - `format_convert.rs` — COO↔CSR↔CSC format conversions
//! - `sparse_highlevel.rs` — high-level ops, reductions, dense conversion

use super::format_convert;
use super::merge::{
    MergeStrategy, OperationSemantics, intersect_coo_impl, merge_coo_impl, merge_csc_impl,
    merge_csr_impl,
};
use super::sparse_highlevel;
use super::spmv::{spmm_csr_impl, spmv_csr_impl};
use super::{CpuClient, CpuRuntime};
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::sparse::SparseOps;
use crate::tensor::Tensor;

impl SparseOps<CpuRuntime> for CpuClient {
    // =========================================================================
    // CSR low-level operations
    // =========================================================================

    fn spmv_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CpuRuntime>> {
        spmv_csr_impl::<T>(self, row_ptrs, col_indices, values, x, shape)
    }

    fn spmm_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CpuRuntime>> {
        spmm_csr_impl::<T>(self, row_ptrs, col_indices, values, b, shape)
    }

    fn add_csr<T: Element>(
        &self,
        a_rp: &Tensor<CpuRuntime>,
        a_ci: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_rp: &Tensor<CpuRuntime>,
        b_ci: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csr_impl::<T, _, _, _>(
            a_rp,
            a_ci,
            a_v,
            b_rp,
            b_ci,
            b_v,
            shape,
            MergeStrategy::Union,
            OperationSemantics::Add,
            |a, b| T::from_f64(a.to_f64() + b.to_f64()),
            |a| a,
            |b| b,
        )
    }

    fn sub_csr<T: Element>(
        &self,
        a_rp: &Tensor<CpuRuntime>,
        a_ci: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_rp: &Tensor<CpuRuntime>,
        b_ci: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csr_impl::<T, _, _, _>(
            a_rp,
            a_ci,
            a_v,
            b_rp,
            b_ci,
            b_v,
            shape,
            MergeStrategy::Union,
            OperationSemantics::Subtract,
            |a, b| T::from_f64(a.to_f64() - b.to_f64()),
            |a| a,
            |b| T::from_f64(-b.to_f64()),
        )
    }

    fn mul_csr<T: Element>(
        &self,
        a_rp: &Tensor<CpuRuntime>,
        a_ci: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_rp: &Tensor<CpuRuntime>,
        b_ci: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csr_impl::<T, _, _, _>(
            a_rp,
            a_ci,
            a_v,
            b_rp,
            b_ci,
            b_v,
            shape,
            MergeStrategy::Intersection,
            OperationSemantics::Multiply,
            |a, b| T::from_f64(a.to_f64() * b.to_f64()),
            |a| a,
            |b| b,
        )
    }

    fn div_csr<T: Element>(
        &self,
        a_rp: &Tensor<CpuRuntime>,
        a_ci: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_rp: &Tensor<CpuRuntime>,
        b_ci: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csr_impl::<T, _, _, _>(
            a_rp,
            a_ci,
            a_v,
            b_rp,
            b_ci,
            b_v,
            shape,
            MergeStrategy::Intersection,
            OperationSemantics::Divide,
            |a, b| T::from_f64(a.to_f64() / b.to_f64()),
            |a| a,
            |b| b,
        )
    }

    // =========================================================================
    // CSC low-level operations
    // =========================================================================

    fn add_csc<T: Element>(
        &self,
        a_cp: &Tensor<CpuRuntime>,
        a_ri: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_cp: &Tensor<CpuRuntime>,
        b_ri: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csc_impl::<T, _, _, _>(
            a_cp,
            a_ri,
            a_v,
            b_cp,
            b_ri,
            b_v,
            shape,
            MergeStrategy::Union,
            OperationSemantics::Add,
            |a, b| T::from_f64(a.to_f64() + b.to_f64()),
            |a| a,
            |b| b,
        )
    }

    fn sub_csc<T: Element>(
        &self,
        a_cp: &Tensor<CpuRuntime>,
        a_ri: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_cp: &Tensor<CpuRuntime>,
        b_ri: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csc_impl::<T, _, _, _>(
            a_cp,
            a_ri,
            a_v,
            b_cp,
            b_ri,
            b_v,
            shape,
            MergeStrategy::Union,
            OperationSemantics::Subtract,
            |a, b| T::from_f64(a.to_f64() - b.to_f64()),
            |a| a,
            |b| T::from_f64(-b.to_f64()),
        )
    }

    fn mul_csc<T: Element>(
        &self,
        a_cp: &Tensor<CpuRuntime>,
        a_ri: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_cp: &Tensor<CpuRuntime>,
        b_ri: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csc_impl::<T, _, _, _>(
            a_cp,
            a_ri,
            a_v,
            b_cp,
            b_ri,
            b_v,
            shape,
            MergeStrategy::Intersection,
            OperationSemantics::Multiply,
            |a, b| T::from_f64(a.to_f64() * b.to_f64()),
            |a| a,
            |b| b,
        )
    }

    fn div_csc<T: Element>(
        &self,
        a_cp: &Tensor<CpuRuntime>,
        a_ri: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_cp: &Tensor<CpuRuntime>,
        b_ri: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csc_impl::<T, _, _, _>(
            a_cp,
            a_ri,
            a_v,
            b_cp,
            b_ri,
            b_v,
            shape,
            MergeStrategy::Intersection,
            OperationSemantics::Divide,
            |a, b| T::from_f64(a.to_f64() / b.to_f64()),
            |a| a,
            |b| b,
        )
    }

    // =========================================================================
    // COO low-level operations
    // =========================================================================

    fn add_coo<T: Element>(
        &self,
        a_ri: &Tensor<CpuRuntime>,
        a_ci: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_ri: &Tensor<CpuRuntime>,
        b_ci: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        _shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_coo_impl::<T, _, _, _>(
            a_ri,
            a_ci,
            a_v,
            b_ri,
            b_ci,
            b_v,
            OperationSemantics::Add,
            |a, b| T::from_f64(a.to_f64() + b.to_f64()),
            |a| a,
            |b| b,
        )
    }

    fn sub_coo<T: Element>(
        &self,
        a_ri: &Tensor<CpuRuntime>,
        a_ci: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_ri: &Tensor<CpuRuntime>,
        b_ci: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        _shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_coo_impl::<T, _, _, _>(
            a_ri,
            a_ci,
            a_v,
            b_ri,
            b_ci,
            b_v,
            OperationSemantics::Subtract,
            |a, b| T::from_f64(a.to_f64() - b.to_f64()),
            |a| a,
            |b| T::from_f64(-b.to_f64()),
        )
    }

    fn mul_coo<T: Element>(
        &self,
        a_ri: &Tensor<CpuRuntime>,
        a_ci: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_ri: &Tensor<CpuRuntime>,
        b_ci: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        _shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        intersect_coo_impl::<T, _>(a_ri, a_ci, a_v, b_ri, b_ci, b_v, |a, b| {
            T::from_f64(a.to_f64() * b.to_f64())
        })
    }

    fn div_coo<T: Element>(
        &self,
        a_ri: &Tensor<CpuRuntime>,
        a_ci: &Tensor<CpuRuntime>,
        a_v: &Tensor<CpuRuntime>,
        b_ri: &Tensor<CpuRuntime>,
        b_ci: &Tensor<CpuRuntime>,
        b_v: &Tensor<CpuRuntime>,
        _shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        intersect_coo_impl::<T, _>(a_ri, a_ci, a_v, b_ri, b_ci, b_v, |a, b| {
            T::from_f64(a.to_f64() / b.to_f64())
        })
    }

    // =========================================================================
    // High-level operations — delegate to sparse_highlevel module
    // =========================================================================

    fn spmv(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::spmv(self, a, x)
    }

    fn spmm(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::spmm(self, a, b)
    }

    fn dsmm(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::dsmm(self, a, b)
    }

    fn sparse_add(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        b: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        sparse_highlevel::sparse_add(a, b)
    }

    fn sparse_sub(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        b: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        sparse_highlevel::sparse_sub(a, b)
    }

    fn sparse_matmul(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        b: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        sparse_highlevel::sparse_matmul(self, a, b)
    }

    fn sparse_mul(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        b: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        sparse_highlevel::sparse_mul(a, b)
    }

    fn sparse_scale(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        scalar: f64,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        sparse_highlevel::sparse_scale(self, a, scalar)
    }

    fn sparse_add_scalar(
        &self,
        _a: &crate::sparse::SparseTensor<CpuRuntime>,
        _scalar: f64,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        Err(Error::Internal(
            "Scalar addition to sparse matrix creates dense result - convert to dense first"
                .to_string(),
        ))
    }

    fn sparse_sum(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::sparse_sum(a)
    }

    fn sparse_sum_rows(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::sparse_sum_rows(a)
    }

    fn sparse_sum_cols(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::sparse_sum_cols(a)
    }

    fn sparse_nnz_per_row(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::sparse_nnz_per_row(a)
    }

    fn sparse_nnz_per_col(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::sparse_nnz_per_col(a)
    }

    fn sparse_to_dense(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sparse_highlevel::sparse_to_dense(a)
    }

    fn dense_to_sparse(
        &self,
        a: &Tensor<CpuRuntime>,
        threshold: f64,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        sparse_highlevel::dense_to_sparse(a, threshold)
    }

    // =========================================================================
    // Format conversions — delegate to format_convert module
    // =========================================================================

    fn coo_to_csr<T: Element>(
        &self,
        row_indices: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        format_convert::coo_to_csr::<T>(row_indices, col_indices, values, shape)
    }

    fn coo_to_csc<T: Element>(
        &self,
        row_indices: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        format_convert::coo_to_csc::<T>(row_indices, col_indices, values, shape)
    }

    fn csr_to_coo<T: Element>(
        &self,
        row_ptrs: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        format_convert::csr_to_coo::<T>(row_ptrs, col_indices, values, shape)
    }

    fn csc_to_coo<T: Element>(
        &self,
        col_ptrs: &Tensor<CpuRuntime>,
        row_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        format_convert::csc_to_coo::<T>(col_ptrs, row_indices, values, shape)
    }

    fn csr_to_csc<T: Element>(
        &self,
        row_ptrs: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        format_convert::csr_to_csc::<T>(row_ptrs, col_indices, values, shape)
    }

    fn csc_to_csr<T: Element>(
        &self,
        col_ptrs: &Tensor<CpuRuntime>,
        row_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        format_convert::csc_to_csr::<T>(col_ptrs, row_indices, values, shape)
    }

    fn extract_diagonal_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CpuRuntime>> {
        format_convert::extract_diagonal_csr::<T>(row_ptrs, col_indices, values, shape)
    }

    fn sparse_transpose(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        sparse_highlevel::sparse_transpose(a)
    }
}
