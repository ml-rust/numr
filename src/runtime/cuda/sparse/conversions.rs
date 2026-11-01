//! Sparse format conversions for CUDA: COO, CSR and CSC to one another.
//!
//! Every conversion runs on the GPU with I64 indices.
//!
//! - **CSR/CSC → COO**: expand the pointers into one index per entry.
//! - **COO → CSR/CSC**: stable sort of the entry ids by the composite key
//!   `primary * extent + secondary`, gather through the sorted ids, then build
//!   the pointers from the sorted primary indices.
//! - **CSR ↔ CSC**: expand the pointers, then the same sort.
//!
//! The output order matches the CPU backend exactly: sorted by (primary,
//! secondary), and duplicates keep their input order.

use super::super::{CudaClient, CudaRuntime};
use super::indices::indices_to_i64;
use super::merge::SparseTriple;
use crate::algorithm::sparse::validate_sparse_operands;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::{
    launch_build_ptrs_from_sorted, launch_coo_compute_keys, launch_coo_gather,
    launch_coo_init_indices, launch_expand_ptrs, launch_thrust_sort_pairs_i64_i32,
};
use crate::tensor::Tensor;

impl CudaClient {
    /// CSR → COO. Returns I64 row indices, I64 column indices and the values.
    pub(crate) fn csr_to_coo_impl<T: Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<SparseTriple> {
        validate_sparse_operands::<T>(&[], &[values.dtype()])?;
        let rows = self.expand_ptrs(row_ptrs, shape[0], values.numel())?;
        Ok((rows, indices_to_i64(self, col_indices)?, values.clone()))
    }

    /// CSC → COO. Returns I64 row indices, I64 column indices and the values.
    pub(crate) fn csc_to_coo_impl<T: Element>(
        &self,
        col_ptrs: &Tensor<CudaRuntime>,
        row_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<SparseTriple> {
        validate_sparse_operands::<T>(&[], &[values.dtype()])?;
        let cols = self.expand_ptrs(col_ptrs, shape[1], values.numel())?;
        Ok((indices_to_i64(self, row_indices)?, cols, values.clone()))
    }

    /// COO → CSR, sorted by (row, column).
    pub(crate) fn coo_to_csr_impl<T: Element>(
        &self,
        row_indices: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<SparseTriple> {
        validate_sparse_operands::<T>(&[], &[values.dtype()])?;
        self.bucket_sort(row_indices, col_indices, values, shape[1], shape[0])
    }

    /// COO → CSC, sorted by (column, row).
    pub(crate) fn coo_to_csc_impl<T: Element>(
        &self,
        row_indices: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<SparseTriple> {
        validate_sparse_operands::<T>(&[], &[values.dtype()])?;
        self.bucket_sort(col_indices, row_indices, values, shape[0], shape[1])
    }

    /// CSR → CSC. Returns I64 column pointers, I64 row indices and the values.
    pub(crate) fn csr_to_csc_impl<T: Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<SparseTriple> {
        validate_sparse_operands::<T>(&[], &[values.dtype()])?;
        let rows = self.expand_ptrs(row_ptrs, shape[0], values.numel())?;
        self.bucket_sort(col_indices, &rows, values, shape[0], shape[1])
    }

    /// CSC → CSR. Returns I64 row pointers, I64 column indices and the values.
    pub(crate) fn csc_to_csr_impl<T: Element>(
        &self,
        col_ptrs: &Tensor<CudaRuntime>,
        row_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<SparseTriple> {
        validate_sparse_operands::<T>(&[], &[values.dtype()])?;
        let cols = self.expand_ptrs(col_ptrs, shape[1], values.numel())?;
        self.bucket_sort(row_indices, &cols, values, shape[1], shape[0])
    }

    /// Expand `n + 1` pointers into `nnz` I64 bucket indices.
    fn expand_ptrs(
        &self,
        ptrs: &Tensor<CudaRuntime>,
        n: usize,
        nnz: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        let device = ptrs.device();
        let out = Tensor::<CudaRuntime>::zeros(&[nnz], DType::I64, device)?;
        if nnz == 0 || n == 0 {
            return Ok(out);
        }
        let ptrs = indices_to_i64(self, &ptrs.contiguous()?)?;
        // SAFETY: `ptrs` holds n + 1 I64 pointers ending at nnz, and `out`
        // holds nnz I64 entries.
        unsafe {
            launch_expand_ptrs(
                &self.context,
                &self.stream,
                self.device.index,
                ptrs.ptr(),
                out.ptr(),
                n,
            )?;
        }
        Ok(out)
    }

    /// Stable sort of entries by (`primary`, `secondary`) into `nbuckets`
    /// buckets, where `secondary < extent`. Returns I64 bucket pointers, the
    /// I64 sorted secondary indices and the values in sorted order.
    fn bucket_sort(
        &self,
        primary: &Tensor<CudaRuntime>,
        secondary: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        extent: usize,
        nbuckets: usize,
    ) -> Result<SparseTriple> {
        let device = values.device();
        let dtype = values.dtype();
        let nnz = values.numel();
        if primary.numel() != nnz || secondary.numel() != nnz {
            return Err(Error::Internal(format!(
                "sparse format conversion (CUDA): index lengths {} and {} differ from the {nnz} values",
                primary.numel(),
                secondary.numel()
            )));
        }
        let ptrs = Tensor::<CudaRuntime>::zeros(&[nbuckets + 1], DType::I64, device)?;
        if nnz == 0 {
            return Ok((
                ptrs,
                Tensor::<CudaRuntime>::zeros(&[0], DType::I64, device)?,
                Tensor::<CudaRuntime>::zeros(&[0], dtype, device)?,
            ));
        }
        if nnz > i32::MAX as usize {
            return Err(Error::Internal(format!(
                "sparse format conversion (CUDA): {nnz} entries exceed the i32 sort ids"
            )));
        }

        let primary = indices_to_i64(self, &primary.contiguous()?)?;
        let secondary = indices_to_i64(self, &secondary.contiguous()?)?;
        let values = values.contiguous()?;
        let keys = Tensor::<CudaRuntime>::zeros(&[nnz], DType::I64, device)?;
        let order = Tensor::<CudaRuntime>::zeros(&[nnz], DType::I32, device)?;
        let sorted_primary = Tensor::<CudaRuntime>::zeros(&[nnz], DType::I64, device)?;
        let sorted_secondary = Tensor::<CudaRuntime>::zeros(&[nnz], DType::I64, device)?;
        let sorted_values = Tensor::<CudaRuntime>::zeros(&[nnz], dtype, device)?;
        let (ctx, stream, index) = (&self.context, &self.stream, self.device.index);

        // SAFETY: every buffer above is a live device tensor of nnz elements
        // (keys, ids and gathers) or nbuckets + 1 pointers.
        unsafe {
            launch_coo_compute_keys(
                ctx,
                stream,
                index,
                primary.ptr(),
                secondary.ptr(),
                keys.ptr(),
                extent as i64,
                nnz,
            )?;
            launch_coo_init_indices(ctx, stream, index, order.ptr(), nnz)?;
            launch_thrust_sort_pairs_i64_i32(
                ctx,
                stream,
                index,
                keys.ptr(),
                order.ptr(),
                nnz as u32,
            )?;
            for (src, dst) in [(&primary, &sorted_primary), (&secondary, &sorted_secondary)] {
                launch_coo_gather(
                    ctx,
                    stream,
                    index,
                    DType::I64,
                    src.ptr(),
                    order.ptr(),
                    dst.ptr(),
                    nnz,
                )?;
            }
            launch_coo_gather(
                ctx,
                stream,
                index,
                dtype,
                values.ptr(),
                order.ptr(),
                sorted_values.ptr(),
                nnz,
            )?;
            launch_build_ptrs_from_sorted(
                ctx,
                stream,
                index,
                sorted_primary.ptr(),
                ptrs.ptr(),
                nnz,
                nbuckets,
            )?;
        }
        Ok((ptrs, sorted_secondary, sorted_values))
    }
}
