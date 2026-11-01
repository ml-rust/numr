//! Sparse format conversions for WebGPU: COO, CSR and CSC to one another,
//! CSR to dense, and dense to COO.
//!
//! Every conversion runs on the GPU. Inputs and outputs carry I64 indices,
//! and the kernels see I32 copies made on the GPU. Values move as raw 32-bit
//! words, so F32, I32 and U32 share one kernel set.
//!
//! # Algorithms
//!
//! - **CSR/CSC → COO**: expand the pointers into one index per entry.
//! - **COO → CSR/CSC**: stable counting sort by (primary, secondary index):
//!   histogram, exclusive scan, scatter of entry ids, rank inside each bucket.
//! - **CSR ↔ CSC**: expand the pointers, then the same counting sort.
//! - **CSR → dense**: one thread per row writes its entries.
//! - **dense → COO**: per-row count, scan, per-row scatter in column order.
//!
//! The output order matches the CPU backend exactly: sorted by (primary,
//! secondary), and duplicates keep their input order.

use super::super::ops::helpers::get_tensor_buffer;
use super::super::shaders::{
    launch_csr_to_dense, launch_dense_row_count, launch_dense_row_scatter, launch_expand_ptrs,
    launch_histogram, launch_rank_gather, launch_scatter_order,
};
use super::super::{WgpuClient, WgpuRuntime};
use super::indices::{
    check_value_dtype, indices_to_i32, indices_to_i64, scan_counts, uniform_words,
};
use super::merge::SparseTriple;
use crate::algorithm::sparse::validate_sparse_operands;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::sparse::{CooData, SparseTensor};
use crate::tensor::Tensor;

impl WgpuClient {
    /// CSR → COO. Returns I64 row indices, I64 column indices and the values.
    pub(crate) fn csr_to_coo_impl<T: Element>(
        &self,
        row_ptrs: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<SparseTriple> {
        validate_sparse_operands::<T>(&[], &[values.dtype()])?;
        let ptrs = indices_to_i32(self, row_ptrs, "csr_to_coo (WebGPU)")?;
        let rows = self.expand_ptrs(&ptrs, shape[0], values.numel())?;
        Ok((
            indices_to_i64(self, &rows)?,
            indices_to_i64(self, col_indices)?,
            values.clone(),
        ))
    }

    /// CSC → COO. Returns I64 row indices, I64 column indices and the values.
    pub(crate) fn csc_to_coo_impl<T: Element>(
        &self,
        col_ptrs: &Tensor<WgpuRuntime>,
        row_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<SparseTriple> {
        validate_sparse_operands::<T>(&[], &[values.dtype()])?;
        let ptrs = indices_to_i32(self, col_ptrs, "csc_to_coo (WebGPU)")?;
        let cols = self.expand_ptrs(&ptrs, shape[1], values.numel())?;
        Ok((
            indices_to_i64(self, row_indices)?,
            indices_to_i64(self, &cols)?,
            values.clone(),
        ))
    }

    /// COO → CSR, sorted by (row, column). Returns I64 row pointers, I64
    /// column indices and the values.
    pub(crate) fn coo_to_csr_impl<T: Element>(
        &self,
        row_indices: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<SparseTriple> {
        let op = "coo_to_csr (WebGPU)";
        validate_sparse_operands::<T>(&[], &[values.dtype()])?;
        let rows = indices_to_i32(self, row_indices, op)?;
        let cols = indices_to_i32(self, col_indices, op)?;
        self.bucket_sort(&rows, &cols, values, shape[0], op)
    }

    /// COO → CSC, sorted by (column, row). Returns I64 column pointers, I64
    /// row indices and the values.
    pub(crate) fn coo_to_csc_impl<T: Element>(
        &self,
        row_indices: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<SparseTriple> {
        let op = "coo_to_csc (WebGPU)";
        validate_sparse_operands::<T>(&[], &[values.dtype()])?;
        let rows = indices_to_i32(self, row_indices, op)?;
        let cols = indices_to_i32(self, col_indices, op)?;
        self.bucket_sort(&cols, &rows, values, shape[1], op)
    }

    /// CSR → CSC. Returns I64 column pointers, I64 row indices and the values.
    pub(crate) fn csr_to_csc_impl<T: Element>(
        &self,
        row_ptrs: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<SparseTriple> {
        let op = "csr_to_csc (WebGPU)";
        validate_sparse_operands::<T>(&[], &[values.dtype()])?;
        let ptrs = indices_to_i32(self, row_ptrs, op)?;
        let rows = self.expand_ptrs(&ptrs, shape[0], values.numel())?;
        let cols = indices_to_i32(self, col_indices, op)?;
        self.bucket_sort(&cols, &rows, values, shape[1], op)
    }

    /// CSC → CSR. Returns I64 row pointers, I64 column indices and the values.
    pub(crate) fn csc_to_csr_impl<T: Element>(
        &self,
        col_ptrs: &Tensor<WgpuRuntime>,
        row_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<SparseTriple> {
        let op = "csc_to_csr (WebGPU)";
        validate_sparse_operands::<T>(&[], &[values.dtype()])?;
        let ptrs = indices_to_i32(self, col_ptrs, op)?;
        let cols = self.expand_ptrs(&ptrs, shape[1], values.numel())?;
        let rows = indices_to_i32(self, row_indices, op)?;
        self.bucket_sort(&rows, &cols, values, shape[0], op)
    }

    /// CSR → dense `[nrows, ncols]`. A repeated position keeps its last entry.
    pub(crate) fn sparse_to_dense_impl<T: Element>(
        &self,
        row_ptrs: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<WgpuRuntime>> {
        let op = "sparse_to_dense (WebGPU)";
        let [nrows, ncols] = shape;
        let dtype = values.dtype();
        check_value_dtype(dtype, op)?;
        validate_sparse_operands::<T>(&[], &[dtype])?;

        let dense = Tensor::<WgpuRuntime>::zeros(&[nrows, ncols], dtype, &self.device_id)?;
        if values.numel() == 0 || nrows == 0 || ncols == 0 {
            return Ok(dense);
        }
        let ptrs = indices_to_i32(self, row_ptrs, op)?;
        let cols = indices_to_i32(self, col_indices, op)?;
        let values = values.contiguous()?;
        let params = uniform_words(
            self,
            "csr_to_dense_params",
            [nrows as u32, ncols as u32, 0, 0],
        );
        launch_csr_to_dense(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &*get_tensor_buffer(&ptrs)?,
            &*get_tensor_buffer(&cols)?,
            &*get_tensor_buffer(&values)?,
            &*get_tensor_buffer(&dense)?,
            &params,
            nrows,
        )?;
        Ok(dense)
    }

    /// Dense `[nrows, ncols]` → COO in row-major order. An entry is kept when
    /// `|value| >= threshold`, as on the CPU backend.
    pub(crate) fn dense_to_coo_impl(
        &self,
        dense: &Tensor<WgpuRuntime>,
        threshold: f64,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        let op = "dense_to_sparse (WebGPU)";
        if dense.ndim() != 2 {
            return Err(Error::Internal(format!(
                "{op}: expected a 2D tensor, got shape {:?}",
                dense.shape()
            )));
        }
        let (nrows, ncols) = (dense.shape()[0], dense.shape()[1]);
        let dtype = dense.dtype();
        check_value_dtype(dtype, op)?;
        if nrows == 0 || ncols == 0 {
            return Ok(SparseTensor::Coo(CooData::empty(
                [nrows, ncols],
                dtype,
                &self.device_id,
            )?));
        }

        let dense = dense.contiguous()?;
        let dense_buf = get_tensor_buffer(&dense)?;
        let params = self.create_uniform_buffer("dense_keep_params", 32);
        self.write_buffer(&params, &keep_params(dtype, threshold, nrows, ncols)?);

        let counts = Tensor::<WgpuRuntime>::zeros(&[nrows], DType::I32, &self.device_id)?;
        launch_dense_row_count(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &dense_buf,
            &*get_tensor_buffer(&counts)?,
            &params,
            nrows,
        )?;
        let (ptrs, total) = scan_counts(self, &counts, nrows)?;
        if total == 0 {
            return Ok(SparseTensor::Coo(CooData::empty(
                [nrows, ncols],
                dtype,
                &self.device_id,
            )?));
        }

        let rows = Tensor::<WgpuRuntime>::zeros(&[total], DType::I32, &self.device_id)?;
        let cols = Tensor::<WgpuRuntime>::zeros(&[total], DType::I32, &self.device_id)?;
        let values = Tensor::<WgpuRuntime>::zeros(&[total], dtype, &self.device_id)?;
        launch_dense_row_scatter(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &dense_buf,
            &*get_tensor_buffer(&ptrs)?,
            &*get_tensor_buffer(&rows)?,
            &*get_tensor_buffer(&cols)?,
            &*get_tensor_buffer(&values)?,
            &params,
            nrows,
        )?;

        let mut coo = CooData::new(
            indices_to_i64(self, &rows)?,
            indices_to_i64(self, &cols)?,
            values,
            [nrows, ncols],
        )?;
        // SAFETY: the scatter writes each row in column order, rows ascending.
        unsafe { coo.set_sorted(true) };
        Ok(SparseTensor::Coo(coo))
    }

    /// Expand `n + 1` I32 pointers into `nnz` I32 bucket indices.
    fn expand_ptrs(
        &self,
        ptrs: &Tensor<WgpuRuntime>,
        n: usize,
        nnz: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        let out = Tensor::<WgpuRuntime>::zeros(&[nnz], DType::I32, &self.device_id)?;
        if nnz == 0 || n == 0 {
            return Ok(out);
        }
        let params = uniform_words(self, "expand_ptrs_params", [n as u32, 0, 0, 0]);
        launch_expand_ptrs(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &*get_tensor_buffer(ptrs)?,
            &*get_tensor_buffer(&out)?,
            &params,
            n,
        )?;
        Ok(out)
    }

    /// Stable counting sort of entries by (`primary`, `secondary`) into
    /// `nbuckets` buckets. Returns I64 bucket pointers, I64 sorted secondary
    /// indices and the values in sorted order.
    fn bucket_sort(
        &self,
        primary: &Tensor<WgpuRuntime>,
        secondary: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        nbuckets: usize,
        op: &'static str,
    ) -> Result<SparseTriple> {
        let dtype = values.dtype();
        check_value_dtype(dtype, op)?;
        let nnz = values.numel();
        if primary.numel() != nnz || secondary.numel() != nnz {
            return Err(Error::Internal(format!(
                "{op}: index lengths {} and {} differ from the {nnz} values",
                primary.numel(),
                secondary.numel()
            )));
        }
        if nnz == 0 || nbuckets == 0 {
            if nnz > 0 {
                return Err(Error::Internal(format!(
                    "{op}: {nnz} entries do not fit a matrix with zero buckets"
                )));
            }
            return Ok((
                Tensor::<WgpuRuntime>::zeros(&[nbuckets + 1], DType::I64, &self.device_id)?,
                Tensor::<WgpuRuntime>::zeros(&[0], DType::I64, &self.device_id)?,
                Tensor::<WgpuRuntime>::zeros(&[0], dtype, &self.device_id)?,
            ));
        }

        let values = values.contiguous()?;
        let primary_buf = get_tensor_buffer(primary)?;
        let secondary_buf = get_tensor_buffer(secondary)?;
        let params = uniform_words(self, "bucket_sort_params", [nnz as u32, 0, 0, 0]);

        let counts = Tensor::<WgpuRuntime>::zeros(&[nbuckets], DType::I32, &self.device_id)?;
        launch_histogram(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &primary_buf,
            &*get_tensor_buffer(&counts)?,
            &params,
            nnz,
        )?;
        let (ptrs, total) = scan_counts(self, &counts, nbuckets)?;
        if total != nnz {
            return Err(Error::Internal(format!(
                "{op}: {total} of {nnz} entries have a bucket index in 0..{nbuckets}"
            )));
        }

        let ptrs_buf = get_tensor_buffer(&ptrs)?;
        let fill = Tensor::<WgpuRuntime>::zeros(&[nbuckets], DType::I32, &self.device_id)?;
        let order = Tensor::<WgpuRuntime>::zeros(&[nnz], DType::I32, &self.device_id)?;
        let order_buf = get_tensor_buffer(&order)?;
        launch_scatter_order(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &primary_buf,
            &ptrs_buf,
            &*get_tensor_buffer(&fill)?,
            &order_buf,
            &params,
            nnz,
        )?;

        let out_secondary = Tensor::<WgpuRuntime>::zeros(&[nnz], DType::I32, &self.device_id)?;
        let out_values = Tensor::<WgpuRuntime>::zeros(&[nnz], dtype, &self.device_id)?;
        launch_rank_gather(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &primary_buf,
            &secondary_buf,
            &*get_tensor_buffer(&values)?,
            &ptrs_buf,
            &order_buf,
            &*get_tensor_buffer(&out_secondary)?,
            &*get_tensor_buffer(&out_values)?,
            &params,
            nnz,
        )?;

        Ok((
            indices_to_i64(self, &ptrs)?,
            indices_to_i64(self, &out_secondary)?,
            out_values,
        ))
    }
}

/// Uniform words of the dense keep predicate `|value| >= threshold`:
/// `{ nrows, ncols, threshold, mode, keep_none, pad, pad, pad }`.
///
/// The host threshold is an `f64`. Each dtype gets the smallest value of its
/// own type that keeps the same entries, so the GPU compare needs no `f64`.
fn keep_params(dtype: DType, threshold: f64, nrows: usize, ncols: usize) -> Result<[u32; 8]> {
    let (bits, mode, keep_none) = match dtype {
        DType::F32 => match f32_at_least(threshold) {
            Some(t) => (t.to_bits(), 0, 0),
            None => (0, 0, 1),
        },
        DType::I32 => match magnitude_at_least(threshold, 1 << 31) {
            Some(t) => (t as u32, 1, 0),
            None => (0, 1, 1),
        },
        DType::U32 => match magnitude_at_least(threshold, u64::from(u32::MAX)) {
            Some(t) => (t as u32, 2, 0),
            None => (0, 2, 1),
        },
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "dense_to_sparse (WebGPU)",
            });
        }
    };
    Ok([nrows as u32, ncols as u32, bits, mode, keep_none, 0, 0, 0])
}

/// Smallest `f32` whose `f64` value is `>= threshold`, or `None` for a NaN
/// threshold, which keeps no entry.
fn f32_at_least(threshold: f64) -> Option<f32> {
    if threshold.is_nan() {
        return None;
    }
    if threshold <= 0.0 {
        return Some(0.0);
    }
    let t = threshold as f32;
    Some(if f64::from(t) < threshold {
        t.next_up()
    } else {
        t
    })
}

/// Smallest integer magnitude `>= threshold`, or `None` when no magnitude up
/// to `max` reaches it.
fn magnitude_at_least(threshold: f64, max: u64) -> Option<u64> {
    if threshold.is_nan() {
        return None;
    }
    if threshold <= 0.0 {
        return Some(0);
    }
    let t = threshold.ceil();
    if t > max as f64 { None } else { Some(t as u64) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_threshold_rounds_up_to_the_next_f32() {
        let t = 0.1f64;
        let got = f32_at_least(t).unwrap();
        assert!(f64::from(got) >= t);
        assert!(f64::from(got.next_down()) < t);
        assert_eq!(f32_at_least(-3.0), Some(0.0));
        assert_eq!(f32_at_least(f64::NAN), None);
        assert_eq!(f32_at_least(f64::INFINITY), Some(f32::INFINITY));
    }

    #[test]
    fn integer_threshold_takes_the_ceiling() {
        assert_eq!(magnitude_at_least(2.5, 1 << 31), Some(3));
        assert_eq!(magnitude_at_least(2.0, 1 << 31), Some(2));
        assert_eq!(magnitude_at_least(0.0, 1 << 31), Some(0));
        assert_eq!(magnitude_at_least((1u64 << 31) as f64 + 0.5, 1 << 31), None);
        assert_eq!(magnitude_at_least(f64::NAN, 10), None);
    }

    #[test]
    fn keep_params_flag_an_unreachable_threshold() {
        let words = keep_params(DType::U32, 5e9, 2, 3).unwrap();
        assert_eq!(&words[..5], &[2, 3, 0, 2, 1]);
        assert!(keep_params(DType::F64, 1.0, 1, 1).is_err());
    }
}
