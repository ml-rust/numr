//! `SparseOps` for the CUDA runtime.
//!
//! Low-level methods take the I64 index tensors the sparse formats store,
//! and return I64 indices. Format changes inside high-level methods run on
//! the GPU. Arithmetic and the dense conversions accept F32, F64, F16 and
//! BF16 values. Format conversions accept any value dtype of 1, 2, 4 or 8
//! bytes. Any other dtype returns `UnsupportedDType`.

use super::super::{CudaClient, CudaRuntime};
use super::formats::threshold_at_least;
use super::indices::check_float_dtype;
use super::merge::SparseTriple;
use crate::algorithm::sparse::{SparseAlgorithms, validate_dtype_match};
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::ops::{ReduceOps, ScalarOps};
use crate::runtime::cuda::kernels::{
    SparseFormat, SparseMergeOp, csc_nnz_per_col_gpu, csc_sum_cols_gpu, csr_nnz_per_row_gpu,
    csr_sum_rows_gpu, csr_to_dense_gpu, dense_to_coo_gpu,
};
use crate::sparse::{CooData, CscData, CsrData, SparseOps, SparseTensor};
use crate::tensor::Tensor;

type T2 = Tensor<CudaRuntime>;
type Sparse = SparseTensor<CudaRuntime>;

/// One low-level merge method: CSR or CSC merges over `shape[$dim]` rows or
/// columns; `coo` merges through CSR.
macro_rules! merge_method {
    ($name:ident, coo, $op:expr) => {
        fn $name<T: Element>(
            &self,
            a0: &T2,
            a1: &T2,
            a2: &T2,
            b0: &T2,
            b1: &T2,
            b2: &T2,
            shape: [usize; 2],
        ) -> Result<SparseTriple> {
            self.coo_merge::<T>($op, (a0, a1, a2), (b0, b1, b2), shape)
        }
    };
    ($name:ident, $format:expr, $dim:expr, $op:expr) => {
        fn $name<T: Element>(
            &self,
            a0: &T2,
            a1: &T2,
            a2: &T2,
            b0: &T2,
            b1: &T2,
            b2: &T2,
            shape: [usize; 2],
        ) -> Result<SparseTriple> {
            self.compressed_merge::<T>($format, $op, (a0, a1, a2), (b0, b1, b2), shape[$dim])
        }
    };
}

/// One low-level format conversion method, delegating to `$impl`.
macro_rules! convert_method {
    ($name:ident, $impl:ident) => {
        fn $name<T: Element>(
            &self,
            i0: &T2,
            i1: &T2,
            v: &T2,
            shape: [usize; 2],
        ) -> Result<SparseTriple> {
            self.$impl::<T>(i0, i1, v, shape)
        }
    };
}

impl SparseOps<CudaRuntime> for CudaClient {
    fn spmv_csr<T: Element>(
        &self,
        rp: &T2,
        ci: &T2,
        v: &T2,
        x: &T2,
        shape: [usize; 2],
    ) -> Result<T2> {
        self.spmv_csr_impl::<T>(rp, ci, v, x, shape)
    }

    fn spmm_csr<T: Element>(
        &self,
        rp: &T2,
        ci: &T2,
        v: &T2,
        b: &T2,
        shape: [usize; 2],
    ) -> Result<T2> {
        self.spmm_csr_impl::<T>(rp, ci, v, b, shape)
    }

    merge_method!(add_csr, SparseFormat::Csr, 0, SparseMergeOp::Add);
    merge_method!(sub_csr, SparseFormat::Csr, 0, SparseMergeOp::Sub);
    merge_method!(mul_csr, SparseFormat::Csr, 0, SparseMergeOp::Mul);
    merge_method!(div_csr, SparseFormat::Csr, 0, SparseMergeOp::Div);
    merge_method!(add_csc, SparseFormat::Csc, 1, SparseMergeOp::Add);
    merge_method!(sub_csc, SparseFormat::Csc, 1, SparseMergeOp::Sub);
    merge_method!(mul_csc, SparseFormat::Csc, 1, SparseMergeOp::Mul);
    merge_method!(div_csc, SparseFormat::Csc, 1, SparseMergeOp::Div);
    merge_method!(add_coo, coo, SparseMergeOp::Add);
    merge_method!(sub_coo, coo, SparseMergeOp::Sub);
    merge_method!(mul_coo, coo, SparseMergeOp::Mul);
    merge_method!(div_coo, coo, SparseMergeOp::Div);

    fn spmv(&self, a: &Sparse, x: &T2) -> Result<T2> {
        validate_dtype_match(a.dtype(), x.dtype())?;
        let csr = self.gpu_csr(a)?;
        crate::dispatch_dtype!(csr.values.dtype(), T => {
            self.spmv_csr_impl::<T>(&csr.row_ptrs, &csr.col_indices, &csr.values, x, csr.shape)
        }, "spmv")
    }

    fn spmm(&self, a: &Sparse, b: &T2) -> Result<T2> {
        validate_dtype_match(a.dtype(), b.dtype())?;
        let csr = self.gpu_csr(a)?;
        crate::dispatch_dtype!(csr.values.dtype(), T => {
            self.spmm_csr_impl::<T>(&csr.row_ptrs, &csr.col_indices, &csr.values, b, csr.shape)
        }, "spmm")
    }

    fn dsmm(&self, a: &T2, b: &Sparse) -> Result<T2> {
        validate_dtype_match(a.dtype(), b.dtype())?;
        check_float_dtype(a.dtype(), "dsmm (CUDA)")?;
        let csc_b = self.gpu_csc(b)?;
        self.column_parallel_dsmm(a, &csc_b)
    }

    fn sparse_add(&self, a: &Sparse, b: &Sparse) -> Result<Sparse> {
        self.sparse_merge_any(a, b, SparseMergeOp::Add)
    }

    fn sparse_sub(&self, a: &Sparse, b: &Sparse) -> Result<Sparse> {
        self.sparse_merge_any(a, b, SparseMergeOp::Sub)
    }

    fn sparse_mul(&self, a: &Sparse, b: &Sparse) -> Result<Sparse> {
        self.sparse_merge_any(a, b, SparseMergeOp::Mul)
    }

    fn sparse_matmul(&self, a: &Sparse, b: &Sparse) -> Result<Sparse> {
        validate_dtype_match(a.dtype(), b.dtype())?;
        check_float_dtype(a.dtype(), "sparse_matmul (CUDA)")?;
        let csr_a = self.gpu_csr(a)?;
        let csr_b = self.gpu_csr(b)?;
        Ok(SparseTensor::Csr(self.esc_spgemm_csr(&csr_a, &csr_b)?))
    }

    fn sparse_scale(&self, a: &Sparse, scalar: f64) -> Result<Sparse> {
        if a.nnz() == 0 {
            return Ok(a.clone());
        }
        self.map_sparse_values(a, |v| self.mul_scalar(v, scalar))
    }

    fn sparse_add_scalar(&self, _a: &Sparse, _scalar: f64) -> Result<Sparse> {
        // Adding a scalar to a sparse matrix would make it dense
        // (all implicit zeros become non-zero)
        Err(Error::Internal(
            "Scalar addition to sparse matrix creates dense result - convert to dense first"
                .to_string(),
        ))
    }

    fn sparse_sum(&self, a: &Sparse) -> Result<T2> {
        // Shape [1], as on the CPU backend.
        if a.nnz() == 0 {
            return Tensor::zeros(&[1], a.dtype(), &self.device);
        }
        let values = match a {
            SparseTensor::Csr(data) => &data.values,
            SparseTensor::Csc(data) => &data.values,
            SparseTensor::Coo(data) => &data.values,
        };
        self.sum(values, &[0], true)
    }

    fn sparse_sum_rows(&self, a: &Sparse) -> Result<T2> {
        let csr = self.gpu_csr(a)?;
        let dtype = csr.values.dtype();
        check_float_dtype(dtype, "sparse_sum_rows (CUDA)")?;
        // SAFETY: CSR arrays of `csr` are live device tensors with I64 pointers.
        unsafe {
            crate::dispatch_dtype!(dtype, T => {
                csr_sum_rows_gpu::<T>(&self.context, &self.stream, self.device.index, &self.device,
                    dtype, &csr.row_ptrs, &csr.values, csr.shape[0])
            }, "sparse_sum_rows")
        }
    }

    fn sparse_sum_cols(&self, a: &Sparse) -> Result<T2> {
        let csc = self.gpu_csc(a)?;
        let dtype = csc.values.dtype();
        check_float_dtype(dtype, "sparse_sum_cols (CUDA)")?;
        // SAFETY: CSC arrays of `csc` are live device tensors with I64 pointers.
        unsafe {
            crate::dispatch_dtype!(dtype, T => {
                csc_sum_cols_gpu::<T>(&self.context, &self.stream, self.device.index, &self.device,
                    dtype, &csc.col_ptrs, &csc.values, csc.shape[1])
            }, "sparse_sum_cols")
        }
    }

    fn sparse_nnz_per_row(&self, a: &Sparse) -> Result<T2> {
        let csr = self.gpu_csr(a)?;
        // SAFETY: `row_ptrs` is a live I64 device tensor of nrows + 1 entries.
        unsafe {
            csr_nnz_per_row_gpu(
                &self.context,
                &self.stream,
                self.device.index,
                &self.device,
                &csr.row_ptrs,
                csr.shape[0],
            )
        }
    }

    fn sparse_nnz_per_col(&self, a: &Sparse) -> Result<T2> {
        let csc = self.gpu_csc(a)?;
        // SAFETY: `col_ptrs` is a live I64 device tensor of ncols + 1 entries.
        unsafe {
            csc_nnz_per_col_gpu(
                &self.context,
                &self.stream,
                self.device.index,
                &self.device,
                &csc.col_ptrs,
                csc.shape[1],
            )
        }
    }

    fn sparse_to_dense(&self, a: &Sparse) -> Result<T2> {
        let csr = self.gpu_csr(a)?;
        let dtype = csr.values.dtype();
        check_float_dtype(dtype, "sparse_to_dense (CUDA)")?;
        // SAFETY: CSR arrays of `csr` are live device tensors with I64 indices.
        unsafe {
            crate::dispatch_dtype!(dtype, T => {
                csr_to_dense_gpu::<T>(&self.context, &self.stream, self.device.index, &self.device,
                    dtype, &csr.row_ptrs, &csr.col_indices, &csr.values, csr.shape)
            }, "sparse_to_dense")
        }
    }

    fn dense_to_sparse(&self, a: &T2, threshold: f64) -> Result<Sparse> {
        if a.ndim() != 2 {
            return Err(Error::Internal(format!(
                "dense_to_sparse (CUDA): expected a 2D tensor, got shape {:?}",
                a.shape()
            )));
        }
        let (nrows, ncols) = (a.shape()[0], a.shape()[1]);
        let dtype = a.dtype();
        check_float_dtype(dtype, "dense_to_sparse (CUDA)")?;
        let a = a.contiguous()?;
        let threshold = threshold_at_least(dtype, threshold);
        // SAFETY: `a` is a contiguous live device tensor of `dtype`.
        let (row_indices, col_indices, values) = unsafe {
            crate::dispatch_dtype!(dtype, T => {
                dense_to_coo_gpu::<T>(&self.context, &self.stream, self.device.index, &self.device,
                    dtype, &a, T::from_f64(threshold))
            }, "dense_to_sparse")?
        };
        let mut coo = CooData::new(row_indices, col_indices, values, [nrows, ncols])?;
        // SAFETY: the extract kernel writes each row in column order, rows ascending.
        unsafe { coo.set_sorted(true) };
        Ok(SparseTensor::Coo(coo))
    }

    fn dense_to_csr(&self, a: &T2, threshold: f64) -> Result<CsrData<CudaRuntime>> {
        let coo = self.dense_to_sparse(a, threshold)?;
        self.gpu_csr(&coo)
    }

    fn dense_to_csc(&self, a: &T2, threshold: f64) -> Result<CscData<CudaRuntime>> {
        let coo = self.dense_to_sparse(a, threshold)?;
        self.gpu_csc(&coo)
    }

    convert_method!(coo_to_csr, coo_to_csr_impl);
    convert_method!(coo_to_csc, coo_to_csc_impl);
    convert_method!(csr_to_coo, csr_to_coo_impl);
    convert_method!(csc_to_coo, csc_to_coo_impl);
    convert_method!(csr_to_csc, csr_to_csc_impl);
    convert_method!(csc_to_csr, csc_to_csr_impl);

    fn extract_diagonal_csr<T: Element>(
        &self,
        rp: &T2,
        ci: &T2,
        v: &T2,
        shape: [usize; 2],
    ) -> Result<T2> {
        self.extract_diagonal_csr_impl::<T>(rp, ci, v, shape)
    }

    fn sparse_extract_diagonal(&self, a: &Sparse) -> Result<T2> {
        let csr = self.gpu_csr(a)?;
        crate::dispatch_dtype!(csr.values.dtype(), T => {
            self.extract_diagonal_csr_impl::<T>(&csr.row_ptrs, &csr.col_indices, &csr.values, csr.shape)
        }, "sparse_extract_diagonal")
    }

    fn sparse_transpose(&self, a: &Sparse) -> Result<Sparse> {
        // The CSR arrays of A are the CSC arrays of A^T, and the reverse. No data moves.
        Ok(match a {
            SparseTensor::Csr(d) => SparseTensor::Csc(d.transpose()),
            SparseTensor::Csc(d) => SparseTensor::Csr(d.transpose()),
            SparseTensor::Coo(d) => SparseTensor::Coo(CooData::new(
                d.col_indices.clone(),
                d.row_indices.clone(),
                d.values.clone(),
                [d.shape[1], d.shape[0]],
            )?),
        })
    }
}
