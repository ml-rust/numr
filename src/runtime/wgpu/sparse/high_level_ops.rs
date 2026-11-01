//! `SparseOps` for the WebGPU runtime.
//!
//! Low-level methods take I64 index tensors, as the sparse formats store
//! them, and return I64 indices. Format changes inside high-level methods
//! run on the GPU. Value dtypes: F32 for SpMV, SpMM, SpGEMM, DSMM, row and
//! column sums, and the diagonal. F32, I32 and U32 for merges and format
//! conversions. Any other dtype returns `UnsupportedDType`.

use super::super::shaders::{CompressedFormat, SparseMergeOp};
use super::super::{WgpuClient, WgpuRuntime};
use super::merge::SparseTriple;
use crate::algorithm::sparse::{SparseAlgorithms, validate_dtype_match};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, TypeConversionOps};
use crate::sparse::{CooData, CscData, CsrData, SparseOps, SparseTensor};
use crate::tensor::Tensor;

type T2 = Tensor<WgpuRuntime>;

/// One low-level merge method: `$format` CSR or CSC merges over `shape[$dim]`
/// rows or columns; `coo` merges through CSR.
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

impl SparseOps<WgpuRuntime> for WgpuClient {
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

    merge_method!(add_csr, CompressedFormat::Csr, 0, SparseMergeOp::Add);
    merge_method!(sub_csr, CompressedFormat::Csr, 0, SparseMergeOp::Sub);
    merge_method!(mul_csr, CompressedFormat::Csr, 0, SparseMergeOp::Mul);
    merge_method!(div_csr, CompressedFormat::Csr, 0, SparseMergeOp::Div);
    merge_method!(add_csc, CompressedFormat::Csc, 1, SparseMergeOp::Add);
    merge_method!(sub_csc, CompressedFormat::Csc, 1, SparseMergeOp::Sub);
    merge_method!(mul_csc, CompressedFormat::Csc, 1, SparseMergeOp::Mul);
    merge_method!(div_csc, CompressedFormat::Csc, 1, SparseMergeOp::Div);
    merge_method!(add_coo, coo, SparseMergeOp::Add);
    merge_method!(sub_coo, coo, SparseMergeOp::Sub);
    merge_method!(mul_coo, coo, SparseMergeOp::Mul);
    merge_method!(div_coo, coo, SparseMergeOp::Div);

    fn spmv(&self, a: &SparseTensor<WgpuRuntime>, x: &T2) -> Result<T2> {
        validate_dtype_match(a.dtype(), x.dtype())?;
        let csr = self.gpu_csr(a)?;
        crate::dispatch_dtype!(csr.values.dtype(), T => {
            self.spmv_csr_impl::<T>(&csr.row_ptrs, &csr.col_indices, &csr.values, x, csr.shape)
        }, "spmv")
    }

    fn spmm(&self, a: &SparseTensor<WgpuRuntime>, b: &T2) -> Result<T2> {
        validate_dtype_match(a.dtype(), b.dtype())?;
        let csr = self.gpu_csr(a)?;
        crate::dispatch_dtype!(csr.values.dtype(), T => {
            self.spmm_csr_impl::<T>(&csr.row_ptrs, &csr.col_indices, &csr.values, b, csr.shape)
        }, "spmm")
    }

    fn dsmm(&self, a: &T2, b: &SparseTensor<WgpuRuntime>) -> Result<T2> {
        validate_dtype_match(a.dtype(), b.dtype())?;
        let csc_b = self.gpu_csc(b)?;
        self.column_parallel_dsmm(a, &csc_b)
    }

    fn sparse_add(
        &self,
        a: &SparseTensor<WgpuRuntime>,
        b: &SparseTensor<WgpuRuntime>,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        self.sparse_merge_any(a, b, SparseMergeOp::Add)
    }

    fn sparse_sub(
        &self,
        a: &SparseTensor<WgpuRuntime>,
        b: &SparseTensor<WgpuRuntime>,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        self.sparse_merge_any(a, b, SparseMergeOp::Sub)
    }

    fn sparse_mul(
        &self,
        a: &SparseTensor<WgpuRuntime>,
        b: &SparseTensor<WgpuRuntime>,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        self.sparse_merge_any(a, b, SparseMergeOp::Mul)
    }

    fn sparse_matmul(
        &self,
        a: &SparseTensor<WgpuRuntime>,
        b: &SparseTensor<WgpuRuntime>,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        validate_dtype_match(a.dtype(), b.dtype())?;
        if a.dtype() != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "sparse_matmul (WebGPU supports F32)",
            });
        }
        let csr_a = self.gpu_csr(a)?;
        let csr_b = self.gpu_csr(b)?;
        Ok(SparseTensor::Csr(self.esc_spgemm_csr(&csr_a, &csr_b)?))
    }

    fn sparse_scale(
        &self,
        a: &SparseTensor<WgpuRuntime>,
        scalar: f64,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        if a.nnz() == 0 {
            return Ok(a.clone());
        }
        self.map_sparse_values(a, |v| self.mul_scalar(v, scalar))
    }

    fn sparse_add_scalar(
        &self,
        _a: &SparseTensor<WgpuRuntime>,
        _scalar: f64,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        // Adding a scalar to a sparse matrix would make it dense
        // (all implicit zeros become non-zero)
        Err(Error::Internal(
            "Scalar addition to sparse matrix creates dense result - convert to dense first"
                .to_string(),
        ))
    }

    fn sparse_sum(&self, a: &SparseTensor<WgpuRuntime>) -> Result<T2> {
        // Shape [1], as on the CPU backend.
        if a.nnz() == 0 {
            return Tensor::zeros(&[1], a.dtype(), &self.device_id);
        }
        let values = match a {
            SparseTensor::Csr(data) => &data.values,
            SparseTensor::Csc(data) => &data.values,
            SparseTensor::Coo(data) => &data.values,
        };
        self.sum(values, &[0], true)
    }

    fn sparse_sum_rows(&self, a: &SparseTensor<WgpuRuntime>) -> Result<T2> {
        // row_sums = A * ones
        let csr = self.gpu_csr(a)?;
        let ones = Tensor::ones(&[csr.shape[1]], csr.values.dtype(), &self.device_id)?;
        crate::dispatch_dtype!(csr.values.dtype(), T => {
            self.spmv_csr_impl::<T>(&csr.row_ptrs, &csr.col_indices, &csr.values, &ones, csr.shape)
        }, "sparse_sum_rows")
    }

    fn sparse_sum_cols(&self, a: &SparseTensor<WgpuRuntime>) -> Result<T2> {
        // A CSC matrix read as CSR is its transpose: col_sums(A) = A^T * ones.
        let csc = self.gpu_csc(a)?;
        let ones = Tensor::ones(&[csc.shape[0]], csc.values.dtype(), &self.device_id)?;
        let transpose_shape = [csc.shape[1], csc.shape[0]];
        crate::dispatch_dtype!(csc.values.dtype(), T => {
            self.spmv_csr_impl::<T>(&csc.col_ptrs, &csc.row_indices, &csc.values, &ones, transpose_shape)
        }, "sparse_sum_cols")
    }

    fn sparse_nnz_per_row(&self, a: &SparseTensor<WgpuRuntime>) -> Result<T2> {
        let csr = self.gpu_csr(a)?;
        self.pointer_spans(&csr.row_ptrs, csr.shape[0])
    }

    fn sparse_nnz_per_col(&self, a: &SparseTensor<WgpuRuntime>) -> Result<T2> {
        let csc = self.gpu_csc(a)?;
        self.pointer_spans(&csc.col_ptrs, csc.shape[1])
    }

    fn sparse_to_dense(&self, a: &SparseTensor<WgpuRuntime>) -> Result<T2> {
        let csr = self.gpu_csr(a)?;
        crate::dispatch_dtype!(csr.values.dtype(), T => {
            self.sparse_to_dense_impl::<T>(&csr.row_ptrs, &csr.col_indices, &csr.values, csr.shape)
        }, "sparse_to_dense")
    }

    fn dense_to_sparse(&self, a: &T2, threshold: f64) -> Result<SparseTensor<WgpuRuntime>> {
        self.dense_to_coo_impl(a, threshold)
    }

    fn dense_to_csr(&self, a: &T2, threshold: f64) -> Result<CsrData<WgpuRuntime>> {
        let coo = self.dense_to_coo_impl(a, threshold)?;
        self.gpu_csr(&coo)
    }

    fn dense_to_csc(&self, a: &T2, threshold: f64) -> Result<CscData<WgpuRuntime>> {
        let coo = self.dense_to_coo_impl(a, threshold)?;
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

    fn sparse_extract_diagonal(&self, a: &SparseTensor<WgpuRuntime>) -> Result<T2> {
        let csr = self.gpu_csr(a)?;
        crate::dispatch_dtype!(csr.values.dtype(), T => {
            self.extract_diagonal_csr_impl::<T>(&csr.row_ptrs, &csr.col_indices, &csr.values, csr.shape)
        }, "sparse_extract_diagonal")
    }

    fn sparse_transpose(&self, a: &SparseTensor<WgpuRuntime>) -> Result<SparseTensor<WgpuRuntime>> {
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

impl WgpuClient {
    /// `ptrs[i + 1] - ptrs[i]` for `i < n`, as I64. WebGPU has no I64
    /// arithmetic, so the difference runs on an I32 copy made on the GPU.
    fn pointer_spans(&self, ptrs: &T2, n: usize) -> Result<T2> {
        if n == 0 {
            return Tensor::zeros(&[0], DType::I64, &self.device_id);
        }
        let ptrs = self.cast(ptrs, DType::I32)?;
        let spans = self.sub(&ptrs.narrow(0, 1, n)?, &ptrs.narrow(0, 0, n)?)?;
        self.cast(&spans, DType::I64)
    }
}
