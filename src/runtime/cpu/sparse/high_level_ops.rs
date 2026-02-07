//! High-level SparseOps trait implementation for CPU runtime
//!
//! This file contains the main SparseOps trait implementation that delegates
//! to specialized submodules for specific operations.

use super::merge::{
    MergeStrategy, OperationSemantics, intersect_coo_impl, merge_coo_impl, merge_csc_impl,
    merge_csr_impl,
};
use super::{CpuClient, CpuRuntime};
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::sparse::SparseOps;
use crate::tensor::Tensor;

impl SparseOps<CpuRuntime> for CpuClient {
    // =========================================================================
    // CSR Operations
    // =========================================================================

    fn spmv_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CpuRuntime>> {
        let [nrows, ncols] = shape;
        let device = values.device();

        // Validate input shapes
        if x.numel() != ncols {
            return Err(Error::ShapeMismatch {
                expected: vec![ncols],
                got: vec![x.numel()],
            });
        }

        // Read CSR data from tensors (CPU: this is just pointer access, no transfer)
        let row_ptrs_data: Vec<i64> = row_ptrs.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();
        let values_data: Vec<T> = values.to_vec();
        let x_data: Vec<T> = x.to_vec();

        // Compute y = A * x
        let mut y_data: Vec<T> = Vec::with_capacity(nrows);

        for row in 0..nrows {
            let start = row_ptrs_data[row] as usize;
            let end = row_ptrs_data[row + 1] as usize;

            // Accumulate in f64 for numerical stability
            let mut sum: f64 = 0.0;
            for j in start..end {
                let col = col_indices_data[j] as usize;
                let val = values_data[j].to_f64();
                let x_val = x_data[col].to_f64();
                sum += val * x_val;
            }
            y_data.push(T::from_f64(sum));
        }

        Ok(Tensor::from_slice(&y_data, &[nrows], device))
    }

    fn spmm_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CpuRuntime>> {
        let [m, k] = shape;
        let device = values.device();

        // Validate B is 2D
        if b.ndim() != 2 {
            return Err(Error::Internal(format!(
                "Expected 2D tensor for SpMM, got {}D",
                b.ndim()
            )));
        }

        let b_shape = b.shape();
        let b_k = b_shape[0];
        let n = b_shape[1];

        // Validate dimensions match
        if b_k != k {
            return Err(Error::ShapeMismatch {
                expected: vec![k],
                got: vec![b_k],
            });
        }

        // Read data from tensors
        let row_ptrs_data: Vec<i64> = row_ptrs.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();
        let a_values: Vec<T> = values.to_vec();
        let b_data: Vec<T> = b.to_vec();

        // Compute C = A * B
        // C is [M, N], stored row-major
        let mut c_data: Vec<T> = vec![T::zero(); m * n];

        for row in 0..m {
            let start = row_ptrs_data[row] as usize;
            let end = row_ptrs_data[row + 1] as usize;

            // For each non-zero in this row of A
            for j in start..end {
                let col = col_indices_data[j] as usize;
                let a_val = a_values[j].to_f64();

                // Update entire row of C
                for col_b in 0..n {
                    let b_val = b_data[col * n + col_b].to_f64();
                    let c_idx = row * n + col_b;
                    let current = c_data[c_idx].to_f64();
                    c_data[c_idx] = T::from_f64(current + a_val * b_val);
                }
            }
        }

        Ok(Tensor::from_slice(&c_data, &[m, n], device))
    }

    fn add_csr<T: Element>(
        &self,
        a_row_ptrs: &Tensor<CpuRuntime>,
        a_col_indices: &Tensor<CpuRuntime>,
        a_values: &Tensor<CpuRuntime>,
        b_row_ptrs: &Tensor<CpuRuntime>,
        b_col_indices: &Tensor<CpuRuntime>,
        b_values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csr_impl::<T, _, _, _>(
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            shape,
            MergeStrategy::Union,
            OperationSemantics::Add,
            |a, b| T::from_f64(a.to_f64() + b.to_f64()), // Both exist: a + b
            |a| a,                                       // Only A: keep as-is
            |b| b,                                       // Only B: keep as-is
        )
    }

    fn sub_csr<T: Element>(
        &self,
        a_row_ptrs: &Tensor<CpuRuntime>,
        a_col_indices: &Tensor<CpuRuntime>,
        a_values: &Tensor<CpuRuntime>,
        b_row_ptrs: &Tensor<CpuRuntime>,
        b_col_indices: &Tensor<CpuRuntime>,
        b_values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csr_impl::<T, _, _, _>(
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            shape,
            MergeStrategy::Union,
            OperationSemantics::Subtract,
            |a, b| T::from_f64(a.to_f64() - b.to_f64()), // Both exist: a - b
            |a| a,                                       // Only A: keep as-is
            |b| T::from_f64(-b.to_f64()),                // Only B: negate it (0 - b = -b)
        )
    }

    fn mul_csr<T: Element>(
        &self,
        a_row_ptrs: &Tensor<CpuRuntime>,
        a_col_indices: &Tensor<CpuRuntime>,
        a_values: &Tensor<CpuRuntime>,
        b_row_ptrs: &Tensor<CpuRuntime>,
        b_col_indices: &Tensor<CpuRuntime>,
        b_values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csr_impl::<T, _, _, _>(
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            shape,
            MergeStrategy::Intersection,
            OperationSemantics::Multiply,
            |a, b| T::from_f64(a.to_f64() * b.to_f64()), // Both exist: a * b
            |a| a, // Only A: unused (Intersection strategy skips)
            |b| b, // Only B: unused (Intersection strategy skips)
        )
    }

    fn div_csr<T: Element>(
        &self,
        a_row_ptrs: &Tensor<CpuRuntime>,
        a_col_indices: &Tensor<CpuRuntime>,
        a_values: &Tensor<CpuRuntime>,
        b_row_ptrs: &Tensor<CpuRuntime>,
        b_col_indices: &Tensor<CpuRuntime>,
        b_values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csr_impl::<T, _, _, _>(
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            shape,
            MergeStrategy::Intersection,
            OperationSemantics::Divide,
            |a, b| T::from_f64(a.to_f64() / b.to_f64()), // Both exist: a / b
            |a| a, // Only A: unused (Intersection strategy skips)
            |b| b, // Only B: unused (Intersection strategy skips)
        )
    }

    // =========================================================================
    // CSC Operations
    // =========================================================================

    fn add_csc<T: Element>(
        &self,
        a_col_ptrs: &Tensor<CpuRuntime>,
        a_row_indices: &Tensor<CpuRuntime>,
        a_values: &Tensor<CpuRuntime>,
        b_col_ptrs: &Tensor<CpuRuntime>,
        b_row_indices: &Tensor<CpuRuntime>,
        b_values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csc_impl::<T, _, _, _>(
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
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
        a_col_ptrs: &Tensor<CpuRuntime>,
        a_row_indices: &Tensor<CpuRuntime>,
        a_values: &Tensor<CpuRuntime>,
        b_col_ptrs: &Tensor<CpuRuntime>,
        b_row_indices: &Tensor<CpuRuntime>,
        b_values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csc_impl::<T, _, _, _>(
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
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
        a_col_ptrs: &Tensor<CpuRuntime>,
        a_row_indices: &Tensor<CpuRuntime>,
        a_values: &Tensor<CpuRuntime>,
        b_col_ptrs: &Tensor<CpuRuntime>,
        b_row_indices: &Tensor<CpuRuntime>,
        b_values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csc_impl::<T, _, _, _>(
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
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
        a_col_ptrs: &Tensor<CpuRuntime>,
        a_row_indices: &Tensor<CpuRuntime>,
        a_values: &Tensor<CpuRuntime>,
        b_col_ptrs: &Tensor<CpuRuntime>,
        b_row_indices: &Tensor<CpuRuntime>,
        b_values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_csc_impl::<T, _, _, _>(
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            shape,
            MergeStrategy::Intersection,
            OperationSemantics::Divide,
            |a, b| T::from_f64(a.to_f64() / b.to_f64()),
            |a| a,
            |b| b,
        )
    }

    // =========================================================================
    // COO Operations
    // =========================================================================

    fn add_coo<T: Element>(
        &self,
        a_row_indices: &Tensor<CpuRuntime>,
        a_col_indices: &Tensor<CpuRuntime>,
        a_values: &Tensor<CpuRuntime>,
        b_row_indices: &Tensor<CpuRuntime>,
        b_col_indices: &Tensor<CpuRuntime>,
        b_values: &Tensor<CpuRuntime>,
        _shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_coo_impl::<T, _, _, _>(
            a_row_indices,
            a_col_indices,
            a_values,
            b_row_indices,
            b_col_indices,
            b_values,
            OperationSemantics::Add,
            |a, b| T::from_f64(a.to_f64() + b.to_f64()), // Both exist: a + b
            |a| a,                                       // A-only: keep a
            |b| b,                                       // B-only: keep b
        )
    }

    fn sub_coo<T: Element>(
        &self,
        a_row_indices: &Tensor<CpuRuntime>,
        a_col_indices: &Tensor<CpuRuntime>,
        a_values: &Tensor<CpuRuntime>,
        b_row_indices: &Tensor<CpuRuntime>,
        b_col_indices: &Tensor<CpuRuntime>,
        b_values: &Tensor<CpuRuntime>,
        _shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        merge_coo_impl::<T, _, _, _>(
            a_row_indices,
            a_col_indices,
            a_values,
            b_row_indices,
            b_col_indices,
            b_values,
            OperationSemantics::Subtract,
            |a, b| T::from_f64(a.to_f64() - b.to_f64()), // Both exist: a - b
            |a| a,                                       // A-only: keep a
            |b| T::from_f64(-b.to_f64()),                // B-only: negate to -b
        )
    }

    fn mul_coo<T: Element>(
        &self,
        a_row_indices: &Tensor<CpuRuntime>,
        a_col_indices: &Tensor<CpuRuntime>,
        a_values: &Tensor<CpuRuntime>,
        b_row_indices: &Tensor<CpuRuntime>,
        b_col_indices: &Tensor<CpuRuntime>,
        b_values: &Tensor<CpuRuntime>,
        _shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        intersect_coo_impl::<T, _>(
            a_row_indices,
            a_col_indices,
            a_values,
            b_row_indices,
            b_col_indices,
            b_values,
            |a, b| T::from_f64(a.to_f64() * b.to_f64()),
        )
    }

    fn div_coo<T: Element>(
        &self,
        a_row_indices: &Tensor<CpuRuntime>,
        a_col_indices: &Tensor<CpuRuntime>,
        a_values: &Tensor<CpuRuntime>,
        b_row_indices: &Tensor<CpuRuntime>,
        b_col_indices: &Tensor<CpuRuntime>,
        b_values: &Tensor<CpuRuntime>,
        _shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        intersect_coo_impl::<T, _>(
            a_row_indices,
            a_col_indices,
            a_values,
            b_row_indices,
            b_col_indices,
            b_values,
            |a, b| T::from_f64(a.to_f64() / b.to_f64()),
        )
    }

    // =========================================================================
    // High-Level Operations
    // =========================================================================
    //
    // Format-agnostic wrappers that convert to optimal format and dispatch
    // to the appropriate low-level implementation.

    fn spmv(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::sparse::SparseTensor;

        // Convert to CSR format (optimal for SpMV)
        let csr = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let shape = csr.shape;
        let dtype = csr.values.dtype();

        // Dispatch to dtype-specific low-level implementation
        crate::dispatch_dtype!(dtype, T => {
            self.spmv_csr::<T>(
                &csr.row_ptrs,
                &csr.col_indices,
                &csr.values,
                x,
                shape,
            )
        }, "spmv")
    }

    fn spmm(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::sparse::SparseTensor;

        // Convert to CSR format (optimal for SpMM)
        let csr = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let shape = csr.shape;
        let dtype = csr.values.dtype();

        // Dispatch to dtype-specific low-level implementation
        crate::dispatch_dtype!(dtype, T => {
            self.spmm_csr::<T>(
                &csr.row_ptrs,
                &csr.col_indices,
                &csr.values,
                b,
                shape,
            )
        }, "spmm")
    }

    fn dsmm(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::algorithm::sparse::SparseAlgorithms;
        use crate::sparse::SparseTensor;

        // Convert to CSC format (optimal for dense * sparse)
        let csc = match b {
            SparseTensor::Csc(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csc()?,
            SparseTensor::Csr(data) => data.to_csc()?,
        };

        // Delegate to algorithm trait (backend-consistent implementation)
        self.column_parallel_dsmm(a, &csc)
    }

    fn sparse_add(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        b: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        use crate::sparse::SparseTensor;

        // Validate shapes match
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: vec![a.shape()[0], a.shape()[1]],
                got: vec![b.shape()[0], b.shape()[1]],
            });
        }

        // Validate dtypes match
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }

        // Convert both to CSR format for efficient addition
        let csr_a = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let csr_b = match b {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        // Perform addition and return as CSR
        let result = csr_a.add(&csr_b)?;
        Ok(SparseTensor::Csr(result))
    }

    fn sparse_sub(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        b: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        use crate::sparse::SparseTensor;

        // Validate shapes match
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: vec![a.shape()[0], a.shape()[1]],
                got: vec![b.shape()[0], b.shape()[1]],
            });
        }

        // Validate dtypes match
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }

        // Convert both to CSR format for efficient subtraction
        let csr_a = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let csr_b = match b {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        // Perform subtraction and return as CSR
        let result = csr_a.sub(&csr_b)?;
        Ok(SparseTensor::Csr(result))
    }

    fn sparse_matmul(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        b: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        use crate::algorithm::sparse::{SparseAlgorithms, validate_dtype_match};
        use crate::sparse::SparseTensor;

        // Convert both to CSR format for efficient row-wise computation
        let csr_a = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let csr_b = match b {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        // Validate dtypes match
        validate_dtype_match(csr_a.values.dtype(), csr_b.values.dtype())?;

        // Delegate to ESC SpGEMM algorithm (backend-consistent implementation)
        let result_csr = self.esc_spgemm_csr(&csr_a, &csr_b)?;

        Ok(SparseTensor::Csr(result_csr))
    }

    fn sparse_mul(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        b: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        use crate::sparse::SparseTensor;

        // Validate shapes match
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: vec![a.shape()[0], a.shape()[1]],
                got: vec![b.shape()[0], b.shape()[1]],
            });
        }

        // Validate dtypes match
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }

        // Convert both to CSR format for efficient element-wise multiplication
        let csr_a = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let csr_b = match b {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        // Perform element-wise multiplication and return as CSR
        let result = csr_a.mul(&csr_b)?;
        Ok(SparseTensor::Csr(result))
    }

    fn sparse_scale(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
        scalar: f64,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        use crate::ops::ScalarOps;
        use crate::sparse::SparseTensor;

        // Handle empty tensors without calling mul_scalar
        if a.nnz() == 0 {
            return Ok(a.clone());
        }

        // mul_scalar already handles dtype dispatch, no need for outer dispatch
        match a {
            SparseTensor::Csr(data) => {
                let scaled_values = self.mul_scalar(&data.values, scalar)?;
                let result = crate::sparse::CsrData {
                    row_ptrs: data.row_ptrs.clone(),
                    col_indices: data.col_indices.clone(),
                    values: scaled_values,
                    shape: data.shape,
                };
                Ok(SparseTensor::Csr(result))
            }
            SparseTensor::Csc(data) => {
                let scaled_values = self.mul_scalar(&data.values, scalar)?;
                let result = crate::sparse::CscData {
                    col_ptrs: data.col_ptrs.clone(),
                    row_indices: data.row_indices.clone(),
                    values: scaled_values,
                    shape: data.shape,
                };
                Ok(SparseTensor::Csc(result))
            }
            SparseTensor::Coo(data) => {
                let scaled_values = self.mul_scalar(&data.values, scalar)?;
                let result = crate::sparse::CooData {
                    row_indices: data.row_indices.clone(),
                    col_indices: data.col_indices.clone(),
                    values: scaled_values,
                    shape: data.shape,
                    sorted: data.sorted,
                };
                Ok(SparseTensor::Coo(result))
            }
        }
    }

    fn sparse_add_scalar(
        &self,
        _a: &crate::sparse::SparseTensor<CpuRuntime>,
        _scalar: f64,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        // Adding a scalar to a sparse matrix would make it dense
        // (all implicit zeros become non-zero)
        Err(Error::Internal(
            "Scalar addition to sparse matrix creates dense result - convert to dense first"
                .to_string(),
        ))
    }

    fn sparse_sum(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::sparse::SparseTensor;

        let dtype = a.dtype();
        let device = match a {
            SparseTensor::Csr(data) => data.values.device(),
            SparseTensor::Csc(data) => data.values.device(),
            SparseTensor::Coo(data) => data.values.device(),
        };

        // Sum all non-zero values
        crate::dispatch_dtype!(dtype, T => {
            let values = match a {
                SparseTensor::Csr(data) => &data.values,
                SparseTensor::Csc(data) => &data.values,
                SparseTensor::Coo(data) => &data.values,
            };

            let values_vec: Vec<T> = values.to_vec();
            let sum: f64 = values_vec.iter().map(|v| v.to_f64()).sum();

            Ok(Tensor::from_slice(&[T::from_f64(sum)], &[1], device))
        }, "sparse_sum")
    }

    fn sparse_sum_rows(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::sparse::SparseTensor;

        // Convert to CSR format (optimal for row operations)
        let csr = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let [nrows, _] = csr.shape;
        let dtype = csr.values.dtype();
        let device = csr.values.device();

        // Sum each row
        crate::dispatch_dtype!(dtype, T => {
            let row_ptrs: Vec<i64> = csr.row_ptrs.to_vec();
            let values: Vec<T> = csr.values.to_vec();

            let mut row_sums: Vec<T> = Vec::with_capacity(nrows);

            for row in 0..nrows {
                let start = row_ptrs[row] as usize;
                let end = row_ptrs[row + 1] as usize;

                let sum: f64 = values[start..end].iter().map(|v| v.to_f64()).sum();
                row_sums.push(T::from_f64(sum));
            }

            Ok(Tensor::from_slice(&row_sums, &[nrows], device))
        }, "sparse_sum_rows")
    }

    fn sparse_sum_cols(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::sparse::SparseTensor;

        // Convert to CSC format (optimal for column operations)
        let csc = match a {
            SparseTensor::Csc(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csc()?,
            SparseTensor::Csr(data) => data.to_csc()?,
        };

        let [_, ncols] = csc.shape;
        let dtype = csc.values.dtype();
        let device = csc.values.device();

        // Sum each column
        crate::dispatch_dtype!(dtype, T => {
            let col_ptrs: Vec<i64> = csc.col_ptrs.to_vec();
            let values: Vec<T> = csc.values.to_vec();

            let mut col_sums: Vec<T> = Vec::with_capacity(ncols);

            for col in 0..ncols {
                let start = col_ptrs[col] as usize;
                let end = col_ptrs[col + 1] as usize;

                let sum: f64 = values[start..end].iter().map(|v| v.to_f64()).sum();
                col_sums.push(T::from_f64(sum));
            }

            Ok(Tensor::from_slice(&col_sums, &[ncols], device))
        }, "sparse_sum_cols")
    }

    fn sparse_nnz_per_row(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::sparse::SparseTensor;

        // Convert to CSR format (optimal for row operations)
        let csr = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let [nrows, _] = csr.shape;
        let device = csr.values.device();

        // Count non-zeros per row
        let row_ptrs: Vec<i64> = csr.row_ptrs.to_vec();
        let mut nnz_counts: Vec<i64> = Vec::with_capacity(nrows);

        for row in 0..nrows {
            let start = row_ptrs[row];
            let end = row_ptrs[row + 1];
            nnz_counts.push(end - start);
        }

        Ok(Tensor::from_slice(&nnz_counts, &[nrows], device))
    }

    fn sparse_nnz_per_col(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::sparse::SparseTensor;

        // Convert to CSC format (optimal for column operations)
        let csc = match a {
            SparseTensor::Csc(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csc()?,
            SparseTensor::Csr(data) => data.to_csc()?,
        };

        let [_, ncols] = csc.shape;
        let device = csc.values.device();

        // Count non-zeros per column
        let col_ptrs: Vec<i64> = csc.col_ptrs.to_vec();
        let mut nnz_counts: Vec<i64> = Vec::with_capacity(ncols);

        for col in 0..ncols {
            let start = col_ptrs[col];
            let end = col_ptrs[col + 1];
            nnz_counts.push(end - start);
        }

        Ok(Tensor::from_slice(&nnz_counts, &[ncols], device))
    }

    fn sparse_to_dense(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::sparse::SparseTensor;

        // Convert to CSR format for efficient conversion
        let csr = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let [nrows, ncols] = csr.shape;
        let dtype = csr.values.dtype();
        let device = csr.values.device();

        // Create dense matrix and fill with sparse values
        crate::dispatch_dtype!(dtype, T => {
            let row_ptrs: Vec<i64> = csr.row_ptrs.to_vec();
            let col_indices: Vec<i64> = csr.col_indices.to_vec();
            let values: Vec<T> = csr.values.to_vec();

            let mut dense: Vec<T> = vec![T::zero(); nrows * ncols];

            for row in 0..nrows {
                let start = row_ptrs[row] as usize;
                let end = row_ptrs[row + 1] as usize;

                for idx in start..end {
                    let col = col_indices[idx] as usize;
                    dense[row * ncols + col] = values[idx];
                }
            }

            Ok(Tensor::from_slice(&dense, &[nrows, ncols], device))
        }, "sparse_to_dense")
    }

    fn dense_to_sparse(
        &self,
        a: &Tensor<CpuRuntime>,
        threshold: f64,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        use crate::sparse::SparseTensor;

        // Validate input is 2D
        if a.ndim() != 2 {
            return Err(Error::Internal(format!(
                "Expected 2D tensor for dense_to_sparse, got {}D",
                a.ndim()
            )));
        }

        let shape_vec = a.shape();
        let nrows = shape_vec[0];
        let ncols = shape_vec[1];
        let dtype = a.dtype();
        let device = a.device();

        // Find non-zero elements and build COO format
        crate::dispatch_dtype!(dtype, T => {
            let data: Vec<T> = a.to_vec();

            let mut row_indices: Vec<i64> = Vec::new();
            let mut col_indices: Vec<i64> = Vec::new();
            let mut values: Vec<T> = Vec::new();

            for row in 0..nrows {
                for col in 0..ncols {
                    let val = data[row * ncols + col];
                    if val.to_f64().abs() >= threshold {
                        row_indices.push(row as i64);
                        col_indices.push(col as i64);
                        values.push(val);
                    }
                }
            }

            // Create COO sparse tensor
            let mut coo = crate::sparse::CooData::from_slices(
                &row_indices,
                &col_indices,
                &values,
                [nrows, ncols],
                device,
            )?;

            // Data is already sorted by row (then column) from our iteration order
            // SAFETY: The row-major iteration order guarantees sorted output
            unsafe { coo.set_sorted(true); }

            Ok(SparseTensor::Coo(coo))
        }, "dense_to_sparse")
    }

    // =========================================================================
    // Format Conversions (Low-Level)
    // =========================================================================

    fn coo_to_csr<T: Element>(
        &self,
        row_indices: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let [nrows, _ncols] = shape;
        let nnz = values.numel();
        let device = values.device();

        // Read COO data
        let row_idx: Vec<i64> = row_indices.to_vec();
        let col_idx: Vec<i64> = col_indices.to_vec();
        let vals: Vec<T> = values.to_vec();

        // Create permutation vector [0, 1, 2, ..., nnz-1]
        let mut perm: Vec<usize> = (0..nnz).collect();

        // Sort by (row, col) using permutation
        perm.sort_by_key(|&i| (row_idx[i], col_idx[i]));

        // Apply permutation to create sorted arrays
        let mut sorted_col_indices = Vec::with_capacity(nnz);
        let mut sorted_values = Vec::with_capacity(nnz);

        for &i in &perm {
            sorted_col_indices.push(col_idx[i]);
            sorted_values.push(vals[i]);
        }

        // Build row pointers via counting
        let mut row_ptrs = vec![0i64; nrows + 1];
        for &i in &perm {
            let row = row_idx[i] as usize;
            row_ptrs[row + 1] += 1;
        }

        // Prefix sum to convert counts to pointers
        for i in 1..=nrows {
            row_ptrs[i] += row_ptrs[i - 1];
        }

        // Create tensors
        let row_ptrs_tensor = Tensor::from_slice(&row_ptrs, &[nrows + 1], device);
        let col_indices_tensor = Tensor::from_slice(&sorted_col_indices, &[nnz], device);
        let values_tensor = Tensor::from_slice(&sorted_values, &[nnz], device);

        Ok((row_ptrs_tensor, col_indices_tensor, values_tensor))
    }

    fn coo_to_csc<T: Element>(
        &self,
        row_indices: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let [_nrows, ncols] = shape;
        let nnz = values.numel();
        let device = values.device();

        // Read COO data
        let row_idx: Vec<i64> = row_indices.to_vec();
        let col_idx: Vec<i64> = col_indices.to_vec();
        let vals: Vec<T> = values.to_vec();

        // Create permutation vector
        let mut perm: Vec<usize> = (0..nnz).collect();

        // Sort by (col, row) for CSC
        perm.sort_by_key(|&i| (col_idx[i], row_idx[i]));

        // Apply permutation
        let mut sorted_row_indices = Vec::with_capacity(nnz);
        let mut sorted_values = Vec::with_capacity(nnz);

        for &i in &perm {
            sorted_row_indices.push(row_idx[i]);
            sorted_values.push(vals[i]);
        }

        // Build column pointers
        let mut col_ptrs = vec![0i64; ncols + 1];
        for &i in &perm {
            let col = col_idx[i] as usize;
            col_ptrs[col + 1] += 1;
        }

        // Prefix sum
        for i in 1..=ncols {
            col_ptrs[i] += col_ptrs[i - 1];
        }

        // Create tensors
        let col_ptrs_tensor = Tensor::from_slice(&col_ptrs, &[ncols + 1], device);
        let row_indices_tensor = Tensor::from_slice(&sorted_row_indices, &[nnz], device);
        let values_tensor = Tensor::from_slice(&sorted_values, &[nnz], device);

        Ok((col_ptrs_tensor, row_indices_tensor, values_tensor))
    }

    fn csr_to_coo<T: Element>(
        &self,
        row_ptrs: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let [nrows, _ncols] = shape;
        let nnz = values.numel();
        let device = values.device();

        // Read row pointers
        let ptrs: Vec<i64> = row_ptrs.to_vec();

        // Expand row pointers to row indices
        let mut row_indices = Vec::with_capacity(nnz);
        for row in 0..nrows {
            let start = ptrs[row] as usize;
            let end = ptrs[row + 1] as usize;
            for _ in start..end {
                row_indices.push(row as i64);
            }
        }

        // Create row indices tensor (col_indices and values stay the same)
        let row_indices_tensor = Tensor::from_slice(&row_indices, &[nnz], device);

        Ok((row_indices_tensor, col_indices.clone(), values.clone()))
    }

    fn csc_to_coo<T: Element>(
        &self,
        col_ptrs: &Tensor<CpuRuntime>,
        row_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let [_nrows, ncols] = shape;
        let nnz = values.numel();
        let device = values.device();

        // Read column pointers
        let ptrs: Vec<i64> = col_ptrs.to_vec();

        // Expand column pointers to column indices
        let mut col_indices = Vec::with_capacity(nnz);
        for col in 0..ncols {
            let start = ptrs[col] as usize;
            let end = ptrs[col + 1] as usize;
            for _ in start..end {
                col_indices.push(col as i64);
            }
        }

        // Create column indices tensor
        let col_indices_tensor = Tensor::from_slice(&col_indices, &[nnz], device);

        Ok((row_indices.clone(), col_indices_tensor, values.clone()))
    }

    fn csr_to_csc<T: Element>(
        &self,
        row_ptrs: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let [nrows, ncols] = shape;
        let nnz = values.numel();
        let device = values.device();

        // Read CSR data
        let row_ptr: Vec<i64> = row_ptrs.to_vec();
        let col_idx: Vec<i64> = col_indices.to_vec();
        let vals: Vec<T> = values.to_vec();

        // Count non-zeros per column
        let mut col_counts = vec![0usize; ncols];
        for &col in &col_idx {
            col_counts[col as usize] += 1;
        }

        // Build column pointers
        let mut col_ptrs = vec![0i64; ncols + 1];
        for col in 0..ncols {
            col_ptrs[col + 1] = col_ptrs[col] + col_counts[col] as i64;
        }

        // Allocate output arrays
        let mut new_row_indices = vec![0i64; nnz];
        let mut new_values = vec![T::from_f64(0.0); nnz];

        // Track current position in each column
        let mut col_positions = col_ptrs[..ncols].to_vec();

        // Scatter CSR entries into CSC format
        for row in 0..nrows {
            let start = row_ptr[row] as usize;
            let end = row_ptr[row + 1] as usize;

            for idx in start..end {
                let col = col_idx[idx] as usize;
                let pos = col_positions[col] as usize;

                new_row_indices[pos] = row as i64;
                new_values[pos] = vals[idx];

                col_positions[col] += 1;
            }
        }

        // Create tensors
        let col_ptrs_tensor = Tensor::from_slice(&col_ptrs, &[ncols + 1], device);
        let row_indices_tensor = Tensor::from_slice(&new_row_indices, &[nnz], device);
        let values_tensor = Tensor::from_slice(&new_values, &[nnz], device);

        Ok((col_ptrs_tensor, row_indices_tensor, values_tensor))
    }

    fn csc_to_csr<T: Element>(
        &self,
        col_ptrs: &Tensor<CpuRuntime>,
        row_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        let [nrows, ncols] = shape;
        let nnz = values.numel();
        let device = values.device();

        // Read CSC data
        let col_ptr: Vec<i64> = col_ptrs.to_vec();
        let row_idx: Vec<i64> = row_indices.to_vec();
        let vals: Vec<T> = values.to_vec();

        // Count non-zeros per row
        let mut row_counts = vec![0usize; nrows];
        for &row in &row_idx {
            row_counts[row as usize] += 1;
        }

        // Build row pointers
        let mut row_ptrs = vec![0i64; nrows + 1];
        for row in 0..nrows {
            row_ptrs[row + 1] = row_ptrs[row] + row_counts[row] as i64;
        }

        // Allocate output arrays
        let mut new_col_indices = vec![0i64; nnz];
        let mut new_values = vec![T::from_f64(0.0); nnz];

        // Track current position in each row
        let mut row_positions = row_ptrs[..nrows].to_vec();

        // Scatter CSC entries into CSR format
        for col in 0..ncols {
            let start = col_ptr[col] as usize;
            let end = col_ptr[col + 1] as usize;

            for idx in start..end {
                let row = row_idx[idx] as usize;
                let pos = row_positions[row] as usize;

                new_col_indices[pos] = col as i64;
                new_values[pos] = vals[idx];

                row_positions[row] += 1;
            }
        }

        // Create tensors
        let row_ptrs_tensor = Tensor::from_slice(&row_ptrs, &[nrows + 1], device);
        let col_indices_tensor = Tensor::from_slice(&new_col_indices, &[nnz], device);
        let values_tensor = Tensor::from_slice(&new_values, &[nnz], device);

        Ok((row_ptrs_tensor, col_indices_tensor, values_tensor))
    }

    fn extract_diagonal_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<CpuRuntime>,
        col_indices: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CpuRuntime>> {
        let [nrows, ncols] = shape;
        let n = nrows.min(ncols);
        let device = values.device();

        if n == 0 {
            return Ok(Tensor::empty(&[0], values.dtype(), device));
        }

        let row_ptrs_data: Vec<i64> = row_ptrs.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();
        let values_data: Vec<T> = values.to_vec();

        let mut diag = vec![T::zero(); n];
        for row in 0..n {
            let start = row_ptrs_data[row] as usize;
            let end = row_ptrs_data[row + 1] as usize;
            for pos in start..end {
                if col_indices_data[pos] as usize == row {
                    diag[row] = values_data[pos];
                    break;
                }
            }
        }

        Ok(Tensor::from_slice(&diag, &[n], device))
    }

    fn sparse_transpose(
        &self,
        a: &crate::sparse::SparseTensor<CpuRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CpuRuntime>> {
        use crate::sparse::SparseTensor;

        // Transpose is efficient format conversion:
        // - CSR -> CSC (swap interpretation)
        // - CSC -> CSR (swap interpretation)
        // - COO -> COO (swap row/col indices)
        match a {
            SparseTensor::Csr(data) => {
                // CSR transpose = CSC with swapped dimensions
                let csc = data.to_csc()?;
                Ok(SparseTensor::Csc(csc))
            }
            SparseTensor::Csc(data) => {
                // CSC transpose = CSR with swapped dimensions
                let csr = data.to_csr()?;
                Ok(SparseTensor::Csr(csr))
            }
            SparseTensor::Coo(data) => {
                // COO transpose: swap row and column indices
                let [nrows, ncols] = data.shape;
                let transposed = crate::sparse::CooData {
                    row_indices: data.col_indices.clone(),
                    col_indices: data.row_indices.clone(),
                    values: data.values.clone(),
                    shape: [ncols, nrows], // Swap dimensions
                    sorted: false,         // After swapping, no longer sorted
                };
                Ok(SparseTensor::Coo(transposed))
            }
        }
    }
}
