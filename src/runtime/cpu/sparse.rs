//! Sparse operations implementation for CPU runtime
//!
//! This module implements the SparseOps trait for CpuRuntime, providing
//! CPU-based sparse matrix operations.

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
        shape: [usize; 2],
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
        shape: [usize; 2],
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
        shape: [usize; 2],
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
        shape: [usize; 2],
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
    // High-Level Operations (Stub implementations for now)
    // =========================================================================
    //
    // These will be properly implemented once we refactor the sparse formats
    // to use the low-level trait methods.

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
        use crate::sparse::SparseTensor;

        // Convert to CSC format (optimal for dense * sparse)
        // A [M, K] * B [K, N] where B is sparse
        let csc = match b {
            SparseTensor::Csc(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csc()?,
            SparseTensor::Csr(data) => data.to_csc()?,
        };

        let [k, n] = csc.shape;
        let dtype = csc.values.dtype();

        // Validate A dimensions
        if a.ndim() != 2 {
            return Err(Error::Internal(format!(
                "Expected 2D tensor for dense matrix, got {}D",
                a.ndim()
            )));
        }

        let a_shape = a.shape();
        let m = a_shape[0];
        let a_k = a_shape[1];

        if a_k != k {
            return Err(Error::ShapeMismatch {
                expected: vec![k],
                got: vec![a_k],
            });
        }

        // Dense * Sparse CSC: For each column in B, multiply with corresponding rows of A
        let device = a.device();

        crate::dispatch_dtype!(dtype, T => {
            let a_data: Vec<T> = a.to_vec();
            let col_ptrs_data: Vec<i64> = csc.col_ptrs.to_vec();
            let row_indices_data: Vec<i64> = csc.row_indices.to_vec();
            let b_values: Vec<T> = csc.values.to_vec();

            // Result C [M, N]
            let mut c_data: Vec<T> = vec![T::zero(); m * n];

            // For each column j in B
            for col in 0..n {
                let start = col_ptrs_data[col] as usize;
                let end = col_ptrs_data[col + 1] as usize;

                // For each non-zero in column j
                for idx in start..end {
                    let row_b = row_indices_data[idx] as usize; // row in B
                    let b_val = b_values[idx].to_f64();

                    // Update column j of C: C[:, j] += A[:, row_b] * b_val
                    for row_a in 0..m {
                        let a_val = a_data[row_a * k + row_b].to_f64();
                        let c_idx = row_a * n + col;
                        let current = c_data[c_idx].to_f64();
                        c_data[c_idx] = T::from_f64(current + a_val * b_val);
                    }
                }
            }

            Ok(Tensor::from_slice(&c_data, &[m, n], device))
        }, "dsmm")
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
        use crate::sparse::SparseTensor;
        use std::collections::HashMap;

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

        let [m, k_a] = csr_a.shape;
        let [k_b, n] = csr_b.shape;

        // Validate dimensions
        if k_a != k_b {
            return Err(Error::ShapeMismatch {
                expected: vec![k_a],
                got: vec![k_b],
            });
        }

        // Validate dtypes match
        if csr_a.values.dtype() != csr_b.values.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: csr_a.values.dtype(),
                rhs: csr_b.values.dtype(),
            });
        }

        let dtype = csr_a.values.dtype();
        let device = csr_a.values.device();

        // Perform CSR × CSR multiplication: C = A * B
        crate::dispatch_dtype!(dtype, T => {
            let a_row_ptrs: Vec<i64> = csr_a.row_ptrs.to_vec();
            let a_col_indices: Vec<i64> = csr_a.col_indices.to_vec();
            let a_values: Vec<T> = csr_a.values.to_vec();

            let b_row_ptrs: Vec<i64> = csr_b.row_ptrs.to_vec();
            let b_col_indices: Vec<i64> = csr_b.col_indices.to_vec();
            let b_values: Vec<T> = csr_b.values.to_vec();

            // Build result CSR
            let mut c_row_ptrs: Vec<i64> = Vec::with_capacity(m + 1);
            let mut c_col_indices: Vec<i64> = Vec::new();
            let mut c_values: Vec<T> = Vec::new();

            c_row_ptrs.push(0);

            // For each row i in A
            for i in 0..m {
                let a_start = a_row_ptrs[i] as usize;
                let a_end = a_row_ptrs[i + 1] as usize;

                // Accumulator for row i of C: maps column index -> value
                let mut row_accum: HashMap<usize, f64> = HashMap::new();

                // For each non-zero A[i, k]
                for a_idx in a_start..a_end {
                    let k = a_col_indices[a_idx] as usize;
                    let a_val = a_values[a_idx].to_f64();

                    // Multiply with row k of B
                    let b_start = b_row_ptrs[k] as usize;
                    let b_end = b_row_ptrs[k + 1] as usize;

                    for b_idx in b_start..b_end {
                        let j = b_col_indices[b_idx] as usize;
                        let b_val = b_values[b_idx].to_f64();

                        *row_accum.entry(j).or_insert(0.0) += a_val * b_val;
                    }
                }

                // Sort and add non-zeros to result
                let mut row_entries: Vec<(usize, f64)> = row_accum.into_iter().collect();
                row_entries.sort_by_key(|&(col, _)| col);

                for (col, val) in row_entries {
                    if val.abs() > zero_tolerance::<T>() {
                        c_col_indices.push(col as i64);
                        c_values.push(T::from_f64(val));
                    }
                }

                c_row_ptrs.push(c_col_indices.len() as i64);
            }

            // Create result CSR tensors
            let result_row_ptrs = Tensor::from_slice(&c_row_ptrs, &[m + 1], device);
            let result_col_indices = Tensor::from_slice(&c_col_indices, &[c_col_indices.len()], device);
            let result_values = Tensor::from_slice(&c_values, &[c_values.len()], device);

            let result_csr = crate::sparse::CsrData {
                row_ptrs: result_row_ptrs,
                col_indices: result_col_indices,
                values: result_values,
                shape: [m, n],
            };

            Ok(SparseTensor::Csr(result_csr))
        }, "sparse_matmul")
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
            let coo = crate::sparse::CooData::from_slices(
                &row_indices,
                &col_indices,
                &values,
                [nrows, ncols],
                device,
            )?;

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

// =============================================================================
// Helper Functions
// =============================================================================

/// Returns dtype-aware tolerance for determining zero values
///
/// Different dtypes have different precision levels:
/// - F64: Machine epsilon ~2e-16, use 1e-15
/// - F32: Machine epsilon ~1e-7, use 1e-7
/// - F16/BF16: Lower precision, use 1e-3
/// - FP8: Very low precision, use 1e-2
#[inline]
/// Returns the zero-elimination tolerance for a given element type.
///
/// This function provides dtype-specific tolerance values used to filter out
/// near-zero values from sparse matrix results. Values with absolute value
/// below the tolerance are treated as zero and eliminated from the sparse
/// representation.
///
/// # Rationale
///
/// Sparse matrix operations can produce tiny values due to floating-point
/// rounding errors, especially when subtracting nearly equal numbers or
/// accumulating many small values. Without zero-elimination:
/// - Memory usage increases unnecessarily
/// - Subsequent operations become slower (more non-zeros to process)
/// - Numerical stability can degrade
///
/// # Tolerance Values
///
/// Tolerances are chosen based on dtype precision and typical numerical error:
///
/// - **8-byte types** (F64, I64, U64): `1e-15`
///   - F64 has ~15-16 decimal digits of precision
///   - Tolerance captures rounding errors from ~15 operations
///
/// - **4-byte types** (F32, I32, U32): `1e-7`
///   - F32 has ~6-7 decimal digits of precision
///   - Tolerance is ~7 ULPs (units in last place)
///
/// - **2-byte types** (F16, BF16, I16, U16): `1e-3`
///   - F16/BF16 have limited mantissa precision
///   - More aggressive filtering due to accumulated errors
///
/// - **1-byte types** (FP8, I8, U8): `1e-2`
///   - Extremely limited precision (FP8: 3-4 bits mantissa)
///   - Very aggressive filtering to maintain sparsity
///
/// # Design Considerations
///
/// 1. **Conservative for high precision**: F64 tolerance is 15 orders of
///    magnitude below typical values, preserving mathematically significant
///    numbers while eliminating numerical noise.
///
/// 2. **Aggressive for low precision**: FP8/F16 have inherent rounding,
///    so we eliminate more aggressively to maintain sparsity benefits.
///
/// 3. **Integer types**: Use same tolerance as corresponding float types
///    to maintain consistent behavior across dtype families.
///
/// # Examples
///
/// ```ignore
/// // F32: eliminates values < 1e-7
/// let result = a - b;  // If result is 1e-8, it's eliminated
///
/// // F64: preserves more precision
/// let result = a - b;  // Values down to 1e-15 are kept
/// ```
fn zero_tolerance<T: Element>() -> f64 {
    use std::mem::size_of;
    match size_of::<T>() {
        8 => 1e-15, // F64, I64, U64: ~machine epsilon
        4 => 1e-7,  // F32, I32, U32: ~7 ULPs
        2 => 1e-3,  // F16, BF16, I16, U16: aggressive due to limited precision
        1 => 1e-2,  // FP8, I8, U8: very aggressive due to extreme quantization
        _ => 1e-15, // Default: fallback to highest precision
    }
}

// =============================================================================
// Merge Strategy and Operation Semantics
// =============================================================================
//
// This section implements a parameterized merge pattern that eliminates
// code duplication across sparse matrix element-wise operations.
//
// # Design Pattern
//
// Instead of having separate implementations for add, subtract, multiply, and
// divide, we have:
// - ONE generic merge function per format (CSR, CSC, COO)
// - Strategy enum to control which positions to keep (union vs intersection)
// - Semantics enum to control empty matrix handling
// - Operation closures to define element-wise computation
//
// # Benefits
//
// 1. **Code Reuse**: Single merge implementation handles all operations
//    - Before: ~900 lines (6 specialized functions)
//    - After: ~300 lines (2 parameterized functions)
//
// 2. **Consistency**: All operations use identical merge logic, reducing bugs
//
// 3. **Extensibility**: New operations (e.g., max, min) require only:
//    - Adding semantics to enum
//    - Writing operation closure
//    - No new merge code needed
//
// 4. **Testability**: Testing merge logic once covers all operations
//
// # Architecture
//
// ```text
// User-facing trait method (add_csr, sub_csr, mul_csr, div_csr)
//       │
//       ├─> Calls merge_csr_impl with:
//       │   - MergeStrategy (Union or Intersection)
//       │   - OperationSemantics (Add, Subtract, Multiply, Divide)
//       │   - Operation closure: |a, b| a + b
//       │   - Transform closures: |a| a, |b| -b
//       │
//       └─> merge_csr_impl:
//           1. Check empty matrices (using semantics)
//           2. Merge based on strategy (union/intersection)
//           3. Apply operations via closures
//           4. Filter near-zeros
// ```
//
// # Example Usage
//
// ```ignore
// // Addition: union merge, identity transforms
// fn add_csr<T>(...) -> Result<...> {
//     merge_csr_impl(
//         ...,
//         MergeStrategy::Union,
//         OperationSemantics::Add,
//         |a, b| a + b,  // Both exist
//         |a| a,         // A-only: keep
//         |b| b,         // B-only: keep
//     )
// }
//
// // Subtraction: union merge, negate B
// fn sub_csr<T>(...) -> Result<...> {
//     merge_csr_impl(
//         ...,
//         MergeStrategy::Union,
//         OperationSemantics::Subtract,
//         |a, b| a - b,  // Both exist
//         |a| a,         // A-only: keep
//         |b| -b,        // B-only: negate
//     )
// }
//
// // Multiplication: intersection merge (0 * x = 0)
// fn mul_csr<T>(...) -> Result<...> {
//     merge_csr_impl(
//         ...,
//         MergeStrategy::Intersection,
//         OperationSemantics::Multiply,
//         |a, b| a * b,  // Both exist
//         |a| a,         // Unused (intersection only)
//         |b| b,         // Unused (intersection only)
//     )
// }
// ```
//
// # Empty Matrix Handling
//
// OperationSemantics controls what happens when one or both matrices are empty:
//
// | Operation | A empty, B not | A not, B empty | Both empty |
// |-----------|----------------|----------------|------------|
// | Add       | Return B       | Return A       | Empty      |
// | Subtract  | Return -B      | Return A       | Empty      |
// | Multiply  | Empty (0*x=0)  | Empty (x*0=0)  | Empty      |
// | Divide    | Empty (0/x=0)  | Error (x/0)    | Empty      |
//
// This logic is centralized in handle_empty_compressed() and avoids
// duplicating empty-check code across operations.

/// Strategy for merging two sparse matrices
///
/// Determines which positions to keep in the result:
/// - Union: Keep positions that exist in EITHER matrix (OR semantics)
/// - Intersection: Keep positions that exist in BOTH matrices (AND semantics)
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum MergeStrategy {
    /// Union semantics: Keep all positions from both matrices
    ///
    /// Example: Addition, Subtraction
    /// - Position in A only: keep A's value
    /// - Position in B only: keep B's value (or -B for subtraction)
    /// - Position in both: apply operation
    Union,

    /// Intersection semantics: Keep only positions where both matrices have values
    ///
    /// Example: Multiplication, Division
    /// - Position in A only: skip (0 * B = 0)
    /// - Position in B only: skip (A * 0 = 0)
    /// - Position in both: apply operation
    Intersection,
}

/// Operation semantics for element-wise operations
///
/// Defines how to handle empty matrices and special cases
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum OperationSemantics {
    /// Addition: 0 + x = x, x + 0 = x
    Add,
    /// Subtraction: x - 0 = x, 0 - x = -x
    Subtract,
    /// Multiplication: 0 * x = 0, x * 0 = 0
    Multiply,
    /// Division: 0 / x = 0, x / 0 = error
    Divide,
}

/// Generic centralized empty matrix handling for CSR/CSC operations
///
/// Returns Some(result) if the operation can be short-circuited due to empty inputs,
/// None if both inputs are non-empty and normal processing should continue.
///
/// # Type Parameters
///
/// * `format_is_csr` - If true, uses row-based pointers (CSR), else column-based (CSC)
fn handle_empty_compressed<T: Element>(
    a_nnz: usize,
    b_nnz: usize,
    shape: [usize; 2],
    device: &<CpuRuntime as crate::runtime::Runtime>::Device,
    a_ptrs: &Tensor<CpuRuntime>,
    a_indices: &Tensor<CpuRuntime>,
    a_values: &Tensor<CpuRuntime>,
    b_ptrs: &Tensor<CpuRuntime>,
    b_indices: &Tensor<CpuRuntime>,
    b_values: &Tensor<CpuRuntime>,
    semantics: OperationSemantics,
    format_is_csr: bool,
) -> Option<Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>> {
    // CSR uses nrows, CSC uses ncols for pointer dimension
    let ptr_dim = if format_is_csr { shape[0] } else { shape[1] };

    let empty_result = || {
        let empty_ptrs = Tensor::from_slice(&vec![0i64; ptr_dim + 1], &[ptr_dim + 1], device);
        let empty_indices = Tensor::from_slice(&Vec::<i64>::new(), &[0], device);
        let empty_vals = Tensor::from_slice(&Vec::<T>::new(), &[0], device);
        (empty_ptrs, empty_indices, empty_vals)
    };

    match (a_nnz, b_nnz, semantics) {
        // Both empty - always return empty
        (0, 0, _) => Some(Ok(empty_result())),

        // A empty, B not empty
        (0, _, OperationSemantics::Add) => {
            // 0 + B = B
            Some(Ok((b_ptrs.clone(), b_indices.clone(), b_values.clone())))
        }
        (0, _, OperationSemantics::Subtract) => {
            // 0 - B = -B
            let b_vals: Vec<T> = b_values.to_vec();
            let negated_vals: Vec<T> = b_vals.iter().map(|&v| T::from_f64(-v.to_f64())).collect();
            let out_vals = Tensor::from_slice(&negated_vals, &[negated_vals.len()], device);
            Some(Ok((b_ptrs.clone(), b_indices.clone(), out_vals)))
        }
        (0, _, OperationSemantics::Multiply) => {
            // 0 * B = 0
            Some(Ok(empty_result()))
        }
        (0, _, OperationSemantics::Divide) => {
            // 0 / B = 0 (mathematically)
            Some(Ok(empty_result()))
        }

        // A not empty, B empty
        (_, 0, OperationSemantics::Add) => {
            // A + 0 = A
            Some(Ok((a_ptrs.clone(), a_indices.clone(), a_values.clone())))
        }
        (_, 0, OperationSemantics::Subtract) => {
            // A - 0 = A
            Some(Ok((a_ptrs.clone(), a_indices.clone(), a_values.clone())))
        }
        (_, 0, OperationSemantics::Multiply) => {
            // A * 0 = 0
            Some(Ok(empty_result()))
        }
        (_, 0, OperationSemantics::Divide) => {
            // A / 0 = undefined (division by zero matrix)
            Some(Err(Error::Internal(
                "Division by zero - B matrix is empty".to_string(),
            )))
        }

        // Both non-empty - continue with normal processing
        (_, _, _) => None,
    }
}

/// Generic CSR merge operation for element-wise ops (add, sub, mul, div)
///
/// This function implements the sorted-merge algorithm for combining two
/// CSR matrices element-wise. The operation is specified by the `op` function
/// and the merge strategy determines which positions to keep.
///
/// # Arguments
///
/// * `strategy` - Whether to use Union (keep all positions) or Intersection (keep only common positions)
/// * `semantics` - Operation semantics for handling empty matrices
/// * `op` - Operation to apply when both matrices have values at a position
/// * `only_a_op` - Transformation for values that only exist in A
/// * `only_b_op` - Transformation for values that only exist in B
///
/// # Algorithm
///
/// For each row:
/// 1. Merge the two sorted lists of column indices
/// 2. Union strategy: Keep positions from either matrix
/// 3. Intersection strategy: Keep only positions where both have values
/// 4. Apply operation when both matrices have values
fn merge_csr_impl<T: Element, F, FA, FB>(
    a_row_ptrs: &Tensor<CpuRuntime>,
    a_col_indices: &Tensor<CpuRuntime>,
    a_values: &Tensor<CpuRuntime>,
    b_row_ptrs: &Tensor<CpuRuntime>,
    b_col_indices: &Tensor<CpuRuntime>,
    b_values: &Tensor<CpuRuntime>,
    shape: [usize; 2],
    strategy: MergeStrategy,
    semantics: OperationSemantics,
    op: F,
    only_a_op: FA,
    only_b_op: FB,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>
where
    F: Fn(T, T) -> T,
    FA: Fn(T) -> T,
    FB: Fn(T) -> T,
{
    let [nrows, _ncols] = shape;
    let device = a_values.device();

    // Handle empty inputs with centralized logic
    if let Some(result) = handle_empty_compressed::<T>(
        a_values.numel(),
        b_values.numel(),
        shape,
        device,
        a_row_ptrs,
        a_col_indices,
        a_values,
        b_row_ptrs,
        b_col_indices,
        b_values,
        semantics,
        true, // CSR format
    ) {
        return result;
    }

    // Read CSR data
    let a_row_ptrs_data: Vec<i64> = a_row_ptrs.to_vec();
    let a_col_indices_data: Vec<i64> = a_col_indices.to_vec();
    let a_values_data: Vec<T> = a_values.to_vec();
    let b_row_ptrs_data: Vec<i64> = b_row_ptrs.to_vec();
    let b_col_indices_data: Vec<i64> = b_col_indices.to_vec();
    let b_values_data: Vec<T> = b_values.to_vec();

    // Build result CSR
    let mut out_row_ptrs: Vec<i64> = Vec::with_capacity(nrows + 1);
    let mut out_col_indices: Vec<i64> = Vec::new();
    let mut out_values: Vec<T> = Vec::new();

    out_row_ptrs.push(0);

    for row in 0..nrows {
        let a_start = a_row_ptrs_data[row] as usize;
        let a_end = a_row_ptrs_data[row + 1] as usize;
        let b_start = b_row_ptrs_data[row] as usize;
        let b_end = b_row_ptrs_data[row + 1] as usize;

        let mut i = a_start;
        let mut j = b_start;

        // Merge strategy determines the loop condition and handling
        match strategy {
            MergeStrategy::Union => {
                // Union: Keep positions from either matrix (|| semantics)
                while i < a_end || j < b_end {
                    let a_col = if i < a_end {
                        a_col_indices_data[i]
                    } else {
                        i64::MAX
                    };
                    let b_col = if j < b_end {
                        b_col_indices_data[j]
                    } else {
                        i64::MAX
                    };

                    if a_col < b_col {
                        // Only A has value at this column - apply only_a_op
                        let result = only_a_op(a_values_data[i]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_col_indices.push(a_col);
                            out_values.push(result);
                        }
                        i += 1;
                    } else if a_col > b_col {
                        // Only B has value at this column - apply only_b_op
                        let result = only_b_op(b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_col_indices.push(b_col);
                            out_values.push(result);
                        }
                        j += 1;
                    } else {
                        // Both have values - apply operation
                        let result = op(a_values_data[i], b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_col_indices.push(a_col);
                            out_values.push(result);
                        }
                        i += 1;
                        j += 1;
                    }
                }
            }
            MergeStrategy::Intersection => {
                // Intersection: Keep only positions where both have values (&& semantics)
                while i < a_end && j < b_end {
                    let a_col = a_col_indices_data[i];
                    let b_col = b_col_indices_data[j];

                    if a_col < b_col {
                        // Only A has value - skip in intersection
                        i += 1;
                    } else if a_col > b_col {
                        // Only B has value - skip in intersection
                        j += 1;
                    } else {
                        // Both have values - apply operation
                        let result = op(a_values_data[i], b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_col_indices.push(a_col);
                            out_values.push(result);
                        }
                        i += 1;
                        j += 1;
                    }
                }
            }
        }

        out_row_ptrs.push(out_col_indices.len() as i64);
    }

    // Create result tensors
    let result_row_ptrs = Tensor::from_slice(&out_row_ptrs, &[nrows + 1], device);
    let result_col_indices = Tensor::from_slice(&out_col_indices, &[out_col_indices.len()], device);
    let result_values = Tensor::from_slice(&out_values, &[out_values.len()], device);

    Ok((result_row_ptrs, result_col_indices, result_values))
}

/// Generic merge implementation for CSC element-wise operations
///
/// Merges two CSC matrices element-wise. The operation is specified by the `op` function.
///
/// # Algorithm
///
/// For each column:
/// 1. Merge the two sorted lists of row indices
/// 2. Apply operation when both matrices have a value at that row
/// 3. Keep values from only one matrix when the other is zero
pub(crate) fn merge_csc_impl<T: Element, F, FA, FB>(
    a_col_ptrs: &Tensor<CpuRuntime>,
    a_row_indices: &Tensor<CpuRuntime>,
    a_values: &Tensor<CpuRuntime>,
    b_col_ptrs: &Tensor<CpuRuntime>,
    b_row_indices: &Tensor<CpuRuntime>,
    b_values: &Tensor<CpuRuntime>,
    shape: [usize; 2],
    strategy: MergeStrategy,
    semantics: OperationSemantics,
    op: F,
    only_a_op: FA,
    only_b_op: FB,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>
where
    F: Fn(T, T) -> T,
    FA: Fn(T) -> T,
    FB: Fn(T) -> T,
{
    let [_nrows, ncols] = shape;
    let device = a_values.device();

    // Handle empty inputs with centralized logic
    if let Some(result) = handle_empty_compressed::<T>(
        a_values.numel(),
        b_values.numel(),
        shape,
        device,
        a_col_ptrs,
        a_row_indices,
        a_values,
        b_col_ptrs,
        b_row_indices,
        b_values,
        semantics,
        false, // CSC format
    ) {
        return result;
    }

    // Read CSC data
    let a_col_ptrs_data: Vec<i64> = a_col_ptrs.to_vec();
    let a_row_indices_data: Vec<i64> = a_row_indices.to_vec();
    let a_values_data: Vec<T> = a_values.to_vec();
    let b_col_ptrs_data: Vec<i64> = b_col_ptrs.to_vec();
    let b_row_indices_data: Vec<i64> = b_row_indices.to_vec();
    let b_values_data: Vec<T> = b_values.to_vec();

    // Build result CSC
    let mut out_col_ptrs: Vec<i64> = Vec::with_capacity(ncols + 1);
    let mut out_row_indices: Vec<i64> = Vec::new();
    let mut out_values: Vec<T> = Vec::new();

    out_col_ptrs.push(0);

    for col in 0..ncols {
        let a_start = a_col_ptrs_data[col] as usize;
        let a_end = a_col_ptrs_data[col + 1] as usize;
        let b_start = b_col_ptrs_data[col] as usize;
        let b_end = b_col_ptrs_data[col + 1] as usize;

        let mut i = a_start;
        let mut j = b_start;

        // Merge strategy determines the loop condition and handling
        match strategy {
            MergeStrategy::Union => {
                // Union: Keep positions from either matrix (|| semantics)
                while i < a_end || j < b_end {
                    let a_row = if i < a_end {
                        a_row_indices_data[i]
                    } else {
                        i64::MAX
                    };
                    let b_row = if j < b_end {
                        b_row_indices_data[j]
                    } else {
                        i64::MAX
                    };

                    if a_row < b_row {
                        // Only A has value at this row - apply only_a_op
                        let result = only_a_op(a_values_data[i]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_row_indices.push(a_row);
                            out_values.push(result);
                        }
                        i += 1;
                    } else if a_row > b_row {
                        // Only B has value at this row - apply only_b_op
                        let result = only_b_op(b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_row_indices.push(b_row);
                            out_values.push(result);
                        }
                        j += 1;
                    } else {
                        // Both have values - apply operation
                        let result = op(a_values_data[i], b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_row_indices.push(a_row);
                            out_values.push(result);
                        }
                        i += 1;
                        j += 1;
                    }
                }
            }
            MergeStrategy::Intersection => {
                // Intersection: Keep only positions where both have values (&& semantics)
                while i < a_end && j < b_end {
                    let a_row = a_row_indices_data[i];
                    let b_row = b_row_indices_data[j];

                    if a_row < b_row {
                        // Only A has value - skip in intersection
                        i += 1;
                    } else if a_row > b_row {
                        // Only B has value - skip in intersection
                        j += 1;
                    } else {
                        // Both have values - apply operation
                        let result = op(a_values_data[i], b_values_data[j]);
                        if result.to_f64().abs() > zero_tolerance::<T>() {
                            out_row_indices.push(a_row);
                            out_values.push(result);
                        }
                        i += 1;
                        j += 1;
                    }
                }
            }
        }

        out_col_ptrs.push(out_row_indices.len() as i64);
    }

    // Create result tensors
    let result_col_ptrs = Tensor::from_slice(&out_col_ptrs, &[ncols + 1], device);
    let result_row_indices = Tensor::from_slice(&out_row_indices, &[out_row_indices.len()], device);
    let result_values = Tensor::from_slice(&out_values, &[out_values.len()], device);

    Ok((result_col_ptrs, result_row_indices, result_values))
}

/// Generic merge implementation for COO element-wise operations
///
/// Merges two COO matrices element-wise with full control over value transformations.
///
/// # Arguments
///
/// * `op` - Operation to apply when both matrices have a value at the same position
/// * `only_a_op` - Transformation for values that only exist in A (identity for add, identity for sub)
/// * `only_b_op` - Transformation for values that only exist in B (identity for add, negate for sub)
///
/// # Algorithm
///
/// 1. Concatenate both matrices' triplets (applying transformations)
/// 2. Sort by (row, col)
/// 3. Merge duplicate positions by applying the operation
pub(crate) fn merge_coo_impl<T: Element, F, FA, FB>(
    a_row_indices: &Tensor<CpuRuntime>,
    a_col_indices: &Tensor<CpuRuntime>,
    a_values: &Tensor<CpuRuntime>,
    b_row_indices: &Tensor<CpuRuntime>,
    b_col_indices: &Tensor<CpuRuntime>,
    b_values: &Tensor<CpuRuntime>,
    semantics: OperationSemantics,
    op: F,
    only_a_op: FA,
    only_b_op: FB,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>
where
    F: Fn(T, T) -> T,
    FA: Fn(T) -> T,
    FB: Fn(T) -> T,
{
    let device = a_values.device();

    // Handle empty inputs early to avoid bytemuck alignment errors
    let a_nnz = a_values.numel();
    let b_nnz = b_values.numel();

    if a_nnz == 0 && b_nnz == 0 {
        // Both empty - return empty result (always empty regardless of semantics)
        let empty_rows = Tensor::from_slice(&Vec::<i64>::new(), &[0], device);
        let empty_cols = Tensor::from_slice(&Vec::<i64>::new(), &[0], device);
        let empty_vals = Tensor::from_slice(&Vec::<T>::new(), &[0], device);
        return Ok((empty_rows, empty_cols, empty_vals));
    } else if a_nnz == 0 {
        // A is empty, B is not - apply B transformation
        let b_vals: Vec<T> = b_values.to_vec();
        let transformed_vals: Vec<T> = b_vals.iter().map(|&v| only_b_op(v)).collect();
        let out_vals = Tensor::from_slice(&transformed_vals, &[transformed_vals.len()], device);
        return Ok((b_row_indices.clone(), b_col_indices.clone(), out_vals));
    } else if b_nnz == 0 {
        // B is empty, A is not - apply A transformation (usually identity)
        let a_vals: Vec<T> = a_values.to_vec();
        let transformed_vals: Vec<T> = a_vals.iter().map(|&v| only_a_op(v)).collect();
        let out_vals = Tensor::from_slice(&transformed_vals, &[transformed_vals.len()], device);
        return Ok((a_row_indices.clone(), a_col_indices.clone(), out_vals));
    }

    // Both non-empty - proceed with merge
    let a_rows: Vec<i64> = a_row_indices.to_vec();
    let a_cols: Vec<i64> = a_col_indices.to_vec();
    let a_vals: Vec<T> = a_values.to_vec();
    let b_rows: Vec<i64> = b_row_indices.to_vec();
    let b_cols: Vec<i64> = b_col_indices.to_vec();
    let b_vals: Vec<T> = b_values.to_vec();

    // Concatenate triplets with source tag (true=from A, false=from B)
    let mut triplets: Vec<(i64, i64, T, bool)> = Vec::new();
    for i in 0..a_rows.len() {
        triplets.push((a_rows[i], a_cols[i], a_vals[i], true));
    }
    for i in 0..b_rows.len() {
        triplets.push((b_rows[i], b_cols[i], b_vals[i], false));
    }

    // Sort by (row, col)
    triplets.sort_by_key(|&(r, c, _, _)| (r, c));

    // Merge duplicates
    let mut result_rows: Vec<i64> = Vec::new();
    let mut result_cols: Vec<i64> = Vec::new();
    let mut result_vals: Vec<T> = Vec::new();

    if triplets.is_empty() {
        // Empty result
        let empty_rows = Tensor::from_slice(&result_rows, &[0], device);
        let empty_cols = Tensor::from_slice(&result_cols, &[0], device);
        let empty_vals = Tensor::from_slice(&result_vals, &[0], device);
        return Ok((empty_rows, empty_cols, empty_vals));
    }

    let mut current_row = triplets[0].0;
    let mut current_col = triplets[0].1;
    let mut current_val = triplets[0].2;
    let mut current_from_a = triplets[0].3;
    let mut current_merged = false; // Track if this value resulted from merging

    for i in 1..triplets.len() {
        let (row, col, val, from_a) = triplets[i];

        if row == current_row && col == current_col {
            // Same position - apply operation (merge)
            current_val = op(current_val, val);
            current_merged = true; // This value is now a merged result
        } else {
            // Different position - save current (with transformation if not merged)
            let final_val = if current_merged {
                // Already merged, no transformation needed
                current_val
            } else {
                // Not merged, apply transformation based on source
                if current_from_a {
                    only_a_op(current_val)
                } else {
                    only_b_op(current_val)
                }
            };

            if final_val.to_f64().abs() > zero_tolerance::<T>() {
                result_rows.push(current_row);
                result_cols.push(current_col);
                result_vals.push(final_val);
            }

            // Start new accumulation
            current_row = row;
            current_col = col;
            current_val = val;
            current_from_a = from_a;
            current_merged = false;
        }
    }

    // Don't forget the last triplet
    let final_val = if current_merged {
        // Already merged, no transformation needed
        current_val
    } else {
        // Not merged, apply transformation based on source
        if current_from_a {
            only_a_op(current_val)
        } else {
            only_b_op(current_val)
        }
    };

    if final_val.to_f64().abs() > zero_tolerance::<T>() {
        result_rows.push(current_row);
        result_cols.push(current_col);
        result_vals.push(final_val);
    }

    // Create result tensors
    let out_rows = Tensor::from_slice(&result_rows, &[result_rows.len()], device);
    let out_cols = Tensor::from_slice(&result_cols, &[result_cols.len()], device);
    let out_vals = Tensor::from_slice(&result_vals, &[result_vals.len()], device);

    Ok((out_rows, out_cols, out_vals))
}

/// Intersection-based merge for COO element-wise operations
///
/// Only keeps positions where BOTH matrices have non-zero values.
/// This is the correct semantics for sparse element-wise multiplication and division.
///
/// Uses a merge-join algorithm on sorted triplets for O(nnz_a + nnz_b) complexity.
pub(crate) fn intersect_coo_impl<T: Element, F>(
    a_row_indices: &Tensor<CpuRuntime>,
    a_col_indices: &Tensor<CpuRuntime>,
    a_values: &Tensor<CpuRuntime>,
    b_row_indices: &Tensor<CpuRuntime>,
    b_col_indices: &Tensor<CpuRuntime>,
    b_values: &Tensor<CpuRuntime>,
    op: F,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>
where
    F: Fn(T, T) -> T,
{
    let device = a_values.device();

    // Handle empty inputs - multiplication by zero gives zero (empty)
    let a_nnz = a_values.numel();
    let b_nnz = b_values.numel();

    if a_nnz == 0 || b_nnz == 0 {
        // If either is empty, result is empty
        let empty_rows = Tensor::from_slice(&Vec::<i64>::new(), &[0], device);
        let empty_cols = Tensor::from_slice(&Vec::<i64>::new(), &[0], device);
        let empty_vals = Tensor::from_slice(&Vec::<T>::new(), &[0], device);
        return Ok((empty_rows, empty_cols, empty_vals));
    }

    // Read COO data
    let a_rows: Vec<i64> = a_row_indices.to_vec();
    let a_cols: Vec<i64> = a_col_indices.to_vec();
    let a_vals: Vec<T> = a_values.to_vec();
    let b_rows: Vec<i64> = b_row_indices.to_vec();
    let b_cols: Vec<i64> = b_col_indices.to_vec();
    let b_vals: Vec<T> = b_values.to_vec();

    // Sort both by (row, col) if not already sorted
    let mut a_triplets: Vec<(i64, i64, T)> = a_rows
        .iter()
        .zip(a_cols.iter())
        .zip(a_vals.iter())
        .map(|((&r, &c), &v)| (r, c, v))
        .collect();
    a_triplets.sort_by_key(|&(r, c, _)| (r, c));

    let mut b_triplets: Vec<(i64, i64, T)> = b_rows
        .iter()
        .zip(b_cols.iter())
        .zip(b_vals.iter())
        .map(|((&r, &c), &v)| (r, c, v))
        .collect();
    b_triplets.sort_by_key(|&(r, c, _)| (r, c));

    // Merge-join to find intersection
    let mut result_rows: Vec<i64> = Vec::new();
    let mut result_cols: Vec<i64> = Vec::new();
    let mut result_vals: Vec<T> = Vec::new();

    let mut i = 0;
    let mut j = 0;

    while i < a_triplets.len() && j < b_triplets.len() {
        let (a_row, a_col, a_val) = a_triplets[i];
        let (b_row, b_col, b_val) = b_triplets[j];

        match (a_row.cmp(&b_row), a_col.cmp(&b_col)) {
            (std::cmp::Ordering::Less, _)
            | (std::cmp::Ordering::Equal, std::cmp::Ordering::Less) => {
                // A's position is before B's - skip it (not in intersection)
                i += 1;
            }
            (std::cmp::Ordering::Greater, _)
            | (std::cmp::Ordering::Equal, std::cmp::Ordering::Greater) => {
                // B's position is before A's - skip it (not in intersection)
                j += 1;
            }
            (std::cmp::Ordering::Equal, std::cmp::Ordering::Equal) => {
                // Same position - apply operation and keep result
                let result = op(a_val, b_val);
                if result.to_f64().abs() > zero_tolerance::<T>() {
                    result_rows.push(a_row);
                    result_cols.push(a_col);
                    result_vals.push(result);
                }
                i += 1;
                j += 1;
            }
        }
    }

    // Create result tensors
    let out_rows = Tensor::from_slice(&result_rows, &[result_rows.len()], device);
    let out_cols = Tensor::from_slice(&result_cols, &[result_cols.len()], device);
    let out_vals = Tensor::from_slice(&result_vals, &[result_vals.len()], device);

    Ok((out_rows, out_cols, out_vals))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;

    #[test]
    fn test_spmv_csr_basic() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        let row_ptrs = Tensor::from_slice(&[0i64, 2, 3, 5], &[4], &device);
        let col_indices = Tensor::from_slice(&[0i64, 2, 2, 0, 1], &[5], &device);
        let values = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);

        // x = [1, 2, 3]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

        // y = A * x
        // y[0] = 1*1 + 2*3 = 7
        // y[1] = 3*3 = 9
        // y[2] = 4*1 + 5*2 = 14
        let y = client
            .spmv_csr::<f32>(&row_ptrs, &col_indices, &values, &x, [3, 3])
            .unwrap();

        assert_eq!(y.shape(), &[3]);
        let y_data: Vec<f32> = y.to_vec();
        assert_eq!(y_data, vec![7.0, 9.0, 14.0]);
    }

    #[test]
    fn test_add_csr_basic() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // A:
        // [1, 0, 2]
        // [0, 3, 0]
        let a_row_ptrs = Tensor::from_slice(&[0i64, 2, 3], &[3], &device);
        let a_col_indices = Tensor::from_slice(&[0i64, 2, 1], &[3], &device);
        let a_values = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

        // B:
        // [0, 4, 0]
        // [5, 0, 6]
        let b_row_ptrs = Tensor::from_slice(&[0i64, 1, 3], &[3], &device);
        let b_col_indices = Tensor::from_slice(&[1i64, 0, 2], &[3], &device);
        let b_values = Tensor::from_slice(&[4.0f32, 5.0, 6.0], &[3], &device);

        // C = A + B:
        // [1, 4, 2]
        // [5, 3, 6]
        let (row_ptrs, col_indices, values) = client
            .add_csr::<f32>(
                &a_row_ptrs,
                &a_col_indices,
                &a_values,
                &b_row_ptrs,
                &b_col_indices,
                &b_values,
                [2, 3],
            )
            .unwrap();

        let row_ptrs_data: Vec<i64> = row_ptrs.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();
        let values_data: Vec<f32> = values.to_vec();

        assert_eq!(row_ptrs_data, vec![0, 3, 6]);
        assert_eq!(col_indices_data, vec![0, 1, 2, 0, 1, 2]);
        assert_eq!(values_data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // =========================================================================
    // Zero-Elimination Tests
    // =========================================================================
    //
    // These tests explicitly verify that values below the dtype-specific
    // tolerance threshold are eliminated from sparse results.

    #[test]
    fn test_zero_elimination_csr_f32() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Test via multiplication which reliably produces small values
        // A = [1e-4, 0]
        //     [0, 2.0]
        let a_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device);
        let a_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let a_values = Tensor::from_slice(&[1e-4f32, 2.0], &[2], &device);

        // B = [1e-5, 0]
        //     [0, 2.0]
        let b_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device);
        let b_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let b_values = Tensor::from_slice(&[1e-5f32, 2.0], &[2], &device);

        // C = A .* B (element-wise multiply)
        // (0,0): 1e-4 * 1e-5 = 1e-9 (< 1e-7, eliminated)
        // (1,1): 2.0 * 2.0 = 4.0 (kept)
        let (_, col_indices, values) = client
            .mul_csr::<f32>(
                &a_row_ptrs,
                &a_col_indices,
                &a_values,
                &b_row_ptrs,
                &b_col_indices,
                &b_values,
                [2, 2],
            )
            .unwrap();

        let values_data: Vec<f32> = values.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();

        // The tiny value (1e-9) should be eliminated, only 4.0 remains
        assert_eq!(
            values_data.len(),
            1,
            "Near-zero values should be eliminated"
        );
        assert!((values_data[0] - 4.0).abs() < 1e-6);
        assert_eq!(col_indices_data, vec![1]);
    }

    #[test]
    fn test_zero_elimination_csr_f32_preserved() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Create matrices that produce values ABOVE the tolerance
        // A = [1.001, 0]
        //     [0, 5.0]
        let a_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device);
        let a_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let a_values = Tensor::from_slice(&[1.001f32, 5.0], &[2], &device);

        // B = [1.0, 0]
        //     [0, 5.0]
        let b_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device);
        let b_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let b_values = Tensor::from_slice(&[1.0f32, 5.0], &[2], &device);

        // C = A - B should keep the first element (1.001 - 1.0 = 0.001)
        // since 0.001 > 1e-7 (F32 tolerance)
        let (row_ptrs, col_indices, values) = client
            .sub_csr::<f32>(
                &a_row_ptrs,
                &a_col_indices,
                &a_values,
                &b_row_ptrs,
                &b_col_indices,
                &b_values,
                [2, 2],
            )
            .unwrap();

        let values_data: Vec<f32> = values.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();

        // The value 0.001 should be preserved (above tolerance)
        assert_eq!(
            values_data.len(),
            1,
            "Values above tolerance should be kept"
        );
        assert!((values_data[0] - 0.001).abs() < 1e-6);
        assert_eq!(col_indices_data, vec![0]);
    }

    #[test]
    fn test_zero_elimination_csr_f64() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // F64 has much tighter tolerance (1e-15)
        // Create values that would be eliminated in F32 but kept in F64
        // A = [1.0 + 1e-8, 0]
        let a_row_ptrs = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let a_col_indices = Tensor::from_slice(&[0i64], &[1], &device);
        let a_values = Tensor::from_slice(&[1.0 + 1e-8], &[1], &device);

        // B = [1.0, 0]
        let b_row_ptrs = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let b_col_indices = Tensor::from_slice(&[0i64], &[1], &device);
        let b_values = Tensor::from_slice(&[1.0], &[1], &device);

        // C = A - B = 1e-8, which is well above F64 tolerance (1e-15)
        let (_, col_indices, values) = client
            .sub_csr::<f64>(
                &a_row_ptrs,
                &a_col_indices,
                &a_values,
                &b_row_ptrs,
                &b_col_indices,
                &b_values,
                [1, 2],
            )
            .unwrap();

        let values_data: Vec<f64> = values.to_vec();

        // F64 should preserve values down to 1e-15
        assert_eq!(values_data.len(), 1, "F64 preserves higher precision");
        assert!((values_data[0] - 1e-8).abs() < 1e-16);
    }

    #[test]
    fn test_zero_elimination_coo_f32() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // COO format test: Use multiplication to create predictable small values
        // A has triplets: (0, 0, 1e-4), (1, 1, 2.0)
        let a_row_indices = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let a_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let a_values = Tensor::from_slice(&[1e-4f32, 2.0], &[2], &device);

        // B has triplets: (0, 0, 1e-5), (1, 1, 2.0)
        let b_row_indices = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let b_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let b_values = Tensor::from_slice(&[1e-5f32, 2.0], &[2], &device);

        // C = A .* B (element-wise multiply)
        // (0,0): 1e-4 * 1e-5 = 1e-9 (< 1e-7, eliminated)
        // (1,1): 2.0 * 2.0 = 4.0 (kept)
        let (row_indices, col_indices, values) = client
            .mul_coo::<f32>(
                &a_row_indices,
                &a_col_indices,
                &a_values,
                &b_row_indices,
                &b_col_indices,
                &b_values,
                [2, 2],
            )
            .unwrap();

        let values_data: Vec<f32> = values.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();

        assert_eq!(
            values_data.len(),
            1,
            "COO zero elimination should remove near-zero values"
        );
        assert!((values_data[0] - 4.0).abs() < 1e-6);
        assert_eq!(col_indices_data, vec![1]);
    }

    #[test]
    fn test_zero_elimination_csc_f32() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // CSC format test: Use multiplication to create predictable small values
        // A (column-major):
        // Col 0: [1e-4]
        // Col 1: [2.0]
        let a_col_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device);
        let a_row_indices = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let a_values = Tensor::from_slice(&[1e-4f32, 2.0], &[2], &device);

        // B (column-major):
        // Col 0: [1e-5]
        // Col 1: [2.0]
        let b_col_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device);
        let b_row_indices = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let b_values = Tensor::from_slice(&[1e-5f32, 2.0], &[2], &device);

        // C = A .* B (element-wise multiply)
        // Col 0: 1e-4 * 1e-5 = 1e-9 (< 1e-7, eliminated)
        // Col 1: 2.0 * 2.0 = 4.0 (kept)
        let (_, row_indices, values) = client
            .mul_csc::<f32>(
                &a_col_ptrs,
                &a_row_indices,
                &a_values,
                &b_col_ptrs,
                &b_row_indices,
                &b_values,
                [2, 2],
            )
            .unwrap();

        let values_data: Vec<f32> = values.to_vec();
        let row_indices_data: Vec<i64> = row_indices.to_vec();

        assert_eq!(
            values_data.len(),
            1,
            "CSC zero elimination should work like CSR"
        );
        assert!((values_data[0] - 4.0).abs() < 1e-6);
        assert_eq!(row_indices_data, vec![1]);
    }

    #[test]
    fn test_zero_elimination_mul_intersection() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Multiplication uses intersection semantics
        // Create small values through multiplication, testing both sides of tolerance
        // A = [1e-3, 0]
        //     [0, 2.0]
        let a_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device);
        let a_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let a_values = Tensor::from_slice(&[1e-3f32, 2.0], &[2], &device);

        // B = [1e-3, 0]
        //     [0, 3.0]
        let b_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device);
        let b_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let b_values = Tensor::from_slice(&[1e-3f32, 3.0], &[2], &device);

        // C = A .* B (element-wise multiply)
        // (0,0): 1e-3 * 1e-3 = 1e-6 (> 1e-7, above tolerance, kept)
        // (1,1): 2.0 * 3.0 = 6.0 (kept)
        let (_, col_indices, values) = client
            .mul_csr::<f32>(
                &a_row_ptrs,
                &a_col_indices,
                &a_values,
                &b_row_ptrs,
                &b_col_indices,
                &b_values,
                [2, 2],
            )
            .unwrap();

        let values_data: Vec<f32> = values.to_vec();
        let col_indices_data: Vec<i64> = col_indices.to_vec();

        assert_eq!(
            values_data.len(),
            2,
            "Both values above F32 tolerance (1e-7)"
        );
        assert!(
            (values_data[0] - 1e-6).abs() < 1e-8,
            "Expected 1e-6, got {}",
            values_data[0]
        );
        assert!((values_data[1] - 6.0).abs() < 1e-6);
        assert_eq!(col_indices_data, vec![0, 1]);
    }

    #[test]
    fn test_zero_elimination_boundary_f32() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Test values right at the boundary of F32 tolerance (1e-7)
        // Use 2x2 matrices to avoid edge cases with single-element matrices

        // Test 1: Below tolerance (1e-5 * 1e-4 = 1e-9 < 1e-7, eliminated)
        // A = [1e-5, 0]    B = [1e-4, 0]
        //     [0, 2.0]         [0, 2.0]
        let a1_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device);
        let a1_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let a1_values = Tensor::from_slice(&[1e-5f32, 2.0], &[2], &device);

        let b1_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device);
        let b1_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let b1_values = Tensor::from_slice(&[1e-4f32, 2.0], &[2], &device);

        let (_, col_indices1, values1) = client
            .mul_csr::<f32>(
                &a1_row_ptrs,
                &a1_col_indices,
                &a1_values,
                &b1_row_ptrs,
                &b1_col_indices,
                &b1_values,
                [2, 2],
            )
            .unwrap();
        let vals1 = values1.to_vec::<f32>();
        assert_eq!(
            vals1.len(),
            1,
            "1e-9 < 1e-7 should be eliminated, only 4.0 kept"
        );
        assert!((vals1[0] - 4.0).abs() < 1e-6);

        // Test 2: Above tolerance (1e-3 * 1e-3 = 1e-6 > 1e-7, kept)
        // A = [1e-3, 0]    B = [1e-3, 0]
        //     [0, 3.0]         [0, 3.0]
        let a2_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device);
        let a2_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let a2_values = Tensor::from_slice(&[1e-3f32, 3.0], &[2], &device);

        let b2_row_ptrs = Tensor::from_slice(&[0i64, 1, 2], &[3], &device);
        let b2_col_indices = Tensor::from_slice(&[0i64, 1], &[2], &device);
        let b2_values = Tensor::from_slice(&[1e-3f32, 3.0], &[2], &device);

        let (_, col_indices2, values2) = client
            .mul_csr::<f32>(
                &a2_row_ptrs,
                &a2_col_indices,
                &a2_values,
                &b2_row_ptrs,
                &b2_col_indices,
                &b2_values,
                [2, 2],
            )
            .unwrap();
        let vals2 = values2.to_vec::<f32>();
        assert_eq!(vals2.len(), 2, "1e-6 > 1e-7 should be kept along with 9.0");
        assert!(
            (vals2[0] - 1e-6).abs() < 1e-8,
            "Expected 1e-6, got {}",
            vals2[0]
        );
        assert!((vals2[1] - 9.0).abs() < 1e-6);
    }
}
