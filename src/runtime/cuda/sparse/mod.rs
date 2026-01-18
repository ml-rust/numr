//! Sparse operations implementation for CUDA runtime
//!
//! This module implements the SparseOps trait for CudaRuntime, providing
//! GPU-accelerated sparse matrix operations using CUDA kernels.
//!
//! ## Module Structure
//!
//! - `spmv` - Sparse matrix-vector and matrix-matrix multiplication
//! - `merge` - Element-wise operations (add/sub/mul/div) for CSR/CSC/COO formats
//! - `conversions` - Format conversions (COO↔CSR, COO↔CSC, CSR↔CSC, transpose)

use super::{CudaClient, CudaRuntime};
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::runtime::cpu;
use crate::runtime::cuda::kernels::{
    coo_add_merge, coo_div_merge, coo_mul_merge, coo_sub_merge, csc_add_merge, csc_div_merge,
    csc_mul_merge, csc_sub_merge, csr_add_merge, csr_div_merge, csr_mul_merge, csr_sub_merge,
    exclusive_scan_i64_gpu, launch_csc_to_csr_transpose, launch_csr_spmm, launch_csr_spmv,
    launch_csr_spmv_warp, launch_csr_to_csc_transpose, launch_expand_ptrs,
    launch_histogram_csc_rows, launch_histogram_csr_columns, should_use_warp_kernel,
};
use crate::sparse::SparseOps;
use crate::tensor::Tensor;

/// Macro to dispatch CSR merge operations based on dtype
/// Eliminates ~60 lines of duplication per operation
macro_rules! sparse_dtype_dispatch_csr {
    ($merge_fn:ident, $self:expr, $a_row_ptrs:expr, $a_col_indices:expr, $a_values:expr,
     $b_row_ptrs:expr, $b_col_indices:expr, $b_values:expr, $shape:expr, $op_name:expr) => {{
        let [nrows, _] = $shape;
        let device = $a_values.device();
        let dtype = $a_values.dtype();

        use crate::dtype::DType;
        match dtype {
            DType::F32 => unsafe {
                $merge_fn::<f32>(
                    &$self.context,
                    &$self.stream,
                    $self.device.index,
                    device,
                    dtype,
                    $a_row_ptrs,
                    $a_col_indices,
                    $a_values,
                    $b_row_ptrs,
                    $b_col_indices,
                    $b_values,
                    nrows,
                )
            },
            DType::F64 => unsafe {
                $merge_fn::<f64>(
                    &$self.context,
                    &$self.stream,
                    $self.device.index,
                    device,
                    dtype,
                    $a_row_ptrs,
                    $a_col_indices,
                    $a_values,
                    $b_row_ptrs,
                    $b_col_indices,
                    $b_values,
                    nrows,
                )
            },
            #[cfg(feature = "f16")]
            DType::F16 => unsafe {
                $merge_fn::<half::f16>(
                    &$self.context,
                    &$self.stream,
                    $self.device.index,
                    device,
                    dtype,
                    $a_row_ptrs,
                    $a_col_indices,
                    $a_values,
                    $b_row_ptrs,
                    $b_col_indices,
                    $b_values,
                    nrows,
                )
            },
            #[cfg(feature = "f16")]
            DType::BF16 => unsafe {
                $merge_fn::<half::bf16>(
                    &$self.context,
                    &$self.stream,
                    $self.device.index,
                    device,
                    dtype,
                    $a_row_ptrs,
                    $a_col_indices,
                    $a_values,
                    $b_row_ptrs,
                    $b_col_indices,
                    $b_values,
                    nrows,
                )
            },
            _ => Err(Error::Internal(format!(
                "Unsupported dtype for CUDA sparse {}: {:?}",
                $op_name, dtype
            ))),
        }
    }};
}

/// Macro to dispatch CSC merge operations based on dtype
macro_rules! sparse_dtype_dispatch_csc {
    ($merge_fn:ident, $self:expr, $a_col_ptrs:expr, $a_row_indices:expr, $a_values:expr,
     $b_col_ptrs:expr, $b_row_indices:expr, $b_values:expr, $shape:expr, $op_name:expr) => {{
        let [_, ncols] = $shape;
        let device = $a_values.device();
        let dtype = $a_values.dtype();

        use crate::dtype::DType;
        match dtype {
            DType::F32 => unsafe {
                $merge_fn::<f32>(
                    &$self.context,
                    &$self.stream,
                    $self.device.index,
                    device,
                    dtype,
                    $a_col_ptrs,
                    $a_row_indices,
                    $a_values,
                    $b_col_ptrs,
                    $b_row_indices,
                    $b_values,
                    ncols,
                )
            },
            DType::F64 => unsafe {
                $merge_fn::<f64>(
                    &$self.context,
                    &$self.stream,
                    $self.device.index,
                    device,
                    dtype,
                    $a_col_ptrs,
                    $a_row_indices,
                    $a_values,
                    $b_col_ptrs,
                    $b_row_indices,
                    $b_values,
                    ncols,
                )
            },
            #[cfg(feature = "f16")]
            DType::F16 => unsafe {
                $merge_fn::<half::f16>(
                    &$self.context,
                    &$self.stream,
                    $self.device.index,
                    device,
                    dtype,
                    $a_col_ptrs,
                    $a_row_indices,
                    $a_values,
                    $b_col_ptrs,
                    $b_row_indices,
                    $b_values,
                    ncols,
                )
            },
            #[cfg(feature = "f16")]
            DType::BF16 => unsafe {
                $merge_fn::<half::bf16>(
                    &$self.context,
                    &$self.stream,
                    $self.device.index,
                    device,
                    dtype,
                    $a_col_ptrs,
                    $a_row_indices,
                    $a_values,
                    $b_col_ptrs,
                    $b_row_indices,
                    $b_values,
                    ncols,
                )
            },
            _ => Err(Error::Internal(format!(
                "Unsupported dtype for CUDA sparse {}: {:?}",
                $op_name, dtype
            ))),
        }
    }};
}

/// Macro to dispatch COO merge operations based on dtype
macro_rules! sparse_dtype_dispatch_coo {
    ($merge_fn:ident, $self:expr, $a_row_indices:expr, $a_col_indices:expr, $a_values:expr,
     $b_row_indices:expr, $b_col_indices:expr, $b_values:expr, $shape:expr, $op_name:expr) => {{
        let device = $a_values.device();
        let dtype = $a_values.dtype();

        use crate::dtype::DType;
        match dtype {
            DType::F32 => unsafe {
                $merge_fn::<f32>(
                    &$self.context,
                    &$self.stream,
                    $self.device.index,
                    device,
                    dtype,
                    $a_row_indices,
                    $a_col_indices,
                    $a_values,
                    $b_row_indices,
                    $b_col_indices,
                    $b_values,
                    $shape,
                )
            },
            DType::F64 => unsafe {
                $merge_fn::<f64>(
                    &$self.context,
                    &$self.stream,
                    $self.device.index,
                    device,
                    dtype,
                    $a_row_indices,
                    $a_col_indices,
                    $a_values,
                    $b_row_indices,
                    $b_col_indices,
                    $b_values,
                    $shape,
                )
            },
            #[cfg(feature = "f16")]
            DType::F16 => unsafe {
                $merge_fn::<half::f16>(
                    &$self.context,
                    &$self.stream,
                    $self.device.index,
                    device,
                    dtype,
                    $a_row_indices,
                    $a_col_indices,
                    $a_values,
                    $b_row_indices,
                    $b_col_indices,
                    $b_values,
                    $shape,
                )
            },
            #[cfg(feature = "f16")]
            DType::BF16 => unsafe {
                $merge_fn::<half::bf16>(
                    &$self.context,
                    &$self.stream,
                    $self.device.index,
                    device,
                    dtype,
                    $a_row_indices,
                    $a_col_indices,
                    $a_values,
                    $b_row_indices,
                    $b_col_indices,
                    $b_values,
                    $shape,
                )
            },
            _ => Err(Error::Internal(format!(
                "Unsupported dtype for CUDA sparse {}: {:?}",
                $op_name, dtype
            ))),
        }
    }};
}

// Submodules
mod conversions;
mod merge;
mod spmv;

// =========================================================================
// High-Level Operations (Stub implementations for now)
// =========================================================================

fn spmv(
    &self,
    a: &crate::sparse::SparseTensor<CudaRuntime>,
    x: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
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
    a: &crate::sparse::SparseTensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
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
    _a: &Tensor<CudaRuntime>,
    _b: &crate::sparse::SparseTensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    // TODO: Implement dense * sparse multiplication for CUDA
    Err(Error::NotImplemented {
        feature: "dsmm (dense*sparse) on CUDA",
    })
}

fn sparse_add(
    &self,
    a: &crate::sparse::SparseTensor<CudaRuntime>,
    b: &crate::sparse::SparseTensor<CudaRuntime>,
) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
    use crate::sparse::SparseTensor;

    // Validate shapes and dtypes
    if a.shape() != b.shape() {
        return Err(Error::ShapeMismatch {
            expected: vec![a.shape()[0], a.shape()[1]],
            got: vec![b.shape()[0], b.shape()[1]],
        });
    }

    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }

    // Convert both to CSR format
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
    a: &crate::sparse::SparseTensor<CudaRuntime>,
    b: &crate::sparse::SparseTensor<CudaRuntime>,
) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
    use crate::sparse::SparseTensor;

    // Validate shapes and dtypes
    if a.shape() != b.shape() {
        return Err(Error::ShapeMismatch {
            expected: vec![a.shape()[0], a.shape()[1]],
            got: vec![b.shape()[0], b.shape()[1]],
        });
    }

    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }

    // Convert both to CSR format
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
    _a: &crate::sparse::SparseTensor<CudaRuntime>,
    _b: &crate::sparse::SparseTensor<CudaRuntime>,
) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
    // Sparse × sparse matrix multiplication not yet implemented on GPU
    //
    // GPU-native implementation requires:
    // 1. Symbolic multiplication to determine output structure
    // 2. Numeric multiplication to compute values
    // 3. Efficient sparse-sparse kernel (e.g., using cusparse or custom kernel)
    //
    // For now, users should:
    // - Use SpMV/SpMM for sparse × dense operations (fully GPU-native)
    // - Convert to CPU for sparse × sparse if needed:
    //   let a_cpu = a.to_cpu()?;
    //   let b_cpu = b.to_cpu()?;
    //   let result = a_cpu.matmul(&b_cpu)?;
    //   let result_gpu = result.to_gpu(device)?;
    Err(Error::Internal(
        "Sparse × sparse matrix multiplication not implemented for CUDA runtime. \
             Use sparse × dense (SpMV/SpMM) or convert to CPU for this operation."
            .to_string(),
    ))
}

fn sparse_mul(
    &self,
    a: &crate::sparse::SparseTensor<CudaRuntime>,
    b: &crate::sparse::SparseTensor<CudaRuntime>,
) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
    use crate::sparse::SparseTensor;

    // Validate shapes and dtypes
    if a.shape() != b.shape() {
        return Err(Error::ShapeMismatch {
            expected: vec![a.shape()[0], a.shape()[1]],
            got: vec![b.shape()[0], b.shape()[1]],
        });
    }

    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }

    // Convert both to CSR format
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
    a: &crate::sparse::SparseTensor<CudaRuntime>,
    scalar: f64,
) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
    use crate::ops::ScalarOps;
    use crate::sparse::SparseTensor;

    // Handle empty tensors without calling mul_scalar
    if a.nnz() == 0 {
        return Ok(a.clone());
    }

    // Scale values tensor directly on GPU using ScalarOps
    match a {
        SparseTensor::Csr(data) => {
            let scaled_values = self.mul_scalar(&data.values, scalar)?;
            Ok(SparseTensor::Csr(crate::sparse::CsrData {
                row_ptrs: data.row_ptrs.clone(),
                col_indices: data.col_indices.clone(),
                values: scaled_values,
                shape: data.shape,
            }))
        }
        SparseTensor::Csc(data) => {
            let scaled_values = self.mul_scalar(&data.values, scalar)?;
            Ok(SparseTensor::Csc(crate::sparse::CscData {
                col_ptrs: data.col_ptrs.clone(),
                row_indices: data.row_indices.clone(),
                values: scaled_values,
                shape: data.shape,
            }))
        }
        SparseTensor::Coo(data) => {
            let scaled_values = self.mul_scalar(&data.values, scalar)?;
            Ok(SparseTensor::Coo(crate::sparse::CooData {
                row_indices: data.row_indices.clone(),
                col_indices: data.col_indices.clone(),
                values: scaled_values,
                shape: data.shape,
                sorted: data.sorted,
            }))
        }
    }
}

fn sparse_add_scalar(
    &self,
    _a: &crate::sparse::SparseTensor<CudaRuntime>,
    _scalar: f64,
) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
    // Adding a scalar to a sparse matrix would make it dense
    Err(Error::Internal(
        "Scalar addition to sparse matrix creates dense result - convert to dense first"
            .to_string(),
    ))
}

fn sparse_sum(&self, a: &crate::sparse::SparseTensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
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
    a: &crate::sparse::SparseTensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
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
    a: &crate::sparse::SparseTensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
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
    a: &crate::sparse::SparseTensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
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
    a: &crate::sparse::SparseTensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
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
    a: &crate::sparse::SparseTensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
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
    a: &Tensor<CudaRuntime>,
    threshold: f64,
) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
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
