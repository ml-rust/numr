//! Sparse operations implementation for CUDA runtime
//!
//! This module implements the SparseOps trait for CudaRuntime, providing
//! GPU-accelerated sparse matrix operations using CUDA kernels.

use super::{CudaClient, CudaRuntime};
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::{
    csr_add_merge, csr_mul_merge, csr_sub_merge, exclusive_scan_i64_gpu,
    launch_csc_to_csr_transpose, launch_csr_spmm, launch_csr_spmv, launch_csr_spmv_warp,
    launch_csr_to_csc_transpose, launch_expand_ptrs, launch_histogram_csc_rows,
    launch_histogram_csr_columns, should_use_warp_kernel,
};
use crate::sparse::SparseOps;
use crate::tensor::Tensor;

impl SparseOps<CudaRuntime> for CudaClient {
    // =========================================================================
    // CSR Operations
    // =========================================================================

    fn spmv_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CudaRuntime>> {
        let [nrows, ncols] = shape;
        let device = values.device();
        let dtype = values.dtype();

        // Validate input shapes
        if x.numel() != ncols {
            return Err(Error::ShapeMismatch {
                expected: vec![ncols],
                got: vec![x.numel()],
            });
        }

        // Allocate output on GPU
        let y = Tensor::<CudaRuntime>::zeros(&[nrows], dtype, device);

        // Get device pointers (no data transfer!)
        let row_ptrs_ptr = row_ptrs.storage().ptr();
        let col_indices_ptr = col_indices.storage().ptr();
        let values_ptr = values.storage().ptr();
        let x_ptr = x.storage().ptr();
        let y_ptr = y.storage().ptr();

        // Choose optimal kernel based on sparsity
        let nnz = values.numel();
        let avg_nnz_per_row = nnz as f32 / nrows.max(1) as f32;

        // Dispatch based on dtype (only F32/F64/F16/BF16 supported on CUDA)
        use crate::dtype::DType;
        match dtype {
            DType::F32 => unsafe {
                if should_use_warp_kernel(avg_nnz_per_row) {
                    launch_csr_spmv_warp::<f32>(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        row_ptrs_ptr,
                        col_indices_ptr,
                        values_ptr,
                        x_ptr,
                        y_ptr,
                        nrows,
                    )?;
                } else {
                    launch_csr_spmv::<f32>(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        row_ptrs_ptr,
                        col_indices_ptr,
                        values_ptr,
                        x_ptr,
                        y_ptr,
                        nrows,
                    )?;
                }
            },
            DType::F64 => unsafe {
                if should_use_warp_kernel(avg_nnz_per_row) {
                    launch_csr_spmv_warp::<f64>(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        row_ptrs_ptr,
                        col_indices_ptr,
                        values_ptr,
                        x_ptr,
                        y_ptr,
                        nrows,
                    )?;
                } else {
                    launch_csr_spmv::<f64>(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        row_ptrs_ptr,
                        col_indices_ptr,
                        values_ptr,
                        x_ptr,
                        y_ptr,
                        nrows,
                    )?;
                }
            },
            #[cfg(feature = "f16")]
            DType::F16 => unsafe {
                use half::f16;
                if should_use_warp_kernel(avg_nnz_per_row) {
                    launch_csr_spmv_warp::<f16>(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        row_ptrs_ptr,
                        col_indices_ptr,
                        values_ptr,
                        x_ptr,
                        y_ptr,
                        nrows,
                    )?;
                } else {
                    launch_csr_spmv::<f16>(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        row_ptrs_ptr,
                        col_indices_ptr,
                        values_ptr,
                        x_ptr,
                        y_ptr,
                        nrows,
                    )?;
                }
            },
            #[cfg(feature = "f16")]
            DType::BF16 => unsafe {
                use half::bf16;
                if should_use_warp_kernel(avg_nnz_per_row) {
                    launch_csr_spmv_warp::<bf16>(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        row_ptrs_ptr,
                        col_indices_ptr,
                        values_ptr,
                        x_ptr,
                        y_ptr,
                        nrows,
                    )?;
                } else {
                    launch_csr_spmv::<bf16>(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        row_ptrs_ptr,
                        col_indices_ptr,
                        values_ptr,
                        x_ptr,
                        y_ptr,
                        nrows,
                    )?;
                }
            },
            _ => {
                return Err(Error::Internal(format!(
                    "Unsupported dtype for CUDA sparse SpMV: {:?}",
                    dtype
                )));
            }
        }

        // Return GPU tensor (no transfer back!)
        Ok(y)
    }

    fn spmm_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CudaRuntime>> {
        let [m, k] = shape;
        let device = values.device();
        let dtype = values.dtype();

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

        // Allocate output on GPU
        let c = Tensor::<CudaRuntime>::zeros(&[m, n], dtype, device);

        // Get device pointers (no data transfer!)
        let row_ptrs_ptr = row_ptrs.storage().ptr();
        let col_indices_ptr = col_indices.storage().ptr();
        let values_ptr = values.storage().ptr();
        let b_ptr = b.storage().ptr();
        let c_ptr = c.storage().ptr();

        // Dispatch based on dtype (only F32/F64/F16/BF16 supported on CUDA)
        use crate::dtype::DType;
        match dtype {
            DType::F32 => unsafe {
                launch_csr_spmm::<f32>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    row_ptrs_ptr,
                    col_indices_ptr,
                    values_ptr,
                    b_ptr,
                    c_ptr,
                    m,
                    n,
                )?;
            },
            DType::F64 => unsafe {
                launch_csr_spmm::<f64>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    row_ptrs_ptr,
                    col_indices_ptr,
                    values_ptr,
                    b_ptr,
                    c_ptr,
                    m,
                    n,
                )?;
            },
            #[cfg(feature = "f16")]
            DType::F16 => unsafe {
                use half::f16;
                launch_csr_spmm::<f16>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    row_ptrs_ptr,
                    col_indices_ptr,
                    values_ptr,
                    b_ptr,
                    c_ptr,
                    m,
                    n,
                )?;
            },
            #[cfg(feature = "f16")]
            DType::BF16 => unsafe {
                use half::bf16;
                launch_csr_spmm::<bf16>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    row_ptrs_ptr,
                    col_indices_ptr,
                    values_ptr,
                    b_ptr,
                    c_ptr,
                    m,
                    n,
                )?;
            },
            _ => {
                return Err(Error::Internal(format!(
                    "Unsupported dtype for CUDA sparse SpMM: {:?}",
                    dtype
                )));
            }
        }

        // Return GPU tensor (no transfer back!)
        Ok(c)
    }

    fn add_csr<T: Element>(
        &self,
        a_row_ptrs: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_ptrs: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let [nrows, _] = shape;
        let device = a_values.device();
        let dtype = a_values.dtype();

        // Two-pass merge algorithm with dtype dispatch
        use crate::dtype::DType;
        match dtype {
            DType::F32 => unsafe {
                csr_add_merge::<f32>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    dtype,
                    a_row_ptrs,
                    a_col_indices,
                    a_values,
                    b_row_ptrs,
                    b_col_indices,
                    b_values,
                    nrows,
                )
            },
            DType::F64 => unsafe {
                csr_add_merge::<f64>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    dtype,
                    a_row_ptrs,
                    a_col_indices,
                    a_values,
                    b_row_ptrs,
                    b_col_indices,
                    b_values,
                    nrows,
                )
            },
            #[cfg(feature = "f16")]
            DType::F16 => unsafe {
                csr_add_merge::<half::f16>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    dtype,
                    a_row_ptrs,
                    a_col_indices,
                    a_values,
                    b_row_ptrs,
                    b_col_indices,
                    b_values,
                    nrows,
                )
            },
            #[cfg(feature = "f16")]
            DType::BF16 => unsafe {
                csr_add_merge::<half::bf16>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    dtype,
                    a_row_ptrs,
                    a_col_indices,
                    a_values,
                    b_row_ptrs,
                    b_col_indices,
                    b_values,
                    nrows,
                )
            },
            _ => Err(Error::Internal(format!(
                "Unsupported dtype for CUDA sparse add: {:?}",
                dtype
            ))),
        }
    }

    fn sub_csr<T: Element>(
        &self,
        a_row_ptrs: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_ptrs: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let [nrows, _] = shape;
        let device = a_values.device();
        let dtype = a_values.dtype();

        // Two-pass merge algorithm with dtype dispatch
        use crate::dtype::DType;
        match dtype {
            DType::F32 => unsafe {
                csr_sub_merge::<f32>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    dtype,
                    a_row_ptrs,
                    a_col_indices,
                    a_values,
                    b_row_ptrs,
                    b_col_indices,
                    b_values,
                    nrows,
                )
            },
            DType::F64 => unsafe {
                csr_sub_merge::<f64>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    dtype,
                    a_row_ptrs,
                    a_col_indices,
                    a_values,
                    b_row_ptrs,
                    b_col_indices,
                    b_values,
                    nrows,
                )
            },
            #[cfg(feature = "f16")]
            DType::F16 => unsafe {
                csr_sub_merge::<half::f16>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    dtype,
                    a_row_ptrs,
                    a_col_indices,
                    a_values,
                    b_row_ptrs,
                    b_col_indices,
                    b_values,
                    nrows,
                )
            },
            #[cfg(feature = "f16")]
            DType::BF16 => unsafe {
                csr_sub_merge::<half::bf16>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    dtype,
                    a_row_ptrs,
                    a_col_indices,
                    a_values,
                    b_row_ptrs,
                    b_col_indices,
                    b_values,
                    nrows,
                )
            },
            _ => Err(Error::Internal(format!(
                "Unsupported dtype for CUDA sparse sub: {:?}",
                dtype
            ))),
        }
    }

    fn mul_csr<T: Element>(
        &self,
        a_row_ptrs: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_ptrs: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let [nrows, _] = shape;
        let device = a_values.device();
        let dtype = a_values.dtype();

        // Two-pass merge algorithm with dtype dispatch
        use crate::dtype::DType;
        match dtype {
            DType::F32 => unsafe {
                csr_mul_merge::<f32>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    dtype,
                    a_row_ptrs,
                    a_col_indices,
                    a_values,
                    b_row_ptrs,
                    b_col_indices,
                    b_values,
                    nrows,
                )
            },
            DType::F64 => unsafe {
                csr_mul_merge::<f64>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    dtype,
                    a_row_ptrs,
                    a_col_indices,
                    a_values,
                    b_row_ptrs,
                    b_col_indices,
                    b_values,
                    nrows,
                )
            },
            #[cfg(feature = "f16")]
            DType::F16 => unsafe {
                csr_mul_merge::<half::f16>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    dtype,
                    a_row_ptrs,
                    a_col_indices,
                    a_values,
                    b_row_ptrs,
                    b_col_indices,
                    b_values,
                    nrows,
                )
            },
            #[cfg(feature = "f16")]
            DType::BF16 => unsafe {
                csr_mul_merge::<half::bf16>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    dtype,
                    a_row_ptrs,
                    a_col_indices,
                    a_values,
                    b_row_ptrs,
                    b_col_indices,
                    b_values,
                    nrows,
                )
            },
            _ => Err(Error::Internal(format!(
                "Unsupported dtype for CUDA sparse mul: {:?}",
                dtype
            ))),
        }
    }

    // =========================================================================
    // CSC Operations (CPU fallback - TODO: Implement GPU-native kernels)
    // =========================================================================

    fn add_csc<T: Element>(
        &self,
        a_col_ptrs: &Tensor<CudaRuntime>,
        a_row_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_col_ptrs: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // TODO: Implement GPU-native CSC merge kernel
        // For now, fallback to CPU implementation
        crate::runtime::fallback::csc_elementwise_fallback::<T, CudaRuntime>(
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            shape,
            |a, b| T::from_f64(a.to_f64() + b.to_f64()),
        )
    }

    fn sub_csc<T: Element>(
        &self,
        a_col_ptrs: &Tensor<CudaRuntime>,
        a_row_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_col_ptrs: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // TODO: Implement GPU-native CSC merge kernel
        crate::runtime::fallback::csc_elementwise_fallback::<T, CudaRuntime>(
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            shape,
            |a, b| T::from_f64(a.to_f64() - b.to_f64()),
        )
    }

    fn mul_csc<T: Element>(
        &self,
        a_col_ptrs: &Tensor<CudaRuntime>,
        a_row_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_col_ptrs: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // TODO: Implement GPU-native CSC merge kernel
        crate::runtime::fallback::csc_elementwise_fallback::<T, CudaRuntime>(
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            shape,
            |a, b| T::from_f64(a.to_f64() * b.to_f64()),
        )
    }

    // =========================================================================
    // COO Operations (CPU fallback - TODO: Implement GPU-native kernels)
    // =========================================================================

    fn add_coo<T: Element>(
        &self,
        a_row_indices: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // TODO: Implement GPU-native COO sort-merge kernel
        crate::runtime::fallback::coo_elementwise_fallback::<T, CudaRuntime>(
            a_row_indices,
            a_col_indices,
            a_values,
            b_row_indices,
            b_col_indices,
            b_values,
            shape,
            |a, b| T::from_f64(a.to_f64() + b.to_f64()),
        )
    }

    fn sub_coo<T: Element>(
        &self,
        a_row_indices: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // TODO: Implement GPU-native COO sort-merge kernel
        crate::runtime::fallback::coo_elementwise_fallback::<T, CudaRuntime>(
            a_row_indices,
            a_col_indices,
            a_values,
            b_row_indices,
            b_col_indices,
            b_values,
            shape,
            |a, b| T::from_f64(a.to_f64() - b.to_f64()),
        )
    }

    fn mul_coo<T: Element>(
        &self,
        a_row_indices: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // TODO: Implement GPU-native COO sort-merge kernel
        crate::runtime::fallback::coo_elementwise_fallback::<T, CudaRuntime>(
            a_row_indices,
            a_col_indices,
            a_values,
            b_row_indices,
            b_col_indices,
            b_values,
            shape,
            |a, b| T::from_f64(a.to_f64() * b.to_f64()),
        )
    }

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
            operation: "dsmm (dense*sparse) on CUDA".to_string(),
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
        a: &crate::sparse::SparseTensor<CudaRuntime>,
        b: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
        // Use CPU fallback for sparse × sparse matrix multiplication
        // TODO: Implement native GPU kernel
        use crate::runtime::fallback::CpuFallbackContext;
        use crate::sparse::{SparseOps, SparseTensor};

        let cpu = CpuFallbackContext::new();
        let dtype = a.dtype();
        let device = a.device();

        // Dispatch based on dtype to handle type conversions properly
        crate::dispatch_dtype!(dtype, T => {
            // Helper to convert GPU SparseTensor to CPU with proper type
            let to_cpu_csr = |data: &crate::sparse::CsrData<CudaRuntime>| -> crate::sparse::CsrData<cpu::CpuRuntime> {
                let row_ptrs_cpu = cpu.tensor_from_gpu::<i64, CudaRuntime>(&data.row_ptrs);
                let col_indices_cpu = cpu.tensor_from_gpu::<i64, CudaRuntime>(&data.col_indices);
                let values_cpu = cpu.tensor_from_gpu::<T, CudaRuntime>(&data.values);
                crate::sparse::CsrData {
                    row_ptrs: row_ptrs_cpu,
                    col_indices: col_indices_cpu,
                    values: values_cpu,
                    shape: data.shape,
                }
            };

            // Convert inputs to CPU CSR format
            let a_csr = match a {
                SparseTensor::Csr(data) => to_cpu_csr(data),
                _ => {
                    let csr = a.to_csr()?;
                    to_cpu_csr(&csr)
                }
            };
            let b_csr = match b {
                SparseTensor::Csr(data) => to_cpu_csr(data),
                _ => {
                    let csr = b.to_csr()?;
                    to_cpu_csr(&csr)
                }
            };

            let a_cpu = SparseTensor::Csr(a_csr);
            let b_cpu = SparseTensor::Csr(b_csr);

            // Execute on CPU
            let result_cpu = cpu.client.sparse_matmul(&a_cpu, &b_cpu)?;

            // Convert result back to GPU
            match result_cpu {
                SparseTensor::Csr(data) => {
                    let row_ptrs_data: Vec<i64> = data.row_ptrs.to_vec();
                    let col_indices_data: Vec<i64> = data.col_indices.to_vec();
                    let values_data: Vec<T> = data.values.to_vec();

                    let row_ptrs_gpu = Tensor::<CudaRuntime>::from_slice(&row_ptrs_data, data.row_ptrs.shape(), device);
                    let col_indices_gpu = Tensor::<CudaRuntime>::from_slice(&col_indices_data, data.col_indices.shape(), device);
                    let values_gpu = Tensor::<CudaRuntime>::from_slice(&values_data, data.values.shape(), device);

                    return Ok(SparseTensor::Csr(crate::sparse::CsrData {
                        row_ptrs: row_ptrs_gpu,
                        col_indices: col_indices_gpu,
                        values: values_gpu,
                        shape: data.shape,
                    }));
                }
                _ => unreachable!("CPU sparse_matmul should return CSR"),
            }
        }, "sparse_matmul")
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

    fn sparse_sum(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Use CPU fallback for reductions
        self.sparse_sum_fallback(a)
    }

    fn sparse_sum_rows(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Use CPU fallback for reductions
        self.sparse_sum_rows_fallback(a)
    }

    fn sparse_sum_cols(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Use CPU fallback for reductions
        self.sparse_sum_cols_fallback(a)
    }

    fn sparse_nnz_per_row(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Use CPU fallback for nnz counting
        self.sparse_nnz_per_row_fallback(a)
    }

    fn sparse_nnz_per_col(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Use CPU fallback for nnz counting
        self.sparse_nnz_per_col_fallback(a)
    }

    fn sparse_to_dense(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Use CPU fallback for conversion
        self.sparse_to_dense_fallback(a)
    }

    fn dense_to_sparse(
        &self,
        a: &Tensor<CudaRuntime>,
        threshold: f64,
    ) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
        // Use CPU fallback for conversion
        self.dense_to_sparse_fallback(a, threshold)
    }

    // =========================================================================
    // Format Conversions (Low-Level)
    // =========================================================================
    //
    // TODO: Implement native GPU kernels using CUB for radix sort + prefix sum.
    // For now, using CPU fallback for correctness. This is acceptable as:
    // 1. Format conversion is typically done once at initialization
    // 2. Most operations auto-convert to optimal format transparently
    // 3. GPU-optimized conversions can be added in Phase 5

    fn coo_to_csr<T: crate::dtype::Element>(
        &self,
        row_indices: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // Use CPU fallback for format conversion
        // TODO: Implement GPU kernel with CUB radix sort
        use crate::runtime::fallback::CpuFallbackContext;
        use crate::sparse::SparseOps;

        let cpu = CpuFallbackContext::new();
        let device = values.device();

        // Copy to CPU
        let row_indices_cpu = cpu.tensor_from_gpu::<i64, CudaRuntime>(row_indices);
        let col_indices_cpu = cpu.tensor_from_gpu::<i64, CudaRuntime>(col_indices);
        let values_cpu = cpu.tensor_from_gpu::<T, CudaRuntime>(values);

        // Execute on CPU
        let (row_ptrs_cpu, col_indices_out_cpu, values_out_cpu) =
            cpu.client
                .coo_to_csr::<T>(&row_indices_cpu, &col_indices_cpu, &values_cpu, shape)?;

        // Copy back to GPU
        let row_ptrs_data: Vec<i64> = row_ptrs_cpu.to_vec();
        let col_indices_data: Vec<i64> = col_indices_out_cpu.to_vec();
        let values_data: Vec<T> = values_out_cpu.to_vec();

        let row_ptrs =
            Tensor::<CudaRuntime>::from_slice(&row_ptrs_data, row_ptrs_cpu.shape(), device);
        let col_indices_out = Tensor::<CudaRuntime>::from_slice(
            &col_indices_data,
            col_indices_out_cpu.shape(),
            device,
        );
        let values_out =
            Tensor::<CudaRuntime>::from_slice(&values_data, values_out_cpu.shape(), device);

        Ok((row_ptrs, col_indices_out, values_out))
    }

    fn coo_to_csc<T: crate::dtype::Element>(
        &self,
        row_indices: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // Use CPU fallback for format conversion
        // TODO: Implement GPU kernel with CUB radix sort
        use crate::runtime::fallback::CpuFallbackContext;
        use crate::sparse::SparseOps;

        let cpu = CpuFallbackContext::new();
        let device = values.device();

        // Copy to CPU
        let row_indices_cpu = cpu.tensor_from_gpu::<i64, CudaRuntime>(row_indices);
        let col_indices_cpu = cpu.tensor_from_gpu::<i64, CudaRuntime>(col_indices);
        let values_cpu = cpu.tensor_from_gpu::<T, CudaRuntime>(values);

        // Execute on CPU
        let (col_ptrs_cpu, row_indices_out_cpu, values_out_cpu) =
            cpu.client
                .coo_to_csc::<T>(&row_indices_cpu, &col_indices_cpu, &values_cpu, shape)?;

        // Copy back to GPU
        let col_ptrs_data: Vec<i64> = col_ptrs_cpu.to_vec();
        let row_indices_data: Vec<i64> = row_indices_out_cpu.to_vec();
        let values_data: Vec<T> = values_out_cpu.to_vec();

        let col_ptrs =
            Tensor::<CudaRuntime>::from_slice(&col_ptrs_data, col_ptrs_cpu.shape(), device);
        let row_indices_out = Tensor::<CudaRuntime>::from_slice(
            &row_indices_data,
            row_indices_out_cpu.shape(),
            device,
        );
        let values_out =
            Tensor::<CudaRuntime>::from_slice(&values_data, values_out_cpu.shape(), device);

        Ok((col_ptrs, row_indices_out, values_out))
    }

    fn csr_to_coo<T: crate::dtype::Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // Native GPU kernel for pointer expansion (CSR → COO)
        let [nrows, _ncols] = shape;
        let nnz = values.numel();
        let device = values.device();
        let dtype = values.dtype();

        // Allocate output row_indices on GPU
        let row_indices = Tensor::<CudaRuntime>::zeros(&[nnz], crate::dtype::DType::I64, device);

        // Get device pointers (no data transfer!)
        let row_ptrs_ptr = row_ptrs.storage().ptr();
        let row_indices_ptr = row_indices.storage().ptr();

        // Launch pointer expansion kernel
        unsafe {
            launch_expand_ptrs(
                &self.context,
                &self.stream,
                self.device.index(),
                row_ptrs_ptr,
                row_indices_ptr,
                nrows,
            )?;
        }

        // Sync stream to ensure kernel completion
        self.stream.synchronize()?;

        // col_indices and values remain unchanged
        let col_indices_out = col_indices.clone();
        let values_out = values.clone();

        Ok((row_indices, col_indices_out, values_out))
    }

    fn csc_to_coo<T: crate::dtype::Element>(
        &self,
        col_ptrs: &Tensor<CudaRuntime>,
        row_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // Native GPU kernel for pointer expansion (CSC → COO)
        let [_nrows, ncols] = shape;
        let nnz = values.numel();
        let device = values.device();
        let dtype = values.dtype();

        // Allocate output col_indices on GPU
        let col_indices = Tensor::<CudaRuntime>::zeros(&[nnz], crate::dtype::DType::I64, device);

        // Get device pointers (no data transfer!)
        let col_ptrs_ptr = col_ptrs.storage().ptr();
        let col_indices_ptr = col_indices.storage().ptr();

        // Launch pointer expansion kernel
        unsafe {
            launch_expand_ptrs(
                &self.context,
                &self.stream,
                self.device.index(),
                col_ptrs_ptr,
                col_indices_ptr,
                ncols,
            )?;
        }

        // Sync stream to ensure kernel completion
        self.stream.synchronize()?;

        // row_indices and values remain unchanged
        let row_indices_out = row_indices.clone();
        let values_out = values.clone();

        Ok((row_indices_out, col_indices, values_out))
    }

    fn csr_to_csc<T: crate::dtype::Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // Native GPU implementation: histogram + scan + scatter
        let [nrows, ncols] = shape;
        let nnz = values.numel();
        let device = values.device();
        let dtype = values.dtype();

        // Step 1: Count NNZ per column (histogram)
        let col_counts = Tensor::<CudaRuntime>::zeros(&[ncols], crate::dtype::DType::I64, device);

        unsafe {
            launch_histogram_csr_columns(
                &self.context,
                &self.stream,
                self.device.index(),
                row_ptrs.storage().ptr(),
                col_indices.storage().ptr(),
                col_counts.storage().ptr(),
                nrows,
            )?;
        }

        // Step 2: Exclusive scan to build column pointers
        let (col_ptrs, _total_nnz) = unsafe {
            exclusive_scan_i64_gpu(
                &self.context,
                &self.stream,
                self.device.index(),
                &self.device,
                &col_counts,
            )?
        };

        // Step 3: Make a copy of column pointers for scatter (will be modified)
        let col_ptrs_working = col_ptrs.clone();

        // Step 4: Allocate output arrays
        let row_indices_out =
            Tensor::<CudaRuntime>::zeros(&[nnz], crate::dtype::DType::I64, device);
        let values_out = Tensor::<CudaRuntime>::zeros(&[nnz], dtype, device);

        // Step 5: Launch scatter kernel
        unsafe {
            launch_csr_to_csc_transpose::<T>(
                &self.context,
                &self.stream,
                self.device.index(),
                row_ptrs.storage().ptr(),
                col_indices.storage().ptr(),
                values.storage().ptr(),
                col_ptrs_working.storage().ptr(),
                row_indices_out.storage().ptr(),
                values_out.storage().ptr(),
                nrows,
                ncols,
            )?;
        }

        // Sync to ensure completion
        self.stream.synchronize()?;

        // Return col_ptrs (unmodified), row_indices_out, values_out
        Ok((col_ptrs, row_indices_out, values_out))
    }

    fn csc_to_csr<T: crate::dtype::Element>(
        &self,
        col_ptrs: &Tensor<CudaRuntime>,
        row_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // Native GPU implementation: histogram + scan + scatter
        let [nrows, ncols] = shape;
        let nnz = values.numel();
        let device = values.device();
        let dtype = values.dtype();

        // Step 1: Count NNZ per row (histogram)
        let row_counts = Tensor::<CudaRuntime>::zeros(&[nrows], crate::dtype::DType::I64, device);

        unsafe {
            launch_histogram_csc_rows(
                &self.context,
                &self.stream,
                self.device.index(),
                col_ptrs.storage().ptr(),
                row_indices.storage().ptr(),
                row_counts.storage().ptr(),
                ncols,
            )?;
        }

        // Step 2: Exclusive scan to build row pointers
        let (row_ptrs, _total_nnz) = unsafe {
            exclusive_scan_i64_gpu(
                &self.context,
                &self.stream,
                self.device.index(),
                &self.device,
                &row_counts,
            )?
        };

        // Step 3: Make a copy of row pointers for scatter (will be modified)
        let row_ptrs_working = row_ptrs.clone();

        // Step 4: Allocate output arrays
        let col_indices_out =
            Tensor::<CudaRuntime>::zeros(&[nnz], crate::dtype::DType::I64, device);
        let values_out = Tensor::<CudaRuntime>::zeros(&[nnz], dtype, device);

        // Step 5: Launch scatter kernel
        unsafe {
            launch_csc_to_csr_transpose::<T>(
                &self.context,
                &self.stream,
                self.device.index(),
                col_ptrs.storage().ptr(),
                row_indices.storage().ptr(),
                values.storage().ptr(),
                row_ptrs_working.storage().ptr(),
                col_indices_out.storage().ptr(),
                values_out.storage().ptr(),
                nrows,
                ncols,
            )?;
        }

        // Sync to ensure completion
        self.stream.synchronize()?;

        // Return row_ptrs (unmodified), col_indices_out, values_out
        Ok((row_ptrs, col_indices_out, values_out))
    }

    fn sparse_transpose(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
        use crate::sparse::SparseTensor;

        // Transpose is efficient format conversion
        match a {
            SparseTensor::Csr(data) => {
                let csc = data.to_csc()?;
                Ok(SparseTensor::Csc(csc))
            }
            SparseTensor::Csc(data) => {
                let csr = data.to_csr()?;
                Ok(SparseTensor::Csr(csr))
            }
            SparseTensor::Coo(data) => {
                let [nrows, ncols] = data.shape;
                let transposed = crate::sparse::CooData {
                    row_indices: data.col_indices.clone(),
                    col_indices: data.row_indices.clone(),
                    values: data.values.clone(),
                    shape: [ncols, nrows],
                    sorted: false,
                };
                Ok(SparseTensor::Coo(transposed))
            }
        }
    }
}
