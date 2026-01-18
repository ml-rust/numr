//! cusparse-backed helpers for CUDA sparse ops.

use super::super::{CudaClient, CudaRuntime};
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::runtime::allocator::Allocator;
use crate::tensor::Tensor;

impl CudaClient {
    /// Cast i64 tensor to i32 for cusparse compatibility (GPU-only, no CPU transfer)
    fn cast_i64_to_i32(&self, tensor: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        use crate::dtype::DType;
        use crate::runtime::cuda::kernels::launch_cast;

        let device = tensor.device();
        let shape_len = tensor.shape()[0];

        // Handle empty tensors
        if shape_len == 0 {
            return Ok(Tensor::zeros(&[0], DType::I32, device));
        }

        // TODO: Add overflow validation kernel instead of assuming matrices < 2^31
        // For now, rely on cusparse validation to catch overflow

        // Allocate output tensor
        let output = Tensor::zeros(&[shape_len], DType::I32, device);

        // Launch cast kernel (stays on GPU)
        unsafe {
            launch_cast(
                &self.context,
                &self.stream,
                self.device.index,
                DType::I64,
                DType::I32,
                tensor.storage().ptr(),
                output.storage().ptr(),
                shape_len,
            )?;
        }

        Ok(output)
    }

    /// Cast i32 tensor back to i64 (numr internal format) (GPU-only, no CPU transfer)
    fn cast_i32_to_i64(&self, tensor: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        use crate::dtype::DType;
        use crate::runtime::cuda::kernels::launch_cast;

        let device = tensor.device();
        let shape_len = tensor.shape()[0];

        // Handle empty tensors
        if shape_len == 0 {
            return Ok(Tensor::zeros(&[0], DType::I64, device));
        }

        // Allocate output tensor
        let output = Tensor::zeros(&[shape_len], DType::I64, device);

        // Launch cast kernel (stays on GPU)
        unsafe {
            launch_cast(
                &self.context,
                &self.stream,
                self.device.index,
                DType::I32,
                DType::I64,
                tensor.storage().ptr(),
                output.storage().ptr(),
                shape_len,
            )?;
        }

        Ok(output)
    }

    /// Filter sparse values below zero_tolerance to match CPU behavior (GPU-only)
    fn filter_sparse_values<
        T: Element + cudarc::types::CudaTypeName + cudarc::driver::DeviceRepr,
    >(
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
        use crate::runtime::cuda::kernels::filter_csr_values_gpu;
        use crate::runtime::sparse_utils::zero_tolerance;

        let threshold = zero_tolerance::<T>();
        eprintln!(
            "[DEBUG] Filter threshold for {:?}: {}",
            std::any::type_name::<T>(),
            threshold
        );

        let device = values.device();
        let dtype = values.dtype();

        // Dispatch to GPU kernel
        unsafe {
            filter_csr_values_gpu::<T>(
                &self.context,
                &self.stream,
                self.device.index,
                device,
                dtype,
                row_ptrs,
                col_indices,
                values,
                shape,
                T::from_f64(threshold),
            )
        }
    }

    /// Sparse × Sparse matrix multiplication using cusparseSpGEMM
    pub(super) fn sparse_matmul_csr_cusparse<
        T: Element + cudarc::types::CudaTypeName + cudarc::driver::DeviceRepr,
    >(
        &self,
        a_row_ptrs: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_ptrs: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        a_shape: [usize; 2],
        b_shape: [usize; 2],
    ) -> Result<crate::sparse::CsrData<CudaRuntime>> {
        use crate::runtime::cuda::cusparse::{
            CsrMatrixDescriptor, SpGEMMDescriptor, check_cusparse, dtype_to_cusparse,
        };
        use cudarc::cusparse::sys::*;
        use std::ptr::null_mut;

        let [m, k] = a_shape;
        let [_, n] = b_shape;

        // Validate matrix dimensions fit in i32 (cusparse requirement)
        if m > i32::MAX as usize || k > i32::MAX as usize || n > i32::MAX as usize {
            return Err(Error::Internal(format!(
                "Matrix dimensions ({}, {}, {}) exceed cusparse i32 limit (2^31 - 1)",
                m, k, n
            )));
        }

        let nnz_a = a_values.shape()[0];
        let nnz_b = b_values.shape()[0];

        // Cast indices from i64 to i32 for cusparse
        let a_row_ptrs_i32 = self.cast_i64_to_i32(a_row_ptrs)?;
        let a_col_indices_i32 = self.cast_i64_to_i32(a_col_indices)?;
        let b_row_ptrs_i32 = self.cast_i64_to_i32(b_row_ptrs)?;
        let b_col_indices_i32 = self.cast_i64_to_i32(b_col_indices)?;

        let device = a_values.device();
        let dtype = a_values.dtype();
        let cuda_dtype = dtype_to_cusparse(dtype);

        // SAFETY: All raw pointers come from live device tensors or allocator buffers,
        // and cusparse descriptors are created with matching sizes/dtypes for those buffers.
        // The CUDA stream/handle outlives the cusparse calls in this scope.
        unsafe {
            let handle = self.cusparse.handle();
            let check =
                |result: std::result::Result<(), crate::runtime::cuda::device::CudaError>,
                 context: &str|
                 -> Result<()> {
                    result.map_err(|e| Error::Internal(format!("{} failed: {:?}", context, e)))
                };

            // Create matrix descriptors
            let mat_a = CsrMatrixDescriptor::new(
                m as i64,
                k as i64,
                nnz_a as i64,
                a_row_ptrs_i32.storage().ptr() as *const i32,
                a_col_indices_i32.storage().ptr() as *const i32,
                a_values.storage().ptr() as *const std::ffi::c_void,
                cuda_dtype,
            )
            .map_err(|e| Error::Internal(format!("cusparse CSR A descriptor failed: {:?}", e)))?;

            let mat_b = CsrMatrixDescriptor::new(
                k as i64,
                n as i64,
                nnz_b as i64,
                b_row_ptrs_i32.storage().ptr() as *const i32,
                b_col_indices_i32.storage().ptr() as *const i32,
                b_values.storage().ptr() as *const std::ffi::c_void,
                cuda_dtype,
            )
            .map_err(|e| Error::Internal(format!("cusparse CSR B descriptor failed: {:?}", e)))?;

            // Create output matrix descriptor (initially empty)
            let mut mat_c = null_mut();
            check(
                check_cusparse(cusparseCreateCsr(
                    &mut mat_c,
                    m as i64,
                    n as i64,
                    0, // nnz unknown initially
                    null_mut(),
                    null_mut(),
                    null_mut(),
                    cusparseIndexType_t::CUSPARSE_INDEX_32I,
                    cusparseIndexType_t::CUSPARSE_INDEX_32I,
                    cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                    cuda_dtype,
                )),
                "cusparseCreateCsr",
            )?;

            // Create SpGEMM descriptor
            let spgemm_desc = SpGEMMDescriptor::new().map_err(|e| {
                Error::Internal(format!("cusparse SpGEMM descriptor failed: {:?}", e))
            })?;

            let alpha: T = T::one();
            let beta: T = T::zero();
            let alpha_ptr = &alpha as *const T as *const std::ffi::c_void;
            let beta_ptr = &beta as *const T as *const std::ffi::c_void;

            // PHASE 1: Work estimation
            let mut buffer_size1: usize = 0;
            check(
                check_cusparse(cusparseSpGEMM_workEstimation(
                    handle,
                    cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                    cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                    alpha_ptr,
                    mat_a.handle(),
                    mat_b.handle(),
                    beta_ptr,
                    mat_c,
                    cuda_dtype,
                    cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_DEFAULT,
                    spgemm_desc.handle(),
                    &mut buffer_size1,
                    null_mut(),
                )),
                "cusparseSpGEMM_workEstimation (size)",
            )?;

            let workspace1 = self.allocator.allocate(buffer_size1);
            check(
                check_cusparse(cusparseSpGEMM_workEstimation(
                    handle,
                    cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                    cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                    alpha_ptr,
                    mat_a.handle(),
                    mat_b.handle(),
                    beta_ptr,
                    mat_c,
                    cuda_dtype,
                    cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_DEFAULT,
                    spgemm_desc.handle(),
                    &mut buffer_size1,
                    workspace1 as *mut std::ffi::c_void,
                )),
                "cusparseSpGEMM_workEstimation",
            )?;

            // PHASE 2: Compute (symbolic + numeric)
            let mut buffer_size2: usize = 0;
            check(
                check_cusparse(cusparseSpGEMM_compute(
                    handle,
                    cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                    cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                    alpha_ptr,
                    mat_a.handle(),
                    mat_b.handle(),
                    beta_ptr,
                    mat_c,
                    cuda_dtype,
                    cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_DEFAULT,
                    spgemm_desc.handle(),
                    &mut buffer_size2,
                    null_mut(),
                )),
                "cusparseSpGEMM_compute (size)",
            )?;

            let workspace2 = self.allocator.allocate(buffer_size2);
            check(
                check_cusparse(cusparseSpGEMM_compute(
                    handle,
                    cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                    cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                    alpha_ptr,
                    mat_a.handle(),
                    mat_b.handle(),
                    beta_ptr,
                    mat_c,
                    cuda_dtype,
                    cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_DEFAULT,
                    spgemm_desc.handle(),
                    &mut buffer_size2,
                    workspace2 as *mut std::ffi::c_void,
                )),
                "cusparseSpGEMM_compute",
            )?;

            // PHASE 3: Get output matrix size
            let mut c_rows: i64 = 0;
            let mut c_cols: i64 = 0;
            let mut c_nnz: i64 = 0;
            let mut c_row_offsets_ptr: *mut std::ffi::c_void = null_mut();
            let mut c_col_indices_ptr: *mut std::ffi::c_void = null_mut();
            let mut c_values_ptr: *mut std::ffi::c_void = null_mut();
            let mut c_row_idx_type = cusparseIndexType_t::CUSPARSE_INDEX_32I;
            let mut c_col_idx_type = cusparseIndexType_t::CUSPARSE_INDEX_32I;
            let mut c_idx_base = cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO;
            let mut c_dtype = cuda_dtype;
            check(
                check_cusparse(cusparseCsrGet(
                    mat_c,
                    &mut c_rows,
                    &mut c_cols,
                    &mut c_nnz,
                    &mut c_row_offsets_ptr,
                    &mut c_col_indices_ptr,
                    &mut c_values_ptr,
                    &mut c_row_idx_type,
                    &mut c_col_idx_type,
                    &mut c_idx_base,
                    &mut c_dtype,
                )),
                "cusparseCsrGet",
            )?;

            // PHASE 4: Allocate and copy result
            let c_row_ptrs_i32 =
                Tensor::zeros(&[c_rows as usize + 1], crate::dtype::DType::I32, device);
            let c_col_indices_i32 =
                Tensor::zeros(&[c_nnz as usize], crate::dtype::DType::I32, device);
            let c_values = Tensor::zeros(&[c_nnz as usize], dtype, device);

            // Only set pointers and copy if result is non-empty
            if c_nnz > 0 {
                check(
                    check_cusparse(cusparseCsrSetPointers(
                        mat_c,
                        c_row_ptrs_i32.storage().ptr() as *mut i32 as *mut std::ffi::c_void,
                        c_col_indices_i32.storage().ptr() as *mut i32 as *mut std::ffi::c_void,
                        c_values.storage().ptr() as *mut std::ffi::c_void,
                    )),
                    "cusparseCsrSetPointers",
                )?;

                check(
                    check_cusparse(cusparseSpGEMM_copy(
                        handle,
                        cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                        cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                        alpha_ptr,
                        mat_a.handle(),
                        mat_b.handle(),
                        beta_ptr,
                        mat_c,
                        cuda_dtype,
                        cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_DEFAULT,
                        spgemm_desc.handle(),
                    )),
                    "cusparseSpGEMM_copy",
                )?;
            }

            // Clean up descriptors
            let _ = cusparseDestroySpMat(mat_c);
            self.allocator.deallocate(workspace1, buffer_size1);
            self.allocator.deallocate(workspace2, buffer_size2);

            // Convert i32 indices back to i64
            let c_row_ptrs = self.cast_i32_to_i64(&c_row_ptrs_i32)?;
            let c_col_indices = self.cast_i32_to_i64(&c_col_indices_i32)?;

            // Debug: Check cusparseSpGEMM output before filtering
            let pre_filter_nnz = c_values.shape()[0];
            eprintln!(
                "[DEBUG] cusparseSpGEMM produced {} non-zeros before filtering",
                pre_filter_nnz
            );

            // Filter small values to match CPU backend behavior (GPU-only)
            let (filtered_row_ptrs, filtered_col_indices, filtered_values) =
                self.filter_sparse_values::<T>(&c_row_ptrs, &c_col_indices, &c_values, [m, n])?;

            // Debug: Check after filtering
            let post_filter_nnz = filtered_values.shape()[0];
            eprintln!(
                "[DEBUG] After filtering: {} non-zeros (removed {})",
                post_filter_nnz,
                pre_filter_nnz - post_filter_nnz
            );

            Ok(crate::sparse::CsrData::new(
                filtered_row_ptrs,
                filtered_col_indices,
                filtered_values,
                [m, n],
            )?)
        }
    }

    /// Dense × Sparse matrix multiplication using cusparseSpMM
    ///
    /// Computes C = A @ B where A is dense [M, K] and B is sparse CSC [K, N].
    pub(super) fn dsmm_cusparse<T: Element>(
        &self,
        a: &Tensor<CudaRuntime>,
        b_col_ptrs: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        a_shape: [usize; 2],
        b_shape: [usize; 2],
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::runtime::cuda::cusparse::{
            CscMatrixDescriptor, DenseMatrixDescriptor, check_cusparse, dtype_to_cusparse,
        };
        use cudarc::cusparse::sys::*;

        let [m, k] = a_shape;
        let [_, n] = b_shape;

        // Validate matrix dimensions fit in i32 (cusparse requirement)
        if m > i32::MAX as usize || k > i32::MAX as usize || n > i32::MAX as usize {
            return Err(Error::Internal(format!(
                "Matrix dimensions ({}, {}, {}) exceed cusparse i32 limit (2^31 - 1)",
                m, k, n
            )));
        }

        let nnz_b = b_values.shape()[0];

        // Ensure A is contiguous for cusparse
        let a_contiguous = if a.is_contiguous() {
            a.clone()
        } else {
            a.contiguous()
        };

        // Cast indices from i64 to i32 for cusparse
        let b_col_ptrs_i32 = self.cast_i64_to_i32(b_col_ptrs)?;
        let b_row_indices_i32 = self.cast_i64_to_i32(b_row_indices)?;

        let device = a.device();
        let dtype = a.dtype();
        let cuda_dtype = dtype_to_cusparse(dtype);

        // Allocate output C [M, N]
        let c = Tensor::zeros(&[m, n], dtype, device);

        // SAFETY: A, B, and C pointers reference live device buffers with layouts that
        // match the cusparse descriptors, and the workspace buffer is valid for the
        // duration of the call. The handle/stream remain valid for this scope.
        unsafe {
            let handle = self.cusparse.handle();
            let check =
                |result: std::result::Result<(), crate::runtime::cuda::device::CudaError>,
                 context: &str|
                 -> Result<()> {
                    result.map_err(|e| Error::Internal(format!("{} failed: {:?}", context, e)))
                };

            // Create dense matrix descriptor for A (row-major)
            let mat_a = DenseMatrixDescriptor::new(
                m as i64,
                k as i64,
                a_contiguous.storage().ptr() as *const std::ffi::c_void,
                cuda_dtype,
                cusparseOrder_t::CUSPARSE_ORDER_ROW,
            )
            .map_err(|e| Error::Internal(format!("cusparse dense A descriptor failed: {:?}", e)))?;

            // Create sparse CSC matrix descriptor for B
            let mat_b = CscMatrixDescriptor::new(
                k as i64,
                n as i64,
                nnz_b as i64,
                b_col_ptrs_i32.storage().ptr() as *const i32,
                b_row_indices_i32.storage().ptr() as *const i32,
                b_values.storage().ptr() as *const std::ffi::c_void,
                cuda_dtype,
            )
            .map_err(|e| Error::Internal(format!("cusparse CSC B descriptor failed: {:?}", e)))?;

            // Create dense matrix descriptor for C (row-major)
            let mat_c = DenseMatrixDescriptor::new(
                n as i64,
                m as i64,
                c.storage().ptr() as *const std::ffi::c_void,
                cuda_dtype,
                cusparseOrder_t::CUSPARSE_ORDER_COL,
            )
            .map_err(|e| Error::Internal(format!("cusparse dense C descriptor failed: {:?}", e)))?;

            let alpha: T = T::one();
            let beta: T = T::zero();
            let alpha_ptr = &alpha as *const T as *const std::ffi::c_void;
            let beta_ptr = &beta as *const T as *const std::ffi::c_void;

            // Query workspace size
            let mut buffer_size: usize = 0;
            check(
                check_cusparse(cusparseSpMM_bufferSize(
                    handle,
                    cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
                    cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
                    alpha_ptr,
                    mat_b.handle(),
                    mat_a.handle(),
                    beta_ptr,
                    mat_c.handle(),
                    cuda_dtype,
                    cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT,
                    &mut buffer_size,
                )),
                "cusparseSpMM_bufferSize",
            )?;

            // Allocate workspace
            let workspace = self.allocator.allocate(buffer_size);

            // Perform multiplication
            check(
                check_cusparse(cusparseSpMM(
                    handle,
                    cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
                    cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
                    alpha_ptr,
                    mat_b.handle(),
                    mat_a.handle(),
                    beta_ptr,
                    mat_c.handle(),
                    cuda_dtype,
                    cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT,
                    workspace as *mut std::ffi::c_void,
                )),
                "cusparseSpMM",
            )?;

            // Clean up workspace
            self.allocator.deallocate(workspace, buffer_size);

            Ok(c)
        }
    }
}
