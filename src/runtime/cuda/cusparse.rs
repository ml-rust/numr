//! cusparse wrapper for sparse matrix operations
//!
//! Provides RAII wrappers around NVIDIA cusparse library for GPU-accelerated sparse operations.

#![allow(unsafe_op_in_unsafe_fn)]

use super::device::CudaError;
use cudarc::cusparse::sys::*;
use cudarc::driver::CudaStream;
use std::ptr::null_mut;
use std::sync::Arc;

/// RAII wrapper for cusparse handle
pub struct CudaSparse {
    handle: cusparseHandle_t,
    stream: Arc<CudaStream>,
}

impl CudaSparse {
    /// Create a new cusparse handle
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CudaError> {
        unsafe {
            let mut handle = null_mut();
            check_cusparse(cusparseCreate(&mut handle))?;

            // Associate handle with stream
            check_cusparse(cusparseSetStream(
                handle,
                stream.cu_stream() as cudaStream_t,
            ))?;

            Ok(Self { handle, stream })
        }
    }

    /// Get the raw cusparse handle
    #[inline]
    pub fn handle(&self) -> cusparseHandle_t {
        self.handle
    }

    /// Get the associated CUDA stream
    #[inline]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl Drop for CudaSparse {
    fn drop(&mut self) {
        unsafe {
            let _ = cusparseDestroy(self.handle);
        }
    }
}

// cusparseHandle_t is a raw pointer, so we need to manually implement Send/Sync
// SAFETY: cusparse operations are synchronized by the associated CUDA stream,
// and the stream ensures proper synchronization across threads
unsafe impl Send for CudaSparse {}
unsafe impl Sync for CudaSparse {}

/// Check cusparse status and convert to Result
pub fn check_cusparse(status: cusparseStatus_t) -> Result<(), CudaError> {
    if status == cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(CudaError::CusparseError(format!(
            "cusparse error: {:?}",
            status
        )))
    }
}

/// Convert DType to cudaDataType for cusparse
pub fn dtype_to_cusparse(dtype: crate::dtype::DType) -> cudaDataType {
    use crate::dtype::DType;
    match dtype {
        DType::F64 => cudaDataType::CUDA_R_64F,
        DType::F32 => cudaDataType::CUDA_R_32F,
        DType::F16 => cudaDataType::CUDA_R_16F,
        DType::BF16 => cudaDataType::CUDA_R_16BF,
        DType::I32 => cudaDataType::CUDA_R_32I,
        _ => panic!("Unsupported dtype for cusparse: {:?}", dtype),
    }
}

/// RAII wrapper for CSR sparse matrix descriptor
pub struct CsrMatrixDescriptor {
    descr: cusparseSpMatDescr_t,
}

impl CsrMatrixDescriptor {
    /// Create a new CSR matrix descriptor
    ///
    /// # Safety
    /// - row_ptrs must point to valid device memory with (rows + 1) elements
    /// - col_indices and values must point to valid device memory with nnz elements
    pub unsafe fn new(
        rows: i64,
        cols: i64,
        nnz: i64,
        row_ptrs: *const i32,
        col_indices: *const i32,
        values: *const std::ffi::c_void,
        data_type: cudaDataType,
    ) -> Result<Self, CudaError> {
        let mut descr = null_mut();
        check_cusparse(cusparseCreateCsr(
            &mut descr,
            rows,
            cols,
            nnz,
            row_ptrs as *mut std::ffi::c_void, // cusparse requires mutable pointers
            col_indices as *mut std::ffi::c_void,
            values as *mut std::ffi::c_void,
            cusparseIndexType_t::CUSPARSE_INDEX_32I,
            cusparseIndexType_t::CUSPARSE_INDEX_32I,
            cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
            data_type,
        ))?;
        Ok(Self { descr })
    }

    /// Get the raw descriptor handle
    #[inline]
    pub fn handle(&self) -> cusparseSpMatDescr_t {
        self.descr
    }
}

impl Drop for CsrMatrixDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _ = cusparseDestroySpMat(self.descr);
        }
    }
}

/// RAII wrapper for CSC sparse matrix descriptor
pub struct CscMatrixDescriptor {
    descr: cusparseSpMatDescr_t,
}

impl CscMatrixDescriptor {
    /// Create a new CSC matrix descriptor
    ///
    /// # Safety
    /// - col_ptrs must point to valid device memory with (cols + 1) elements
    /// - row_indices and values must point to valid device memory with nnz elements
    pub unsafe fn new(
        rows: i64,
        cols: i64,
        nnz: i64,
        col_ptrs: *const i32,
        row_indices: *const i32,
        values: *const std::ffi::c_void,
        data_type: cudaDataType,
    ) -> Result<Self, CudaError> {
        let mut descr = null_mut();
        check_cusparse(cusparseCreateCsc(
            &mut descr,
            rows,
            cols,
            nnz,
            col_ptrs as *mut std::ffi::c_void, // cusparse requires mutable pointers
            row_indices as *mut std::ffi::c_void,
            values as *mut std::ffi::c_void,
            cusparseIndexType_t::CUSPARSE_INDEX_32I,
            cusparseIndexType_t::CUSPARSE_INDEX_32I,
            cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
            data_type,
        ))?;
        Ok(Self { descr })
    }

    /// Get the raw descriptor handle
    #[inline]
    pub fn handle(&self) -> cusparseSpMatDescr_t {
        self.descr
    }
}

impl Drop for CscMatrixDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _ = cusparseDestroySpMat(self.descr);
        }
    }
}

/// RAII wrapper for dense matrix descriptor
pub struct DenseMatrixDescriptor {
    descr: cusparseDnMatDescr_t,
}

impl DenseMatrixDescriptor {
    /// Create a new dense matrix descriptor
    ///
    /// # Safety
    /// - values must point to valid device memory with rows * cols elements
    pub unsafe fn new(
        rows: i64,
        cols: i64,
        values: *const std::ffi::c_void,
        data_type: cudaDataType,
        order: cusparseOrder_t,
    ) -> Result<Self, CudaError> {
        let mut descr = null_mut();
        let ld = if order == cusparseOrder_t::CUSPARSE_ORDER_ROW {
            cols // Row-major: leading dimension is number of columns
        } else {
            rows // Column-major: leading dimension is number of rows
        };

        check_cusparse(cusparseCreateDnMat(
            &mut descr,
            rows,
            cols,
            ld,
            values as *mut std::ffi::c_void,
            data_type,
            order,
        ))?;
        Ok(Self { descr })
    }

    /// Get the raw descriptor handle
    #[inline]
    pub fn handle(&self) -> cusparseDnMatDescr_t {
        self.descr
    }
}

impl Drop for DenseMatrixDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _ = cusparseDestroyDnMat(self.descr);
        }
    }
}

/// RAII wrapper for SpGEMM descriptor
pub struct SpGEMMDescriptor {
    descr: cusparseSpGEMMDescr_t,
}

impl SpGEMMDescriptor {
    /// Create a new SpGEMM descriptor
    pub fn new() -> Result<Self, CudaError> {
        unsafe {
            let mut descr = null_mut();
            check_cusparse(cusparseSpGEMM_createDescr(&mut descr))?;
            Ok(Self { descr })
        }
    }

    /// Get the raw descriptor handle
    #[inline]
    pub fn handle(&self) -> cusparseSpGEMMDescr_t {
        self.descr
    }
}

impl Drop for SpGEMMDescriptor {
    fn drop(&mut self) {
        unsafe {
            let _ = cusparseSpGEMM_destroyDescr(self.descr);
        }
    }
}
