//! TensorOps, ScalarOps, and CompareOps implementations for CUDA runtime
//!
//! This module implements tensor operations for CUDA using:
//! - cuBLAS for matrix multiplication (matmul)
//! - CPU fallback for other operations (can be optimized with custom CUDA kernels later)
//!
//! # Performance Note
//!
//! Operations other than matmul currently use CPU fallback, which involves
//! Host-Device memory transfers. This is acceptable for Phase 2; custom CUDA
//! kernels will be implemented in future phases for better performance.

use super::{CudaClient, CudaRuntime};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    BinaryOp, CompareOp, CompareOps, ReduceOp, ScalarOps, TensorOps, UnaryOp, matmul_output_shape,
};
use crate::runtime::fallback::{
    activation_fallback, binary_op_fallback, compare_op_fallback, matmul_fallback,
    reduce_op_fallback, scalar_op_fallback, softmax_fallback, unary_op_fallback,
    validate_binary_dtypes,
};
use crate::tensor::Tensor;

use cudarc::cublas::Gemm;
use cudarc::cublas::sys::{cublasOperation_t, cublasSgemmStridedBatched};

#[cfg(feature = "f16")]
use cudarc::cublas::sys::{cublasComputeType_t, cublasGemmEx, cudaDataType_t};

// ============================================================================
// Helper Functions
// ============================================================================

/// Ensure a tensor is contiguous.
#[inline]
fn ensure_contiguous(tensor: &Tensor<CudaRuntime>) -> Tensor<CudaRuntime> {
    if tensor.is_contiguous() {
        tensor.clone()
    } else {
        tensor.contiguous()
    }
}

// ============================================================================
// cuBLAS GEMM Implementation
// ============================================================================

/// Matrix multiplication using cuBLAS for F32
///
/// Computes C = A @ B using cuBLAS SGEMM.
/// Handles row-major to column-major conversion internally.
fn matmul_f32_cublas(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor<CudaRuntime>> {
    use cudarc::cublas::GemmConfig;

    let a_contig = ensure_contiguous(a);
    let b_contig = ensure_contiguous(b);

    // Get output shape
    let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
        expected: a.shape().to_vec(),
        got: b.shape().to_vec(),
    })?;

    // Create output tensor
    let out = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, &client.device);

    let m = m as i32;
    let k = k as i32;
    let n = n as i32;

    // Handle row-major (C-style) vs column-major (FORTRAN/cuBLAS) mismatch
    // For row-major: C = A @ B becomes col-major: C^T = B^T @ A^T
    // We call cuBLAS with swapped matrices and NO transpose flags

    let transa = cublasOperation_t::CUBLAS_OP_N;
    let transb = cublasOperation_t::CUBLAS_OP_N;

    // For the swapped operation B @ A:
    let cublas_m = n;
    let cublas_n = m;
    let cublas_k = k;

    // Leading dimensions for row-major storage
    let lda = n;
    let ldb = k;
    let ldc = n;

    let cfg = GemmConfig {
        transa,
        transb,
        m: cublas_m,
        n: cublas_n,
        k: cublas_k,
        alpha: 1.0f32,
        lda,
        ldb,
        beta: 0.0f32,
        ldc,
    };

    // Create temporary device slices from raw pointers
    let a_ptr = a_contig.storage().ptr();
    let b_ptr = b_contig.storage().ptr();
    let out_ptr = out.storage().ptr();
    let numel_a = a_contig.numel();
    let numel_b = b_contig.numel();
    let numel_out = out.numel();

    // Swap B and A for cuBLAS (B becomes first arg, A becomes second)
    let b_slice = unsafe { client.stream.upgrade_device_ptr::<f32>(b_ptr, numel_b) };
    let a_slice = unsafe { client.stream.upgrade_device_ptr::<f32>(a_ptr, numel_a) };
    let mut out_slice = unsafe { client.stream.upgrade_device_ptr::<f32>(out_ptr, numel_out) };

    // Perform GEMM using cuBLAS
    unsafe {
        client
            .cublas
            .gemm(cfg, &b_slice, &a_slice, &mut out_slice)
            .map_err(|e| Error::Internal(format!("cuBLAS GEMM failed: {:?}", e)))?;
    }

    // Prevent drop from deallocating since we don't own the memory
    std::mem::forget(a_slice);
    std::mem::forget(b_slice);
    std::mem::forget(out_slice);

    Ok(out)
}

/// Batched matrix multiplication using cuBLAS for F32
fn matmul_batched_f32_cublas(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor<CudaRuntime>> {
    let a_contig = ensure_contiguous(a);
    let b_contig = ensure_contiguous(b);

    let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
        expected: a.shape().to_vec(),
        got: b.shape().to_vec(),
    })?;

    let out = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, &client.device);

    let batch = batch as i32;
    let m_i32 = m as i32;
    let k_i32 = k as i32;
    let n_i32 = n as i32;

    // Row-major to column-major conversion
    let transa = cublasOperation_t::CUBLAS_OP_N;
    let transb = cublasOperation_t::CUBLAS_OP_N;

    let cublas_m = n_i32;
    let cublas_n = m_i32;
    let cublas_k = k_i32;

    let lda = n_i32;
    let ldb = k_i32;
    let ldc = n_i32;

    // Strides between batches (in elements)
    let stride_a = (m * k) as i64;
    let stride_b = (n * k) as i64;
    let stride_c = (m * n) as i64;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    let a_ptr = a_contig.storage().ptr();
    let b_ptr = b_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    unsafe {
        let handle = *client.cublas.handle();

        let status = cublasSgemmStridedBatched(
            handle,
            transa,
            transb,
            cublas_m,
            cublas_n,
            cublas_k,
            &alpha,
            b_ptr as *const f32, // B becomes first matrix
            lda,
            stride_b,
            a_ptr as *const f32, // A becomes second matrix
            ldb,
            stride_a,
            &beta,
            out_ptr as *mut f32,
            ldc,
            stride_c,
            batch,
        );

        if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(Error::Internal(format!(
                "cuBLAS strided batched GEMM failed: {:?}",
                status
            )));
        }
    }

    Ok(out)
}

/// Matrix multiplication using cuBLAS GemmEx for F16/BF16
///
/// # Requirements
/// - F16 requires Compute Capability >= 5.3 (Maxwell+)
/// - BF16 requires Compute Capability >= 8.0 (Ampere+)
#[cfg(feature = "f16")]
fn matmul_half_cublas(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    m: usize,
    k: usize,
    n: usize,
    dtype: DType,
) -> Result<Tensor<CudaRuntime>> {
    // Validate compute capability for half-precision support
    if let Ok((major, minor)) = client.device.compute_capability() {
        let cc = major * 10 + minor;
        match dtype {
            DType::F16 if cc < 53 => {
                return Err(Error::Internal(format!(
                    "F16 matmul requires Compute Capability >= 5.3, got {}.{}",
                    major, minor
                )));
            }
            DType::BF16 if cc < 80 => {
                return Err(Error::Internal(format!(
                    "BF16 matmul requires Compute Capability >= 8.0, got {}.{}",
                    major, minor
                )));
            }
            _ => {}
        }
    }

    let a_contig = ensure_contiguous(a);
    let b_contig = ensure_contiguous(b);

    let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
        expected: a.shape().to_vec(),
        got: b.shape().to_vec(),
    })?;

    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);

    let m = m as i32;
    let k = k as i32;
    let n = n as i32;

    // Row-major to column-major conversion
    let transa = cublasOperation_t::CUBLAS_OP_N;
    let transb = cublasOperation_t::CUBLAS_OP_N;

    let cublas_m = n;
    let cublas_n = m;
    let cublas_k = k;

    let lda = n;
    let ldb = k;
    let ldc = n;

    // Select data type for cuBLAS
    let cuda_dtype = match dtype {
        DType::BF16 => cudaDataType_t::CUDA_R_16BF,
        DType::F16 => cudaDataType_t::CUDA_R_16F,
        _ => unreachable!(),
    };

    let compute_type = cublasComputeType_t::CUBLAS_COMPUTE_32F;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    let a_ptr = a_contig.storage().ptr();
    let b_ptr = b_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    unsafe {
        let handle = *client.cublas.handle();

        let status = cublasGemmEx(
            handle,
            transa,
            transb,
            cublas_m,
            cublas_n,
            cublas_k,
            &alpha as *const f32 as *const std::ffi::c_void,
            b_ptr as *const std::ffi::c_void,
            cuda_dtype,
            lda,
            a_ptr as *const std::ffi::c_void,
            cuda_dtype,
            ldb,
            &beta as *const f32 as *const std::ffi::c_void,
            out_ptr as *mut std::ffi::c_void,
            cuda_dtype,
            ldc,
            compute_type,
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        );

        if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(Error::Internal(format!(
                "cuBLAS GemmEx ({:?}) failed: {:?}",
                dtype, status
            )));
        }
    }

    Ok(out)
}

// ============================================================================
// TensorOps Implementation
// ============================================================================

impl TensorOps<CudaRuntime> for CudaClient {
    // ===== Binary Operations (CPU Fallback) =====

    fn add(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        binary_op_fallback(a, b, BinaryOp::Add, &self.device, "add")
    }

    fn sub(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        binary_op_fallback(a, b, BinaryOp::Sub, &self.device, "sub")
    }

    fn mul(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        binary_op_fallback(a, b, BinaryOp::Mul, &self.device, "mul")
    }

    fn div(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        binary_op_fallback(a, b, BinaryOp::Div, &self.device, "div")
    }

    fn pow(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        binary_op_fallback(a, b, BinaryOp::Pow, &self.device, "pow")
    }

    fn maximum(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        binary_op_fallback(a, b, BinaryOp::Max, &self.device, "maximum")
    }

    fn minimum(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        binary_op_fallback(a, b, BinaryOp::Min, &self.device, "minimum")
    }

    // ===== Unary Operations (CPU Fallback) =====

    fn neg(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_fallback(a, UnaryOp::Neg, &self.device, "neg")
    }

    fn abs(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_fallback(a, UnaryOp::Abs, &self.device, "abs")
    }

    fn sqrt(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_fallback(a, UnaryOp::Sqrt, &self.device, "sqrt")
    }

    fn exp(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_fallback(a, UnaryOp::Exp, &self.device, "exp")
    }

    fn log(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_fallback(a, UnaryOp::Log, &self.device, "log")
    }

    fn sin(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_fallback(a, UnaryOp::Sin, &self.device, "sin")
    }

    fn cos(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_fallback(a, UnaryOp::Cos, &self.device, "cos")
    }

    fn tan(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_fallback(a, UnaryOp::Tan, &self.device, "tan")
    }

    fn tanh(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_fallback(a, UnaryOp::Tanh, &self.device, "tanh")
    }

    fn recip(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_fallback(a, UnaryOp::Recip, &self.device, "recip")
    }

    fn square(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_fallback(a, UnaryOp::Square, &self.device, "square")
    }

    fn floor(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_fallback(a, UnaryOp::Floor, &self.device, "floor")
    }

    fn ceil(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_fallback(a, UnaryOp::Ceil, &self.device, "ceil")
    }

    fn round(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_fallback(a, UnaryOp::Round, &self.device, "round")
    }

    // ===== Matrix Operations (cuBLAS) =====

    fn matmul(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate dtypes match
        let dtype = validate_binary_dtypes(a, b)?;

        // Get matrix dimensions (last two dims)
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

        // Validate inner dimensions match
        let k_b = if b_shape.len() >= 2 {
            b_shape[b_shape.len() - 2]
        } else {
            b_shape[b_shape.len() - 1]
        };
        if k != k_b {
            return Err(Error::ShapeMismatch {
                expected: a_shape.to_vec(),
                got: b_shape.to_vec(),
            });
        }

        // Calculate batch size
        let out_shape = matmul_output_shape(a_shape, b_shape).ok_or(Error::ShapeMismatch {
            expected: a_shape.to_vec(),
            got: b_shape.to_vec(),
        })?;

        let batch_size: usize = out_shape
            .iter()
            .take(out_shape.len().saturating_sub(2))
            .product();
        let batch_size = batch_size.max(1);

        match dtype {
            DType::F32 => {
                if batch_size > 1 {
                    matmul_batched_f32_cublas(self, a, b, batch_size, m, k, n)
                } else {
                    matmul_f32_cublas(self, a, b, m, k, n)
                }
            }
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => matmul_half_cublas(self, a, b, m, k, n, dtype),
            _ => {
                // Fallback to CPU for other dtypes
                matmul_fallback(a, b, &out_shape, &self.device, "matmul")
            }
        }
    }

    // ===== Reductions (CPU Fallback) =====

    fn sum(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        reduce_op_fallback(a, ReduceOp::Sum, dims, keepdim, &self.device, "sum")
    }

    fn mean(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        reduce_op_fallback(a, ReduceOp::Mean, dims, keepdim, &self.device, "mean")
    }

    fn max(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        reduce_op_fallback(a, ReduceOp::Max, dims, keepdim, &self.device, "max")
    }

    fn min(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        reduce_op_fallback(a, ReduceOp::Min, dims, keepdim, &self.device, "min")
    }

    // ===== Activations (CPU Fallback) =====

    fn relu(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        activation_fallback(a, &self.device, "relu", |client, a_cpu| client.relu(a_cpu))
    }

    fn sigmoid(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        activation_fallback(a, &self.device, "sigmoid", |client, a_cpu| {
            client.sigmoid(a_cpu)
        })
    }

    fn softmax(&self, a: &Tensor<CudaRuntime>, dim: isize) -> Result<Tensor<CudaRuntime>> {
        softmax_fallback(a, dim, &self.device, "softmax")
    }
}

// ============================================================================
// ScalarOps Implementation
// ============================================================================

impl ScalarOps<CudaRuntime> for CudaClient {
    fn add_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        scalar_op_fallback(a, BinaryOp::Add, scalar, &self.device, "add_scalar")
    }

    fn sub_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        scalar_op_fallback(a, BinaryOp::Sub, scalar, &self.device, "sub_scalar")
    }

    fn mul_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        scalar_op_fallback(a, BinaryOp::Mul, scalar, &self.device, "mul_scalar")
    }

    fn div_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        scalar_op_fallback(a, BinaryOp::Div, scalar, &self.device, "div_scalar")
    }

    fn pow_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        scalar_op_fallback(a, BinaryOp::Pow, scalar, &self.device, "pow_scalar")
    }
}

// ============================================================================
// CompareOps Implementation (CPU Fallback)
// ============================================================================

impl CompareOps<CudaRuntime> for CudaClient {
    fn eq(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        compare_op_fallback(a, b, CompareOp::Eq, &self.device, "eq")
    }

    fn ne(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        compare_op_fallback(a, b, CompareOp::Ne, &self.device, "ne")
    }

    fn lt(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        compare_op_fallback(a, b, CompareOp::Lt, &self.device, "lt")
    }

    fn le(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        compare_op_fallback(a, b, CompareOp::Le, &self.device, "le")
    }

    fn gt(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        compare_op_fallback(a, b, CompareOp::Gt, &self.device, "gt")
    }

    fn ge(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        compare_op_fallback(a, b, CompareOp::Ge, &self.device, "ge")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::runtime::cuda::CudaDevice;

    #[test]
    fn test_cuda_tensor_add() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
        let b = Tensor::<CudaRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);

        let c = client.add(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_cuda_tensor_matmul_2x2() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C = A @ B = [[19, 22], [43, 50]]
        let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
        let b = Tensor::<CudaRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);

        let c = TensorOps::matmul(&client, &a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_cuda_tensor_matmul_3x2_2x4() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // A = [[1, 2], [3, 4], [5, 6]] (3x2)
        // B = [[1, 2, 3, 4], [5, 6, 7, 8]] (2x4)
        // C = A @ B (3x4)
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);
        let b = Tensor::<CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
            &device,
        );

        let c = TensorOps::matmul(&client, &a, &b).unwrap();

        assert_eq!(c.shape(), &[3, 4]);
        let result: Vec<f32> = c.to_vec();
        // Row 0: [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11, 14, 17, 20]
        // Row 1: [3*1+4*5, 3*2+4*6, 3*3+4*7, 3*4+4*8] = [23, 30, 37, 44]
        // Row 2: [5*1+6*5, 5*2+6*6, 5*3+6*7, 5*4+6*8] = [35, 46, 57, 68]
        assert_eq!(
            result,
            [
                11.0, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0
            ]
        );
    }

    #[test]
    fn test_cuda_tensor_relu() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a = Tensor::<CudaRuntime>::from_slice(&[-1.0f32, 0.0, 1.0, -2.0], &[4], &device);
        let b = client.relu(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_cuda_tensor_sum() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // Shape [2, 3] -> sum over dim 1 -> shape [2]
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

        let b = client.sum(&a, &[1], false).unwrap();

        assert_eq!(b.shape(), &[2]);
        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [6.0, 15.0]); // [1+2+3, 4+5+6]
    }
}
