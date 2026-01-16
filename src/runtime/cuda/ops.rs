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

#![allow(unreachable_code)] // dispatch_dtype! macro uses early returns in all branches

use super::{CudaClient, CudaRuntime};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::ops::{
    BinaryOp, CompareOps, ReduceOp, ScalarOps, TensorOps, UnaryOp, broadcast_shape,
    matmul_output_shape, reduce_output_shape,
};
use crate::runtime::{Runtime, cpu};
use crate::tensor::Tensor;

use cudarc::cublas::Gemm;
use cudarc::cublas::sys::{cublasOperation_t, cublasSgemmStridedBatched};

#[cfg(feature = "f16")]
use cudarc::cublas::sys::{cublasComputeType_t, cublasGemmEx, cudaDataType_t};

// ============================================================================
// DType Dispatch Macro
// ============================================================================

/// Macro for dtype dispatch to typed operations
#[allow(unused_macros)] // Macro used within this module
macro_rules! dispatch_dtype {
    ($dtype:expr, $T:ident => $body:block, $error_op:expr) => {
        match $dtype {
            DType::F64 => {
                type $T = f64;
                $body
            }
            DType::F32 => {
                type $T = f32;
                $body
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                type $T = half::f16;
                $body
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                type $T = half::bf16;
                $body
            }
            DType::I64 => {
                type $T = i64;
                $body
            }
            DType::I32 => {
                type $T = i32;
                $body
            }
            DType::I16 => {
                type $T = i16;
                $body
            }
            DType::I8 => {
                type $T = i8;
                $body
            }
            DType::U64 => {
                type $T = u64;
                $body
            }
            DType::U32 => {
                type $T = u32;
                $body
            }
            DType::U16 => {
                type $T = u16;
                $body
            }
            DType::U8 => {
                type $T = u8;
                $body
            }
            #[cfg(not(feature = "f16"))]
            DType::F16 | DType::BF16 => {
                return Err(Error::UnsupportedDType {
                    dtype: $dtype,
                    op: $error_op,
                })
            }
            DType::Bool => {
                return Err(Error::UnsupportedDType {
                    dtype: $dtype,
                    op: $error_op,
                })
            }
        }
    };
}

// ============================================================================
// Helper Functions
// ============================================================================

/// CPU fallback context for operations not yet implemented in CUDA.
///
/// This struct holds the CPU device and client needed for fallback operations.
/// Using a struct avoids the repeated `cpu::CpuDevice::new()` and
/// `cpu::CpuRuntime::default_client()` boilerplate in every fallback function.
struct CpuFallback {
    device: cpu::CpuDevice,
    client: cpu::CpuClient,
}

impl CpuFallback {
    /// Create a new CPU fallback context.
    #[inline]
    fn new() -> Self {
        let device = cpu::CpuDevice::new();
        let client = cpu::CpuRuntime::default_client(&device);
        Self { device, client }
    }

    /// Create a CPU tensor from GPU tensor data.
    #[inline]
    fn tensor_from_gpu<T: Element>(&self, tensor: &Tensor<CudaRuntime>) -> Tensor<cpu::CpuRuntime> {
        let data: Vec<T> = tensor.to_vec();
        Tensor::<cpu::CpuRuntime>::from_slice(&data, tensor.shape(), &self.device)
    }
}

/// Create a GPU tensor from CPU data.
#[inline]
fn tensor_from_cpu<T: Element>(
    data: &[T],
    shape: &[usize],
    device: &super::CudaDevice,
) -> Tensor<CudaRuntime> {
    Tensor::<CudaRuntime>::from_slice(data, shape, device)
}

/// Validate that two tensors have matching dtypes for binary operations.
#[inline]
fn validate_binary_dtypes(a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<DType> {
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    Ok(a.dtype())
}

/// Compute broadcast shape for binary operations.
#[inline]
fn compute_broadcast_shape(a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Vec<usize>> {
    broadcast_shape(a.shape(), b.shape()).ok_or_else(|| Error::BroadcastError {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })
}

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
// CPU Fallback Operations
// ============================================================================

/// Perform a binary operation using CPU fallback.
fn binary_op_cpu_fallback(
    client: &CudaClient,
    op: BinaryOp,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    op_name: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;
    let cpu = CpuFallback::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T>(a);
        let b_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T>(b);

        let result_cpu = match op {
            BinaryOp::Add => cpu.client.add(&a_cpu, &b_cpu)?,
            BinaryOp::Sub => cpu.client.sub(&a_cpu, &b_cpu)?,
            BinaryOp::Mul => cpu.client.mul(&a_cpu, &b_cpu)?,
            BinaryOp::Div => cpu.client.div(&a_cpu, &b_cpu)?,
            BinaryOp::Pow => cpu.client.pow(&a_cpu, &b_cpu)?,
            BinaryOp::Max => cpu.client.maximum(&a_cpu, &b_cpu)?,
            BinaryOp::Min => cpu.client.minimum(&a_cpu, &b_cpu)?,
        };

        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(tensor_from_cpu(&result_data, &out_shape, &client.device));
    }, op_name);

    unreachable!()
}

/// Perform a unary operation using CPU fallback.
fn unary_op_cpu_fallback(
    client: &CudaClient,
    op: UnaryOp,
    a: &Tensor<CudaRuntime>,
    op_name: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();
    let cpu = CpuFallback::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T>(a);

        let result_cpu = match op {
            UnaryOp::Neg => cpu.client.neg(&a_cpu)?,
            UnaryOp::Abs => cpu.client.abs(&a_cpu)?,
            UnaryOp::Sqrt => cpu.client.sqrt(&a_cpu)?,
            UnaryOp::Exp => cpu.client.exp(&a_cpu)?,
            UnaryOp::Log => cpu.client.log(&a_cpu)?,
            UnaryOp::Sin => cpu.client.sin(&a_cpu)?,
            UnaryOp::Cos => cpu.client.cos(&a_cpu)?,
            UnaryOp::Tan => cpu.client.tan(&a_cpu)?,
            UnaryOp::Tanh => cpu.client.tanh(&a_cpu)?,
            UnaryOp::Recip => cpu.client.recip(&a_cpu)?,
            UnaryOp::Square => cpu.client.square(&a_cpu)?,
            UnaryOp::Floor => cpu.client.floor(&a_cpu)?,
            UnaryOp::Ceil => cpu.client.ceil(&a_cpu)?,
            UnaryOp::Round => cpu.client.round(&a_cpu)?,
        };

        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(tensor_from_cpu(&result_data, a.shape(), &client.device));
    }, op_name);

    unreachable!()
}

/// Perform a scalar operation using CPU fallback.
fn scalar_op_cpu_fallback(
    client: &CudaClient,
    op: BinaryOp,
    a: &Tensor<CudaRuntime>,
    scalar: f64,
    op_name: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();
    let cpu = CpuFallback::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T>(a);

        let result_cpu = match op {
            BinaryOp::Add => cpu.client.add_scalar(&a_cpu, scalar)?,
            BinaryOp::Sub => cpu.client.sub_scalar(&a_cpu, scalar)?,
            BinaryOp::Mul => cpu.client.mul_scalar(&a_cpu, scalar)?,
            BinaryOp::Div => cpu.client.div_scalar(&a_cpu, scalar)?,
            BinaryOp::Pow => cpu.client.pow_scalar(&a_cpu, scalar)?,
            _ => return Err(Error::UnsupportedDType { dtype, op: op_name }),
        };

        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(tensor_from_cpu(&result_data, a.shape(), &client.device));
    }, op_name);

    unreachable!()
}

/// Perform a reduce operation using CPU fallback.
fn reduce_cpu_fallback(
    client: &CudaClient,
    op: ReduceOp,
    a: &Tensor<CudaRuntime>,
    dims: &[usize],
    keepdim: bool,
    op_name: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();
    let out_shape = reduce_output_shape(a.shape(), dims, keepdim);
    let cpu = CpuFallback::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T>(a);

        let result_cpu = match op {
            ReduceOp::Sum => cpu.client.sum(&a_cpu, dims, keepdim)?,
            ReduceOp::Mean => cpu.client.mean(&a_cpu, dims, keepdim)?,
            ReduceOp::Max => cpu.client.max(&a_cpu, dims, keepdim)?,
            ReduceOp::Min => cpu.client.min(&a_cpu, dims, keepdim)?,
            _ => return Err(Error::UnsupportedDType { dtype, op: op_name }),
        };

        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(tensor_from_cpu(&result_data, &out_shape, &client.device));
    }, op_name);

    unreachable!()
}

/// Perform an activation operation using CPU fallback.
///
/// This is a generic helper for activation functions (relu, sigmoid, etc.)
/// that share the same pattern: copy to CPU, apply function, copy back.
fn activation_cpu_fallback<F>(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    op_name: &'static str,
    op_fn: F,
) -> Result<Tensor<CudaRuntime>>
where
    F: Fn(&CpuFallback, &Tensor<cpu::CpuRuntime>) -> Result<Tensor<cpu::CpuRuntime>>,
{
    let dtype = a.dtype();
    let cpu = CpuFallback::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T>(a);
        let result_cpu = op_fn(&cpu, &a_cpu)?;
        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(tensor_from_cpu(&result_data, a.shape(), &client.device));
    }, op_name);

    unreachable!()
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
        binary_op_cpu_fallback(self, BinaryOp::Add, a, b, "add")
    }

    fn sub(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        binary_op_cpu_fallback(self, BinaryOp::Sub, a, b, "sub")
    }

    fn mul(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        binary_op_cpu_fallback(self, BinaryOp::Mul, a, b, "mul")
    }

    fn div(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        binary_op_cpu_fallback(self, BinaryOp::Div, a, b, "div")
    }

    // ===== Unary Operations (CPU Fallback) =====

    fn neg(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_cpu_fallback(self, UnaryOp::Neg, a, "neg")
    }

    fn abs(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_cpu_fallback(self, UnaryOp::Abs, a, "abs")
    }

    fn sqrt(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_cpu_fallback(self, UnaryOp::Sqrt, a, "sqrt")
    }

    fn exp(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_cpu_fallback(self, UnaryOp::Exp, a, "exp")
    }

    fn log(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_cpu_fallback(self, UnaryOp::Log, a, "log")
    }

    fn sin(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_cpu_fallback(self, UnaryOp::Sin, a, "sin")
    }

    fn cos(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_cpu_fallback(self, UnaryOp::Cos, a, "cos")
    }

    fn tanh(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_cpu_fallback(self, UnaryOp::Tanh, a, "tanh")
    }

    fn tan(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_cpu_fallback(self, UnaryOp::Tan, a, "tan")
    }

    fn recip(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_cpu_fallback(self, UnaryOp::Recip, a, "recip")
    }

    fn square(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_cpu_fallback(self, UnaryOp::Square, a, "square")
    }

    fn floor(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_cpu_fallback(self, UnaryOp::Floor, a, "floor")
    }

    fn ceil(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_cpu_fallback(self, UnaryOp::Ceil, a, "ceil")
    }

    fn round(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        unary_op_cpu_fallback(self, UnaryOp::Round, a, "round")
    }

    // ===== Element-wise Binary (extended, CPU Fallback) =====

    fn pow(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        binary_op_cpu_fallback(self, BinaryOp::Pow, a, b, "pow")
    }

    fn maximum(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        binary_op_cpu_fallback(self, BinaryOp::Max, a, b, "maximum")
    }

    fn minimum(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        binary_op_cpu_fallback(self, BinaryOp::Min, a, b, "minimum")
    }

    // ===== Matrix Operations (cuBLAS) =====

    fn matmul(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate dtypes match
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }

        let dtype = a.dtype();

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
                let cpu = CpuFallback::new();

                dispatch_dtype!(dtype, T => {
                    let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T>(a);
                    let b_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T>(b);

                    let result_cpu = cpu.client.matmul(&a_cpu, &b_cpu)?;
                    let result_data: Vec<T> = result_cpu.to_vec();

                    return Ok(tensor_from_cpu(&result_data, &out_shape, &self.device));
                }, "matmul");

                unreachable!()
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
        reduce_cpu_fallback(self, ReduceOp::Sum, a, dims, keepdim, "sum")
    }

    fn mean(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        reduce_cpu_fallback(self, ReduceOp::Mean, a, dims, keepdim, "mean")
    }

    fn max(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        reduce_cpu_fallback(self, ReduceOp::Max, a, dims, keepdim, "max")
    }

    fn min(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        reduce_cpu_fallback(self, ReduceOp::Min, a, dims, keepdim, "min")
    }

    // ===== Activations (CPU Fallback) =====

    fn relu(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        activation_cpu_fallback(self, a, "relu", |cpu, a_cpu| cpu.client.relu(a_cpu))
    }

    fn sigmoid(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        activation_cpu_fallback(self, a, "sigmoid", |cpu, a_cpu| cpu.client.sigmoid(a_cpu))
    }

    fn softmax(&self, a: &Tensor<CudaRuntime>, dim: isize) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let cpu = CpuFallback::new();

        dispatch_dtype!(dtype, T => {
            let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T>(a);
            let result_cpu = cpu.client.softmax(&a_cpu, dim)?;
            let result_data: Vec<T> = result_cpu.to_vec();
            return Ok(tensor_from_cpu(&result_data, a.shape(), &self.device));
        }, "softmax");

        unreachable!()
    }
}

// ============================================================================
// ScalarOps Implementation
// ============================================================================

impl ScalarOps<CudaRuntime> for CudaClient {
    fn add_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        scalar_op_cpu_fallback(self, BinaryOp::Add, a, scalar, "add_scalar")
    }

    fn sub_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        scalar_op_cpu_fallback(self, BinaryOp::Sub, a, scalar, "sub_scalar")
    }

    fn mul_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        scalar_op_cpu_fallback(self, BinaryOp::Mul, a, scalar, "mul_scalar")
    }

    fn div_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        scalar_op_cpu_fallback(self, BinaryOp::Div, a, scalar, "div_scalar")
    }

    fn pow_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        scalar_op_cpu_fallback(self, BinaryOp::Pow, a, scalar, "pow_scalar")
    }
}

// ============================================================================
// CompareOps Implementation (CPU Fallback)
// ============================================================================

impl CompareOps<CudaRuntime> for CudaClient {
    fn eq(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        compare_op_cpu_fallback(self, "eq", a, b)
    }

    fn ne(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        compare_op_cpu_fallback(self, "ne", a, b)
    }

    fn lt(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        compare_op_cpu_fallback(self, "lt", a, b)
    }

    fn le(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        compare_op_cpu_fallback(self, "le", a, b)
    }

    fn gt(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        compare_op_cpu_fallback(self, "gt", a, b)
    }

    fn ge(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        compare_op_cpu_fallback(self, "ge", a, b)
    }
}

fn compare_op_cpu_fallback(
    client: &CudaClient,
    op_name: &'static str,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;
    let cpu = CpuFallback::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T>(a);
        let b_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T>(b);

        let result_cpu = match op_name {
            "eq" => cpu.client.eq(&a_cpu, &b_cpu)?,
            "ne" => cpu.client.ne(&a_cpu, &b_cpu)?,
            "lt" => cpu.client.lt(&a_cpu, &b_cpu)?,
            "le" => cpu.client.le(&a_cpu, &b_cpu)?,
            "gt" => cpu.client.gt(&a_cpu, &b_cpu)?,
            "ge" => cpu.client.ge(&a_cpu, &b_cpu)?,
            _ => return Err(Error::Internal(format!("Unknown compare op: {}", op_name))),
        };

        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(tensor_from_cpu(&result_data, &out_shape, &client.device));
    }, op_name);

    unreachable!()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
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
