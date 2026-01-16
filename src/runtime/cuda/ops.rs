//! TensorOps, ScalarOps, and CompareOps implementations for CUDA runtime
//!
//! This module implements tensor operations for CUDA using:
//! - Native CUDA kernels for element-wise, unary, scalar, reduction, and activation ops
//! - cuBLAS for matrix multiplication (matmul)
//!
//! Kernels are compiled from .cu files by build.rs and loaded at runtime.
//!
//! # Performance Characteristics
//!
//! ## Native GPU Operations (Fast Path)
//!
//! Operations on tensors with matching shapes run entirely on GPU:
//! - Binary ops (add, sub, mul, div, pow, max, min)
//! - Unary ops (neg, abs, sqrt, exp, log, sin, cos, tan, tanh, etc.)
//! - Scalar ops (add_scalar, mul_scalar, etc.)
//! - Reductions (sum, max, min) - single dimension
//! - Activations (relu, sigmoid, softmax)
//! - Matrix multiplication (via cuBLAS)
//!
//! ## CPU Fallback (Slow Path)
//!
//! The following operations trigger GPU→CPU→GPU transfers, causing significant overhead:
//!
//! 1. **Broadcasting binary operations**: When tensor shapes don't match (e.g., `[3, 4] + [4]`),
//!    the operation falls back to CPU. This involves:
//!    - Copying both tensors from GPU to CPU
//!    - Computing the result on CPU
//!    - Copying the result back to GPU
//!
//! 2. **Multi-dimension reductions**: Reducing over multiple dimensions at once
//!    (e.g., `sum(&[0, 1])`) falls back to CPU.
//!
//! 3. **Unsupported dtypes for scalar ops**: Non-F32/F64 scalar operations use CPU.
//!
//! ## Recommendations
//!
//! - Pre-broadcast tensors to matching shapes before binary operations
//! - Use single-dimension reductions and chain them if needed
//! - Use F32 or F64 for best GPU performance

use super::kernels::{
    AccumulationPrecision, launch_argmax_dim, launch_argmin_dim, launch_binary_op, launch_cast,
    launch_compare_op, launch_gelu, launch_layer_norm, launch_reduce_dim_op, launch_relu,
    launch_rms_norm, launch_scalar_op_f32, launch_scalar_op_f64, launch_sigmoid, launch_silu,
    launch_softmax, launch_softmax_dim, launch_unary_op,
};
use super::{CudaClient, CudaRuntime};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    CompareOps, ReduceOp, ScalarOps, TensorOps, compute_reduce_strides, matmul_output_shape,
    normalize_softmax_dim, reduce_dim_output_shape, reduce_output_shape,
};
use crate::runtime::fallback::{compute_broadcast_shape, matmul_fallback, validate_binary_dtypes};
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

    let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or(Error::ShapeMismatch {
        expected: a.shape().to_vec(),
        got: b.shape().to_vec(),
    })?;

    let out = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, &client.device);

    let m = m as i32;
    let k = k as i32;
    let n = n as i32;

    let transa = cublasOperation_t::CUBLAS_OP_N;
    let transb = cublasOperation_t::CUBLAS_OP_N;

    let cublas_m = n;
    let cublas_n = m;
    let cublas_k = k;

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

    let a_ptr = a_contig.storage().ptr();
    let b_ptr = b_contig.storage().ptr();
    let out_ptr = out.storage().ptr();
    let numel_a = a_contig.numel();
    let numel_b = b_contig.numel();
    let numel_out = out.numel();

    let b_slice = unsafe { client.stream.upgrade_device_ptr::<f32>(b_ptr, numel_b) };
    let a_slice = unsafe { client.stream.upgrade_device_ptr::<f32>(a_ptr, numel_a) };
    let mut out_slice = unsafe { client.stream.upgrade_device_ptr::<f32>(out_ptr, numel_out) };

    unsafe {
        client
            .cublas
            .gemm(cfg, &b_slice, &a_slice, &mut out_slice)
            .map_err(|e| Error::Internal(format!("cuBLAS GEMM failed: {:?}", e)))?;
    }

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

    let transa = cublasOperation_t::CUBLAS_OP_N;
    let transb = cublasOperation_t::CUBLAS_OP_N;

    let cublas_m = n_i32;
    let cublas_n = m_i32;
    let cublas_k = k_i32;

    let lda = n_i32;
    let ldb = k_i32;
    let ldc = n_i32;

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
            b_ptr as *const f32,
            lda,
            stride_b,
            a_ptr as *const f32,
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

    let transa = cublasOperation_t::CUBLAS_OP_N;
    let transb = cublasOperation_t::CUBLAS_OP_N;

    let cublas_m = n;
    let cublas_n = m;
    let cublas_k = k;

    let lda = n;
    let ldb = k;
    let ldc = n;

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
// Native Kernel Helpers
// ============================================================================

/// Launch a native binary operation on GPU.
///
/// # Performance
///
/// - **Same shape**: Runs entirely on GPU (fast)
/// - **Different shapes**: Falls back to CPU with GPU↔CPU transfers (slow)
///
/// For broadcasting operations, consider pre-expanding tensors to matching shapes
/// using `broadcast_to()` or similar operations to avoid CPU fallback.
fn native_binary_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    op: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    // Native CUDA kernels require same-shape tensors.
    // Broadcasting requires strided kernel access patterns not yet implemented.
    if a.shape() != b.shape() {
        // CPU fallback: involves GPU→CPU transfer, CPU compute, CPU→GPU transfer
        use crate::ops::BinaryOp;
        use crate::runtime::fallback::binary_op_fallback;
        let binary_op = match op {
            "add" => BinaryOp::Add,
            "sub" => BinaryOp::Sub,
            "mul" => BinaryOp::Mul,
            "div" => BinaryOp::Div,
            "pow" => BinaryOp::Pow,
            "max" => BinaryOp::Max,
            "min" => BinaryOp::Min,
            _ => {
                return Err(Error::Internal(format!(
                    "Unsupported binary operation: {}",
                    op
                )));
            }
        };
        return binary_op_fallback(a, b, binary_op, &client.device, op);
    }

    let a_contig = ensure_contiguous(a);
    let b_contig = ensure_contiguous(b);
    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);

    unsafe {
        launch_binary_op(
            &client.context,
            &client.stream,
            client.device.index,
            op,
            dtype,
            a_contig.storage().ptr(),
            b_contig.storage().ptr(),
            out.storage().ptr(),
            out.numel(),
        )?;
    }

    Ok(out)
}

/// Launch a native unary operation.
fn native_unary_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    op: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &client.device);

    unsafe {
        launch_unary_op(
            &client.context,
            &client.stream,
            client.device.index,
            op,
            dtype,
            a_contig.storage().ptr(),
            out.storage().ptr(),
            out.numel(),
        )?;
    }

    Ok(out)
}

/// Launch a native scalar operation.
fn native_scalar_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    op: &'static str,
    scalar: f64,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &client.device);

    unsafe {
        match dtype {
            DType::F32 => launch_scalar_op_f32(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                a_contig.storage().ptr(),
                scalar as f32,
                out.storage().ptr(),
                out.numel(),
            )?,
            DType::F64 => launch_scalar_op_f64(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                a_contig.storage().ptr(),
                scalar,
                out.storage().ptr(),
                out.numel(),
            )?,
            _ => {
                // Fall back to CPU for unsupported dtypes
                use crate::ops::BinaryOp;
                use crate::runtime::fallback::scalar_op_fallback;
                let binary_op = match op {
                    "add_scalar" => BinaryOp::Add,
                    "sub_scalar" => BinaryOp::Sub,
                    "mul_scalar" => BinaryOp::Mul,
                    "div_scalar" => BinaryOp::Div,
                    "pow_scalar" => BinaryOp::Pow,
                    _ => {
                        return Err(Error::Internal(format!(
                            "Unsupported scalar operation: {}",
                            op
                        )));
                    }
                };
                return scalar_op_fallback(a, binary_op, scalar, &client.device, op);
            }
        }
    }

    Ok(out)
}

/// Launch a native reduce operation with optional accumulation precision.
fn native_reduce_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    op: &'static str,
    dims: &[usize],
    keepdim: bool,
    precision: Option<AccumulationPrecision>,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();
    let out_shape = reduce_output_shape(a.shape(), dims, keepdim);
    let acc_precision = precision.unwrap_or_default();

    // For single-dimension reduction, use optimized kernel
    if dims.len() == 1 {
        let dim = dims[0];
        let shape = a.shape();

        // Calculate outer, reduce, inner sizes
        let outer_size: usize = shape[..dim].iter().product();
        let reduce_size = shape[dim];
        let inner_size: usize = shape[dim + 1..].iter().product();

        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);

        unsafe {
            launch_reduce_dim_op(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                outer_size,
                reduce_size,
                inner_size,
                acc_precision,
            )?;
        }

        return Ok(out);
    }

    // For multiple dimensions or unsupported dtypes, fall back to CPU
    use crate::runtime::fallback::reduce_op_fallback;
    let reduce_op = match op {
        "sum" => ReduceOp::Sum,
        "max" => ReduceOp::Max,
        "min" => ReduceOp::Min,
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported reduce operation: {}",
                op
            )));
        }
    };
    reduce_op_fallback(a, reduce_op, dims, keepdim, &client.device, op)
}

/// Launch a native comparison operation on GPU.
///
/// # Performance
///
/// - **Same shape**: Runs entirely on GPU (fast)
/// - **Different shapes**: Falls back to CPU with GPU↔CPU transfers (slow)
fn native_compare_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    op: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    // Native CUDA kernels require same-shape tensors
    if a.shape() != b.shape() {
        use crate::ops::CompareOp;
        use crate::runtime::fallback::compare_op_fallback;
        let compare_op = match op {
            "eq" => CompareOp::Eq,
            "ne" => CompareOp::Ne,
            "lt" => CompareOp::Lt,
            "le" => CompareOp::Le,
            "gt" => CompareOp::Gt,
            "ge" => CompareOp::Ge,
            _ => {
                return Err(Error::Internal(format!(
                    "Unsupported compare operation: {}",
                    op
                )));
            }
        };
        return compare_op_fallback(a, b, compare_op, &client.device, op);
    }

    let a_contig = ensure_contiguous(a);
    let b_contig = ensure_contiguous(b);
    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);

    unsafe {
        launch_compare_op(
            &client.context,
            &client.stream,
            client.device.index,
            op,
            dtype,
            a_contig.storage().ptr(),
            b_contig.storage().ptr(),
            out.storage().ptr(),
            out.numel(),
        )?;
    }

    Ok(out)
}

// ============================================================================
// TensorOps Implementation
// ============================================================================

impl TensorOps<CudaRuntime> for CudaClient {
    // ===== Binary Operations (Native CUDA Kernels) =====

    fn add(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "add")
    }

    fn sub(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "sub")
    }

    fn mul(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "mul")
    }

    fn div(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "div")
    }

    fn pow(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "pow")
    }

    fn maximum(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "max")
    }

    fn minimum(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "min")
    }

    // ===== Unary Operations (Native CUDA Kernels) =====

    fn neg(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "neg")
    }

    fn abs(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "abs")
    }

    fn sqrt(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "sqrt")
    }

    fn exp(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "exp")
    }

    fn log(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "log")
    }

    fn sin(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "sin")
    }

    fn cos(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "cos")
    }

    fn tan(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "tan")
    }

    fn tanh(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "tanh")
    }

    fn recip(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "recip")
    }

    fn square(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "square")
    }

    fn floor(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "floor")
    }

    fn ceil(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "ceil")
    }

    fn round(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "round")
    }

    // ===== Matrix Operations (cuBLAS) =====

    fn matmul(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = validate_binary_dtypes(a, b)?;

        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = if a_shape.len() >= 2 {
            a_shape[a_shape.len() - 2]
        } else {
            1
        };
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];

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
            _ => matmul_fallback(a, b, &out_shape, &self.device, "matmul"),
        }
    }

    // ===== Reductions (Native CUDA Kernels) =====

    fn sum(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "sum", dims, keepdim, None)
    }

    fn sum_with_precision(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "sum", dims, keepdim, Some(precision))
    }

    fn mean(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        // Mean = sum / count
        let sum_result = self.sum(a, dims, keepdim)?;
        let count: usize = dims.iter().map(|&d| a.shape()[d]).product();
        self.div_scalar(&sum_result, count as f64)
    }

    fn max(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "max", dims, keepdim, None)
    }

    fn max_with_precision(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "max", dims, keepdim, Some(precision))
    }

    fn min(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "min", dims, keepdim, None)
    }

    fn min_with_precision(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "min", dims, keepdim, Some(precision))
    }

    // ===== Activations (Native CUDA Kernels) =====

    fn relu(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_relu(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn sigmoid(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_sigmoid(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn silu(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_silu(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn gelu(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        unsafe {
            launch_gelu(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn softmax(&self, a: &Tensor<CudaRuntime>, dim: isize) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let ndim = a.ndim();

        let dim_idx =
            normalize_softmax_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device);

        let shape = a.shape();
        let outer_size: usize = shape[..dim_idx].iter().product();
        let dim_size = shape[dim_idx];
        let inner_size: usize = shape[dim_idx + 1..].iter().product();

        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        unsafe {
            if dim_idx == ndim - 1 {
                // Softmax over last dimension (optimized)
                launch_softmax(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    outer_size,
                    dim_size,
                )?;
            } else {
                // Softmax over non-last dimension
                launch_softmax_dim(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    a_contig.storage().ptr(),
                    out.storage().ptr(),
                    outer_size,
                    dim_size,
                    inner_size,
                )?;
            }
        }

        Ok(out)
    }

    // ===== Normalization =====

    fn rms_norm(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        eps: f32,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = input.dtype();

        // Validate dtypes match
        if weight.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: weight.dtype(),
            });
        }

        // Weight must be 1D with size matching input's last dimension
        let input_shape = input.shape();
        let hidden_size = input_shape[input_shape.len() - 1];
        if weight.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: weight.shape().to_vec(),
            });
        }

        // Compute batch_size as product of all dimensions except last
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();
        let batch_size = batch_size.max(1); // Handle 1D case

        let input_contig = ensure_contiguous(input);
        let weight_contig = ensure_contiguous(weight);
        let out = Tensor::<CudaRuntime>::empty(input_shape, dtype, &self.device);

        unsafe {
            launch_rms_norm(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                input_contig.storage().ptr(),
                weight_contig.storage().ptr(),
                out.storage().ptr(),
                batch_size,
                hidden_size,
                eps,
            )?;
        }

        Ok(out)
    }

    fn layer_norm(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
        eps: f32,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = input.dtype();

        // Validate dtypes match
        if weight.dtype() != dtype || bias.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: if weight.dtype() != dtype {
                    weight.dtype()
                } else {
                    bias.dtype()
                },
            });
        }

        // Weight and bias must be 1D with size matching input's last dimension
        let input_shape = input.shape();
        let hidden_size = input_shape[input_shape.len() - 1];
        if weight.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: weight.shape().to_vec(),
            });
        }
        if bias.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: bias.shape().to_vec(),
            });
        }

        // Compute batch_size as product of all dimensions except last
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();
        let batch_size = batch_size.max(1); // Handle 1D case

        let input_contig = ensure_contiguous(input);
        let weight_contig = ensure_contiguous(weight);
        let bias_contig = ensure_contiguous(bias);
        let out = Tensor::<CudaRuntime>::empty(input_shape, dtype, &self.device);

        unsafe {
            launch_layer_norm(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                input_contig.storage().ptr(),
                weight_contig.storage().ptr(),
                bias_contig.storage().ptr(),
                out.storage().ptr(),
                batch_size,
                hidden_size,
                eps,
            )?;
        }

        Ok(out)
    }

    // ===== Index Operations =====

    fn argmax(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let shape = a.shape();
        let ndim = shape.len();

        // Validate dimension
        if dim >= ndim {
            return Err(Error::InvalidDimension {
                dim: dim as isize,
                ndim,
            });
        }

        let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim);
        let out_shape = reduce_dim_output_shape(shape, dim, keepdim);

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(&out_shape, DType::I64, &self.device);

        unsafe {
            launch_argmax_dim(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                outer_size,
                reduce_size,
                inner_size,
            )?;
        }

        Ok(out)
    }

    fn argmin(
        &self,
        a: &Tensor<CudaRuntime>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let shape = a.shape();
        let ndim = shape.len();

        // Validate dimension
        if dim >= ndim {
            return Err(Error::InvalidDimension {
                dim: dim as isize,
                ndim,
            });
        }

        let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim);
        let out_shape = reduce_dim_output_shape(shape, dim, keepdim);

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(&out_shape, DType::I64, &self.device);

        unsafe {
            launch_argmin_dim(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                outer_size,
                reduce_size,
                inner_size,
            )?;
        }

        Ok(out)
    }

    // ===== Type Casting =====

    fn cast(&self, a: &Tensor<CudaRuntime>, target_dtype: DType) -> Result<Tensor<CudaRuntime>> {
        let src_dtype = a.dtype();

        // No-op if types match
        if src_dtype == target_dtype {
            return Ok(a.clone());
        }

        let shape = a.shape();
        let numel = a.numel();
        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(shape, target_dtype, &self.device);

        unsafe {
            launch_cast(
                &self.context,
                &self.stream,
                self.device.index,
                src_dtype,
                target_dtype,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }
}

// ============================================================================
// ScalarOps Implementation
// ============================================================================

impl ScalarOps<CudaRuntime> for CudaClient {
    fn add_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "add_scalar", scalar)
    }

    fn sub_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "sub_scalar", scalar)
    }

    fn mul_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "mul_scalar", scalar)
    }

    fn div_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "div_scalar", scalar)
    }

    fn pow_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "pow_scalar", scalar)
    }
}

// ============================================================================
// CompareOps Implementation
// ============================================================================

impl CompareOps<CudaRuntime> for CudaClient {
    fn eq(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_compare_op(self, a, b, "eq")
    }

    fn ne(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_compare_op(self, a, b, "ne")
    }

    fn lt(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_compare_op(self, a, b, "lt")
    }

    fn le(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_compare_op(self, a, b, "le")
    }

    fn gt(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_compare_op(self, a, b, "gt")
    }

    fn ge(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_compare_op(self, a, b, "ge")
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

        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

        let b = client.sum(&a, &[1], false).unwrap();

        assert_eq!(b.shape(), &[2]);
        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [6.0, 15.0]);
    }

    #[test]
    fn test_cuda_tensor_silu() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a = Tensor::<CudaRuntime>::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], &device);
        let b = client.silu(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        // SiLU(x) = x / (1 + exp(-x))
        // SiLU(0) = 0
        // SiLU(1) ≈ 0.731
        // SiLU(-1) ≈ -0.269
        assert!((result[2] - 0.0).abs() < 1e-5); // SiLU(0) = 0
        assert!((result[3] - 0.7310586).abs() < 1e-4); // SiLU(1) ≈ 0.731
        assert!((result[1] - (-0.2689414)).abs() < 1e-4); // SiLU(-1) ≈ -0.269
    }

    #[test]
    fn test_cuda_tensor_gelu() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a = Tensor::<CudaRuntime>::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], &device);
        let b = client.gelu(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        // GELU(0) = 0
        // GELU is approximately x for large positive x
        // GELU is approximately 0 for large negative x
        assert!((result[2] - 0.0).abs() < 1e-5); // GELU(0) = 0
        assert!((result[3] - 0.8413).abs() < 0.01); // GELU(1) ≈ 0.841
        assert!((result[4] - 1.9545).abs() < 0.01); // GELU(2) ≈ 1.955
    }

    #[test]
    fn test_cuda_tensor_rms_norm() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // Input: 2 rows, 4 features each
        let input = Tensor::<CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
            &[2, 4],
            &device,
        );
        let weight = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device);

        let out = client.rms_norm(&input, &weight, 1e-5).unwrap();
        let result: Vec<f32> = out.to_vec();

        // Row 1: [1, 2, 3, 4], RMS = sqrt((1+4+9+16)/4) = sqrt(30/4) = sqrt(7.5) ≈ 2.739
        let rms1 = (30.0f32 / 4.0 + 1e-5).sqrt();
        assert!((result[0] - 1.0 / rms1).abs() < 1e-3); // Wider tolerance for GPU
        assert!((result[1] - 2.0 / rms1).abs() < 1e-3);
        assert!((result[2] - 3.0 / rms1).abs() < 1e-3);
        assert!((result[3] - 4.0 / rms1).abs() < 1e-3);

        // Row 2: [2, 4, 6, 8]
        let rms2 = (120.0f32 / 4.0 + 1e-5).sqrt();
        assert!((result[4] - 2.0 / rms2).abs() < 1e-3);
    }

    #[test]
    fn test_cuda_tensor_layer_norm() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // Input: 2 rows, 4 features each
        let input = Tensor::<CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
            &[2, 4],
            &device,
        );
        let weight = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device);
        let bias = Tensor::<CudaRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device);

        let out = client.layer_norm(&input, &weight, &bias, 1e-5).unwrap();
        let result: Vec<f32> = out.to_vec();

        // Row 1: [1, 2, 3, 4], mean = 2.5, var = 1.25, std = 1.118
        let mean1 = 2.5f32;
        let var1 = ((1.0 - mean1).powi(2)
            + (2.0 - mean1).powi(2)
            + (3.0 - mean1).powi(2)
            + (4.0 - mean1).powi(2))
            / 4.0;
        let std1 = (var1 + 1e-5).sqrt();
        assert!((result[0] - (1.0 - mean1) / std1).abs() < 1e-3); // Wider tolerance for GPU
        assert!((result[1] - (2.0 - mean1) / std1).abs() < 1e-3);
        assert!((result[2] - (3.0 - mean1) / std1).abs() < 1e-3);
        assert!((result[3] - (4.0 - mean1) / std1).abs() < 1e-3);

        // Verify normalized outputs sum to approximately 0 (zero-centered)
        let row1_sum: f32 = result[0..4].iter().sum();
        assert!(row1_sum.abs() < 1e-3);
    }

    #[test]
    fn test_cuda_tensor_argmax() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // 2D tensor: [[1, 5, 3], [4, 2, 6]]
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &device);

        // argmax along dim=1 (find max index in each row)
        let out = client.argmax(&a, 1, false).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[2]);
        assert_eq!(result, [1, 2]); // Row 0: max at index 1 (5.0), Row 1: max at index 2 (6.0)

        // argmax along dim=0 (find max index in each column)
        let out = client.argmax(&a, 0, false).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[3]);
        assert_eq!(result, [1, 0, 1]); // Col 0: max at 1 (4.0), Col 1: max at 0 (5.0), Col 2: max at 1 (6.0)

        // Test keepdim=true
        let out = client.argmax(&a, 1, true).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[2, 1]);
        assert_eq!(result, [1, 2]);
    }

    #[test]
    fn test_cuda_tensor_argmin() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // 2D tensor: [[1, 5, 3], [4, 2, 6]]
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &device);

        // argmin along dim=1 (find min index in each row)
        let out = client.argmin(&a, 1, false).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[2]);
        assert_eq!(result, [0, 1]); // Row 0: min at index 0 (1.0), Row 1: min at index 1 (2.0)

        // argmin along dim=0 (find min index in each column)
        let out = client.argmin(&a, 0, false).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[3]);
        assert_eq!(result, [0, 1, 0]); // Col 0: min at 0 (1.0), Col 1: min at 1 (2.0), Col 2: min at 0 (3.0)

        // Test keepdim=true
        let out = client.argmin(&a, 1, true).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[2, 1]);
        assert_eq!(result, [0, 1]);
    }
}
