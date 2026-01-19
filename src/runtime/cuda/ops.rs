//! TensorOps, ScalarOps, and CompareOps implementations for CUDA runtime
//!
//! This module implements tensor operations for CUDA using native CUDA kernels:
//! - Element-wise, unary, scalar, reduction, and activation ops
//! - Native tiled matrix multiplication (shared memory optimization)
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
//! - Matrix multiplication (native tiled GEMM with shared memory)
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
    launch_compare_op, launch_fill_with_f64, launch_gelu, launch_isinf_op, launch_isnan_op,
    launch_layer_norm, launch_logical_and_op, launch_logical_not_op, launch_logical_or_op,
    launch_logical_xor_op, launch_matmul_batched_kernel, launch_matmul_kernel, launch_rand,
    launch_randn, launch_reduce_dim_op, launch_relu, launch_rms_norm, launch_scalar_op_f32,
    launch_scalar_op_f64, launch_sigmoid, launch_silu, launch_softmax, launch_softmax_dim,
    launch_unary_op, launch_where_broadcast_op, launch_where_op,
};
use super::{CudaClient, CudaRuntime};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    CompareOps, LogicalOps, ScalarOps, TensorOps, compute_reduce_strides, matmul_output_shape,
    normalize_softmax_dim, reduce_dim_output_shape, reduce_output_shape,
};
use crate::runtime::fallback::{compute_broadcast_shape, matmul_fallback, validate_binary_dtypes};
use crate::tensor::Tensor;

// ============================================================================
// Helper Functions
// ============================================================================

/// Ensure a tensor is contiguous in memory.
///
/// If the tensor is already contiguous (elements laid out consecutively),
/// returns a clone (zero-copy, just increments refcount). Otherwise,
/// creates a new contiguous copy of the data.
///
/// This is required before passing tensors to CUDA kernels
/// that expect contiguous memory layout.
#[inline]
fn ensure_contiguous(tensor: &Tensor<CudaRuntime>) -> Tensor<CudaRuntime> {
    if tensor.is_contiguous() {
        tensor.clone()
    } else {
        tensor.contiguous()
    }
}

// ============================================================================
// Native Tiled GEMM Implementation
// ============================================================================

/// Native matrix multiplication using tiled CUDA kernel.
///
/// Uses shared memory tiling for cache efficiency. This is the default
/// implementation that works without any vendor dependencies.
fn matmul_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    dtype: DType,
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

    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);

    unsafe {
        launch_matmul_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.storage().ptr(),
            b_contig.storage().ptr(),
            out.storage().ptr(),
            m,
            n,
            k,
        )?;
    }

    Ok(out)
}

/// Native batched matrix multiplication using tiled CUDA kernel.
fn matmul_batched_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    dtype: DType,
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

    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);

    unsafe {
        launch_matmul_batched_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.storage().ptr(),
            b_contig.storage().ptr(),
            out.storage().ptr(),
            batch,
            m,
            n,
            k,
        )?;
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
    use super::kernels::launch_broadcast_binary_op;

    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    // For same-shape tensors, use the optimized element-wise kernel
    if a.shape() == b.shape() {
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

        return Ok(out);
    }

    // For different shapes, use the broadcast kernel (stays on GPU)
    let a_contig = ensure_contiguous(a);
    let b_contig = ensure_contiguous(b);
    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);

    unsafe {
        launch_broadcast_binary_op(
            &client.context,
            &client.stream,
            client.device.index,
            &client.device,
            op,
            dtype,
            a_contig.storage().ptr(),
            b_contig.storage().ptr(),
            out.storage().ptr(),
            a.shape(),
            b.shape(),
            &out_shape,
        )?;
    }

    Ok(out)
}

/// Launch a native CUDA unary operation (element-wise, single input).
///
/// Dispatches to CUDA kernels for operations like neg, abs, sqrt, exp, log,
/// sin, cos, sigmoid, relu, etc. The operation runs entirely on GPU.
///
/// # Arguments
/// * `op` - Operation name (must match kernel function suffix, e.g., "neg", "exp")
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

/// Launch a native CUDA tensor-scalar operation.
///
/// Dispatches to CUDA kernels for operations like add_scalar, mul_scalar, etc.
/// For F32/F64, runs on GPU. For other dtypes, falls back to CPU.
///
/// # Arguments
/// * `op` - Operation name (e.g., "add_scalar", "mul_scalar")
/// * `scalar` - Scalar value to apply to each element
fn native_scalar_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    op: &'static str,
    scalar: f64,
) -> Result<Tensor<CudaRuntime>> {
    #[cfg(any(feature = "f16", feature = "fp8"))]
    use super::kernels::launch_scalar_op_half;
    use super::kernels::{launch_scalar_op_i32, launch_scalar_op_i64};

    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &client.device);

    // Check if pow is supported for this dtype (integers don't have pow kernel)
    if op == "pow_scalar" && matches!(dtype, DType::I32 | DType::I64) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "pow_scalar",
        });
    }

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
            DType::I32 => launch_scalar_op_i32(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                a_contig.storage().ptr(),
                scalar as i32,
                out.storage().ptr(),
                out.numel(),
            )?,
            DType::I64 => launch_scalar_op_i64(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                a_contig.storage().ptr(),
                scalar as i64,
                out.storage().ptr(),
                out.numel(),
            )?,
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => launch_scalar_op_half(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                dtype,
                a_contig.storage().ptr(),
                scalar as f32,
                out.storage().ptr(),
                out.numel(),
            )?,
            #[cfg(feature = "fp8")]
            DType::FP8E4M3 | DType::FP8E5M2 => launch_scalar_op_half(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                dtype,
                a_contig.storage().ptr(),
                scalar as f32,
                out.storage().ptr(),
                out.numel(),
            )?,
            _ => {
                // Remaining types (U8, Bool, etc.) return unsupported
                return Err(Error::UnsupportedDType { dtype, op });
            }
        }
    }

    Ok(out)
}

/// Launch a native CUDA reduction operation (sum, max, min along dimensions).
///
/// # Performance
///
/// - **Single dimension**: Uses optimized CUDA kernel with warp-level reductions (fast)
/// - **Multiple dimensions**: Falls back to CPU with GPU↔CPU transfers (slow)
///
/// # Arguments
/// * `op` - Operation name ("sum", "max", "min")
/// * `dims` - Dimensions to reduce over
/// * `keepdim` - Whether to keep reduced dimensions as size 1
/// * `precision` - Optional accumulation precision (higher precision for sum)
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

    // For multiple dimensions: chain single-dimension reductions on GPU
    // This keeps all computation on the GPU instead of falling back to CPU

    // Sort dimensions from highest to lowest to avoid index shifting issues
    let mut sorted_dims: Vec<usize> = dims.to_vec();
    sorted_dims.sort_unstable();
    sorted_dims.reverse();

    // Reduce one dimension at a time
    let mut current = a.clone();
    for (i, &dim) in sorted_dims.iter().enumerate() {
        // For all but the last dimension, always keepdim to preserve indexing
        let keep = if i == sorted_dims.len() - 1 {
            keepdim
        } else {
            true
        };
        current = native_reduce_op(client, &current, op, &[dim], keep, precision)?;
    }

    // If keepdim was false but we kept dims during iteration, squeeze them now
    if !keepdim && sorted_dims.len() > 1 {
        // The output shape is already correct from the final reduction with keepdim=false
        // We just need to return what we have
    }

    Ok(current)
}

/// Launch a native comparison operation on GPU.
///
/// # Performance
///
/// - **Same shape**: Uses optimized element-wise kernel (fast)
/// - **Different shapes**: Uses broadcast kernel with strided access (stays on GPU)
fn native_compare_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    op: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    use super::kernels::launch_broadcast_compare_op;

    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    // For same-shape tensors, use the optimized element-wise kernel
    if a.shape() == b.shape() {
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

        return Ok(out);
    }

    // For different shapes, use the broadcast kernel (stays on GPU)
    let a_contig = ensure_contiguous(a);
    let b_contig = ensure_contiguous(b);
    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);

    unsafe {
        launch_broadcast_compare_op(
            &client.context,
            &client.stream,
            client.device.index,
            &client.device,
            op,
            dtype,
            a_contig.storage().ptr(),
            b_contig.storage().ptr(),
            out.storage().ptr(),
            a.shape(),
            b.shape(),
            &out_shape,
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

    fn sign(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "sign")
    }

    fn isnan(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        // Output is always U8 (boolean)
        let out = Tensor::<CudaRuntime>::empty(a.shape(), DType::U8, &self.device);

        unsafe {
            launch_isnan_op(
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

    fn isinf(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a);
        // Output is always U8 (boolean)
        let out = Tensor::<CudaRuntime>::empty(a.shape(), DType::U8, &self.device);

        unsafe {
            launch_isinf_op(
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

    // ===== Matrix Operations (Native CUDA Kernels) =====

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

        // Native tiled CUDA kernel
        match dtype {
            DType::F32 | DType::F64 => {
                if batch_size > 1 {
                    matmul_batched_native(self, a, b, dtype, batch_size, m, k, n)
                } else {
                    matmul_native(self, a, b, dtype, m, k, n)
                }
            }
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => {
                if batch_size > 1 {
                    matmul_batched_native(self, a, b, dtype, batch_size, m, k, n)
                } else {
                    matmul_native(self, a, b, dtype, m, k, n)
                }
            }
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
        // When dims is empty, reduce over all dimensions
        let count: usize = if dims.is_empty() {
            a.numel()
        } else {
            dims.iter().map(|&d| a.shape()[d]).product()
        };

        // For empty dims, we need to reduce all dimensions
        let actual_dims: Vec<usize> = if dims.is_empty() {
            (0..a.shape().len()).collect()
        } else {
            dims.to_vec()
        };

        let sum_result = self.sum(a, &actual_dims, keepdim)?;
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

    // ===== Conditional Operations =====

    fn where_cond(
        &self,
        cond: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate that x and y have the same dtype
        let dtype = validate_binary_dtypes(x, y)?;

        // Validate condition tensor is U8 (boolean)
        if cond.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: cond.dtype(),
            });
        }

        // For same shapes, use optimized element-wise kernel on GPU
        if cond.shape() == x.shape() && x.shape() == y.shape() {
            let cond_contig = ensure_contiguous(cond);
            let x_contig = ensure_contiguous(x);
            let y_contig = ensure_contiguous(y);
            let out = Tensor::<CudaRuntime>::empty(x.shape(), dtype, &self.device);

            unsafe {
                launch_where_op(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    cond_contig.storage().ptr(),
                    x_contig.storage().ptr(),
                    y_contig.storage().ptr(),
                    out.storage().ptr(),
                    out.numel(),
                )?;
            }

            return Ok(out);
        }

        // For different shapes, use the broadcast kernel (stays on GPU)
        // Compute broadcast shape for all three tensors
        let xy_shape = compute_broadcast_shape(x, y)?;
        let out_shape = crate::ops::broadcast_shape(cond.shape(), &xy_shape).ok_or_else(|| {
            Error::BroadcastError {
                lhs: cond.shape().to_vec(),
                rhs: xy_shape.clone(),
            }
        })?;

        let cond_contig = ensure_contiguous(cond);
        let x_contig = ensure_contiguous(x);
        let y_contig = ensure_contiguous(y);
        let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &self.device);

        unsafe {
            launch_where_broadcast_op(
                &self.context,
                &self.stream,
                self.device.index,
                &self.device,
                dtype,
                cond_contig.storage().ptr(),
                x_contig.storage().ptr(),
                y_contig.storage().ptr(),
                out.storage().ptr(),
                cond.shape(),
                x.shape(),
                y.shape(),
                &out_shape,
            )?;
        }

        Ok(out)
    }

    // ===== Utility Operations =====

    fn clamp(
        &self,
        a: &Tensor<CudaRuntime>,
        min_val: f64,
        max_val: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        // Use native CUDA implementation via composition of maximum and minimum
        // clamp(x, min, max) = min(max(x, min), max)
        // This approach uses existing optimized kernels

        // Create scalar tensors for min and max
        let min_scalar = self.fill(&[], min_val, a.dtype())?;
        let max_scalar = self.fill(&[], max_val, a.dtype())?;

        // First: max(x, min_val)
        let clamped_low = self.maximum(a, &min_scalar)?;

        // Then: min(result, max_val)
        self.minimum(&clamped_low, &max_scalar)
    }

    fn fill(&self, shape: &[usize], value: f64, dtype: DType) -> Result<Tensor<CudaRuntime>> {
        let numel: usize = shape.iter().product();
        if numel == 0 {
            // Empty tensor - just allocate
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        // Allocate output tensor
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);

        // Launch native CUDA fill kernel
        unsafe {
            launch_fill_with_f64(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                value,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    // ===== Statistical Operations =====

    fn var(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        // Variance implementation using existing ops
        // var(x) = mean((x - mean(x))^2) * N / (N - correction)

        let shape = a.shape();

        // When dims is empty, reduce over all dimensions
        let actual_dims: Vec<usize> = if dims.is_empty() {
            (0..shape.len()).collect()
        } else {
            dims.to_vec()
        };

        // Compute count of elements being reduced
        let count: usize = if dims.is_empty() {
            a.numel()
        } else {
            dims.iter().map(|&d| shape[d]).product()
        };

        // Compute mean (mean already handles empty dims internally)
        let mean_val = self.mean(a, dims, true)?;

        // Compute (x - mean)
        let diff = self.sub(a, &mean_val)?;

        // Compute (x - mean)^2
        let diff_squared = self.square(&diff)?;

        // Compute sum of squared differences over all dims when dims is empty
        let sum_sq = self.sum(&diff_squared, &actual_dims, keepdim)?;

        // Divide by (N - correction)
        let divisor = (count.saturating_sub(correction)).max(1) as f64;
        self.div_scalar(&sum_sq, divisor)
    }

    fn std(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        // Standard deviation is sqrt of variance
        let variance = self.var(a, dims, keepdim, correction)?;
        self.sqrt(&variance)
    }

    // ===== Random Operations =====

    fn rand(&self, shape: &[usize], dtype: DType) -> Result<Tensor<CudaRuntime>> {
        // Only F32 and F64 have native CUDA kernels
        if !matches!(dtype, DType::F32 | DType::F64) {
            return Err(Error::UnsupportedDType { dtype, op: "rand" });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            // Empty tensor - just allocate
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        // Allocate output tensor
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);

        // Generate seed using atomic counter + time for better entropy
        let seed = generate_random_seed();

        // Launch native CUDA rand kernel
        unsafe {
            launch_rand(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                seed,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn randn(&self, shape: &[usize], dtype: DType) -> Result<Tensor<CudaRuntime>> {
        // Only F32 and F64 have native CUDA kernels
        if !matches!(dtype, DType::F32 | DType::F64) {
            return Err(Error::UnsupportedDType { dtype, op: "randn" });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            // Empty tensor - just allocate
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device));
        }

        // Allocate output tensor
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device);

        // Generate seed using atomic counter + time for better entropy
        let seed = generate_random_seed();

        // Launch native CUDA randn kernel (uses Box-Muller transform)
        unsafe {
            launch_randn(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                seed,
                out.storage().ptr(),
                numel,
            )?;
        }

        Ok(out)
    }
}

// ============================================================================
// Random Seed Generation
// ============================================================================

use std::sync::atomic::{AtomicU64, Ordering};

/// Global atomic counter for generating unique seeds
static SEED_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a random seed combining atomic counter and system time.
///
/// This provides good entropy for parallel random number generation:
/// - Atomic counter ensures uniqueness across calls
/// - System time adds unpredictability
#[inline]
fn generate_random_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    let time_component = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    // Combine counter and time using splitmix64-style mixing
    let mut z = counter.wrapping_add(time_component);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
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
// LogicalOps Implementation
// ============================================================================

impl LogicalOps<CudaRuntime> for CudaClient {
    fn logical_and(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate both tensors are U8 (boolean)
        if a.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: a.dtype(),
            });
        }
        if b.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: b.dtype(),
            });
        }

        // Validate same shape
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }

        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), DType::U8, &self.device);

        unsafe {
            launch_logical_and_op(
                &self.context,
                &self.stream,
                self.device.index,
                a_contig.storage().ptr(),
                b_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn logical_or(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate both tensors are U8 (boolean)
        if a.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: a.dtype(),
            });
        }
        if b.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: b.dtype(),
            });
        }

        // Validate same shape
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }

        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), DType::U8, &self.device);

        unsafe {
            launch_logical_or_op(
                &self.context,
                &self.stream,
                self.device.index,
                a_contig.storage().ptr(),
                b_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn logical_xor(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate both tensors are U8 (boolean)
        if a.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: a.dtype(),
            });
        }
        if b.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: b.dtype(),
            });
        }

        // Validate same shape
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }

        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), DType::U8, &self.device);

        unsafe {
            launch_logical_xor_op(
                &self.context,
                &self.stream,
                self.device.index,
                a_contig.storage().ptr(),
                b_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn logical_not(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        // Validate tensor is U8 (boolean)
        if a.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: a.dtype(),
            });
        }

        let a_contig = ensure_contiguous(a);
        let out = Tensor::<CudaRuntime>::empty(a.shape(), DType::U8, &self.device);

        unsafe {
            launch_logical_not_op(
                &self.context,
                &self.stream,
                self.device.index,
                a_contig.storage().ptr(),
                out.storage().ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
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
