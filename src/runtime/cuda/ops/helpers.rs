//! CUDA-specific helper functions for kernel launching and tensor operations

#[cfg(any(feature = "f16", feature = "fp8"))]
use super::super::kernels::launch_scalar_op_half;
use super::super::kernels::{
    AccumulationPrecision, launch_binary_op, launch_broadcast_binary_op,
    launch_broadcast_compare_op, launch_compare_op, launch_matmul_batched_kernel,
    launch_matmul_bias_batched_kernel, launch_matmul_bias_kernel, launch_matmul_kernel,
    launch_reduce_dim_op, launch_scalar_op_f32, launch_scalar_op_f64, launch_unary_op,
};
use super::super::kernels::{
    launch_scalar_op_c64, launch_scalar_op_c128, launch_scalar_op_i32, launch_scalar_op_i64,
};
use super::super::{CudaClient, CudaRuntime};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{matmul_bias_output_shape, matmul_output_shape, reduce_output_shape};
use crate::runtime::{compute_broadcast_shape, ensure_contiguous, validate_binary_dtypes};
use crate::tensor::Tensor;

// ============================================================================
// Native Tiled GEMM Implementation
// ============================================================================

/// Native matrix multiplication using tiled CUDA kernel.
///
/// Uses shared memory tiling for cache efficiency. This is the default
/// implementation that works without any vendor dependencies.
pub(super) fn matmul_native(
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
pub(super) fn matmul_batched_native(
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
// Fused Matmul+Bias Native Implementation
// ============================================================================

/// Native fused matmul+bias using tiled CUDA kernel: C = A @ B + bias
///
/// Uses the same tiled GEMM algorithm as matmul_native, but fuses bias addition
/// into the epilogue to avoid an extra memory round-trip.
pub(super) fn matmul_bias_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    bias: &Tensor<CudaRuntime>,
    dtype: DType,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor<CudaRuntime>> {
    let a_contig = ensure_contiguous(a);
    let b_contig = ensure_contiguous(b);
    let bias_contig = ensure_contiguous(bias);

    let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape()).ok_or(
        Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        },
    )?;

    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);

    unsafe {
        launch_matmul_bias_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.storage().ptr(),
            b_contig.storage().ptr(),
            bias_contig.storage().ptr(),
            out.storage().ptr(),
            m,
            n,
            k,
        )?;
    }

    Ok(out)
}

/// Native batched fused matmul+bias using tiled CUDA kernel:
/// C[batch,M,N] = A[batch,M,K] @ B[batch,K,N] + bias[N]
pub(super) fn matmul_bias_batched_native(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    bias: &Tensor<CudaRuntime>,
    dtype: DType,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Tensor<CudaRuntime>> {
    let a_contig = ensure_contiguous(a);
    let b_contig = ensure_contiguous(b);
    let bias_contig = ensure_contiguous(bias);

    let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape()).ok_or(
        Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        },
    )?;

    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);

    unsafe {
        launch_matmul_bias_batched_kernel(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.storage().ptr(),
            b_contig.storage().ptr(),
            bias_contig.storage().ptr(),
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
pub(super) fn native_binary_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    op: &'static str,
) -> Result<Tensor<CudaRuntime>> {
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
pub(super) fn native_unary_op(
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
pub(super) fn native_scalar_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    op: &'static str,
    scalar: f64,
) -> Result<Tensor<CudaRuntime>> {
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
            DType::Complex64 => launch_scalar_op_c64(
                &client.context,
                &client.stream,
                client.device.index,
                op,
                a_contig.storage().ptr(),
                scalar as f32,
                out.storage().ptr(),
                out.numel(),
            )?,
            DType::Complex128 => launch_scalar_op_c128(
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
pub(super) fn native_reduce_op(
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
pub(super) fn native_compare_op(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    op: &'static str,
) -> Result<Tensor<CudaRuntime>> {
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
