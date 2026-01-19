//! CPU kernel implementations
//!
//! This module provides low-level compute kernels for CPU operations.
//! Kernels are generic over `T: Element` and dispatch based on operation type.

#![allow(unsafe_op_in_unsafe_fn)] // Kernels are already marked unsafe, inner unsafe is redundant

use crate::dtype::Element;
use crate::ops::{AccumulationPrecision, BinaryOp, CompareOp, ReduceOp, UnaryOp};

// ============================================================================
// Binary Operations
// ============================================================================

/// Execute a binary operation element-wise
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` elements
/// - `out` must not overlap with `a` or `b` unless they are the same pointer
#[inline]
pub unsafe fn binary_op_kernel<T: Element>(
    op: BinaryOp,
    a: *const T,
    b: *const T,
    out: *mut T,
    len: usize,
) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let b_slice = std::slice::from_raw_parts(b, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    match op {
        BinaryOp::Add => {
            for i in 0..len {
                out_slice[i] = a_slice[i] + b_slice[i];
            }
        }
        BinaryOp::Sub => {
            for i in 0..len {
                out_slice[i] = a_slice[i] - b_slice[i];
            }
        }
        BinaryOp::Mul => {
            for i in 0..len {
                out_slice[i] = a_slice[i] * b_slice[i];
            }
        }
        BinaryOp::Div => {
            for i in 0..len {
                out_slice[i] = a_slice[i] / b_slice[i];
            }
        }
        BinaryOp::Pow => {
            // Pow requires conversion to f64 and back
            for i in 0..len {
                let base = a_slice[i].to_f64();
                let exp = b_slice[i].to_f64();
                out_slice[i] = T::from_f64(base.powf(exp));
            }
        }
        BinaryOp::Max => {
            for i in 0..len {
                out_slice[i] = if a_slice[i] > b_slice[i] {
                    a_slice[i]
                } else {
                    b_slice[i]
                };
            }
        }
        BinaryOp::Min => {
            for i in 0..len {
                out_slice[i] = if a_slice[i] < b_slice[i] {
                    a_slice[i]
                } else {
                    b_slice[i]
                };
            }
        }
    }
}

// ============================================================================
// Strided Binary Operations (for broadcasting)
// ============================================================================

/// Execute a binary operation with broadcasting support
///
/// Uses strides to handle arbitrary broadcasting patterns. Stride of 0 means
/// the dimension is broadcast (all indices access the same element).
///
/// # Arguments
/// * `op` - Binary operation to perform
/// * `a` - Pointer to first input tensor data
/// * `b` - Pointer to second input tensor data
/// * `out` - Pointer to output tensor data
/// * `out_shape` - Shape of output tensor
/// * `a_strides` - Strides for tensor a (0 = broadcast dim)
/// * `b_strides` - Strides for tensor b (0 = broadcast dim)
/// * `a_offset` - Starting offset for tensor a
/// * `b_offset` - Starting offset for tensor b
///
/// # Safety
/// - All pointers must be valid for the specified shapes and strides
/// - `out` must not overlap with `a` or `b`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn binary_op_strided_kernel<T: Element>(
    op: BinaryOp,
    a: *const T,
    b: *const T,
    out: *mut T,
    out_shape: &[usize],
    a_strides: &[isize],
    b_strides: &[isize],
    a_offset: usize,
    b_offset: usize,
) {
    let ndim = out_shape.len();
    let total = out_shape.iter().product::<usize>();

    if total == 0 {
        return;
    }

    // Optimize for common case: both inputs are contiguous and same shape
    // (strides are standard row-major and no broadcasting)
    let is_simple = ndim > 0 && {
        let mut expected_stride = 1isize;
        let mut simple = true;
        for i in (0..ndim).rev() {
            if a_strides[i] != expected_stride || b_strides[i] != expected_stride {
                simple = false;
                break;
            }
            expected_stride *= out_shape[i] as isize;
        }
        simple && a_offset == 0 && b_offset == 0
    };

    if is_simple {
        // Fast path: use contiguous kernel
        binary_op_kernel(op, a, b, out, total);
        return;
    }

    // General strided iteration with incremental offset updates
    // (avoids O(ndim) recalculation per element)
    let mut indices = vec![0usize; ndim];
    let mut a_idx = a_offset as isize;
    let mut b_idx = b_offset as isize;

    for out_idx in 0..total {
        let a_val = *a.offset(a_idx);
        let b_val = *b.offset(b_idx);

        let result = match op {
            BinaryOp::Add => a_val + b_val,
            BinaryOp::Sub => a_val - b_val,
            BinaryOp::Mul => a_val * b_val,
            BinaryOp::Div => a_val / b_val,
            BinaryOp::Pow => T::from_f64(a_val.to_f64().powf(b_val.to_f64())),
            BinaryOp::Max => {
                if a_val > b_val {
                    a_val
                } else {
                    b_val
                }
            }
            BinaryOp::Min => {
                if a_val < b_val {
                    a_val
                } else {
                    b_val
                }
            }
        };

        *out.add(out_idx) = result;

        // Increment multi-dimensional index with incremental offset updates
        for dim in (0..ndim).rev() {
            indices[dim] += 1;
            a_idx += a_strides[dim];
            b_idx += b_strides[dim];

            if indices[dim] < out_shape[dim] {
                break;
            }

            // Reset this dimension and adjust offsets
            indices[dim] = 0;
            a_idx -= (out_shape[dim] as isize) * a_strides[dim];
            b_idx -= (out_shape[dim] as isize) * b_strides[dim];
        }
    }
}

// ============================================================================
// Unary Operations
// ============================================================================

/// Execute a unary operation element-wise
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn unary_op_kernel<T: Element>(op: UnaryOp, a: *const T, out: *mut T, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    match op {
        UnaryOp::Neg => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(-v);
            }
        }
        UnaryOp::Abs => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.abs());
            }
        }
        UnaryOp::Sqrt => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.sqrt());
            }
        }
        UnaryOp::Exp => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.exp());
            }
        }
        UnaryOp::Log => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.ln());
            }
        }
        UnaryOp::Sin => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.sin());
            }
        }
        UnaryOp::Cos => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.cos());
            }
        }
        UnaryOp::Tan => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.tan());
            }
        }
        UnaryOp::Tanh => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.tanh());
            }
        }
        UnaryOp::Recip => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(1.0 / v);
            }
        }
        UnaryOp::Square => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v * v);
            }
        }
        UnaryOp::Floor => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.floor());
            }
        }
        UnaryOp::Ceil => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.ceil());
            }
        }
        UnaryOp::Round => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.round());
            }
        }
        UnaryOp::Sign => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                let sign = if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                out_slice[i] = T::from_f64(sign);
            }
        }
    }
}

// ============================================================================
// Activation Operations (special unary ops)
// ============================================================================

/// ReLU activation: max(0, x)
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn relu_kernel<T: Element>(a: *const T, out: *mut T, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);
    let zero = T::zero();

    for i in 0..len {
        out_slice[i] = if a_slice[i] > zero { a_slice[i] } else { zero };
    }
}

/// Sigmoid activation: 1 / (1 + exp(-x))
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn sigmoid_kernel<T: Element>(a: *const T, out: *mut T, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let v = a_slice[i].to_f64();
        let sig = 1.0 / (1.0 + (-v).exp());
        out_slice[i] = T::from_f64(sig);
    }
}

/// SiLU (Swish) activation: x / (1 + exp(-x)) = x * sigmoid(x)
///
/// Used in LLaMA, Mistral, and other modern transformer architectures.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn silu_kernel<T: Element>(a: *const T, out: *mut T, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let x = a_slice[i].to_f64();
        // SiLU(x) = x / (1 + exp(-x)) = x * sigmoid(x)
        let result = x / (1.0 + (-x).exp());
        out_slice[i] = T::from_f64(result);
    }
}

/// GELU (Gaussian Error Linear Unit) activation using tanh approximation
///
/// Computes: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// Used in GPT, BERT, and other transformer architectures.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn gelu_kernel<T: Element>(a: *const T, out: *mut T, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    // GELU constants
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654; // sqrt(2/pi)
    const TANH_COEF: f64 = 0.044715;

    for i in 0..len {
        let x = a_slice[i].to_f64();
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let inner = SQRT_2_OVER_PI * (x + TANH_COEF * x * x * x);
        let result = 0.5 * x * (1.0 + inner.tanh());
        out_slice[i] = T::from_f64(result);
    }
}

/// Leaky ReLU activation: max(negative_slope * x, x)
///
/// # Safety
/// - `a` must be valid pointer to `len` elements
/// - `out` must be valid pointer to `len` elements (may alias `a`)
pub unsafe fn leaky_relu_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    len: usize,
    negative_slope: f64,
) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);
    let zero = T::zero();

    for i in 0..len {
        let x = a_slice[i];
        out_slice[i] = if x > zero {
            x
        } else {
            T::from_f64(x.to_f64() * negative_slope)
        };
    }
}

/// ELU activation: x if x > 0, else alpha * (exp(x) - 1)
///
/// # Safety
/// - `a` must be valid pointer to `len` elements
/// - `out` must be valid pointer to `len` elements (may alias `a`)
pub unsafe fn elu_kernel<T: Element>(a: *const T, out: *mut T, len: usize, alpha: f64) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);
    let zero = T::zero();

    for i in 0..len {
        let x = a_slice[i];
        out_slice[i] = if x > zero {
            x
        } else {
            T::from_f64(alpha * (x.to_f64().exp() - 1.0))
        };
    }
}

// ============================================================================
// Normalization
// ============================================================================

/// RMS Normalization: output = input * rsqrt(mean(input^2) + eps) * weight
///
/// Normalizes over the last dimension. Used in LLaMA and other modern transformers.
///
/// # Arguments
/// * `input` - Input tensor data, shape [batch_size, hidden_size] flattened
/// * `weight` - Weight tensor, shape [hidden_size]
/// * `out` - Output tensor, same shape as input
/// * `batch_size` - Number of rows
/// * `hidden_size` - Size of each row (dimension to normalize over)
/// * `eps` - Small constant for numerical stability
///
/// # Safety
/// - `input` and `out` must be valid pointers to `batch_size * hidden_size` elements
/// - `weight` must be valid pointer to `hidden_size` elements
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn rms_norm_kernel<T: Element>(
    input: *const T,
    weight: *const T,
    out: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let weight_slice = std::slice::from_raw_parts(weight, hidden_size);
    let eps = eps as f64;

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        // Compute sum of squares
        let mut sum_sq = 0.0f64;
        for i in 0..hidden_size {
            let x = (*input.add(row_start + i)).to_f64();
            sum_sq += x * x;
        }

        // Compute RMS: sqrt(mean(x^2) + eps)
        let rms = (sum_sq / hidden_size as f64 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Apply normalization and weight
        for (i, &w) in weight_slice.iter().enumerate() {
            let x = (*input.add(row_start + i)).to_f64();
            let result = x * inv_rms * w.to_f64();
            *out.add(row_start + i) = T::from_f64(result);
        }
    }
}

/// Layer Normalization: output = (input - mean) / sqrt(variance + eps) * weight + bias
///
/// Normalizes over the last dimension.
///
/// # Arguments
/// * `input` - Input tensor data, shape [batch_size, hidden_size] flattened
/// * `weight` - Weight (gamma) tensor, shape [hidden_size]
/// * `bias` - Bias (beta) tensor, shape [hidden_size]
/// * `out` - Output tensor, same shape as input
/// * `batch_size` - Number of rows
/// * `hidden_size` - Size of each row (dimension to normalize over)
/// * `eps` - Small constant for numerical stability
///
/// # Safety
/// - `input` and `out` must be valid pointers to `batch_size * hidden_size` elements
/// - `weight` and `bias` must be valid pointers to `hidden_size` elements
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn layer_norm_kernel<T: Element>(
    input: *const T,
    weight: *const T,
    bias: *const T,
    out: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let weight_slice = std::slice::from_raw_parts(weight, hidden_size);
    let bias_slice = std::slice::from_raw_parts(bias, hidden_size);
    let eps = eps as f64;

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        // Compute mean
        let mut sum = 0.0f64;
        for i in 0..hidden_size {
            sum += (*input.add(row_start + i)).to_f64();
        }
        let mean = sum / hidden_size as f64;

        // Compute variance
        let mut var_sum = 0.0f64;
        for i in 0..hidden_size {
            let x = (*input.add(row_start + i)).to_f64();
            let diff = x - mean;
            var_sum += diff * diff;
        }
        let variance = var_sum / hidden_size as f64;

        // Compute inverse standard deviation
        let inv_std = 1.0 / (variance + eps).sqrt();

        // Apply normalization, weight, and bias
        for i in 0..hidden_size {
            let x = (*input.add(row_start + i)).to_f64();
            let w = weight_slice[i].to_f64();
            let b = bias_slice[i].to_f64();
            let result = (x - mean) * inv_std * w + b;
            *out.add(row_start + i) = T::from_f64(result);
        }
    }
}

// ============================================================================
// Matrix Multiplication
// ============================================================================

/// Matrix multiplication with cache-optimized loop ordering: C = A @ B
///
/// Uses i-k-j loop order for better cache locality with row-major matrices.
/// The innermost loop accesses B and C sequentially, maximizing cache hits.
///
/// # Arguments
/// * `a` - Pointer to matrix A (m × k), row-major with leading dimension lda
/// * `b` - Pointer to matrix B (k × n), row-major with leading dimension ldb
/// * `out` - Pointer to output matrix C (m × n), row-major with leading dimension ldc
/// * `m`, `n`, `k` - Matrix dimensions
/// * `lda`, `ldb`, `ldc` - Leading dimensions (row stride in elements)
///
/// # Safety
/// - All pointers must be valid for the specified dimensions and strides
/// - `out` must not alias with `a` or `b`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_kernel<T: Element>(
    a: *const T,
    b: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    // Zero output first
    for i in 0..m {
        for j in 0..n {
            *out.add(i * ldc + j) = T::zero();
        }
    }

    // ikj order: better cache locality for B
    for i in 0..m {
        for kk in 0..k {
            let a_val = *a.add(i * lda + kk);
            for j in 0..n {
                let b_val = *b.add(kk * ldb + j);
                let out_ptr = out.add(i * ldc + j);
                *out_ptr = *out_ptr + a_val * b_val;
            }
        }
    }
}

// ============================================================================
// Reductions
// ============================================================================

/// Reduce along contiguous dimension
///
/// # Arguments
/// * `op` - Reduction operation
/// * `a` - Input pointer (reduce_size * outer_size elements)
/// * `out` - Output pointer (outer_size elements)
/// * `reduce_size` - Number of elements to reduce over
/// * `outer_size` - Number of independent reductions
///
/// # Safety
/// - `a` must point to `reduce_size * outer_size` elements
/// - `out` must point to `outer_size` elements
#[inline]
pub unsafe fn reduce_kernel<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    match op {
        ReduceOp::Sum => {
            for o in 0..outer_size {
                let mut sum = T::zero();
                for r in 0..reduce_size {
                    sum = sum + *a.add(o * reduce_size + r);
                }
                *out.add(o) = sum;
            }
        }
        ReduceOp::Mean => {
            let scale = 1.0 / reduce_size as f64;
            for o in 0..outer_size {
                let mut sum = T::zero();
                for r in 0..reduce_size {
                    sum = sum + *a.add(o * reduce_size + r);
                }
                *out.add(o) = T::from_f64(sum.to_f64() * scale);
            }
        }
        ReduceOp::Max => {
            for o in 0..outer_size {
                let mut max_val = *a.add(o * reduce_size);
                for r in 1..reduce_size {
                    let val = *a.add(o * reduce_size + r);
                    if val > max_val {
                        max_val = val;
                    }
                }
                *out.add(o) = max_val;
            }
        }
        ReduceOp::Min => {
            for o in 0..outer_size {
                let mut min_val = *a.add(o * reduce_size);
                for r in 1..reduce_size {
                    let val = *a.add(o * reduce_size + r);
                    if val < min_val {
                        min_val = val;
                    }
                }
                *out.add(o) = min_val;
            }
        }
        ReduceOp::Prod => {
            for o in 0..outer_size {
                let mut prod = T::one();
                for r in 0..reduce_size {
                    prod = prod * *a.add(o * reduce_size + r);
                }
                *out.add(o) = prod;
            }
        }
        ReduceOp::All | ReduceOp::Any => {
            // Boolean reductions - convert to/from f64 (0.0 = false, non-zero = true)
            let is_any = matches!(op, ReduceOp::Any);
            for o in 0..outer_size {
                let mut result = !is_any; // All starts true, Any starts false
                for r in 0..reduce_size {
                    let val = (*a.add(o * reduce_size + r)).to_f64() != 0.0;
                    if is_any {
                        result = result || val;
                    } else {
                        result = result && val;
                    }
                }
                *out.add(o) = T::from_f64(if result { 1.0 } else { 0.0 });
            }
        }
    }
}

/// Reduce kernel with explicit accumulation precision
///
/// For reduced-precision types (F16, BF16, FP8), this allows accumulating
/// in a higher precision format for better numerical stability.
///
/// # Arguments
/// * `op` - Reduction operation
/// * `a` - Input pointer (reduce_size * outer_size elements)
/// * `out` - Output pointer (outer_size elements)
/// * `reduce_size` - Number of elements to reduce over
/// * `outer_size` - Number of independent reductions
/// * `precision` - Accumulation precision
///
/// # Safety
/// - `a` must point to `reduce_size * outer_size` elements
/// - `out` must point to `outer_size` elements
#[inline]
pub unsafe fn reduce_kernel_with_precision<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
    precision: AccumulationPrecision,
) {
    match precision {
        AccumulationPrecision::Native => {
            // Use native type accumulation (existing behavior)
            reduce_kernel(op, a, out, reduce_size, outer_size);
        }
        AccumulationPrecision::FP32 | AccumulationPrecision::BF16 => {
            // Accumulate in f32 for better precision
            // BF16 uses f32 on CPU since there's no native bf16 arithmetic
            reduce_kernel_f32_acc(op, a, out, reduce_size, outer_size);
        }
        AccumulationPrecision::FP64 => {
            // Accumulate in f64 for maximum precision (math/science)
            reduce_kernel_f64_acc(op, a, out, reduce_size, outer_size);
        }
    }
}

// ============================================================================
// Accumulator Trait for Generic Precision Reduction
// ============================================================================

/// Trait for accumulation types (f32, f64) used in precision-aware reductions.
///
/// This allows a single generic implementation for both FP32 and FP64 accumulation,
/// avoiding code duplication while maintaining type safety and performance.
///
/// Uses `Into<f64>` for output conversion, `acc_in` for input (f64 -> Self).
pub trait Accumulator: Copy + PartialOrd + PartialEq + Into<f64> {
    const ZERO: Self;
    const ONE: Self;
    /// Convert f64 input to accumulator type
    fn acc_in(v: f64) -> Self;
    fn acc_add(self, other: Self) -> Self;
    fn acc_mul(self, other: Self) -> Self;
    fn acc_div(self, n: usize) -> Self;
}

impl Accumulator for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn acc_in(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn acc_add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn acc_mul(self, other: Self) -> Self {
        self * other
    }
    #[inline]
    fn acc_div(self, n: usize) -> Self {
        self / n as f32
    }
}

impl Accumulator for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn acc_in(v: f64) -> Self {
        v
    }
    #[inline]
    fn acc_add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn acc_mul(self, other: Self) -> Self {
        self * other
    }
    #[inline]
    fn acc_div(self, n: usize) -> Self {
        self / n as f64
    }
}

/// Generic reduce kernel with configurable accumulation precision.
///
/// Converts input elements to accumulator type A, performs reduction, then converts back to T.
#[inline]
unsafe fn reduce_kernel_acc<T: Element, A: Accumulator>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    match op {
        ReduceOp::Sum => {
            for o in 0..outer_size {
                let mut sum = A::ZERO;
                for r in 0..reduce_size {
                    sum = sum.acc_add(A::acc_in((*a.add(o * reduce_size + r)).to_f64()));
                }
                *out.add(o) = T::from_f64(sum.into());
            }
        }
        ReduceOp::Mean => {
            for o in 0..outer_size {
                let mut sum = A::ZERO;
                for r in 0..reduce_size {
                    sum = sum.acc_add(A::acc_in((*a.add(o * reduce_size + r)).to_f64()));
                }
                *out.add(o) = T::from_f64(sum.acc_div(reduce_size).into());
            }
        }
        ReduceOp::Max => {
            for o in 0..outer_size {
                let mut max_val = A::acc_in((*a.add(o * reduce_size)).to_f64());
                for r in 1..reduce_size {
                    let val = A::acc_in((*a.add(o * reduce_size + r)).to_f64());
                    if val > max_val {
                        max_val = val;
                    }
                }
                *out.add(o) = T::from_f64(max_val.into());
            }
        }
        ReduceOp::Min => {
            for o in 0..outer_size {
                let mut min_val = A::acc_in((*a.add(o * reduce_size)).to_f64());
                for r in 1..reduce_size {
                    let val = A::acc_in((*a.add(o * reduce_size + r)).to_f64());
                    if val < min_val {
                        min_val = val;
                    }
                }
                *out.add(o) = T::from_f64(min_val.into());
            }
        }
        ReduceOp::Prod => {
            for o in 0..outer_size {
                let mut prod = A::ONE;
                for r in 0..reduce_size {
                    prod = prod.acc_mul(A::acc_in((*a.add(o * reduce_size + r)).to_f64()));
                }
                *out.add(o) = T::from_f64(prod.into());
            }
        }
        ReduceOp::All | ReduceOp::Any => {
            // Boolean reductions don't benefit from higher precision accumulation
            reduce_kernel(op, a, out, reduce_size, outer_size);
        }
    }
}

/// Reduce kernel with f32 accumulation (convenience wrapper)
#[inline]
unsafe fn reduce_kernel_f32_acc<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    reduce_kernel_acc::<T, f32>(op, a, out, reduce_size, outer_size)
}

/// Reduce kernel with f64 accumulation (convenience wrapper)
#[inline]
unsafe fn reduce_kernel_f64_acc<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    reduce_kernel_acc::<T, f64>(op, a, out, reduce_size, outer_size)
}

// ============================================================================
// Argmax (Index Reduction)
// ============================================================================

/// Argmax along a dimension - returns indices of maximum values
///
/// # Arguments
/// * `a` - Input pointer (outer_size * reduce_size * inner_size elements)
/// * `out` - Output pointer (outer_size * inner_size elements) for i64 indices
/// * `outer_size` - Product of dimensions before the reduction dimension
/// * `reduce_size` - Size of the dimension being reduced
/// * `inner_size` - Product of dimensions after the reduction dimension
///
/// # Safety
/// - `a` must point to `outer_size * reduce_size * inner_size` elements
/// - `out` must point to `outer_size * inner_size` i64 elements
#[inline]
pub unsafe fn argmax_kernel<T: Element>(
    a: *const T,
    out: *mut i64,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Base index for this (outer, inner) position
            let base_idx = outer * reduce_size * inner_size + inner;

            // Find index of maximum value
            let mut max_val = (*a.add(base_idx)).to_f64();
            let mut max_idx: i64 = 0;

            for r in 1..reduce_size {
                let idx = base_idx + r * inner_size;
                let val = (*a.add(idx)).to_f64();
                if val > max_val {
                    max_val = val;
                    max_idx = r as i64;
                }
            }

            *out.add(outer * inner_size + inner) = max_idx;
        }
    }
}

/// Argmin along a dimension - returns indices of minimum values
///
/// # Arguments
/// * `a` - Input pointer (outer_size * reduce_size * inner_size elements)
/// * `out` - Output pointer (outer_size * inner_size elements) for i64 indices
/// * `outer_size` - Product of dimensions before the reduction dimension
/// * `reduce_size` - Size of the dimension being reduced
/// * `inner_size` - Product of dimensions after the reduction dimension
///
/// # Safety
/// - `a` must point to `outer_size * reduce_size * inner_size` elements
/// - `out` must point to `outer_size * inner_size` i64 elements
#[inline]
pub unsafe fn argmin_kernel<T: Element>(
    a: *const T,
    out: *mut i64,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Base index for this (outer, inner) position
            let base_idx = outer * reduce_size * inner_size + inner;

            // Find index of minimum value
            let mut min_val = (*a.add(base_idx)).to_f64();
            let mut min_idx: i64 = 0;

            for r in 1..reduce_size {
                let idx = base_idx + r * inner_size;
                let val = (*a.add(idx)).to_f64();
                if val < min_val {
                    min_val = val;
                    min_idx = r as i64;
                }
            }

            *out.add(outer * inner_size + inner) = min_idx;
        }
    }
}

// ============================================================================
// Scalar Operations
// ============================================================================

/// Binary operation with a scalar (tensor op scalar)
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn scalar_op_kernel<T: Element>(
    op: BinaryOp,
    a: *const T,
    scalar: f64,
    out: *mut T,
    len: usize,
) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);
    let s = T::from_f64(scalar);

    match op {
        BinaryOp::Add => {
            for i in 0..len {
                out_slice[i] = a_slice[i] + s;
            }
        }
        BinaryOp::Sub => {
            for i in 0..len {
                out_slice[i] = a_slice[i] - s;
            }
        }
        BinaryOp::Mul => {
            for i in 0..len {
                out_slice[i] = a_slice[i] * s;
            }
        }
        BinaryOp::Div => {
            for i in 0..len {
                out_slice[i] = a_slice[i] / s;
            }
        }
        BinaryOp::Pow => {
            for i in 0..len {
                let base = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(base.powf(scalar));
            }
        }
        BinaryOp::Max => {
            for i in 0..len {
                out_slice[i] = if a_slice[i] > s { a_slice[i] } else { s };
            }
        }
        BinaryOp::Min => {
            for i in 0..len {
                out_slice[i] = if a_slice[i] < s { a_slice[i] } else { s };
            }
        }
    }
}

// ============================================================================
// Comparison Operations
// ============================================================================

/// Execute a comparison operation element-wise
///
/// Returns 1.0 for true, 0.0 for false (stored in output type)
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn compare_op_kernel<T: Element>(
    op: CompareOp,
    a: *const T,
    b: *const T,
    out: *mut T,
    len: usize,
) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let b_slice = std::slice::from_raw_parts(b, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    let one = T::one();
    let zero = T::zero();

    match op {
        CompareOp::Eq => {
            for i in 0..len {
                out_slice[i] = if a_slice[i] == b_slice[i] { one } else { zero };
            }
        }
        CompareOp::Ne => {
            for i in 0..len {
                out_slice[i] = if a_slice[i] != b_slice[i] { one } else { zero };
            }
        }
        CompareOp::Lt => {
            for i in 0..len {
                out_slice[i] = if a_slice[i] < b_slice[i] { one } else { zero };
            }
        }
        CompareOp::Le => {
            for i in 0..len {
                out_slice[i] = if a_slice[i] <= b_slice[i] { one } else { zero };
            }
        }
        CompareOp::Gt => {
            for i in 0..len {
                out_slice[i] = if a_slice[i] > b_slice[i] { one } else { zero };
            }
        }
        CompareOp::Ge => {
            for i in 0..len {
                out_slice[i] = if a_slice[i] >= b_slice[i] { one } else { zero };
            }
        }
    }
}

/// Execute a comparison operation with broadcasting support
///
/// Uses strides to handle arbitrary broadcasting patterns. Stride of 0 means
/// the dimension is broadcast (all indices access the same element).
///
/// # Safety
/// - All pointers must be valid for the specified shapes and strides
/// - `out` must not overlap with `a` or `b`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn compare_op_strided_kernel<T: Element>(
    op: CompareOp,
    a: *const T,
    b: *const T,
    out: *mut T,
    out_shape: &[usize],
    a_strides: &[isize],
    b_strides: &[isize],
    a_offset: usize,
    b_offset: usize,
) {
    let ndim = out_shape.len();
    let total = out_shape.iter().product::<usize>();

    if total == 0 {
        return;
    }

    // Optimize for common case: both inputs are contiguous and same shape
    let is_simple = ndim > 0 && {
        let mut expected_stride = 1isize;
        let mut simple = true;
        for i in (0..ndim).rev() {
            if a_strides[i] != expected_stride || b_strides[i] != expected_stride {
                simple = false;
                break;
            }
            expected_stride *= out_shape[i] as isize;
        }
        simple && a_offset == 0 && b_offset == 0
    };

    if is_simple {
        compare_op_kernel(op, a, b, out, total);
        return;
    }

    let one = T::one();
    let zero = T::zero();

    // General strided iteration with incremental offset updates
    let mut indices = vec![0usize; ndim];
    let mut a_idx = a_offset as isize;
    let mut b_idx = b_offset as isize;

    for out_idx in 0..total {
        let a_val = *a.offset(a_idx);
        let b_val = *b.offset(b_idx);

        let result = match op {
            CompareOp::Eq => {
                if a_val == b_val {
                    one
                } else {
                    zero
                }
            }
            CompareOp::Ne => {
                if a_val != b_val {
                    one
                } else {
                    zero
                }
            }
            CompareOp::Lt => {
                if a_val < b_val {
                    one
                } else {
                    zero
                }
            }
            CompareOp::Le => {
                if a_val <= b_val {
                    one
                } else {
                    zero
                }
            }
            CompareOp::Gt => {
                if a_val > b_val {
                    one
                } else {
                    zero
                }
            }
            CompareOp::Ge => {
                if a_val >= b_val {
                    one
                } else {
                    zero
                }
            }
        };

        *out.add(out_idx) = result;

        // Increment multi-dimensional index with incremental offset updates
        for dim in (0..ndim).rev() {
            indices[dim] += 1;
            a_idx += a_strides[dim];
            b_idx += b_strides[dim];

            if indices[dim] < out_shape[dim] {
                break;
            }

            // Reset this dimension and adjust offsets
            indices[dim] = 0;
            a_idx -= (out_shape[dim] as isize) * a_strides[dim];
            b_idx -= (out_shape[dim] as isize) * b_strides[dim];
        }
    }
}

// ============================================================================
// Memory Operations
// ============================================================================

/// Fill buffer with a constant value
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
#[inline]
pub unsafe fn fill_kernel<T: Element>(out: *mut T, value: T, len: usize) {
    let out_slice = std::slice::from_raw_parts_mut(out, len);
    out_slice.fill(value);
}

/// Copy elements from src to dst
///
/// # Safety
/// - `src` and `dst` must be valid pointers to `len` elements
/// - `dst` must not overlap with `src`
#[inline]
pub unsafe fn copy_kernel<T: Element>(src: *const T, dst: *mut T, len: usize) {
    std::ptr::copy_nonoverlapping(src, dst, len);
}

// ============================================================================
// Softmax (special reduction + element-wise)
// ============================================================================

/// Softmax along the last dimension
///
/// softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
///
/// # Arguments
/// * `a` - Input pointer (outer_size * dim_size elements)
/// * `out` - Output pointer (outer_size * dim_size elements)
/// * `outer_size` - Number of independent softmax operations
/// * `dim_size` - Size of the softmax dimension
///
/// # Safety
/// - `a` and `out` must point to `outer_size * dim_size` elements
#[inline]
pub unsafe fn softmax_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    outer_size: usize,
    dim_size: usize,
) {
    for o in 0..outer_size {
        let base = o * dim_size;

        // Find max for numerical stability
        let mut max_val = (*a.add(base)).to_f64();
        for d in 1..dim_size {
            let val = (*a.add(base + d)).to_f64();
            if val > max_val {
                max_val = val;
            }
        }

        // Compute exp(x - max) and sum
        let mut sum = 0.0f64;
        for d in 0..dim_size {
            let val = (*a.add(base + d)).to_f64();
            let exp_val = (val - max_val).exp();
            *out.add(base + d) = T::from_f64(exp_val);
            sum += exp_val;
        }

        // Normalize by sum
        let inv_sum = 1.0 / sum;
        for d in 0..dim_size {
            let val = (*out.add(base + d)).to_f64();
            *out.add(base + d) = T::from_f64(val * inv_sum);
        }
    }
}

// ============================================================================
// Type Casting
// ============================================================================

/// Cast tensor data from one dtype to another.
///
/// Converts elements by going through f64 as an intermediate representation,
/// which works for all numeric types via the Element trait.
///
/// # Safety
/// - `src` must be valid pointer to `len` elements of `src_dtype`
/// - `dst` must be valid pointer to `len` elements of `dst_dtype`
/// - `src` and `dst` must not overlap
#[inline]
pub unsafe fn cast_kernel(
    src: *const u8,
    dst: *mut u8,
    len: usize,
    src_dtype: crate::dtype::DType,
    dst_dtype: crate::dtype::DType,
) -> crate::error::Result<()> {
    use crate::dtype::DType;
    use crate::error::Error;

    // Helper macro to cast from a known source type to any destination type
    macro_rules! cast_from {
        ($src_ty:ty, $src_ptr:expr, $dst_ptr:expr, $len:expr, $dst_dtype:expr) => {{
            let src_slice = std::slice::from_raw_parts($src_ptr as *const $src_ty, $len);
            match $dst_dtype {
                DType::F64 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut f64, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64();
                    }
                }
                DType::F32 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut f32, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as f32;
                    }
                }
                DType::F16 => {
                    #[cfg(feature = "f16")]
                    {
                        let dst_slice =
                            std::slice::from_raw_parts_mut($dst_ptr as *mut half::f16, $len);
                        for i in 0..$len {
                            dst_slice[i] = half::f16::from_f64(src_slice[i].to_f64());
                        }
                    }
                    #[cfg(not(feature = "f16"))]
                    {
                        return Err(Error::UnsupportedDType {
                            dtype: DType::F16,
                            op: "cast",
                        });
                    }
                }
                DType::BF16 => {
                    #[cfg(feature = "f16")]
                    {
                        let dst_slice =
                            std::slice::from_raw_parts_mut($dst_ptr as *mut half::bf16, $len);
                        for i in 0..$len {
                            dst_slice[i] = half::bf16::from_f64(src_slice[i].to_f64());
                        }
                    }
                    #[cfg(not(feature = "f16"))]
                    {
                        return Err(Error::UnsupportedDType {
                            dtype: DType::BF16,
                            op: "cast",
                        });
                    }
                }
                DType::FP8E4M3 => {
                    #[cfg(feature = "fp8")]
                    {
                        let dst_slice = std::slice::from_raw_parts_mut(
                            $dst_ptr as *mut crate::dtype::FP8E4M3,
                            $len,
                        );
                        for i in 0..$len {
                            dst_slice[i] =
                                crate::dtype::FP8E4M3::from_f32(src_slice[i].to_f64() as f32);
                        }
                    }
                    #[cfg(not(feature = "fp8"))]
                    {
                        return Err(Error::UnsupportedDType {
                            dtype: DType::FP8E4M3,
                            op: "cast",
                        });
                    }
                }
                DType::FP8E5M2 => {
                    #[cfg(feature = "fp8")]
                    {
                        let dst_slice = std::slice::from_raw_parts_mut(
                            $dst_ptr as *mut crate::dtype::FP8E5M2,
                            $len,
                        );
                        for i in 0..$len {
                            dst_slice[i] =
                                crate::dtype::FP8E5M2::from_f32(src_slice[i].to_f64() as f32);
                        }
                    }
                    #[cfg(not(feature = "fp8"))]
                    {
                        return Err(Error::UnsupportedDType {
                            dtype: DType::FP8E5M2,
                            op: "cast",
                        });
                    }
                }
                DType::I64 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut i64, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as i64;
                    }
                }
                DType::I32 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut i32, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as i32;
                    }
                }
                DType::I16 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut i16, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as i16;
                    }
                }
                DType::I8 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut i8, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as i8;
                    }
                }
                DType::U64 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut u64, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as u64;
                    }
                }
                DType::U32 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut u32, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as u32;
                    }
                }
                DType::U16 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut u16, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as u16;
                    }
                }
                DType::U8 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut u8, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as u8;
                    }
                }
                DType::Bool => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut u8, $len);
                    for i in 0..$len {
                        dst_slice[i] = if src_slice[i].to_f64() != 0.0 { 1 } else { 0 };
                    }
                }
            }
        }};
    }

    // Dispatch based on source dtype
    match src_dtype {
        DType::F64 => cast_from!(f64, src, dst, len, dst_dtype),
        DType::F32 => cast_from!(f32, src, dst, len, dst_dtype),
        DType::F16 => {
            #[cfg(feature = "f16")]
            {
                cast_from!(half::f16, src, dst, len, dst_dtype)
            }
            #[cfg(not(feature = "f16"))]
            {
                return Err(Error::UnsupportedDType {
                    dtype: DType::F16,
                    op: "cast",
                });
            }
        }
        DType::BF16 => {
            #[cfg(feature = "f16")]
            {
                cast_from!(half::bf16, src, dst, len, dst_dtype)
            }
            #[cfg(not(feature = "f16"))]
            {
                return Err(Error::UnsupportedDType {
                    dtype: DType::BF16,
                    op: "cast",
                });
            }
        }
        DType::FP8E4M3 => {
            #[cfg(feature = "fp8")]
            {
                cast_from!(crate::dtype::FP8E4M3, src, dst, len, dst_dtype)
            }
            #[cfg(not(feature = "fp8"))]
            {
                return Err(Error::UnsupportedDType {
                    dtype: DType::FP8E4M3,
                    op: "cast",
                });
            }
        }
        DType::FP8E5M2 => {
            #[cfg(feature = "fp8")]
            {
                cast_from!(crate::dtype::FP8E5M2, src, dst, len, dst_dtype)
            }
            #[cfg(not(feature = "fp8"))]
            {
                return Err(Error::UnsupportedDType {
                    dtype: DType::FP8E5M2,
                    op: "cast",
                });
            }
        }
        DType::I64 => cast_from!(i64, src, dst, len, dst_dtype),
        DType::I32 => cast_from!(i32, src, dst, len, dst_dtype),
        DType::I16 => cast_from!(i16, src, dst, len, dst_dtype),
        DType::I8 => cast_from!(i8, src, dst, len, dst_dtype),
        DType::U64 => cast_from!(u64, src, dst, len, dst_dtype),
        DType::U32 => cast_from!(u32, src, dst, len, dst_dtype),
        DType::U16 => cast_from!(u16, src, dst, len, dst_dtype),
        DType::U8 => cast_from!(u8, src, dst, len, dst_dtype),
        DType::Bool => {
            // Bool is stored as u8 (0 or 1)
            cast_from!(u8, src, dst, len, dst_dtype)
        }
    }

    Ok(())
}

// ============================================================================
// Special Unary Operations (with different output dtype)
// ============================================================================

/// Check for NaN values element-wise
///
/// Returns 1 (u8) if the value is NaN, 0 otherwise.
///
/// # Safety
/// - `a` must be valid pointer to `len` elements
/// - `out` must be valid pointer to `len` u8 elements
#[inline]
pub unsafe fn isnan_kernel<T: Element>(a: *const T, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let v = a_slice[i].to_f64();
        out_slice[i] = if v.is_nan() { 1 } else { 0 };
    }
}

/// Check for Inf values element-wise
///
/// Returns 1 (u8) if the value is infinite (positive or negative), 0 otherwise.
///
/// # Safety
/// - `a` must be valid pointer to `len` elements
/// - `out` must be valid pointer to `len` u8 elements
#[inline]
pub unsafe fn isinf_kernel<T: Element>(a: *const T, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let v = a_slice[i].to_f64();
        out_slice[i] = if v.is_infinite() { 1 } else { 0 };
    }
}

// ============================================================================
// Ternary Operations
// ============================================================================

/// Where (conditional select): out[i] = cond[i] ? x[i] : y[i]
///
/// # Safety
/// - `cond` must be valid pointer to `len` u8 elements
/// - `x`, `y`, and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn where_kernel<T: Element>(
    cond: *const u8,
    x: *const T,
    y: *const T,
    out: *mut T,
    len: usize,
) {
    let cond_slice = std::slice::from_raw_parts(cond, len);
    let x_slice = std::slice::from_raw_parts(x, len);
    let y_slice = std::slice::from_raw_parts(y, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        out_slice[i] = if cond_slice[i] != 0 {
            x_slice[i]
        } else {
            y_slice[i]
        };
    }
}

/// Where (conditional select) with broadcasting support
///
/// Uses strides to handle arbitrary broadcasting patterns. Stride of 0 means
/// the dimension is broadcast (all indices access the same element).
///
/// # Arguments
/// * `cond` - Pointer to condition tensor data (U8)
/// * `x` - Pointer to "true" values tensor data
/// * `y` - Pointer to "false" values tensor data
/// * `out` - Pointer to output tensor data
/// * `out_shape` - Shape of output tensor
/// * `cond_strides` - Strides for cond tensor (0 = broadcast dim)
/// * `x_strides` - Strides for x tensor (0 = broadcast dim)
/// * `y_strides` - Strides for y tensor (0 = broadcast dim)
/// * `cond_offset`, `x_offset`, `y_offset` - Starting offsets for each tensor
///
/// # Safety
/// - All pointers must be valid for the specified shapes and strides
/// - `out` must not overlap with input tensors
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn where_strided_kernel<T: Element>(
    cond: *const u8,
    x: *const T,
    y: *const T,
    out: *mut T,
    out_shape: &[usize],
    cond_strides: &[isize],
    x_strides: &[isize],
    y_strides: &[isize],
    cond_offset: usize,
    x_offset: usize,
    y_offset: usize,
) {
    let ndim = out_shape.len();
    let total = out_shape.iter().product::<usize>();

    if total == 0 {
        return;
    }

    // Optimize for common case: all inputs are contiguous and same shape
    let is_simple = ndim > 0 && {
        let mut expected_stride = 1isize;
        let mut simple = true;
        for i in (0..ndim).rev() {
            if cond_strides[i] != expected_stride
                || x_strides[i] != expected_stride
                || y_strides[i] != expected_stride
            {
                simple = false;
                break;
            }
            expected_stride *= out_shape[i] as isize;
        }
        simple && cond_offset == 0 && x_offset == 0 && y_offset == 0
    };

    if is_simple {
        // Fast path: use contiguous kernel
        where_kernel(cond, x, y, out, total);
        return;
    }

    // General strided iteration with incremental offset updates
    let mut indices = vec![0usize; ndim];
    let mut cond_idx = cond_offset as isize;
    let mut x_idx = x_offset as isize;
    let mut y_idx = y_offset as isize;

    for out_idx in 0..total {
        let cond_val = *cond.offset(cond_idx);
        let result = if cond_val != 0 {
            *x.offset(x_idx)
        } else {
            *y.offset(y_idx)
        };

        *out.add(out_idx) = result;

        // Increment multi-dimensional index with incremental offset updates
        for dim in (0..ndim).rev() {
            indices[dim] += 1;
            cond_idx += cond_strides[dim];
            x_idx += x_strides[dim];
            y_idx += y_strides[dim];

            if indices[dim] < out_shape[dim] {
                break;
            }

            // Reset this dimension and adjust offsets
            indices[dim] = 0;
            cond_idx -= (out_shape[dim] as isize) * cond_strides[dim];
            x_idx -= (out_shape[dim] as isize) * x_strides[dim];
            y_idx -= (out_shape[dim] as isize) * y_strides[dim];
        }
    }
}

// ============================================================================
// Logical Operations (U8 only)
// ============================================================================

/// Logical AND: out[i] = a[i] && b[i]
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` u8 elements
#[inline]
pub unsafe fn logical_and_kernel(a: *const u8, b: *const u8, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let b_slice = std::slice::from_raw_parts(b, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        out_slice[i] = if a_slice[i] != 0 && b_slice[i] != 0 {
            1
        } else {
            0
        };
    }
}

/// Logical OR: out[i] = a[i] || b[i]
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` u8 elements
#[inline]
pub unsafe fn logical_or_kernel(a: *const u8, b: *const u8, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let b_slice = std::slice::from_raw_parts(b, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        out_slice[i] = if a_slice[i] != 0 || b_slice[i] != 0 {
            1
        } else {
            0
        };
    }
}

/// Logical XOR: out[i] = a[i] ^ b[i]
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` u8 elements
#[inline]
pub unsafe fn logical_xor_kernel(a: *const u8, b: *const u8, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let b_slice = std::slice::from_raw_parts(b, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        // XOR: true if exactly one is true
        let a_bool = a_slice[i] != 0;
        let b_bool = b_slice[i] != 0;
        out_slice[i] = if a_bool != b_bool { 1 } else { 0 };
    }
}

/// Logical NOT: out[i] = !a[i]
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` u8 elements
#[inline]
pub unsafe fn logical_not_kernel(a: *const u8, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        out_slice[i] = if a_slice[i] == 0 { 1 } else { 0 };
    }
}

// ============================================================================
// Clamp Operation
// ============================================================================

/// Clamp values to a range: out[i] = min(max(a[i], min_val), max_val)
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn clamp_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    len: usize,
    min_val: f64,
    max_val: f64,
) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let val = a_slice[i].to_f64();
        let clamped = if val < min_val {
            min_val
        } else if val > max_val {
            max_val
        } else {
            val
        };
        out_slice[i] = T::from_f64(clamped);
    }
}

// ============================================================================
// Variance and Standard Deviation
// ============================================================================

/// Compute variance along a dimension
///
/// variance = sum((x - mean)^2) / (N - correction)
///
/// # Arguments
/// * `a` - Input data pointer
/// * `out` - Output pointer (for variance values)
/// * `outer_size` - Product of dimensions before the reduce dimension
/// * `reduce_size` - Size of the dimension being reduced
/// * `inner_size` - Product of dimensions after the reduce dimension
/// * `correction` - Degrees of freedom correction (0 for population, 1 for sample)
///
/// # Safety
/// - `a` must point to `outer_size * reduce_size * inner_size` elements
/// - `out` must point to `outer_size * inner_size` elements
#[inline]
pub unsafe fn variance_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
    correction: usize,
) {
    let total_size = reduce_size;
    let divisor = (total_size.saturating_sub(correction)).max(1) as f64;

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // First pass: compute mean
            let mut sum = 0.0f64;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                sum += (*a.add(idx)).to_f64();
            }
            let mean = sum / (reduce_size as f64);

            // Second pass: compute variance
            let mut var_sum = 0.0f64;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                let diff = (*a.add(idx)).to_f64() - mean;
                var_sum += diff * diff;
            }

            let out_idx = outer * inner_size + inner;
            *out.add(out_idx) = T::from_f64(var_sum / divisor);
        }
    }
}

// ============================================================================
// Random Number Generation
// ============================================================================

/// Fill output with uniform random values in [0, 1)
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
#[inline]
pub unsafe fn rand_uniform_kernel<T: Element>(out: *mut T, len: usize) {
    use rand::Rng;
    let mut rng = rand::rng();
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let val: f64 = rng.random();
        *elem = T::from_f64(val);
    }
}

/// Fill output with standard normal random values (mean=0, std=1)
///
/// Uses the Box-Muller transform for generating normally distributed values.
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
#[inline]
pub unsafe fn rand_normal_kernel<T: Element>(out: *mut T, len: usize) {
    use rand_distr::{Distribution, StandardNormal};

    let mut rng = rand::rng();
    let normal = StandardNormal;
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let val: f64 = normal.sample(&mut rng);
        *elem = T::from_f64(val);
    }
}

// ============================================================================
// Indexing Operations
// ============================================================================

/// Gather elements along a dimension using an index tensor.
///
/// For a 3D tensor with dim=1:
/// `out[i][j][k] = input[i][index[i][j][k]][k]`
///
/// # Arguments
/// * `a` - Input data pointer
/// * `indices` - Index tensor pointer (i64 values)
/// * `out` - Output pointer
/// * `shape` - Shape of input tensor
/// * `index_shape` - Shape of index tensor (same as output shape)
/// * `dim` - Dimension along which to gather
///
/// # Safety
/// - All pointers must be valid for the specified shapes
/// - `indices` must contain valid indices within bounds of `shape[dim]`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gather_kernel<T: Element>(
    a: *const T,
    indices: *const i64,
    out: *mut T,
    shape: &[usize],
    index_shape: &[usize],
    dim: usize,
) {
    let ndim = shape.len();
    if ndim == 0 {
        return;
    }

    // Compute strides for input tensor (row-major)
    let mut a_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        a_strides[i] = a_strides[i + 1] * shape[i + 1];
    }

    // Compute strides for index/output tensor (row-major)
    let mut idx_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        idx_strides[i] = idx_strides[i + 1] * index_shape[i + 1];
    }

    let total = index_shape.iter().product::<usize>();

    // Iterate over all output positions
    for out_idx in 0..total {
        // Convert linear index to multi-dimensional indices
        let mut remaining = out_idx;
        let mut multi_idx = vec![0usize; ndim];
        for d in 0..ndim {
            multi_idx[d] = remaining / idx_strides[d];
            remaining %= idx_strides[d];
        }

        // Get the index value from the indices tensor
        let index_val = *indices.add(out_idx);
        if index_val < 0 || index_val as usize >= shape[dim] {
            // Out of bounds - set to zero (could also panic)
            *out.add(out_idx) = T::zero();
            continue;
        }

        // Compute source position: replace multi_idx[dim] with index_val
        let mut src_offset = 0;
        for d in 0..ndim {
            let coord = if d == dim {
                index_val as usize
            } else {
                multi_idx[d]
            };
            src_offset += coord * a_strides[d];
        }

        *out.add(out_idx) = *a.add(src_offset);
    }
}

/// Scatter values into a tensor at positions specified by an index tensor.
///
/// For a 3D tensor with dim=1:
/// `out[i][index[i][j][k]][k] = src[i][j][k]`
///
/// First copies `a` to `out`, then scatters `src` values.
///
/// # Arguments
/// * `a` - Base tensor to scatter into
/// * `indices` - Index tensor pointer (i64 values)
/// * `src` - Source values to scatter
/// * `out` - Output pointer (must be separate from a)
/// * `shape` - Shape of input/output tensor
/// * `index_shape` - Shape of index/src tensors
/// * `dim` - Dimension along which to scatter
///
/// # Safety
/// - All pointers must be valid for the specified shapes
/// - `out` must not alias with `a`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn scatter_kernel<T: Element>(
    a: *const T,
    indices: *const i64,
    src: *const T,
    out: *mut T,
    shape: &[usize],
    index_shape: &[usize],
    dim: usize,
) {
    let ndim = shape.len();
    if ndim == 0 {
        return;
    }

    let a_numel: usize = shape.iter().product();

    // First, copy a to out
    std::ptr::copy_nonoverlapping(a, out, a_numel);

    // Compute strides for output tensor (row-major)
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        out_strides[i] = out_strides[i + 1] * shape[i + 1];
    }

    // Compute strides for index/src tensor (row-major)
    let mut idx_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        idx_strides[i] = idx_strides[i + 1] * index_shape[i + 1];
    }

    let total = index_shape.iter().product::<usize>();

    // Scatter src values to out at index positions
    for src_idx in 0..total {
        // Convert linear index to multi-dimensional indices
        let mut remaining = src_idx;
        let mut multi_idx = vec![0usize; ndim];
        for d in 0..ndim {
            multi_idx[d] = remaining / idx_strides[d];
            remaining %= idx_strides[d];
        }

        // Get the index value from the indices tensor
        let index_val = *indices.add(src_idx);
        if index_val < 0 || index_val as usize >= shape[dim] {
            // Out of bounds - skip
            continue;
        }

        // Compute destination position: replace multi_idx[dim] with index_val
        let mut dst_offset = 0;
        for d in 0..ndim {
            let coord = if d == dim {
                index_val as usize
            } else {
                multi_idx[d]
            };
            dst_offset += coord * out_strides[d];
        }

        *out.add(dst_offset) = *src.add(src_idx);
    }
}

/// Select elements along a dimension using a 1D index tensor.
///
/// Simpler than gather - the index tensor is 1D and applies uniformly
/// to all positions in the specified dimension.
///
/// # Arguments
/// * `a` - Input data pointer
/// * `indices` - 1D index tensor pointer (i64 values), length = index_len
/// * `out` - Output pointer
/// * `shape` - Shape of input tensor
/// * `dim` - Dimension along which to select
/// * `index_len` - Length of the 1D index tensor
///
/// # Safety
/// - All pointers must be valid for the specified shapes
/// - `indices` must contain valid indices within bounds of `shape[dim]`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn index_select_kernel<T: Element>(
    a: *const T,
    indices: *const i64,
    out: *mut T,
    shape: &[usize],
    dim: usize,
    index_len: usize,
) {
    let ndim = shape.len();
    if ndim == 0 {
        return;
    }

    // Compute sizes: outer * dim_size * inner
    let outer_size: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner_size: usize = shape[dim + 1..].iter().product();

    // For each outer position
    for outer in 0..outer_size.max(1) {
        // For each selected index
        for (sel_idx, &idx_ptr) in std::slice::from_raw_parts(indices, index_len)
            .iter()
            .enumerate()
        {
            let idx = idx_ptr as usize;
            if idx >= dim_size {
                // Out of bounds - fill with zeros
                for inner in 0..inner_size.max(1) {
                    let out_offset =
                        outer * index_len * inner_size.max(1) + sel_idx * inner_size.max(1) + inner;
                    *out.add(out_offset) = T::zero();
                }
                continue;
            }

            // Copy the entire inner slice
            for inner in 0..inner_size.max(1) {
                let src_offset =
                    outer * dim_size * inner_size.max(1) + idx * inner_size.max(1) + inner;
                let out_offset =
                    outer * index_len * inner_size.max(1) + sel_idx * inner_size.max(1) + inner;
                *out.add(out_offset) = *a.add(src_offset);
            }
        }
    }
}

/// Count elements where mask is true.
///
/// Returns the count of non-zero elements in the mask.
///
/// # Safety
/// - `mask` must be valid pointer to `numel` u8 elements
#[inline]
pub unsafe fn masked_count_kernel(mask: *const u8, numel: usize) -> usize {
    let mask_slice = std::slice::from_raw_parts(mask, numel);
    mask_slice.iter().filter(|&&m| m != 0).count()
}

/// Select elements where mask is true, returning a flattened result.
///
/// # Arguments
/// * `a` - Input data pointer
/// * `mask` - Mask tensor pointer (u8: 0=false, non-zero=true)
/// * `out` - Output pointer (must be sized for count of true elements)
/// * `numel` - Number of elements in input/mask
///
/// # Safety
/// - All pointers must be valid for the specified size
/// - `out` must have enough space for all selected elements
#[inline]
pub unsafe fn masked_select_kernel<T: Element>(
    a: *const T,
    mask: *const u8,
    out: *mut T,
    numel: usize,
) {
    let a_slice = std::slice::from_raw_parts(a, numel);
    let mask_slice = std::slice::from_raw_parts(mask, numel);

    let mut out_idx = 0;
    for i in 0..numel {
        if mask_slice[i] != 0 {
            *out.add(out_idx) = a_slice[i];
            out_idx += 1;
        }
    }
}

/// Fill elements where mask is true with a scalar value.
///
/// # Arguments
/// * `a` - Input data pointer
/// * `mask` - Mask tensor pointer (u8: 0=false, non-zero=true)
/// * `out` - Output pointer
/// * `numel` - Number of elements
/// * `value` - Value to fill where mask is true
///
/// # Safety
/// - All pointers must be valid for the specified size
#[inline]
pub unsafe fn masked_fill_kernel<T: Element>(
    a: *const T,
    mask: *const u8,
    out: *mut T,
    numel: usize,
    value: f64,
) {
    let a_slice = std::slice::from_raw_parts(a, numel);
    let mask_slice = std::slice::from_raw_parts(mask, numel);
    let out_slice = std::slice::from_raw_parts_mut(out, numel);

    let fill_val = T::from_f64(value);

    for i in 0..numel {
        out_slice[i] = if mask_slice[i] != 0 {
            fill_val
        } else {
            a_slice[i]
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_add() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];

        unsafe {
            binary_op_kernel(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4);
        }

        assert_eq!(out, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_binary_mul() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [2.0f32, 3.0, 4.0, 5.0];
        let mut out = [0.0f32; 4];

        unsafe {
            binary_op_kernel(BinaryOp::Mul, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4);
        }

        assert_eq!(out, [2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_unary_neg() {
        let a = [1.0f32, -2.0, 3.0, -4.0];
        let mut out = [0.0f32; 4];

        unsafe {
            unary_op_kernel(UnaryOp::Neg, a.as_ptr(), out.as_mut_ptr(), 4);
        }

        assert_eq!(out, [-1.0, 2.0, -3.0, 4.0]);
    }

    #[test]
    fn test_unary_sqrt() {
        let a = [1.0f32, 4.0, 9.0, 16.0];
        let mut out = [0.0f32; 4];

        unsafe {
            unary_op_kernel(UnaryOp::Sqrt, a.as_ptr(), out.as_mut_ptr(), 4);
        }

        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_relu() {
        let a = [-1.0f32, 0.0, 1.0, -2.0];
        let mut out = [0.0f32; 4];

        unsafe {
            relu_kernel(a.as_ptr(), out.as_mut_ptr(), 4);
        }

        assert_eq!(out, [0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_sigmoid() {
        let a = [0.0f32];
        let mut out = [0.0f32; 1];

        unsafe {
            sigmoid_kernel(a.as_ptr(), out.as_mut_ptr(), 1);
        }

        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_2x2() {
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C = A @ B = [[19, 22], [43, 50]]
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];

        unsafe {
            matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 2, 2, 2, 2, 2);
        }

        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_3x2_2x4() {
        // A = [[1, 2], [3, 4], [5, 6]] (3x2)
        // B = [[1, 2, 3, 4], [5, 6, 7, 8]] (2x4)
        // C = A @ B (3x4)
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 12];

        unsafe {
            matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 3, 4, 2, 2, 4, 4);
        }

        // Row 0: [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11, 14, 17, 20]
        // Row 1: [3*1+4*5, 3*2+4*6, 3*3+4*7, 3*4+4*8] = [23, 30, 37, 44]
        // Row 2: [5*1+6*5, 5*2+6*6, 5*3+6*7, 5*4+6*8] = [35, 46, 57, 68]
        assert_eq!(
            c,
            [
                11.0, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0
            ]
        );
    }

    #[test]
    fn test_reduce_sum() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 2];

        unsafe {
            // Reduce 3 elements per output, 2 outputs
            reduce_kernel(ReduceOp::Sum, a.as_ptr(), out.as_mut_ptr(), 3, 2);
        }

        assert_eq!(out, [6.0, 15.0]); // [1+2+3, 4+5+6]
    }

    #[test]
    fn test_reduce_mean() {
        let a = [1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        let mut out = [0.0f32; 2];

        unsafe {
            reduce_kernel(ReduceOp::Mean, a.as_ptr(), out.as_mut_ptr(), 3, 2);
        }

        assert_eq!(out, [2.0, 20.0]); // [6/3, 60/3]
    }

    #[test]
    fn test_reduce_max() {
        let a = [1.0f32, 5.0, 3.0, 2.0, 8.0, 4.0];
        let mut out = [0.0f32; 2];

        unsafe {
            reduce_kernel(ReduceOp::Max, a.as_ptr(), out.as_mut_ptr(), 3, 2);
        }

        assert_eq!(out, [5.0, 8.0]);
    }

    #[test]
    fn test_softmax() {
        let a = [1.0f32, 2.0, 3.0];
        let mut out = [0.0f32; 3];

        unsafe {
            softmax_kernel(a.as_ptr(), out.as_mut_ptr(), 1, 3);
        }

        // Check that outputs sum to 1
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check monotonicity: out[0] < out[1] < out[2]
        assert!(out[0] < out[1]);
        assert!(out[1] < out[2]);
    }

    #[test]
    fn test_fill() {
        let mut out = [0.0f32; 4];

        unsafe {
            fill_kernel(out.as_mut_ptr(), 7.5f32, 4);
        }

        assert_eq!(out, [7.5, 7.5, 7.5, 7.5]);
    }

    #[test]
    fn test_copy() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut dst = [0.0f32; 4];

        unsafe {
            copy_kernel(src.as_ptr(), dst.as_mut_ptr(), 4);
        }

        assert_eq!(dst, src);
    }

    #[test]
    fn test_i32_binary_add() {
        let a = [1i32, 2, 3, 4];
        let b = [5i32, 6, 7, 8];
        let mut out = [0i32; 4];

        unsafe {
            binary_op_kernel(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4);
        }

        assert_eq!(out, [6, 8, 10, 12]);
    }
}
