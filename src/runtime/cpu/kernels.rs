//! CPU kernel implementations
//!
//! This module provides low-level compute kernels for CPU operations.
//! Kernels are generic over `T: Element` and dispatch based on operation type.

#![allow(unsafe_op_in_unsafe_fn)] // Kernels are already marked unsafe, inner unsafe is redundant

use crate::dtype::Element;
use crate::ops::{BinaryOp, ReduceOp, UnaryOp};

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

/// Comparison operation kind
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompareOp {
    /// Equal: a == b
    Eq,
    /// Not equal: a != b
    Ne,
    /// Less than: a < b
    Lt,
    /// Less than or equal: a <= b
    Le,
    /// Greater than: a > b
    Gt,
    /// Greater than or equal: a >= b
    Ge,
}

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
