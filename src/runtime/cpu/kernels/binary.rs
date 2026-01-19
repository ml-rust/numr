//! Binary operations kernels

use crate::dtype::Element;
use crate::ops::BinaryOp;

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
