//! Comparison operation kernels

use crate::dtype::Element;
use crate::ops::CompareOp;

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
