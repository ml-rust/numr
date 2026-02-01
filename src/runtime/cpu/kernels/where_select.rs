//! Where (conditional select) kernels
//!
//! Provides optimized kernels for conditional selection operations:
//! - `where_kernel<T>` - U8 condition with SIMD optimization (AVX2/AVX512)
//! - `where_kernel_generic<C, T>` - Generic condition type
//! - `where_strided_kernel<T>` - U8 condition with broadcasting
//! - `where_strided_kernel_generic<C, T>` - Generic condition with broadcasting

use crate::dtype::{DType, Element};

/// Where (conditional select): out[i] = cond[i] ? x[i] : y[i]
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64 when
/// condition is U8:
/// - AVX-512: 16 f32s or 8 f64s per iteration
/// - AVX2: 8 f32s or 4 f64s per iteration
/// - Scalar fallback for other types or non-x86 platforms
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
    // Dispatch to SIMD for f32/f64 on x86-64
    #[cfg(target_arch = "x86_64")]
    {
        use super::simd::where_select;

        match T::DTYPE {
            DType::F32 => {
                where_select::where_f32(
                    cond,
                    x as *const f32,
                    y as *const f32,
                    out as *mut f32,
                    len,
                );
                return;
            }
            DType::F64 => {
                where_select::where_f64(
                    cond,
                    x as *const f64,
                    y as *const f64,
                    out as *mut f64,
                    len,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback - delegate to generic kernel with u8 condition
    where_kernel_generic::<u8, T>(cond, x, y, out, len);
}

/// Where (conditional select) with generic condition type
///
/// Accepts any Element type for condition, treating non-zero as true.
/// This allows using comparison results directly without dtype conversion.
///
/// # Safety
/// - `cond` must be valid pointer to `len` elements of type C
/// - `x`, `y`, and `out` must be valid pointers to `len` elements of type T
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn where_kernel_generic<C: Element, T: Element>(
    cond: *const C,
    x: *const T,
    y: *const T,
    out: *mut T,
    len: usize,
) {
    let cond_slice = std::slice::from_raw_parts(cond, len);
    let x_slice = std::slice::from_raw_parts(x, len);
    let y_slice = std::slice::from_raw_parts(y, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    let zero = C::zero();
    for i in 0..len {
        out_slice[i] = if cond_slice[i] != zero {
            x_slice[i]
        } else {
            y_slice[i]
        };
    }
}

/// Check if strides represent a contiguous layout with no offsets.
///
/// Returns true if all tensors are contiguous (row-major) with zero offsets.
#[inline]
fn is_contiguous_layout(
    ndim: usize,
    out_shape: &[usize],
    cond_strides: &[isize],
    x_strides: &[isize],
    y_strides: &[isize],
    cond_offset: usize,
    x_offset: usize,
    y_offset: usize,
) -> bool {
    if ndim == 0 {
        return false;
    }

    let mut expected_stride = 1isize;
    for i in (0..ndim).rev() {
        if cond_strides[i] != expected_stride
            || x_strides[i] != expected_stride
            || y_strides[i] != expected_stride
        {
            return false;
        }
        expected_stride *= out_shape[i] as isize;
    }

    cond_offset == 0 && x_offset == 0 && y_offset == 0
}

/// Shared strided iteration logic for where operations.
///
/// Iterates over output positions using multi-dimensional indexing with
/// incremental offset updates for efficient strided access.
///
/// # Safety
/// - All pointers must be valid for the specified shapes and strides
/// - Caller must ensure the condition check function is valid
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn where_strided_impl<C, T: Element, F>(
    cond: *const C,
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
    is_true: F,
) where
    F: Fn(*const C, isize) -> bool,
{
    let ndim = out_shape.len();
    let total = out_shape.iter().product::<usize>();

    if total == 0 {
        return;
    }

    // General strided iteration with incremental offset updates
    let mut indices = vec![0usize; ndim];
    let mut cond_idx = cond_offset as isize;
    let mut x_idx = x_offset as isize;
    let mut y_idx = y_offset as isize;

    for out_idx in 0..total {
        let result = if is_true(cond, cond_idx) {
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

    // Fast path: use contiguous kernel with SIMD optimization
    if is_contiguous_layout(
        ndim,
        out_shape,
        cond_strides,
        x_strides,
        y_strides,
        cond_offset,
        x_offset,
        y_offset,
    ) {
        where_kernel(cond, x, y, out, total);
        return;
    }

    // Strided path using shared implementation
    where_strided_impl(
        cond,
        x,
        y,
        out,
        out_shape,
        cond_strides,
        x_strides,
        y_strides,
        cond_offset,
        x_offset,
        y_offset,
        |cond_ptr, idx| *cond_ptr.offset(idx) != 0,
    );
}

/// Where (conditional select) with generic condition type and broadcasting support
///
/// Accepts any Element type for condition, treating non-zero as true.
/// This allows using comparison results directly without dtype conversion.
///
/// # Safety
/// - All pointers must be valid for the specified shapes and strides
/// - `out` must not overlap with other inputs
#[inline]
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn where_strided_kernel_generic<C: Element, T: Element>(
    cond: *const C,
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

    // Fast path: use contiguous kernel
    if is_contiguous_layout(
        ndim,
        out_shape,
        cond_strides,
        x_strides,
        y_strides,
        cond_offset,
        x_offset,
        y_offset,
    ) {
        where_kernel_generic(cond, x, y, out, total);
        return;
    }

    // Strided path using shared implementation with generic zero check
    let zero = C::zero();
    where_strided_impl(
        cond,
        x,
        y,
        out,
        out_shape,
        cond_strides,
        x_strides,
        y_strides,
        cond_offset,
        x_offset,
        y_offset,
        |cond_ptr, idx| *cond_ptr.offset(idx) != zero,
    );
}
