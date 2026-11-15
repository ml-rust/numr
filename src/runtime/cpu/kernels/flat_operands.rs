//! Flat fast path for strided element-wise kernels.
//!
//! A broadcast against one element, or a view that walks the output in
//! row-major order, needs no multi-dimensional index. Each operand then
//! advances by zero or one element per output position.
//!
//! The loops call the strided kernels' own element functions, so every output
//! element has the same bits as the general strided loop.

use super::binary_elem::binary_elem;
use super::compare::compare_elem;
use crate::dtype::Element;
use crate::ops::{BinaryOp, CompareOp};

/// Per-element step of one operand.
///
/// Returns one when the operand walks the output in row-major order, zero when
/// every output position reads the same element, and `None` otherwise.
fn flat_step(out_shape: &[usize], strides: &[isize]) -> Option<isize> {
    let mut contiguous = true;
    let mut repeated = true;
    let mut expected: isize = 1;
    for d in (0..out_shape.len()).rev() {
        let extent = out_shape[d];
        if extent == 1 {
            continue;
        }
        if strides[d] != expected {
            contiguous = false;
        }
        if strides[d] != 0 {
            repeated = false;
        }
        expected = expected.checked_mul(isize::try_from(extent).ok()?)?;
    }
    if repeated {
        Some(0)
    } else if contiguous {
        Some(1)
    } else {
        None
    }
}

/// Apply `f` over two operands with steps of zero or one.
///
/// # Safety
///
/// `a` and `b` point at the first element each operand reads. A step of one
/// requires `total` readable elements. `out` holds `total` elements and
/// overlaps neither input.
#[inline(always)]
unsafe fn run_pair<T: Element, F: Fn(T, T) -> T>(
    a: *const T,
    b: *const T,
    out: *mut T,
    total: usize,
    steps: (isize, isize),
    f: F,
) {
    match steps {
        (1, 0) => {
            let b = *b;
            for i in 0..total {
                *out.add(i) = f(*a.add(i), b);
            }
        }
        (0, 1) => {
            let a = *a;
            for i in 0..total {
                *out.add(i) = f(a, *b.add(i));
            }
        }
        (1, 1) => {
            for i in 0..total {
                *out.add(i) = f(*a.add(i), *b.add(i));
            }
        }
        _ => {
            let value = f(*a, *b);
            for i in 0..total {
                *out.add(i) = value;
            }
        }
    }
}

/// Steps and output length when both operands are flat.
fn pair_steps(
    out_shape: &[usize],
    a_strides: &[isize],
    b_strides: &[isize],
) -> Option<((isize, isize), usize)> {
    Some((
        (
            flat_step(out_shape, a_strides)?,
            flat_step(out_shape, b_strides)?,
        ),
        out_shape.iter().product(),
    ))
}

/// Run a strided comparison on flat operands.
///
/// Returns `false` without writing when an operand is not flat.
///
/// # Safety
///
/// Same contract as [`super::compare::compare_op_strided_kernel`].
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn try_compare_flat<T: Element>(
    op: CompareOp,
    a: *const T,
    b: *const T,
    out: *mut T,
    out_shape: &[usize],
    a_strides: &[isize],
    b_strides: &[isize],
    a_offset: usize,
    b_offset: usize,
) -> bool {
    let Some((steps, total)) = pair_steps(out_shape, a_strides, b_strides) else {
        return false;
    };
    let (a, b) = (a.add(a_offset), b.add(b_offset));
    // Constant operators let each loop specialize its comparison.
    macro_rules! with_op {
        ($($variant:ident),*) => {
            match op {
                $(CompareOp::$variant => run_pair(a, b, out, total, steps, |x, y| {
                    compare_elem(CompareOp::$variant, x, y)
                }),)*
            }
        };
    }
    with_op!(Eq, Ne, Lt, Le, Gt, Ge);
    true
}

/// Run a strided binary operation on flat operands.
///
/// Returns `false` without writing when an operand is not flat.
///
/// # Safety
///
/// Same contract as [`super::binary::binary_op_strided_kernel`].
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn try_binary_flat<T: Element>(
    op: BinaryOp,
    a: *const T,
    b: *const T,
    out: *mut T,
    out_shape: &[usize],
    a_strides: &[isize],
    b_strides: &[isize],
    a_offset: usize,
    b_offset: usize,
) -> bool {
    let Some((steps, total)) = pair_steps(out_shape, a_strides, b_strides) else {
        return false;
    };
    let (a, b) = (a.add(a_offset), b.add(b_offset));
    macro_rules! with_op {
        ($($variant:ident),*) => {
            match op {
                $(BinaryOp::$variant => run_pair(a, b, out, total, steps, |x, y| {
                    binary_elem(BinaryOp::$variant, x, y)
                }),)*
            }
        };
    }
    with_op!(Add, Sub, Mul, Div, Pow, Max, Min, Atan2);
    true
}

/// Run a strided selection on flat operands.
///
/// Strides and offsets are ordered condition, true value, false value.
/// Returns `false` without writing when an operand is not flat.
///
/// # Safety
///
/// Same contract as the strided where kernels in [`super::where_select`].
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn try_where_flat<C, T: Element, F>(
    cond: *const C,
    x: *const T,
    y: *const T,
    out: *mut T,
    out_shape: &[usize],
    strides: [&[isize]; 3],
    offsets: [usize; 3],
    is_true: &F,
) -> bool
where
    F: Fn(*const C, isize) -> bool,
{
    let (Some(sc), Some(sx), Some(sy)) = (
        flat_step(out_shape, strides[0]),
        flat_step(out_shape, strides[1]),
        flat_step(out_shape, strides[2]),
    ) else {
        return false;
    };
    let total: usize = out_shape.iter().product();
    let (cond, x, y) = (cond.add(offsets[0]), x.add(offsets[1]), y.add(offsets[2]));
    // Constant steps let each loop drop its index arithmetic.
    macro_rules! with_steps {
        ($(($c:literal, $x:literal, $y:literal)),*) => {
            match (sc, sx, sy) {
                $(($c, $x, $y) => where_loop::<C, T, F, $c, $x, $y>(cond, x, y, out, total, is_true),)*
                _ => return false,
            }
        };
    }
    with_steps!(
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1)
    );
    true
}

/// Select with constant operand steps of zero or one.
///
/// # Safety
///
/// Pointers start at each operand's first element. A step of one requires
/// `total` readable elements. `out` holds `total` elements.
#[inline(always)]
unsafe fn where_loop<C, T: Element, F, const SC: isize, const SX: isize, const SY: isize>(
    cond: *const C,
    x: *const T,
    y: *const T,
    out: *mut T,
    total: usize,
    is_true: &F,
) where
    F: Fn(*const C, isize) -> bool,
{
    for i in 0..total {
        let i = i as isize;
        *out.offset(i) = if is_true(cond, i * SC) {
            *x.offset(i * SX)
        } else {
            *y.offset(i * SY)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::super::binary::binary_op_strided_kernel;
    use super::super::compare::compare_op_strided_kernel;
    use super::super::where_select::where_strided_kernel_generic;
    use super::*;

    /// Row-major reference over explicit strides and offsets.
    fn index(flat: usize, shape: &[usize], strides: &[isize], offset: usize) -> usize {
        let mut rem = flat;
        let mut at = offset as isize;
        for d in (0..shape.len()).rev() {
            at += (rem % shape[d]) as isize * strides[d];
            rem /= shape[d];
        }
        at as usize
    }

    #[test]
    fn flat_steps_classify_contiguous_repeated_and_general_layouts() {
        assert_eq!(flat_step(&[2, 3], &[3, 1]), Some(1));
        assert_eq!(flat_step(&[2, 3], &[0, 0]), Some(0));
        assert_eq!(flat_step(&[1, 3], &[7, 1]), Some(1));
        assert_eq!(flat_step(&[2, 1], &[1, 9]), Some(1));
        assert_eq!(flat_step(&[2, 3], &[0, 1]), None);
        assert_eq!(flat_step(&[2, 3], &[1, 2]), None);
        assert_eq!(flat_step(&[], &[]), Some(0));
    }

    #[test]
    fn flat_compare_and_binary_keep_reference_bits_with_special_values() {
        let values = [
            -0.0f64,
            0.0,
            f64::NAN,
            1.5,
            f64::INFINITY,
            -2.0,
            f64::MIN_POSITIVE / 4.0,
        ];
        let shape = [values.len()];
        let scalar = [9.0, 0.0, -0.0, f64::NAN];
        for (layout, (a_strides, a_offset), (b_strides, b_offset)) in [
            ("contiguous-scalar", ([1isize], 0usize), ([0isize], 1usize)),
            ("scalar-contiguous", ([0], 2), ([1], 1)),
            ("contiguous-contiguous-offset", ([1], 0), ([1], 1)),
            ("scalar-scalar", ([0], 3), ([0], 1)),
        ] {
            let a_source: Vec<f64> = if a_strides[0] == 0 {
                scalar.to_vec()
            } else {
                values.to_vec()
            };
            let b_source: Vec<f64> = if b_strides[0] == 0 {
                scalar.to_vec()
            } else {
                // A leading pad keeps the offset view off the contiguous fast path.
                std::iter::once(99.0)
                    .chain(values.iter().rev().copied())
                    .collect()
            };
            for op in [
                CompareOp::Eq,
                CompareOp::Ne,
                CompareOp::Lt,
                CompareOp::Le,
                CompareOp::Gt,
                CompareOp::Ge,
            ] {
                let mut out = vec![7.0; values.len()];
                unsafe {
                    compare_op_strided_kernel(
                        op,
                        a_source.as_ptr(),
                        b_source.as_ptr(),
                        out.as_mut_ptr(),
                        &shape,
                        &a_strides,
                        &b_strides,
                        a_offset,
                        b_offset,
                    );
                }
                for (i, value) in out.iter().enumerate() {
                    let expected = compare_elem(
                        op,
                        a_source[index(i, &shape, &a_strides, a_offset)],
                        b_source[index(i, &shape, &b_strides, b_offset)],
                    );
                    assert_eq!(value.to_bits(), expected.to_bits(), "{layout} {op:?} {i}");
                }
            }
            for op in [
                BinaryOp::Add,
                BinaryOp::Sub,
                BinaryOp::Mul,
                BinaryOp::Div,
                BinaryOp::Pow,
                BinaryOp::Max,
                BinaryOp::Min,
                BinaryOp::Atan2,
            ] {
                let mut out = vec![7.0; values.len()];
                unsafe {
                    binary_op_strided_kernel(
                        op,
                        a_source.as_ptr(),
                        b_source.as_ptr(),
                        out.as_mut_ptr(),
                        &shape,
                        &a_strides,
                        &b_strides,
                        a_offset,
                        b_offset,
                    );
                }
                for (i, value) in out.iter().enumerate() {
                    let expected = binary_elem(
                        op,
                        a_source[index(i, &shape, &a_strides, a_offset)],
                        b_source[index(i, &shape, &b_strides, b_offset)],
                    );
                    assert_eq!(value.to_bits(), expected.to_bits(), "{layout} {op:?} {i}");
                }
            }
        }
    }

    #[test]
    fn flat_integer_binary_keeps_wrapping_reference() {
        let a = [i32::MAX, i32::MIN, -7, 0, 5];
        let b = [3i32];
        let shape = [2usize, 2];
        let strides = [2isize, 1];
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Max] {
            let mut out = [0i32; 4];
            unsafe {
                binary_op_strided_kernel(
                    op,
                    a.as_ptr(),
                    b.as_ptr(),
                    out.as_mut_ptr(),
                    &shape,
                    &strides,
                    &[0, 0],
                    1,
                    0,
                );
            }
            for (i, value) in out.iter().enumerate() {
                assert_eq!(*value, binary_elem(op, a[i + 1], b[0]), "{op:?} {i}");
            }
        }
    }

    #[test]
    fn flat_where_matches_reference_for_scalar_and_offset_operands() {
        let cond = [0.0f64, 1.0, -0.0, f64::NAN, 2.0, 0.0];
        let x = [10.0f64, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let y = [-1.0f64, -0.0];
        let shape = [2usize, 3];
        let cases: [([isize; 2], usize, [isize; 2], usize, [isize; 2], usize); 4] = [
            ([3, 1], 0, [3, 1], 1, [0, 0], 1),
            ([0, 0], 1, [3, 1], 0, [0, 0], 0),
            ([3, 1], 0, [0, 0], 6, [0, 0], 1),
            ([0, 0], 0, [0, 0], 2, [0, 0], 1),
        ];
        for (case, (cs, co, xs, xo, ys, yo)) in cases.into_iter().enumerate() {
            let mut out = [7.0f64; 6];
            unsafe {
                where_strided_kernel_generic(
                    cond.as_ptr(),
                    x.as_ptr(),
                    y.as_ptr(),
                    out.as_mut_ptr(),
                    &shape,
                    &cs,
                    &xs,
                    &ys,
                    co,
                    xo,
                    yo,
                );
            }
            for (i, value) in out.iter().enumerate() {
                let expected = if cond[index(i, &shape, &cs, co)] != 0.0 {
                    x[index(i, &shape, &xs, xo)]
                } else {
                    y[index(i, &shape, &ys, yo)]
                };
                assert_eq!(value.to_bits(), expected.to_bits(), "case {case} {i}");
            }
        }
    }
}
