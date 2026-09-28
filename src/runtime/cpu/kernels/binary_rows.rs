//! Row-wise fast path for broadcast binary operations.
//!
//! A bias add `[N, D] + [D]` or a positional add `[B, T, D] + [T, D]` has a
//! trailing block that is contiguous in both operands, repeated across outer
//! dimensions (the broadcast operand has stride 0 there). This path walks the
//! outer dimensions once per row and hands each row to the contiguous SIMD
//! kernel [`binary_op_kernel`].
//!
//! The path only accepts F32 and F64 with Add, Sub, Mul or Div. Every
//! backend of `binary_op_kernel` evaluates those ops with the exactly rounded
//! IEEE instruction, the same as the scalar strided loop, so each output
//! element has the same bits either way. Max/Min (NEON propagates NaN, the
//! scalar loop does not), Pow and Atan2 (different formulas) stay on the
//! scalar strided loop.

use super::binary::binary_op_kernel;
use crate::dtype::{DType, Element};
use crate::ops::BinaryOp;

/// Shortest row worth a kernel call. Below this the per-row call costs more
/// than the scalar strided loop it replaces.
const MIN_ROW_LEN: usize = 8;

/// Number of trailing dimensions contiguous in both operands, and their extent product.
fn shared_contiguous_suffix(
    out_shape: &[usize],
    a_strides: &[isize],
    b_strides: &[isize],
) -> (usize, usize) {
    let mut expected: isize = 1;
    let mut dims = 0;
    for d in (0..out_shape.len()).rev() {
        if out_shape[d] != 1 && (a_strides[d] != expected || b_strides[d] != expected) {
            break;
        }
        dims += 1;
        expected *= out_shape[d] as isize;
    }
    (dims, expected as usize)
}

/// Run the operation row by row when a shared contiguous trailing block exists.
///
/// Returns `false` without writing anything when the path does not apply; the
/// caller then runs its general strided loop.
///
/// # Safety
///
/// Same contract as [`super::binary::binary_op_strided_kernel`]: every offset
/// reachable from the shape, strides and offsets is in bounds, and `out` holds
/// `out_shape.iter().product()` elements that overlap neither input.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn try_binary_rows<T: Element>(
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
    if !matches!(T::DTYPE, DType::F32 | DType::F64)
        || !matches!(
            op,
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
        )
    {
        return false;
    }

    let (suffix, row_len) = shared_contiguous_suffix(out_shape, a_strides, b_strides);
    if suffix == 0 || row_len < MIN_ROW_LEN {
        return false;
    }

    let outer_dims = out_shape.len() - suffix;
    let rows: usize = out_shape[..outer_dims].iter().product();
    let mut idx = vec![0usize; outer_dims];
    let mut a_idx = a_offset as isize;
    let mut b_idx = b_offset as isize;

    for row in 0..rows {
        binary_op_kernel(
            op,
            a.offset(a_idx),
            b.offset(b_idx),
            out.add(row * row_len),
            row_len,
        );

        // Row-major odometer over the outer dimensions.
        for d in (0..outer_dims).rev() {
            idx[d] += 1;
            a_idx += a_strides[d];
            b_idx += b_strides[d];
            if idx[d] < out_shape[d] {
                break;
            }
            idx[d] = 0;
            a_idx -= out_shape[d] as isize * a_strides[d];
            b_idx -= out_shape[d] as isize * b_strides[d];
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::super::binary::binary_op_strided_kernel;
    use super::*;

    /// The general strided loop, element by element, as the bit-exact oracle.
    fn reference<T: Element>(
        op: BinaryOp,
        a: &[T],
        b: &[T],
        out_shape: &[usize],
        a_strides: &[isize],
        b_strides: &[isize],
    ) -> Vec<T> {
        let total: usize = out_shape.iter().product();
        let ndim = out_shape.len();
        let mut out = Vec::with_capacity(total);
        for flat in 0..total {
            let mut rem = flat;
            let (mut ai, mut bi) = (0isize, 0isize);
            for d in (0..ndim).rev() {
                let i = (rem % out_shape[d]) as isize;
                rem /= out_shape[d];
                ai += i * a_strides[d];
                bi += i * b_strides[d];
            }
            let (x, y) = (a[ai as usize], b[bi as usize]);
            out.push(match op {
                BinaryOp::Add => x + y,
                BinaryOp::Sub => x - y,
                BinaryOp::Mul => x * y,
                BinaryOp::Div => x / y,
                _ => unreachable!("reference covers the fast-path ops only"),
            });
        }
        out
    }

    fn values(n: usize, seed: f64) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f64 * 0.731 + seed).sin() * 3.7 + 0.01) as f32)
            .collect()
    }

    fn check_f32(
        out_shape: &[usize],
        a_strides: &[isize],
        b_strides: &[isize],
        a_len: usize,
        b_len: usize,
    ) {
        let a = values(a_len, 0.3);
        let b = values(b_len, 1.9);
        let total: usize = out_shape.iter().product();
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
            let want = reference(op, &a, &b, out_shape, a_strides, b_strides);
            let mut got = vec![0.0f32; total];
            unsafe {
                binary_op_strided_kernel(
                    op,
                    a.as_ptr(),
                    b.as_ptr(),
                    got.as_mut_ptr(),
                    out_shape,
                    a_strides,
                    b_strides,
                    0,
                    0,
                );
            }
            for i in 0..total {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "{op:?} {out_shape:?} i={i}"
                );
            }
        }
    }

    #[test]
    fn test_bias_add_rows_bit_identical() {
        // [N, D] + [D], row lengths below and above the SIMD threshold.
        for &d in &[8usize, 31, 32, 100] {
            check_f32(&[5, d], &[d as isize, 1], &[0, 1], 5 * d, d);
            // Broadcast operand on the left.
            check_f32(&[5, d], &[0, 1], &[d as isize, 1], d, 5 * d);
        }
    }

    #[test]
    fn test_positional_add_rows_bit_identical() {
        // [B, T, D] + [T, D]: the shared block is T * D.
        check_f32(&[3, 4, 16], &[64, 16, 1], &[0, 16, 1], 192, 64);
        // [B, T, D] + [B, 1, D]: the shared block is D only.
        check_f32(&[3, 4, 16], &[64, 16, 1], &[16, 0, 1], 192, 48);
    }

    #[test]
    fn test_rows_path_declines_and_general_path_matches() {
        // Innermost broadcast: no shared contiguous suffix.
        check_f32(&[5, 16], &[16, 1], &[1, 0], 80, 5);
        // Row shorter than MIN_ROW_LEN.
        check_f32(&[5, 4], &[4, 1], &[0, 1], 20, 4);
        assert_eq!(
            shared_contiguous_suffix(&[5, 16], &[16, 1], &[1, 0]),
            (0, 1)
        );
        assert_eq!(
            shared_contiguous_suffix(&[3, 4, 16], &[64, 16, 1], &[0, 16, 1]),
            (2, 64)
        );
    }

    #[test]
    fn test_rows_path_f64_bit_identical() {
        let a: Vec<f64> = (0..6 * 40).map(|i| (i as f64 * 0.37).cos() * 2.5).collect();
        let b: Vec<f64> = (0..40).map(|i| (i as f64 * 1.3).sin() + 0.02).collect();
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
            let want = reference(op, &a, &b, &[6, 40], &[40, 1], &[0, 1]);
            let mut got = vec![0.0f64; 240];
            unsafe {
                binary_op_strided_kernel(
                    op,
                    a.as_ptr(),
                    b.as_ptr(),
                    got.as_mut_ptr(),
                    &[6, 40],
                    &[40, 1],
                    &[0, 1],
                    0,
                    0,
                );
            }
            for i in 0..240 {
                assert_eq!(got[i].to_bits(), want[i].to_bits(), "{op:?} i={i}");
            }
        }
    }
}
