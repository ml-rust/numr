//! Element rule shared by the strided and flat binary kernels.

use super::binary_int::binary_int_elem;
use super::ipow::pow_elem;
use crate::dtype::Element;
use crate::ops::BinaryOp;

/// One strided binary element, with the wrapping integer path first.
#[inline(always)]
pub(super) fn binary_elem<T: Element>(op: BinaryOp, a_val: T, b_val: T) -> T {
    // Integer dtypes take the wrapping path; see `binary_int`.
    match binary_int_elem(op, a_val, b_val) {
        Some(v) => v,
        None => match op {
            BinaryOp::Add => a_val + b_val,
            BinaryOp::Sub => a_val - b_val,
            BinaryOp::Mul => a_val * b_val,
            BinaryOp::Div => a_val / b_val,
            BinaryOp::Pow => pow_elem(a_val, b_val),
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
            BinaryOp::Atan2 => T::from_f64(a_val.to_f64().atan2(b_val.to_f64())),
        },
    }
}
