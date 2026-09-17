//! Generic reduce kernel with configurable accumulation precision.

use super::accumulator::Accumulator;
use super::dispatch::reduce_kernel;
use crate::dtype::Element;
use crate::ops::ReduceOp;

/// Generic reduce kernel with configurable accumulation precision.
///
/// Converts input elements to accumulator type A, performs reduction, then converts back to T.
#[inline]
pub(super) unsafe fn reduce_kernel_acc<T: Element, A: Accumulator>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    // Same out-of-bounds guard as `reduce_kernel`: `Max` and `Min` seed
    // themselves from `a[o * 0]`, which reads past the end of the empty
    // allocation. See the comment there.
    if reduce_size == 0 && matches!(op, ReduceOp::Max | ReduceOp::Min) {
        reduce_kernel(op, a, out, reduce_size, outer_size);
        return;
    }

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
