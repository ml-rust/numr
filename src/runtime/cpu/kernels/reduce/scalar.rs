//! Scalar reduce kernel covering every `Element` type.

use crate::dtype::Element;
use crate::ops::ReduceOp;

/// Scalar reduce kernel for all Element types
#[inline]
pub(super) unsafe fn reduce_kernel_scalar<T: Element>(
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
