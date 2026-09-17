//! Scalar fallbacks for reduction operations.

use crate::ops::ReduceOp;

/// Scalar reduction for f32
#[inline]
pub(super) unsafe fn reduce_scalar_f32(
    op: ReduceOp,
    a: *const f32,
    out: *mut f32,
    reduce_size: usize,
    outer_size: usize,
) {
    match op {
        ReduceOp::Sum => {
            for o in 0..outer_size {
                let mut sum = 0.0f32;
                for r in 0..reduce_size {
                    sum += *a.add(o * reduce_size + r);
                }
                *out.add(o) = sum;
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
                let mut prod = 1.0f32;
                for r in 0..reduce_size {
                    prod *= *a.add(o * reduce_size + r);
                }
                *out.add(o) = prod;
            }
        }
        ReduceOp::Mean => {
            let scale = 1.0 / reduce_size as f32;
            for o in 0..outer_size {
                let mut sum = 0.0f32;
                for r in 0..reduce_size {
                    sum += *a.add(o * reduce_size + r);
                }
                *out.add(o) = sum * scale;
            }
        }
        ReduceOp::All | ReduceOp::Any => {
            let is_any = matches!(op, ReduceOp::Any);
            for o in 0..outer_size {
                let mut result = !is_any;
                for r in 0..reduce_size {
                    let val = *a.add(o * reduce_size + r) != 0.0;
                    if is_any {
                        result = result || val;
                    } else {
                        result = result && val;
                    }
                }
                *out.add(o) = if result { 1.0 } else { 0.0 };
            }
        }
    }
}

/// Scalar reduction for f64
#[inline]
pub(super) unsafe fn reduce_scalar_f64(
    op: ReduceOp,
    a: *const f64,
    out: *mut f64,
    reduce_size: usize,
    outer_size: usize,
) {
    match op {
        ReduceOp::Sum => {
            for o in 0..outer_size {
                let mut sum = 0.0f64;
                for r in 0..reduce_size {
                    sum += *a.add(o * reduce_size + r);
                }
                *out.add(o) = sum;
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
                let mut prod = 1.0f64;
                for r in 0..reduce_size {
                    prod *= *a.add(o * reduce_size + r);
                }
                *out.add(o) = prod;
            }
        }
        ReduceOp::Mean => {
            let scale = 1.0 / reduce_size as f64;
            for o in 0..outer_size {
                let mut sum = 0.0f64;
                for r in 0..reduce_size {
                    sum += *a.add(o * reduce_size + r);
                }
                *out.add(o) = sum * scale;
            }
        }
        ReduceOp::All | ReduceOp::Any => {
            let is_any = matches!(op, ReduceOp::Any);
            for o in 0..outer_size {
                let mut result = !is_any;
                for r in 0..reduce_size {
                    let val = *a.add(o * reduce_size + r) != 0.0;
                    if is_any {
                        result = result || val;
                    } else {
                        result = result && val;
                    }
                }
                *out.add(o) = if result { 1.0 } else { 0.0 };
            }
        }
    }
}
