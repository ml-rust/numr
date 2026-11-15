//! Complete per-lane interval operations: outward rounding, then optional tightening.

use super::lane_float::LaneFloat;
use super::lane_interval::{
    Flagged, Span, add_outward, div_positive_interval_outward, div_positive_outward,
    divisor_invalid, mul_outward, quotient_valid, restrict_div_sign, restrict_mul_sign,
    sub_outward, tighten_div, tighten_mul, tighten_sum,
};

/// Rule switches shared by every lane of one call.
#[derive(Debug, Clone, Copy)]
pub struct LaneMode {
    /// Apply singleton retention and sign restriction.
    pub tight: bool,
    /// The runtime declares IEEE fused multiply-add.
    pub fused: bool,
}

/// Interval sum.
#[inline(always)]
pub fn add<T: LaneFloat>(a: Span<T>, b: Span<T>, mode: LaneMode) -> Flagged<T> {
    let (outward, flag) = add_outward(a, b);
    if !mode.tight {
        return (outward, flag);
    }
    let (out, tight_flag) = tighten_sum(a, b, outward, false);
    (out, flag || tight_flag)
}

/// Interval difference.
#[inline(always)]
pub fn sub<T: LaneFloat>(a: Span<T>, b: Span<T>, mode: LaneMode) -> Flagged<T> {
    let (outward, flag) = sub_outward(a, b);
    if !mode.tight {
        return (outward, flag);
    }
    let (out, tight_flag) = tighten_sum(a, b, outward, true);
    (out, flag || tight_flag)
}

/// Interval product.
#[inline(always)]
pub fn mul<T: LaneFloat>(a: Span<T>, b: Span<T>, mode: LaneMode) -> Flagged<T> {
    let (outward, flag) = mul_outward(a, b);
    if !mode.tight {
        return (outward, flag);
    }
    let (tightened, tight_flag) = if mode.fused {
        tighten_mul(a, b, outward)
    } else {
        (outward, false)
    };
    let (out, sign_flag) = restrict_mul_sign(a, b, tightened);
    (out, flag || tight_flag || sign_flag)
}

/// Interval quotient by a represented positive divisor.
#[inline(always)]
pub fn div_positive<T: LaneFloat>(a: Span<T>, divisor: T, mode: LaneMode) -> Flagged<T> {
    let input_flag = divisor_invalid(divisor);
    let (outward, flag) = div_positive_outward(a, divisor);
    let flag = input_flag || flag;
    if !mode.tight {
        return (outward, flag);
    }
    let (tightened, tight_flag) = if mode.fused {
        tighten_div(a, Span::point(divisor), outward)
    } else {
        (outward, false)
    };
    let (out, sign_flag) = restrict_div_sign(a, tightened);
    (out, flag || tight_flag || sign_flag)
}

/// Interval quotient by a positive interval.
#[inline(always)]
pub fn div_positive_interval<T: LaneFloat>(a: Span<T>, b: Span<T>, mode: LaneMode) -> Flagged<T> {
    let (a_valid, b_valid) = quotient_valid(a, b);
    let (outward, flag) = div_positive_interval_outward(a, b);
    let flag = !a_valid || !b_valid || flag;
    if !mode.tight {
        return (outward, flag);
    }
    let (tightened, tight_flag) = if mode.fused {
        tighten_div(a, b, outward)
    } else {
        (outward, false)
    };
    let (out, sign_flag) = restrict_div_sign(a, tightened);
    (out, flag || tight_flag || sign_flag)
}
