//! Per-lane expansion quotient and interval lift matching the composed reference.

use super::lane_expansion::{FlaggedParts, Parts, normalize, product, project, sum};
use super::lane_float::{LaneFloat, max, midpoint, select, two_sum};
use super::lane_interval::Span;
use super::lane_public::{self, LaneMode};

const OUTWARD: LaneMode = LaneMode {
    tight: false,
    fused: true,
};
const TIGHT: LaneMode = LaneMode {
    tight: true,
    fused: true,
};

/// Corrected quotient center certified through an independent residual enclosure.
///
/// A point divisor takes a copy of the same rule with its constant components folded.
#[inline(always)]
pub fn quotient<T: LaneFloat>(a: Parts<T>, b: Parts<T>) -> FlaggedParts<T> {
    if b.point() {
        let b = b.as_point();
        if b.high == T::ONE {
            let one = Parts { high: T::ONE, ..b };
            general_quotient(a, one)
        } else {
            general_quotient(a, b)
        }
    } else {
        general_quotient(a, b)
    }
}

#[inline(always)]
fn general_quotient<T: LaneFloat>(a: Parts<T>, b: Parts<T>) -> FlaggedParts<T> {
    let (den, mut flag) = project(b);
    let positive = den.lo > T::ZERO;
    let valid =
        positive && den.lo.finite() && den.hi.finite() && b.high.finite() && b.high > T::ZERO;
    flag = flag || !positive || !valid;
    // Mask invalid denominator lanes before eager quotient evaluation.
    let denominator = select(valid, b.high, T::ONE);
    let q0 = select(valid, a.high, T::ZERO) / denominator;
    let provisional = Parts {
        high: q0,
        low: T::ZERO,
        error: T::ZERO,
    };
    let (provisional_product, f0) = product(provisional, b);
    let (first_residual, f1) = sum(a, provisional_product, true);
    let q1 = select(valid, first_residual.high, T::ZERO) / denominator;
    let (high, low) = two_sum(q0, q1);
    let candidate = Parts {
        high,
        low,
        error: T::ZERO,
    };
    let (candidate_product, f2) = product(candidate, b);
    let (residual, f3) = sum(a, candidate_product, true);
    let (residual_span, f4) = project(residual);
    let magnitude = max(residual_span.lo.magnitude(), residual_span.hi.magnitude());
    let (radius, f5) = lane_public::div_positive(
        Span::point(magnitude),
        select(valid, den.lo, T::ONE),
        OUTWARD,
    );
    let out = Parts {
        high,
        low,
        error: radius.hi,
    };
    flag = flag
        || provisional.invalid()
        || f0
        || f1
        || !q1.finite()
        || candidate.invalid()
        || f2
        || f3
        || f4
        || !magnitude.finite()
        || f5
        || out.invalid();
    (out, flag)
}

/// Lift an interval into its midpoint center and a covering radius.
#[inline(always)]
pub fn lift<T: LaneFloat>(value: Span<T>) -> FlaggedParts<T> {
    let equal = value.lo == value.hi;
    let center = select(equal, value.lo, midpoint(value.lo, value.hi));
    let (lower_distance, f0) = lane_public::sub(Span::point(center), Span::point(value.lo), TIGHT);
    let (upper_distance, f1) = lane_public::sub(Span::point(value.hi), Span::point(center), TIGHT);
    let radius = select(equal, T::ZERO, max(lower_distance.hi, upper_distance.hi));
    let (out, f2) = normalize(Parts {
        high: center,
        low: T::ZERO,
        error: radius,
    });
    let flag = value.invalid() || !center.finite() || f0 || f1 || f2;
    (out, flag)
}

#[cfg(test)]
mod tests {
    use super::super::lane_expansion::tests::{bits, lanes};
    use super::*;

    #[test]
    fn quotient_shortcuts_match_the_general_rule_bit_for_bit() {
        let lanes = lanes();
        for &a in &lanes {
            for &b in &lanes {
                assert_eq!(
                    bits(quotient(a, b)),
                    bits(general_quotient(a, b)),
                    "a={a:?} b={b:?}"
                );
            }
        }
    }
}
