//! Per-lane certified expansion rules matching the composed reference.

use super::lane_float::{LaneFloat, max, select, two_sum};
use super::lane_interval::{
    Span, add_outward, add_outward_upper, mul_outward_upper, residual_exact, sub_outward,
};

/// One lane of expansion components.
#[derive(Debug, Clone, Copy)]
pub struct Parts<T> {
    /// High center component.
    pub high: T,
    /// Low center component.
    pub low: T,
    /// Absolute error radius.
    pub error: T,
}

impl<T: LaneFloat> Parts<T> {
    /// The status rule of an owned expansion: a nonfinite component or a negative radius.
    #[inline(always)]
    pub fn invalid(self) -> bool {
        !self.high.finite() || !self.low.finite() || !self.error.finite() || self.error < T::ZERO
    }

    #[inline(always)]
    fn exact_one(self) -> bool {
        self.high == T::ONE && self.low == T::ZERO && self.error == T::ZERO
    }

    /// Both trailing components hold the bit pattern of positive zero.
    #[inline(always)]
    pub(super) fn point(self) -> bool {
        self.low.positive_zero() && self.error.positive_zero()
    }

    /// The same lane with its trailing components as constants.
    #[inline(always)]
    pub(super) fn as_point(self) -> Self {
        Self {
            high: self.high,
            low: T::ZERO,
            error: T::ZERO,
        }
    }
}

/// Expansion result and status of one lane.
pub type FlaggedParts<T> = (Parts<T>, bool);

/// TwoSum normalization of supplied components.
#[inline(always)]
pub fn normalize<T: LaneFloat>(value: Parts<T>) -> FlaggedParts<T> {
    let flag = !value.high.finite() || !value.low.finite();
    let low_zero = value.low == T::ZERO;
    let (high, low) = two_sum(value.high, value.low);
    let out = Parts {
        high: select(low_zero, value.high, high),
        low: select(low_zero, value.low, low),
        error: value.error,
    };
    (out, flag || out.invalid())
}

/// Compress terms with TwoSum while bounding every discarded residual.
#[inline(always)]
pub fn compress<T: LaneFloat, const N: usize>(terms: [T; N], radius: T) -> FlaggedParts<T> {
    let mut flag = terms.iter().any(|term| !term.finite()) || !radius.finite() || radius < T::ZERO;
    let mut high = terms[0];
    let mut low = T::ZERO;
    let mut radius = radius;
    for &term in &terms[1..] {
        let (sum, residual) = two_sum(high, term);
        let (low_sum, discarded) = two_sum(low, residual);
        let (next_high, next_low) = two_sum(sum, low_sum);
        let discarded_magnitude = discarded.magnitude();
        let (sum_radius, sum_flag) = add_outward_upper(radius, discarded_magnitude);
        flag = flag
            || !discarded.finite()
            || !radius.finite()
            || !discarded_magnitude.finite()
            || sum_flag;
        radius = sum_radius;
        high = next_high;
        low = next_low;
    }
    let out = Parts {
        high,
        low,
        error: radius,
    };
    (out, flag || out.invalid())
}

/// Center sum or difference.
///
/// A point operand takes a copy of the same rule with its zero components folded.
#[inline(always)]
pub fn sum<T: LaneFloat>(a: Parts<T>, b: Parts<T>, subtract: bool) -> FlaggedParts<T> {
    if b.point() {
        general_sum(a, b.as_point(), subtract)
    } else if a.point() {
        general_sum(a.as_point(), b, subtract)
    } else {
        general_sum(a, b, subtract)
    }
}

#[inline(always)]
fn general_sum<T: LaneFloat>(a: Parts<T>, b: Parts<T>, subtract: bool) -> FlaggedParts<T> {
    let (radius, radius_flag) = add_outward_upper(a.error, b.error);
    let flag = !a.error.finite() || !b.error.finite() || radius_flag;
    let (b_high, b_low) = if subtract {
        (-b.high, -b.low)
    } else {
        (b.high, b.low)
    };
    let (out, compress_flag) = compress([a.high, b_high, a.low, b_low], radius);
    (out, flag || compress_flag)
}

#[inline(always)]
fn component_product<T: LaneFloat>(a: T, b: T) -> (T, T, T, bool) {
    let product = a * b;
    let residual = a.fused(b, -product);
    let exact = residual_exact(a, b, product);
    // Adjacent differences are exact, including gaps around zero and subnormal residuals.
    let gap = max(residual - residual.down(), residual.up() - residual);
    let flag = !product.finite() || !residual.finite() || !gap.finite();
    (product, residual, select(exact, T::ZERO, gap), flag)
}

#[inline(always)]
fn radius_product<T: LaneFloat>(component: T, radius: T) -> (T, bool) {
    let magnitude = component.magnitude();
    let (out, flag) = mul_outward_upper(magnitude, radius);
    (out, flag || !magnitude.finite() || !radius.finite())
}

/// Center product with fused residual and radius enclosures.
///
/// An exact one returns the other operand. The general rule masks both operands to one there,
/// so its own status is clear and only the returned lane can flag.
/// A point operand takes a copy of the same rule with its zero components folded.
#[inline(always)]
pub fn product<T: LaneFloat>(a: Parts<T>, b: Parts<T>) -> FlaggedParts<T> {
    let a_one = a.exact_one();
    if a_one || b.exact_one() {
        let out = if a_one { b } else { a };
        (out, out.invalid())
    } else if b.point() {
        general_product(a, b.as_point())
    } else if a.point() {
        general_product(a.as_point(), b)
    } else {
        general_product(a, b)
    }
}

#[inline(always)]
fn general_product<T: LaneFloat>(a: Parts<T>, b: Parts<T>) -> FlaggedParts<T> {
    let (a_one, b_one) = (a.exact_one(), b.exact_one());
    let identity = a_one || b_one;
    let mask = |value: Parts<T>| Parts {
        high: select(identity, T::ONE, value.high),
        low: select(identity, T::ZERO, value.low),
        error: select(identity, T::ZERO, value.error),
    };
    let (ma, mb) = (mask(a), mask(b));
    let (hh, hh_low, hh_error, f0) = component_product(ma.high, mb.high);
    let (hl, hl_low, hl_error, f1) = component_product(ma.high, mb.low);
    let (lh, lh_low, lh_error, f2) = component_product(ma.low, mb.high);
    let (ll, ll_low, ll_error, f3) = component_product(ma.low, mb.low);
    let mut flag = f0 || f1 || f2 || f3;
    let mut terms = [T::ZERO; 9];
    for (slot, (component, radius)) in terms.iter_mut().zip([
        (ma.high, mb.error),
        (ma.low, mb.error),
        (mb.high, ma.error),
        (mb.low, ma.error),
        (ma.error, mb.error),
    ]) {
        let (value, radius_flag) = radius_product(component, radius);
        *slot = value;
        flag = flag || radius_flag;
    }
    terms[5] = hh_error;
    terms[6] = hl_error;
    terms[7] = lh_error;
    terms[8] = ll_error;
    flag = flag || terms.iter().any(|term| !term.finite());
    let mut radius = Span::point(terms[0]);
    for &term in &terms[1..] {
        let (next, add_flag) = add_outward(radius, Span::point(term));
        radius = next;
        flag = flag || add_flag;
    }
    let (general, compress_flag) =
        compress([hh, hh_low, hl, hl_low, lh, lh_low, ll, ll_low], radius.hi);
    let choose = |x: T, y: T, g: T| select(identity, select(a_one, y, x), g);
    let out = Parts {
        high: choose(a.high, b.high, general.high),
        low: choose(a.low, b.low, general.low),
        error: choose(a.error, b.error, general.error),
    };
    (out, flag || compress_flag || out.invalid())
}

/// Outward projection around the rounded center.
///
/// A point lane takes a copy of the same rule with its zero components folded.
#[inline(always)]
pub fn project<T: LaneFloat>(value: Parts<T>) -> (Span<T>, bool) {
    if value.point() {
        general_project(value.as_point())
    } else {
        general_project(value)
    }
}

#[inline(always)]
fn general_project<T: LaneFloat>(value: Parts<T>) -> (Span<T>, bool) {
    let (center, center_flag) = add_outward(Span::point(value.high), Span::point(value.low));
    let radius = Span::point(value.error);
    let (lower, lower_flag) = sub_outward(center, radius);
    let (upper, upper_flag) = add_outward(center, radius);
    let out = Span {
        lo: lower.lo,
        hi: upper.hi,
    };
    let flag = !value.high.finite()
        || !value.low.finite()
        || !value.error.finite()
        || center_flag
        || lower_flag
        || upper_flag
        || out.invalid();
    (out, flag)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// Every lane built from a set of signed, special, and subnormal components.
    pub(in super::super) fn lanes() -> Vec<Parts<f64>> {
        let values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.1,
            3.0,
            f64::from_bits(1),
            f64::INFINITY,
            f64::NAN,
        ];
        let mut out = Vec::new();
        for &high in &values {
            for &low in &values {
                for &error in &values {
                    out.push(Parts { high, low, error });
                }
            }
        }
        out
    }

    /// The result components and status as raw bits.
    pub(in super::super) fn bits(value: FlaggedParts<f64>) -> [u64; 4] {
        let (parts, flag) = value;
        [
            parts.high.to_bits(),
            parts.low.to_bits(),
            parts.error.to_bits(),
            flag as u64,
        ]
    }

    #[test]
    fn product_shortcuts_match_the_general_rule_bit_for_bit() {
        let lanes = lanes();
        for &a in &lanes {
            for &b in &lanes {
                assert_eq!(
                    bits(product(a, b)),
                    bits(general_product(a, b)),
                    "a={a:?} b={b:?}"
                );
            }
        }
    }

    #[test]
    fn project_shortcut_matches_the_general_rule_bit_for_bit() {
        for value in lanes() {
            let (span, flag) = project(value);
            let (general, general_flag) = general_project(value);
            assert_eq!(
                [span.lo.to_bits(), span.hi.to_bits(), flag as u64],
                [
                    general.lo.to_bits(),
                    general.hi.to_bits(),
                    general_flag as u64
                ],
                "{value:?}"
            );
        }
    }

    #[test]
    fn sum_shortcuts_match_the_general_rule_bit_for_bit() {
        let lanes = lanes();
        for &a in &lanes {
            for &b in &lanes {
                for subtract in [false, true] {
                    assert_eq!(
                        bits(sum(a, b, subtract)),
                        bits(general_sum(a, b, subtract)),
                        "a={a:?} b={b:?} subtract={subtract}"
                    );
                }
            }
        }
    }
}
