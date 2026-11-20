//! Per-lane interval rules matching the composed reference in `impl_generic::interval`.

use super::lane_float::{LaneFloat, guarded, max, min, positive, select, two_sum};

/// One lane of interval endpoints.
#[derive(Debug, Clone, Copy)]
pub struct Span<T> {
    /// Lower endpoint.
    pub lo: T,
    /// Upper endpoint.
    pub hi: T,
}

impl<T: LaneFloat> Span<T> {
    /// Singleton interval.
    #[inline(always)]
    pub fn point(value: T) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }

    #[inline(always)]
    fn zero(self) -> bool {
        self.lo == T::ZERO && self.hi == T::ZERO
    }

    #[inline(always)]
    fn finite_ordered(self) -> bool {
        self.lo.finite() && self.hi.finite() && self.lo <= self.hi
    }

    /// The status rule of `bounds`: a nonfinite endpoint or `lo > hi`.
    #[inline(always)]
    pub fn invalid(self) -> bool {
        !self.lo.finite() || !self.hi.finite() || self.lo > self.hi
    }

    /// `invalid` through float compares only, so dense lane loops can vectorize.
    ///
    /// A magnitude at most `MAX` is exactly a finite value, and NaN fails every compare.
    #[inline(always)]
    pub fn invalid_by_compare(self) -> bool {
        let finite = (self.lo.magnitude() <= T::MAX) & (self.hi.magnitude() <= T::MAX);
        !(finite & (self.lo <= self.hi))
    }
}

/// Rounded result and status of one lane.
pub type Flagged<T> = (Span<T>, bool);

/// Outward sum with exact-zero identities.
#[inline(always)]
pub fn add_outward<T: LaneFloat>(a: Span<T>, b: Span<T>) -> Flagged<T> {
    let (az, bz) = (a.zero(), b.zero());
    let lo = (a.lo + b.lo).down();
    let hi = (a.hi + b.hi).up();
    let out = Span {
        lo: select(az, b.lo, select(bz, a.lo, lo)),
        hi: select(az, b.hi, select(bz, a.hi, hi)),
    };
    (out, out.invalid())
}

/// Upper endpoint and status of `add_outward` over two singletons.
///
/// The lower endpoint only feeds the status. Stepping down leaves the finite range only from
/// the most negative finite value, so the status needs no second step.
#[inline(always)]
pub fn add_outward_upper<T: LaneFloat>(a: T, b: T) -> (T, bool) {
    let (az, bz) = (a == T::ZERO, b == T::ZERO);
    singleton_upper(az, b, bz, a, a + b)
}

/// Upper endpoint and status of `mul_outward` over two singletons.
///
/// Every corner of two singletons is the same product, so the status follows the sum rule.
#[inline(always)]
pub fn mul_outward_upper<T: LaneFloat>(a: T, b: T) -> (T, bool) {
    let (az, bz) = (a == T::ZERO, b == T::ZERO);
    // A zero operand is returned itself, the first one when both are zero.
    singleton_upper(az, a, bz, b, a * b)
}

/// `first` returns `on_first`, then `second` returns `on_second`, else the step above `nominal`.
#[inline(always)]
fn singleton_upper<T: LaneFloat>(
    first: bool,
    on_first: T,
    second: bool,
    on_second: T,
    nominal: T,
) -> (T, bool) {
    let edge = nominal.magnitude() == T::MAX;
    let upper = select(first, on_first, select(second, on_second, nominal.up()));
    let flag = select_flag(
        first,
        !on_first.finite(),
        select_flag(second, !on_second.finite(), !nominal.finite() || edge),
    );
    (upper, flag)
}

#[inline(always)]
fn select_flag(condition: bool, x: bool, y: bool) -> bool {
    if condition { x } else { y }
}

/// Outward difference with exact-zero identities.
#[inline(always)]
pub fn sub_outward<T: LaneFloat>(a: Span<T>, b: Span<T>) -> Flagged<T> {
    let (az, bz) = (a.zero(), b.zero());
    let lo = (a.lo - b.hi).down();
    let hi = (a.hi - b.lo).up();
    let out = Span {
        lo: select(bz, a.lo, select(az, -b.hi, lo)),
        hi: select(bz, a.hi, select(az, -b.lo, hi)),
    };
    (out, out.invalid())
}

#[inline(always)]
fn corners<T: LaneFloat>(ll: T, lu: T, ul: T, uu: T) -> Span<T> {
    Span {
        lo: min(min(ll, lu), min(ul, uu)).down(),
        hi: max(max(ll, lu), max(ul, uu)).up(),
    }
}

/// Outward product over four endpoint products.
#[inline(always)]
pub fn mul_outward<T: LaneFloat>(a: Span<T>, b: Span<T>) -> Flagged<T> {
    let (az, bz) = (a.zero(), b.zero());
    let zp = az || bz;
    let al = select(zp, T::ZERO, a.lo);
    let au = select(zp, T::ZERO, a.hi);
    let bl = select(zp, T::ONE, b.lo);
    let bu = select(zp, T::ONE, b.hi);
    let c = corners(al * bl, al * bu, au * bl, au * bu);
    let out = Span {
        lo: select(zp, select(az, a.lo, b.lo), c.lo),
        hi: select(zp, select(az, a.hi, b.hi), c.hi),
    };
    (out, out.invalid())
}

/// Divisor status of `div_positive`, on the divisor's own lane.
#[inline(always)]
pub fn divisor_invalid<T: LaneFloat>(divisor: T) -> bool {
    !divisor.finite() || !positive(divisor)
}

/// Outward quotient by a represented divisor.
#[inline(always)]
pub fn div_positive_outward<T: LaneFloat>(a: Span<T>, d: T) -> Flagged<T> {
    let den = select(d > T::ZERO, d, T::ONE);
    let identity = d == T::ONE;
    let al = select(identity, T::ONE, a.lo);
    let au = select(identity, T::ONE, a.hi);
    let den = select(identity, T::ONE, den);
    let exact = a.zero() || identity;
    let out = Span {
        lo: select(exact, a.lo, (al / den).down()),
        hi: select(exact, a.hi, (au / den).up()),
    };
    (out, out.invalid())
}

/// Validity of interval quotient operands: `(numerator, divisor)`.
#[inline(always)]
pub fn quotient_valid<T: LaneFloat>(a: Span<T>, b: Span<T>) -> (bool, bool) {
    (a.finite_ordered(), b.finite_ordered() && b.lo > T::ZERO)
}

/// Outward quotient by an interval with a strictly positive lower endpoint.
#[inline(always)]
pub fn div_positive_interval_outward<T: LaneFloat>(a: Span<T>, b: Span<T>) -> Flagged<T> {
    let (av, bv) = quotient_valid(a, b);
    let identity = b.lo == T::ONE && b.hi == T::ONE;
    let al = select(identity, T::ONE, select(av, a.lo, T::ZERO));
    let au = select(identity, T::ONE, select(av, a.hi, T::ZERO));
    let bl = select(identity, T::ONE, select(bv, b.lo, T::ONE));
    let bu = select(identity, T::ONE, select(bv, b.hi, T::ONE));
    let c = corners(al / bl, al / bu, au / bl, au / bu);
    let exact = a.zero() || identity;
    let out = Span {
        lo: select(exact, a.lo, c.lo),
        hi: select(exact, a.hi, c.hi),
    };
    (out, out.invalid())
}

/// Outward square roots, with the domain status of the input lane.
#[inline(always)]
pub fn sqrt_nonnegative<T: LaneFloat>(a: Span<T>) -> Flagged<T> {
    let valid = a.finite_ordered() && a.lo >= T::ZERO;
    let li = select(valid, a.lo, T::ZERO);
    let ui = select(valid, a.hi, T::ZERO);
    let out = Span {
        lo: select(li == T::ZERO, li, li.root().down()),
        hi: select(ui == T::ZERO, ui, ui.root().up()),
    };
    (out, !valid || out.invalid())
}

#[inline(always)]
fn exact_sum<T: LaneFloat>(a: T, b: T, outward: T, valid: bool) -> T {
    // The upper guard prevents overflow inside TwoSum. Gradual underflow preserves additive residuals.
    let candidate = valid && a.magnitude() <= T::GUARD_MAX && b.magnitude() <= T::GUARD_MAX;
    let (sum, residual) = two_sum(select(candidate, a, T::ZERO), select(candidate, b, T::ZERO));
    select(candidate && residual == T::ZERO, sum, outward)
}

/// Retain singleton sums or differences that TwoSum proves exact.
#[inline(always)]
pub fn tighten_sum<T: LaneFloat>(
    a: Span<T>,
    b: Span<T>,
    outward: Span<T>,
    subtract: bool,
) -> Flagged<T> {
    let valid = outward.finite_ordered() && !(a.zero() || b.zero());
    let (bl, bu) = if subtract {
        (-b.hi, -b.lo)
    } else {
        (b.lo, b.hi)
    };
    let out = Span {
        lo: exact_sum(a.lo, bl, outward.lo, valid),
        hi: exact_sum(a.hi, bu, outward.hi, valid),
    };
    (out, out.invalid())
}

/// Lanes whose fused product residual is exact.
#[inline(always)]
pub fn residual_exact<T: LaneFloat>(a: T, b: T, product: T) -> bool {
    let (fa, fb) = (a.finite(), b.finite());
    let sa = select(fa, a, T::ZERO);
    let sb = select(fb, b, T::ZERO);
    // An integer factor keeps the exact product on the minimum-subnormal lattice.
    let proof = (guarded(a) && guarded(b) && guarded(product))
        || a == T::ZERO
        || b == T::ZERO
        || sa == sa.truncated()
        || sb == sb.truncated();
    proof && fa && fb && product.finite()
}

#[inline(always)]
fn replace_exact<T: LaneFloat>(
    outward: Span<T>,
    candidate: bool,
    nominal: T,
    residual: T,
) -> Flagged<T> {
    let exact = candidate && residual == T::ZERO;
    let out = Span {
        lo: select(exact, nominal, outward.lo),
        hi: select(exact, nominal, outward.hi),
    };
    (out, out.invalid())
}

/// Retain singleton products that a fused residual proves exact.
#[inline(always)]
pub fn tighten_mul<T: LaneFloat>(a: Span<T>, b: Span<T>, outward: Span<T>) -> Flagged<T> {
    let candidate = a.lo == a.hi
        && b.lo == b.hi
        && a.finite_ordered()
        && b.finite_ordered()
        && outward.finite_ordered()
        && !(a.zero() || b.zero());
    let sa = select(candidate, a.lo, T::ONE);
    let sb = select(candidate, b.lo, T::ONE);
    let product = sa * sb;
    let candidate = candidate && residual_exact(sa, sb, product);
    // Guard the residual separately: a rounded zero can conceal a nonzero exact product.
    let sa = select(candidate, sa, T::ONE);
    let sb = select(candidate, sb, T::ONE);
    let sp = select(candidate, product, T::ONE);
    replace_exact(outward, candidate, product, sa.fused(sb, -sp))
}

/// Retain singleton quotients that a fused residual proves exact.
#[inline(always)]
pub fn tighten_div<T: LaneFloat>(a: Span<T>, b: Span<T>, outward: Span<T>) -> Flagged<T> {
    let candidate = a.lo == a.hi
        && guarded(a.lo)
        && b.lo == b.hi
        && guarded(b.lo)
        && outward.finite_ordered()
        && b.lo > T::ZERO;
    let sa = select(candidate, a.lo, T::ONE);
    let sb = select(candidate, b.lo, T::ONE);
    let quotient = sa / sb;
    let candidate = candidate && guarded(quotient);
    let sa = select(candidate, sa, T::ONE);
    let sb = select(candidate, sb, T::ONE);
    let sq = select(candidate, quotient, T::ONE);
    replace_exact(outward, candidate, quotient, (-sq).fused(sb, sa))
}

#[inline(always)]
fn restrict_sign<T: LaneFloat>(
    outward: Span<T>,
    nonnegative: bool,
    nonpositive: bool,
) -> Flagged<T> {
    // Strict comparisons preserve represented zero endpoints, including their sign bits.
    let out = Span {
        lo: select(nonnegative && outward.lo < T::ZERO, T::ZERO, outward.lo),
        hi: select(nonpositive && outward.hi > T::ZERO, T::ZERO, outward.hi),
    };
    (out, out.invalid())
}

/// Clamp product bounds to signs that operand endpoints prove.
#[inline(always)]
pub fn restrict_mul_sign<T: LaneFloat>(a: Span<T>, b: Span<T>, outward: Span<T>) -> Flagged<T> {
    let (ann, anp) = (a.lo >= T::ZERO, a.hi <= T::ZERO);
    let (bnn, bnp) = (b.lo >= T::ZERO, b.hi <= T::ZERO);
    restrict_sign(
        outward,
        (ann && bnn) || (anp && bnp),
        (ann && bnp) || (anp && bnn),
    )
}

/// Clamp quotient bounds to the numerator's proved sign.
#[inline(always)]
pub fn restrict_div_sign<T: LaneFloat>(a: Span<T>, outward: Span<T>) -> Flagged<T> {
    restrict_sign(outward, a.lo >= T::ZERO, a.hi <= T::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_uppers_match_the_span_rules() {
        let values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            3.0,
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            f64::MAX,
            -f64::MAX,
            f64::MAX * 0.5,
            f64::MAX.next_down(),
            -f64::MAX.next_down(),
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for &a in &values {
            for &b in &values {
                let (sum, sum_flag) = add_outward(Span::point(a), Span::point(b));
                let (upper, flag) = add_outward_upper(a, b);
                assert_eq!(
                    (sum.hi.to_bits(), sum_flag),
                    (upper.to_bits(), flag),
                    "{a} + {b}"
                );
                let (product, product_flag) = mul_outward(Span::point(a), Span::point(b));
                let (upper, flag) = mul_outward_upper(a, b);
                assert_eq!(
                    (product.hi.to_bits(), product_flag),
                    (upper.to_bits(), flag),
                    "{a} * {b}"
                );
            }
        }
    }

    #[test]
    fn compare_status_matches_the_bit_test_status() {
        let values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::from_bits(1),
            f64::MAX,
            f64::MIN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
        ];
        for &lo in &values {
            for &hi in &values {
                let span = Span { lo, hi };
                assert_eq!(span.invalid(), span.invalid_by_compare(), "{lo} {hi}");
                let narrow = Span {
                    lo: lo as f32,
                    hi: hi as f32,
                };
                assert_eq!(narrow.invalid(), narrow.invalid_by_compare(), "{lo} {hi}");
            }
        }
    }
}
