// Per-lane certified expansion rules. Mirrors lane_expansion.rs and lane_division.rs.

#ifndef NUMR_EXPANSION_LANE_CUH
#define NUMR_EXPANSION_LANE_CUH

#include "interval_lane.cuh"

template <typename T> struct Parts {
    T high;
    T low;
    T error;
    __device__ __forceinline__ bool invalid() const {
        return !lane_finite(high) || !lane_finite(low) || !lane_finite(error) || error < (T)0;
    }
    __device__ __forceinline__ bool exact_one() const {
        return high == (T)1 && low == (T)0 && error == (T)0;
    }
};

template <typename T> __device__ __forceinline__ Parts<T> make_parts(T h, T l, T e) { Parts<T> p = {h, l, e}; return p; }

template <typename T>
__device__ __forceinline__ Parts<T> normalize(Parts<T> v, bool& flag) {
    flag = flag || !lane_finite(v.high) || !lane_finite(v.low);
    const bool low_zero = v.low == (T)0;
    T high, low;
    lane_two_sum(v.high, v.low, high, low);
    Parts<T> out = make_parts(lane_select(low_zero, v.high, high), lane_select(low_zero, v.low, low), v.error);
    flag = flag || out.invalid();
    return out;
}

template <typename T, int N>
__device__ __forceinline__ Parts<T> compress(const T (&terms)[N], T radius, bool& flag) {
    typedef LaneTraits<T> L;
    for (int k = 0; k < N; ++k) flag = flag || !lane_finite(terms[k]);
    flag = flag || !lane_finite(radius) || radius < (T)0;
    T high = terms[0], low = (T)0;
    for (int k = 1; k < N; ++k) {
        T sum, residual, low_sum, discarded, next_high, next_low;
        lane_two_sum(high, terms[k], sum, residual);
        lane_two_sum(low, residual, low_sum, discarded);
        lane_two_sum(sum, low_sum, next_high, next_low);
        const T magnitude = L::magnitude(discarded);
        flag = flag || !lane_finite(discarded) || !lane_finite(radius) || !lane_finite(magnitude);
        radius = add_outward(span_point(radius), span_point(magnitude), flag).hi;
        high = next_high;
        low = next_low;
    }
    Parts<T> out = make_parts(high, low, radius);
    flag = flag || out.invalid();
    return out;
}

template <typename T>
__device__ __forceinline__ Parts<T> expansion_sum(Parts<T> a, Parts<T> b, bool subtract, bool& flag) {
    flag = flag || !lane_finite(a.error) || !lane_finite(b.error);
    const T radius = add_outward(span_point(a.error), span_point(b.error), flag).hi;
    const T terms[4] = {a.high, subtract ? -b.high : b.high, a.low, subtract ? -b.low : b.low};
    return compress(terms, radius, flag);
}

template <typename T>
__device__ __forceinline__ T component_product(T a, T b, T& residual, T& error, bool& flag) {
    typedef LaneTraits<T> L;
    const T product = a * b;
    residual = L::fused(a, b, -product);
    const bool exact = residual_exact(a, b, product);
    const T gap = lane_max(residual - lane_down(residual), lane_up(residual) - residual);
    flag = flag || !lane_finite(product) || !lane_finite(residual) || !lane_finite(gap);
    error = lane_select(exact, (T)0, gap);
    return product;
}

template <typename T>
__device__ __forceinline__ T radius_product(T component, T radius, bool& flag) {
    const T magnitude = LaneTraits<T>::magnitude(component);
    flag = flag || !lane_finite(magnitude) || !lane_finite(radius);
    return mul_outward(span_point(magnitude), span_point(radius), flag).hi;
}

template <typename T>
__device__ __forceinline__ Parts<T> expansion_product(Parts<T> a, Parts<T> b, bool& flag) {
    const bool a_one = a.exact_one(), b_one = b.exact_one(), identity = a_one || b_one;
    const Parts<T> ma = make_parts(lane_select(identity, (T)1, a.high), lane_select(identity, (T)0, a.low),
                                   lane_select(identity, (T)0, a.error));
    const Parts<T> mb = make_parts(lane_select(identity, (T)1, b.high), lane_select(identity, (T)0, b.low),
                                   lane_select(identity, (T)0, b.error));
    T terms[9], products[8];
    products[0] = component_product(ma.high, mb.high, products[1], terms[5], flag);
    products[2] = component_product(ma.high, mb.low, products[3], terms[6], flag);
    products[4] = component_product(ma.low, mb.high, products[5], terms[7], flag);
    products[6] = component_product(ma.low, mb.low, products[7], terms[8], flag);
    terms[0] = radius_product(ma.high, mb.error, flag);
    terms[1] = radius_product(ma.low, mb.error, flag);
    terms[2] = radius_product(mb.high, ma.error, flag);
    terms[3] = radius_product(mb.low, ma.error, flag);
    terms[4] = radius_product(ma.error, mb.error, flag);
    for (int k = 0; k < 9; ++k) flag = flag || !lane_finite(terms[k]);
    Span<T> radius = span_point(terms[0]);
    for (int k = 1; k < 9; ++k) radius = add_outward(radius, span_point(terms[k]), flag);
    const Parts<T> general = compress(products, radius.hi, flag);
    Parts<T> out = make_parts(lane_select(identity, lane_select(a_one, b.high, a.high), general.high),
                              lane_select(identity, lane_select(a_one, b.low, a.low), general.low),
                              lane_select(identity, lane_select(a_one, b.error, a.error), general.error));
    flag = flag || out.invalid();
    return out;
}

template <typename T>
__device__ __forceinline__ Span<T> expansion_project(Parts<T> v, bool& flag) {
    flag = flag || !lane_finite(v.high) || !lane_finite(v.low) || !lane_finite(v.error);
    const Span<T> center = add_outward(span_point(v.high), span_point(v.low), flag);
    const Span<T> radius = span_point(v.error);
    const T lo = sub_outward(center, radius, flag).lo;
    const T hi = add_outward(center, radius, flag).hi;
    Span<T> out = {lo, hi};
    flag = flag || out.invalid();
    return out;
}

template <typename T>
__device__ __forceinline__ Parts<T> expansion_quotient(Parts<T> a, Parts<T> b, bool& flag) {
    typedef LaneTraits<T> L;
    const Span<T> den = expansion_project(b, flag);
    const bool positive = den.lo > (T)0;
    const bool valid = positive && lane_finite(den.lo) && lane_finite(den.hi) && lane_finite(b.high) && b.high > (T)0;
    flag = flag || !positive || !valid;
    const T denominator = lane_select(valid, b.high, (T)1);
    const T q0 = L::div(lane_select(valid, a.high, (T)0), denominator);
    const Parts<T> provisional = make_parts(q0, (T)0, (T)0);
    flag = flag || provisional.invalid();
    const Parts<T> first = expansion_sum(a, expansion_product(provisional, b, flag), true, flag);
    const T q1 = L::div(lane_select(valid, first.high, (T)0), denominator);
    flag = flag || !lane_finite(q1);
    T high, low;
    lane_two_sum(q0, q1, high, low);
    const Parts<T> candidate = make_parts(high, low, (T)0);
    flag = flag || candidate.invalid();
    const Parts<T> residual = expansion_sum(a, expansion_product(candidate, b, flag), true, flag);
    const Span<T> residual_span = expansion_project(residual, flag);
    const T magnitude = lane_max(L::magnitude(residual_span.lo), L::magnitude(residual_span.hi));
    flag = flag || !lane_finite(magnitude);
    const T radius = div_positive_outward(span_point(magnitude), lane_select(valid, den.lo, (T)1), flag).hi;
    Parts<T> out = make_parts(high, low, radius);
    flag = flag || out.invalid();
    return out;
}

template <typename T>
__device__ __forceinline__ Parts<T> expansion_lift(Span<T> v, bool& flag) {
    flag = flag || v.invalid();
    const bool equal = v.lo == v.hi;
    const T center = lane_select(equal, v.lo, lane_midpoint(v.lo, v.hi));
    flag = flag || !lane_finite(center);
    const Span<T> c = span_point(center);
    Span<T> lower = sub_outward(c, span_point(v.lo), flag);
    lower = tighten_sum(c, span_point(v.lo), lower, true, flag);
    Span<T> upper = sub_outward(span_point(v.hi), c, flag);
    upper = tighten_sum(span_point(v.hi), c, upper, true, flag);
    const T radius = lane_select(equal, (T)0, lane_max(lower.hi, upper.hi));
    return normalize(make_parts(center, (T)0, radius), flag);
}

#endif
