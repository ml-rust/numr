// Singleton retention, sign restriction, and complete per-lane interval
// operations. Mirrors lane_interval.rs and lane_public.rs.

#ifndef NUMR_INTERVAL_TIGHTEN_CUH
#define NUMR_INTERVAL_TIGHTEN_CUH

template <typename T>
__device__ __forceinline__ T exact_sum(T a, T b, T outward, bool valid) {
    typedef LaneTraits<T> L;
    const bool candidate = valid && L::magnitude(a) <= L::guard_max() && L::magnitude(b) <= L::guard_max();
    T sum, residual;
    lane_two_sum(lane_select(candidate, a, (T)0), lane_select(candidate, b, (T)0), sum, residual);
    return lane_select(candidate && residual == (T)0, sum, outward);
}

template <typename T>
__device__ __forceinline__ Span<T> tighten_sum(Span<T> a, Span<T> b, Span<T> outward, bool subtract, bool& flag) {
    const bool valid = outward.finite_ordered() && !(a.zero() || b.zero());
    const T bl = subtract ? -b.hi : b.lo;
    const T bu = subtract ? -b.lo : b.hi;
    Span<T> out = {exact_sum(a.lo, bl, outward.lo, valid), exact_sum(a.hi, bu, outward.hi, valid)};
    flag = flag || out.invalid();
    return out;
}

template <typename T>
__device__ __forceinline__ bool residual_exact(T a, T b, T product) {
    typedef LaneTraits<T> L;
    const bool fa = lane_finite(a), fb = lane_finite(b);
    const T sa = lane_select(fa, a, (T)0), sb = lane_select(fb, b, (T)0);
    const bool proof = (lane_guarded(a) && lane_guarded(b) && lane_guarded(product)) || a == (T)0 ||
                       b == (T)0 || sa == L::truncated(sa) || sb == L::truncated(sb);
    return proof && fa && fb && lane_finite(product);
}

template <typename T>
__device__ __forceinline__ Span<T> replace_exact(Span<T> outward, bool candidate, T nominal, T residual, bool& flag) {
    const bool exact = candidate && residual == (T)0;
    Span<T> out = {lane_select(exact, nominal, outward.lo), lane_select(exact, nominal, outward.hi)};
    flag = flag || out.invalid();
    return out;
}

template <typename T>
__device__ __forceinline__ Span<T> tighten_mul(Span<T> a, Span<T> b, Span<T> outward, bool& flag) {
    typedef LaneTraits<T> L;
    bool candidate = a.lo == a.hi && b.lo == b.hi && a.finite_ordered() && b.finite_ordered() &&
                     outward.finite_ordered() && !(a.zero() || b.zero());
    T sa = lane_select(candidate, a.lo, (T)1), sb = lane_select(candidate, b.lo, (T)1);
    const T product = sa * sb;
    candidate = candidate && residual_exact(sa, sb, product);
    sa = lane_select(candidate, sa, (T)1);
    sb = lane_select(candidate, sb, (T)1);
    const T sp = lane_select(candidate, product, (T)1);
    return replace_exact(outward, candidate, product, L::fused(sa, sb, -sp), flag);
}

template <typename T>
__device__ __forceinline__ Span<T> tighten_div(Span<T> a, Span<T> b, Span<T> outward, bool& flag) {
    typedef LaneTraits<T> L;
    bool candidate = a.lo == a.hi && lane_guarded(a.lo) && b.lo == b.hi && lane_guarded(b.lo) &&
                     outward.finite_ordered() && b.lo > (T)0;
    T sa = lane_select(candidate, a.lo, (T)1), sb = lane_select(candidate, b.lo, (T)1);
    const T quotient = L::div(sa, sb);
    candidate = candidate && lane_guarded(quotient);
    sa = lane_select(candidate, sa, (T)1);
    sb = lane_select(candidate, sb, (T)1);
    const T sq = lane_select(candidate, quotient, (T)1);
    return replace_exact(outward, candidate, quotient, L::fused(-sq, sb, sa), flag);
}

template <typename T>
__device__ __forceinline__ Span<T> restrict_sign(Span<T> outward, bool nonnegative, bool nonpositive, bool& flag) {
    Span<T> out = {lane_select(nonnegative && outward.lo < (T)0, (T)0, outward.lo),
                   lane_select(nonpositive && outward.hi > (T)0, (T)0, outward.hi)};
    flag = flag || out.invalid();
    return out;
}

template <typename T>
__device__ __forceinline__ Span<T> restrict_mul_sign(Span<T> a, Span<T> b, Span<T> outward, bool& flag) {
    const bool ann = a.lo >= (T)0, anp = a.hi <= (T)0, bnn = b.lo >= (T)0, bnp = b.hi <= (T)0;
    return restrict_sign(outward, (ann && bnn) || (anp && bnp), (ann && bnp) || (anp && bnn), flag);
}

template <typename T>
__device__ __forceinline__ Span<T> restrict_div_sign(Span<T> a, Span<T> outward, bool& flag) {
    return restrict_sign(outward, a.lo >= (T)0, a.hi <= (T)0, flag);
}

// Operation codes shared with the Rust launcher.
#define INTERVAL_ADD 0
#define INTERVAL_SUB 1
#define INTERVAL_MUL 2
#define INTERVAL_DIV_POSITIVE 3
#define INTERVAL_DIV_POSITIVE_INTERVAL 4

template <typename T>
__device__ __forceinline__ Span<T> interval_lane(int op, Span<T> a, Span<T> b, bool tight, bool fused, bool& flag) {
    Span<T> out;
    switch (op) {
    case INTERVAL_ADD:
        out = add_outward(a, b, flag);
        return tight ? tighten_sum(a, b, out, false, flag) : out;
    case INTERVAL_SUB:
        out = sub_outward(a, b, flag);
        return tight ? tighten_sum(a, b, out, true, flag) : out;
    case INTERVAL_MUL:
        out = mul_outward(a, b, flag);
        if (!tight) return out;
        if (fused) out = tighten_mul(a, b, out, flag);
        return restrict_mul_sign(a, b, out, flag);
    case INTERVAL_DIV_POSITIVE:
        out = div_positive_outward(a, b.lo, flag);
        if (!tight) return out;
        if (fused) out = tighten_div(a, span_point(b.lo), out, flag);
        return restrict_div_sign(a, out, flag);
    default:
        out = div_positive_interval_outward(a, b, flag);
        if (!tight) return out;
        if (fused) out = tighten_div(a, b, out, flag);
        return restrict_div_sign(a, out, flag);
    }
}

#endif
