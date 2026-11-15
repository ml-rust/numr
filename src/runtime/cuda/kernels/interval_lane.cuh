// Per-lane interval rules. Each function mirrors the CPU lane code in
// src/runtime/cpu/kernels/interval/, which matches the composed tensor
// reference bit for bit. Every arithmetic expression uses the same intrinsic
// as the matching standalone CUDA kernel, and no product feeds an addition,
// unless an opaque barrier separates them, so contraction cannot fuse a step
// that the composed reference rounds.

#ifndef NUMR_INTERVAL_LANE_CUH
#define NUMR_INTERVAL_LANE_CUH

#include <stdint.h>

template <typename T> struct LaneTraits;

template <> struct LaneTraits<float> {
    typedef unsigned int Bits;
    static __device__ __forceinline__ float guard_min() { return __uint_as_float((127u - 40u) << 23); }
    static __device__ __forceinline__ float guard_max() { return __uint_as_float((127u + 40u) << 23); }
    static __device__ __forceinline__ float half_max() { return __uint_as_float(0x7effffffu); }
    static __device__ __forceinline__ float div(float a, float b) { return __fdiv_rn(a, b); }
    // The empty asm hides the product, so it never contracts into a following addition.
    static __device__ __forceinline__ float mul(float a, float b) { float p = a * b; asm volatile("" : "+f"(p)); return p; }
    static __device__ __forceinline__ float root(float a) { return __fsqrt_rn(a); }
    static __device__ __forceinline__ float fused(float a, float b, float c) { return __fmaf_rn(a, b, c); }
    static __device__ __forceinline__ float truncated(float a) { return truncf(a); }
    static __device__ __forceinline__ float magnitude(float a) { return fabsf(a); }
    static __device__ __forceinline__ Bits bits(float a) { return __float_as_uint(a); }
    static __device__ __forceinline__ float from_bits(Bits a) { return __uint_as_float(a); }
    static const Bits SIGN = 0x80000000u;
    static const Bits INF = 0x7f800000u;
};

template <> struct LaneTraits<double> {
    typedef unsigned long long Bits;
    static __device__ __forceinline__ double guard_min() { return __longlong_as_double((long long)((1023ull - 450ull) << 52)); }
    static __device__ __forceinline__ double guard_max() { return __longlong_as_double((long long)((1023ull + 450ull) << 52)); }
    static __device__ __forceinline__ double half_max() { return __longlong_as_double(0x7fdfffffffffffffll); }
    static __device__ __forceinline__ double div(double a, double b) { return a / b; }
    static __device__ __forceinline__ double mul(double a, double b) { double p = a * b; asm volatile("" : "+d"(p)); return p; }
    static __device__ __forceinline__ double root(double a) { return __dsqrt_rn(a); }
    static __device__ __forceinline__ double fused(double a, double b, double c) { return __fma_rn(a, b, c); }
    static __device__ __forceinline__ double truncated(double a) { return trunc(a); }
    static __device__ __forceinline__ double magnitude(double a) { return fabs(a); }
    static __device__ __forceinline__ Bits bits(double a) { return (Bits)__double_as_longlong(a); }
    static __device__ __forceinline__ double from_bits(Bits a) { return __longlong_as_double((long long)a); }
    static const Bits SIGN = 0x8000000000000000ull;
    static const Bits INF = 0x7ff0000000000000ull;
};

// Integer stepping, identical to next_representable.cu.
template <typename T, bool UP>
__device__ __forceinline__ T lane_step(T value) {
    typedef LaneTraits<T> L;
    typename L::Bits bits = L::bits(value);
    const typename L::Bits magnitude = bits & ~L::SIGN;
    if (magnitude > L::INF) return value;
    if (magnitude == 0) return L::from_bits(UP ? 1 : (L::SIGN | 1));
    if (UP && bits == L::INF) return value;
    if (!UP && bits == (L::SIGN | L::INF)) return value;
    bits = ((bits & L::SIGN) != 0) == UP ? bits - 1 : bits + 1;
    return L::from_bits(bits);
}
template <typename T> __device__ __forceinline__ T lane_up(T v) { return lane_step<T, true>(v); }
template <typename T> __device__ __forceinline__ T lane_down(T v) { return lane_step<T, false>(v); }

template <typename T> __device__ __forceinline__ bool lane_finite(T v) { return !isnan(v) && !isinf(v); }
template <typename T> __device__ __forceinline__ T lane_max(T a, T b) { return a > b ? a : b; }
template <typename T> __device__ __forceinline__ T lane_min(T a, T b) { return a < b ? a : b; }
template <typename T> __device__ __forceinline__ T lane_select(bool c, T x, T y) { return c ? x : y; }

template <typename T>
__device__ __forceinline__ void lane_two_sum(T a, T b, T& sum, T& residual) {
    sum = a + b;
    const T b_virtual = sum - a;
    const T a_virtual = sum - b_virtual;
    const T b_roundoff = b - b_virtual;
    const T a_roundoff = a - a_virtual;
    residual = a_roundoff + b_roundoff;
}

template <typename T>
__device__ __forceinline__ bool lane_guarded(T v) {
    const T m = LaneTraits<T>::magnitude(v);
    return m >= LaneTraits<T>::guard_min() && m <= LaneTraits<T>::guard_max();
}

template <typename T>
__device__ __forceinline__ T lane_midpoint(T a, T b) {
    typedef LaneTraits<T> L;
    const T half = (T)0.5;
    const bool small = L::magnitude(a) <= L::half_max() && L::magnitude(b) <= L::half_max();
    const T sum = L::mul(lane_select(small, a, (T)0) + lane_select(small, b, (T)0), half);
    return lane_select(small, sum, L::mul(a, half) + L::mul(b, half));
}

template <typename T> struct Span {
    T lo;
    T hi;
    __device__ __forceinline__ bool zero() const { return lo == (T)0 && hi == (T)0; }
    __device__ __forceinline__ bool finite_ordered() const {
        return lane_finite(lo) && lane_finite(hi) && lo <= hi;
    }
    __device__ __forceinline__ bool invalid() const {
        return !lane_finite(lo) || !lane_finite(hi) || lo > hi;
    }
};

template <typename T> __device__ __forceinline__ Span<T> span_point(T v) { Span<T> s = {v, v}; return s; }

template <typename T>
__device__ __forceinline__ Span<T> add_outward(Span<T> a, Span<T> b, bool& flag) {
    const bool az = a.zero(), bz = b.zero();
    const T lo = lane_down(a.lo + b.lo);
    const T hi = lane_up(a.hi + b.hi);
    Span<T> out = {lane_select(az, b.lo, lane_select(bz, a.lo, lo)),
                   lane_select(az, b.hi, lane_select(bz, a.hi, hi))};
    flag = flag || out.invalid();
    return out;
}

template <typename T>
__device__ __forceinline__ Span<T> sub_outward(Span<T> a, Span<T> b, bool& flag) {
    const bool az = a.zero(), bz = b.zero();
    const T lo = lane_down(a.lo - b.hi);
    const T hi = lane_up(a.hi - b.lo);
    Span<T> out = {lane_select(bz, a.lo, lane_select(az, -b.hi, lo)),
                   lane_select(bz, a.hi, lane_select(az, -b.lo, hi))};
    flag = flag || out.invalid();
    return out;
}

template <typename T>
__device__ __forceinline__ Span<T> corners(T ll, T lu, T ul, T uu) {
    Span<T> s = {lane_down(lane_min(lane_min(ll, lu), lane_min(ul, uu))),
                 lane_up(lane_max(lane_max(ll, lu), lane_max(ul, uu)))};
    return s;
}

template <typename T>
__device__ __forceinline__ Span<T> mul_outward(Span<T> a, Span<T> b, bool& flag) {
    const bool az = a.zero(), bz = b.zero();
    const bool zp = az || bz;
    const T al = lane_select(zp, (T)0, a.lo), au = lane_select(zp, (T)0, a.hi);
    const T bl = lane_select(zp, (T)1, b.lo), bu = lane_select(zp, (T)1, b.hi);
    const Span<T> c = corners(al * bl, al * bu, au * bl, au * bu);
    Span<T> out = {lane_select(zp, lane_select(az, a.lo, b.lo), c.lo),
                   lane_select(zp, lane_select(az, a.hi, b.hi), c.hi)};
    flag = flag || out.invalid();
    return out;
}

template <typename T>
__device__ __forceinline__ Span<T> div_positive_outward(Span<T> a, T d, bool& flag) {
    typedef LaneTraits<T> L;
    flag = flag || !lane_finite(d) || !(d > (T)0);
    const T den0 = lane_select(d > (T)0, d, (T)1);
    const bool identity = d == (T)1;
    const T al = lane_select(identity, (T)1, a.lo), au = lane_select(identity, (T)1, a.hi);
    const T den = lane_select(identity, (T)1, den0);
    const bool exact = a.zero() || identity;
    Span<T> out = {lane_select(exact, a.lo, lane_down(L::div(al, den))),
                   lane_select(exact, a.hi, lane_up(L::div(au, den)))};
    flag = flag || out.invalid();
    return out;
}

template <typename T>
__device__ __forceinline__ Span<T> div_positive_interval_outward(Span<T> a, Span<T> b, bool& flag) {
    typedef LaneTraits<T> L;
    const bool av = a.finite_ordered();
    const bool bv = b.finite_ordered() && b.lo > (T)0;
    flag = flag || !av || !bv;
    const bool identity = b.lo == (T)1 && b.hi == (T)1;
    const T al = lane_select(identity, (T)1, lane_select(av, a.lo, (T)0));
    const T au = lane_select(identity, (T)1, lane_select(av, a.hi, (T)0));
    const T bl = lane_select(identity, (T)1, lane_select(bv, b.lo, (T)1));
    const T bu = lane_select(identity, (T)1, lane_select(bv, b.hi, (T)1));
    const Span<T> c = corners(L::div(al, bl), L::div(al, bu), L::div(au, bl), L::div(au, bu));
    const bool exact = a.zero() || identity;
    Span<T> out = {lane_select(exact, a.lo, c.lo), lane_select(exact, a.hi, c.hi)};
    flag = flag || out.invalid();
    return out;
}

template <typename T>
__device__ __forceinline__ Span<T> sqrt_nonnegative(Span<T> a, bool& flag) {
    typedef LaneTraits<T> L;
    const bool valid = a.finite_ordered() && a.lo >= (T)0;
    const T li = lane_select(valid, a.lo, (T)0), ui = lane_select(valid, a.hi, (T)0);
    Span<T> out = {lane_select(li == (T)0, li, lane_down(L::root(li))),
                   lane_select(ui == (T)0, ui, lane_up(L::root(ui)))};
    flag = flag || !valid || out.invalid();
    return out;
}

#include "interval_tighten.cuh"

#endif
