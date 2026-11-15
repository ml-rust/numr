// Single-pass interval and expansion kernels with one shared status flag.
//
// Each operand advances by `step` elements per lane: zero repeats one element,
// one walks a contiguous buffer. Any flagged lane sets *status to one.

#include "expansion_lane.cuh"

template <typename T> struct Lane {
    const T* data;
    unsigned int step;
    __device__ __forceinline__ T operator[](unsigned int i) const { return data[(unsigned long long)i * step]; }
};

__device__ __forceinline__ void raise_status(unsigned int* status, bool flag) {
    if (flag) atomicOr(status, 1u);
}

#define LANE_INDEX                                                    \
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;     \
    if (i >= n) return;

template <typename T>
__device__ void interval_binary(int op, int tight, int fused, Lane<T> alo, Lane<T> ahi, Lane<T> blo, Lane<T> bhi,
                                T* lo, T* hi, unsigned int* status, unsigned int n) {
    LANE_INDEX
    bool flag = false;
    const Span<T> a = {alo[i], ahi[i]};
    const Span<T> b = {blo[i], bhi[i]};
    const Span<T> out = interval_lane(op, a, b, tight != 0, fused != 0, flag);
    lo[i] = out.lo;
    hi[i] = out.hi;
    raise_status(status, flag);
}

template <typename T>
__device__ void interval_sqrt(Lane<T> alo, Lane<T> ahi, T* lo, T* hi, unsigned int* status, unsigned int n) {
    LANE_INDEX
    bool flag = false;
    const Span<T> a = {alo[i], ahi[i]};
    const Span<T> out = sqrt_nonnegative(a, flag);
    lo[i] = out.lo;
    hi[i] = out.hi;
    raise_status(status, flag);
}

// Bit copies keep NaN payloads; float views only compute the status.
template <typename T>
__device__ void interval_capture(Lane<T> alo, Lane<T> ahi, T* lo, T* hi, int same, unsigned int* status,
                                 unsigned int n) {
    LANE_INDEX
    typedef typename LaneTraits<T>::Bits B;
    const Span<T> a = {alo[i], ahi[i]};
    reinterpret_cast<B*>(lo)[i] = LaneTraits<T>::bits(a.lo);
    if (!same) reinterpret_cast<B*>(hi)[i] = LaneTraits<T>::bits(a.hi);
    raise_status(status, a.invalid());
}

template <typename T>
__device__ void interval_flag(Lane<T> alo, Lane<T> ahi, int positive, unsigned int* status, unsigned int n) {
    LANE_INDEX
    const Span<T> a = {alo[i], ahi[i]};
    raise_status(status, a.invalid() || (positive != 0 && !(a.lo > (T)0)));
}

template <typename T>
__device__ void flag_nonzero(Lane<T> mask, unsigned int* status, unsigned int n) {
    LANE_INDEX
    raise_status(status, mask[i] != (T)0);
}

// Expansion operation codes shared with the Rust launcher.
#define EXPANSION_ADD 0
#define EXPANSION_SUB 1
#define EXPANSION_MUL 2
#define EXPANSION_DIV_POSITIVE 3
#define EXPANSION_POINT 4
#define EXPANSION_NORMALIZE 5
#define EXPANSION_FLAG 6
#define EXPANSION_PROJECT 7
#define EXPANSION_LIFT 8

template <typename T>
__device__ void expansion(int op, Lane<T> ah, Lane<T> al, Lane<T> ae, Lane<T> bh, Lane<T> bl, Lane<T> be, T* oh,
                          T* ol, T* oe, unsigned int* status, unsigned int n) {
    LANE_INDEX
    bool flag = false;
    const Parts<T> a = make_parts(ah[i], al[i], ae[i]);
    Parts<T> out;
    switch (op) {
    case EXPANSION_ADD: out = expansion_sum(a, make_parts(bh[i], bl[i], be[i]), false, flag); break;
    case EXPANSION_SUB: out = expansion_sum(a, make_parts(bh[i], bl[i], be[i]), true, flag); break;
    case EXPANSION_MUL: out = expansion_product(a, make_parts(bh[i], bl[i], be[i]), flag); break;
    case EXPANSION_DIV_POSITIVE: out = expansion_quotient(a, make_parts(bh[i], bl[i], be[i]), flag); break;
    case EXPANSION_POINT:
        out = make_parts(a.high, (T)0, (T)0);
        flag = !lane_finite(a.high);
        break;
    case EXPANSION_NORMALIZE: out = normalize(a, flag); break;
    case EXPANSION_FLAG:
        raise_status(status, a.invalid());
        return;
    case EXPANSION_PROJECT: {
        const Span<T> s = expansion_project(a, flag);
        oh[i] = s.lo;
        ol[i] = s.hi;
        raise_status(status, flag);
        return;
    }
    default: {
        const Span<T> v = {a.high, a.low};
        out = expansion_lift(v, flag);
        break;
    }
    }
    oh[i] = out.high;
    ol[i] = out.low;
    oe[i] = out.error;
    raise_status(status, flag);
}

// Envelope probes take runtime operands, so the compiler cannot fold them.
template <typename T>
__device__ void envelope(T epsilon, T eta, T minimum_normal, T even_next, T two_epsilon, unsigned int* out) {
    typedef LaneTraits<T> L;
    const T zero = (T)0, one = (T)1, two = (T)2, half = (T)0.5;
    const T half_epsilon = L::mul(epsilon, half);
    const T odd = one + epsilon;
    T ph, pl, nh, nl, rh, rl, ch, cl;
    lane_two_sum(one, half_epsilon, ph, pl);
    lane_two_sum(-one, -half_epsilon, nh, nl);
    lane_two_sum(one, -eta, rh, rl);
    lane_two_sum(odd, -one, ch, cl);
    const bool failures[8] = {
        one + half_epsilon != one || odd + half_epsilon != even_next,
        -one + -half_epsilon != -one || -odd + -half_epsilon != -even_next,
        L::div(eta + eta, minimum_normal) != two_epsilon,
        L::div(L::mul(minimum_normal, half), minimum_normal) != half,
        L::div(L::div(minimum_normal, two), minimum_normal) != half,
        ph != one || pl != half_epsilon,
        nh != -one || nl != -half_epsilon,
        rh != one || L::div(rl, minimum_normal) != -epsilon || ch != epsilon || cl != zero,
    };
    unsigned int bits = 0;
    for (int k = 0; k < 8; ++k) bits |= (failures[k] ? 1u : 0u) << k;
    *out = bits;
}

#define INTERVAL_KERNELS(T, SUFFIX)                                                                              \
    __global__ void interval_binary_##SUFFIX(int op, int tight, int fused, const T* alo, unsigned int alo_step, \
                                             const T* ahi, unsigned int ahi_step, const T* blo,                   \
                                             unsigned int blo_step, const T* bhi, unsigned int bhi_step, T* lo,  \
                                             T* hi, unsigned int* status, unsigned int n) {                       \
        interval_binary<T>(op, tight, fused, Lane<T>{alo, alo_step}, Lane<T>{ahi, ahi_step},                     \
                           Lane<T>{blo, blo_step}, Lane<T>{bhi, bhi_step}, lo, hi, status, n);                   \
    }                                                                                                            \
    __global__ void interval_sqrt_##SUFFIX(const T* alo, unsigned int alo_step, const T* ahi,                    \
                                           unsigned int ahi_step, T* lo, T* hi, unsigned int* status,            \
                                           unsigned int n) {                                                     \
        interval_sqrt<T>(Lane<T>{alo, alo_step}, Lane<T>{ahi, ahi_step}, lo, hi, status, n);                     \
    }                                                                                                            \
    __global__ void interval_capture_##SUFFIX(const T* alo, unsigned int alo_step, const T* ahi,                 \
                                              unsigned int ahi_step, T* lo, T* hi, int same,                     \
                                              unsigned int* status, unsigned int n) {                            \
        interval_capture<T>(Lane<T>{alo, alo_step}, Lane<T>{ahi, ahi_step}, lo, hi, same, status, n);           \
    }                                                                                                            \
    __global__ void interval_flag_##SUFFIX(const T* alo, unsigned int alo_step, const T* ahi,                    \
                                           unsigned int ahi_step, int positive, unsigned int* status,            \
                                           unsigned int n) {                                                     \
        interval_flag<T>(Lane<T>{alo, alo_step}, Lane<T>{ahi, ahi_step}, positive, status, n);                  \
    }                                                                                                            \
    __global__ void expansion_##SUFFIX(int op, const T* ah, unsigned int ah_step, const T* al,                   \
                                       unsigned int al_step, const T* ae, unsigned int ae_step, const T* bh,     \
                                       unsigned int bh_step, const T* bl, unsigned int bl_step, const T* be,     \
                                       unsigned int be_step, T* oh, T* ol, T* oe, unsigned int* status,          \
                                       unsigned int n) {                                                         \
        expansion<T>(op, Lane<T>{ah, ah_step}, Lane<T>{al, al_step}, Lane<T>{ae, ae_step}, Lane<T>{bh, bh_step}, \
                     Lane<T>{bl, bl_step}, Lane<T>{be, be_step}, oh, ol, oe, status, n);                         \
    }                                                                                                            \
    __global__ void interval_envelope_##SUFFIX(T epsilon, T eta, T minimum_normal, T even_next, T two_epsilon,   \
                                               unsigned int* out) {                                              \
        if (blockIdx.x == 0 && threadIdx.x == 0) envelope<T>(epsilon, eta, minimum_normal, even_next,            \
                                                             two_epsilon, out);                                  \
    }

#define FLAG_NONZERO_KERNEL(T, SUFFIX)                                                                    \
    __global__ void interval_flag_nonzero_##SUFFIX(const T* mask, unsigned int step, unsigned int* status, \
                                                   unsigned int n) {                                      \
        flag_nonzero<T>(Lane<T>{mask, step}, status, n);                                                  \
    }

extern "C" {
INTERVAL_KERNELS(float, f32)
INTERVAL_KERNELS(double, f64)
FLAG_NONZERO_KERNEL(float, f32)
FLAG_NONZERO_KERNEL(double, f64)
FLAG_NONZERO_KERNEL(long long, i64)
FLAG_NONZERO_KERNEL(int, i32)
FLAG_NONZERO_KERNEL(unsigned int, u32)
FLAG_NONZERO_KERNEL(unsigned char, u8)
}
