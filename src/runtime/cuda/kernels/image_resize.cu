// Separable pass kernels of resize (ImageOps::resize).
//
// This file is PTX module "image_resize" (IMAGE_RESIZE_MODULE in
// image_resize.rs).
//
// Kernel naming: resize_pass_{src}_{dst} for the element pairs a schedule
// uses: f32_f32, f64_f64, and for F16 and BF16 the widening pass {t}_f32,
// the narrowing pass f32_{t} and the single pass {t}_{t}. F64 accumulates in
// double, every other dtype in float.
//
// One thread owns one destination element, in a grid-stride loop with 64-bit
// indices. The linear index decodes into canonical (n, c, h, w): W fastest
// for NCHW order, C fastest for NHWC order (`channels_last`). The pass
// interpolates along `axis` (2 = H, 3 = W) and keeps the other coordinates.
// Both sides are addressed through canonical [N, C, H, W] strides in
// elements, which can be negative.
//
// Each output computes its own window and weights from its index, so no
// table is uploaded. The rules match the CPU kernels in
// runtime/cpu/kernels/image_resize_taps.rs bit for bit. The build passes
// --use_fast_math, so every arithmetic step uses an _rn intrinsic: nvcc
// neither contracts nor approximates those. The fused steps call the fma
// intrinsics. --ftz=true still flushes float subnormals.
//
// Modes (ResizeMode::kernel_mode):
//   0 copy, 1 linear, 2 cubic (a = -0.75), 3 AA linear, 4 AA cubic (a = -0.5)

#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename A>
struct ResizeMath;

template <>
struct ResizeMath<float> {
    static constexpr long long AA_BLOCK = 4;
    __device__ static __forceinline__ float add(float a, float b) { return __fadd_rn(a, b); }
    __device__ static __forceinline__ float sub(float a, float b) { return __fsub_rn(a, b); }
    __device__ static __forceinline__ float mul(float a, float b) { return __fmul_rn(a, b); }
    __device__ static __forceinline__ float div(float a, float b) { return __fdiv_rn(a, b); }
    __device__ static __forceinline__ float fma(float a, float b, float c) {
        return __fmaf_rn(a, b, c);
    }
    __device__ static __forceinline__ float from_ll(long long v) { return __ll2float_rn(v); }
    __device__ static __forceinline__ long long floor_index(float x) {
        return (long long)floorf(x);
    }
    __device__ static __forceinline__ long long ceil_count(float x) {
        return (long long)ceilf(x);
    }
    // scale * (i + 0.5) with the product in double, as the C++ literal promotes it.
    __device__ static __forceinline__ float aa_center(float scale, long long i) {
        return __double2float_rn(__dmul_rn((double)scale, __dadd_rn(__ll2double_rn(i), 0.5)));
    }
    __device__ static __forceinline__ long long aa_bound(float x) {
        return (long long)__dadd_rn((double)x, 0.5);
    }
    __device__ static __forceinline__ float aa_arg(float d, float invscale) {
        return __double2float_rn(__dmul_rn(__dadd_rn((double)d, 0.5), (double)invscale));
    }
};

template <>
struct ResizeMath<double> {
    static constexpr long long AA_BLOCK = 2;
    __device__ static __forceinline__ double add(double a, double b) { return __dadd_rn(a, b); }
    __device__ static __forceinline__ double sub(double a, double b) { return __dsub_rn(a, b); }
    __device__ static __forceinline__ double mul(double a, double b) { return __dmul_rn(a, b); }
    __device__ static __forceinline__ double div(double a, double b) { return __ddiv_rn(a, b); }
    __device__ static __forceinline__ double fma(double a, double b, double c) {
        return __fma_rn(a, b, c);
    }
    __device__ static __forceinline__ double from_ll(long long v) { return __ll2double_rn(v); }
    // PyTorch calls floorf on the double index, so it rounds to float first.
    __device__ static __forceinline__ long long floor_index(double x) {
        return (long long)floorf(__double2float_rn(x));
    }
    __device__ static __forceinline__ long long ceil_count(double x) {
        return (long long)ceil(x);
    }
    __device__ static __forceinline__ double aa_center(double scale, long long i) {
        return __dmul_rn(scale, __dadd_rn(__ll2double_rn(i), 0.5));
    }
    __device__ static __forceinline__ long long aa_bound(double x) {
        return (long long)__dadd_rn(x, 0.5);
    }
    __device__ static __forceinline__ double aa_arg(double d, double invscale) {
        return __dmul_rn(__dadd_rn(d, 0.5), invscale);
    }
};

__device__ __forceinline__ float resize_load(float v, float) { return v; }
__device__ __forceinline__ double resize_load(double v, double) { return v; }
__device__ __forceinline__ float resize_load(__half v, float) { return __half2float(v); }
__device__ __forceinline__ float resize_load(__nv_bfloat16 v, float) {
    return __bfloat162float(v);
}

template <typename O, typename A>
__device__ __forceinline__ O resize_store(A v);
template <>
__device__ __forceinline__ float resize_store<float, float>(float v) { return v; }
template <>
__device__ __forceinline__ double resize_store<double, double>(double v) { return v; }
template <>
__device__ __forceinline__ __half resize_store<__half, float>(float v) {
    return __float2half_rn(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 resize_store<__nv_bfloat16, float>(float v) {
    return __float2bfloat16_rn(v);
}

// std::max(a, b) and std::min(a, b).
template <typename A>
__device__ __forceinline__ A resize_cmax(A a, A b) { return a < b ? b : a; }
template <typename A>
__device__ __forceinline__ A resize_cmin(A a, A b) { return b < a ? b : a; }

// cubic_convolution1(x, -0.75), every step rounded.
template <typename A>
__device__ __forceinline__ A resize_cubic1(A x) {
    typedef ResizeMath<A> M;
    A t = M::sub(M::mul((A)1.25, x), (A)2.25);
    return M::add(M::mul(M::mul(t, x), x), (A)1.0);
}

// cubic_convolution2(x, -0.75), every step rounded.
template <typename A>
__device__ __forceinline__ A resize_cubic2(A x) {
    typedef ResizeMath<A> M;
    A u = M::sub(M::mul((A)-0.75, x), (A)-3.75);
    A v = M::add(M::mul(u, x), (A)-6.0);
    return M::sub(M::mul(v, x), (A)-3.0);
}

// HelperInterpLinear::aa_filter (mode 3) or HelperInterpCubic::aa_filter with
// a = -0.5 and the polynomial fused (mode 4).
template <typename A>
__device__ __forceinline__ A resize_aa_filter(A x, unsigned int mode) {
    typedef ResizeMath<A> M;
    x = x < (A)0 ? -x : x;
    if (mode == 3u) {
        return x < (A)1.0 ? M::sub((A)1.0, x) : (A)0.0;
    }
    if (x < (A)1.0) {
        A t = M::fma((A)1.5, x, (A)-2.5);
        return M::fma(M::mul(t, x), x, (A)1.0);
    }
    if (x < (A)2.0) {
        A u = M::fma((A)-0.5, x, (A)2.5);
        A v = M::fma(u, x, (A)-4.0);
        return M::fma(v, x, (A)2.0);
    }
    return (A)0.0;
}

// One destination value: interpolate src[base + k * step] over the window of
// output index `o`.
template <typename I, typename A>
__device__ __forceinline__ A resize_value(
    const I* __restrict__ src, long long base, long long step, unsigned int mode,
    long long o, long long in_size, long long out_size) {
    typedef ResizeMath<A> M;
    const A zero = (A)0.0;
    const A one = (A)1.0;
    const A half = (A)0.5;
    const long long last = in_size - 1;
#define RESIZE_X(k) resize_load(src[base + (k) * step], (A)0)

    if (mode == 0u) {
        return RESIZE_X(o);
    }
    const A scale = M::div(M::from_ll(in_size), M::from_ll(out_size));

    if (mode == 1u) {
        long long i0 = o;
        long long i1 = o;
        A w0 = one;
        A w1 = zero;
        if (in_size != out_size) {
            A s = M::sub(M::mul(scale, M::add(M::from_ll(o), half)), half);
            s = s < zero ? zero : s;
            long long fl = M::floor_index(s);
            i0 = fl < last ? fl : last;
            w1 = resize_cmin(resize_cmax(M::sub(s, M::from_ll(i0)), zero), one);
            i1 = i0 + (i0 < last ? 1 : 0);
            w0 = M::sub(one, w1);
        }
        return M::fma(RESIZE_X(i0), w0, M::mul(RESIZE_X(i1), w1));
    }

    if (mode == 2u) {
        A s = M::fma(scale, M::add(M::from_ll(o), half), -half);
        long long fl = M::floor_index(s);
        long long i = fl < last ? fl : last;
        A t = resize_cmin(resize_cmax(M::sub(s, M::from_ll(i)), zero), one);
        A t2 = M::sub(one, t);
        A c[4] = {resize_cubic2(M::add(t, one)), resize_cubic1(t), resize_cubic1(t2),
                  resize_cubic2(M::add(t2, one))};
        long long k[4];
        for (int j = 0; j < 4; ++j) {
            long long v = i + j - 1;
            k[j] = v < 0 ? 0 : (v > last ? last : v);
        }
        A acc = M::fma(RESIZE_X(k[0]), c[0], M::mul(RESIZE_X(k[1]), c[1]));
        acc = M::fma(RESIZE_X(k[2]), c[2], acc);
        return M::fma(RESIZE_X(k[3]), c[3], acc);
    }

    // Antialias window: HelperInterpBase::_compute_indices_min_size_weights_aa.
    const bool wide = scale >= one;
    const A half_taps = mode == 3u ? (A)1.0 : (A)2.0;
    const A support = wide ? M::mul(half_taps, scale) : half_taps;
    const long long max_taps = M::ceil_count(support) * 2 + 1;
    const A invscale = wide ? M::div(one, scale) : one;
    const A center = M::aa_center(scale, o);
    long long xmin = M::aa_bound(M::sub(center, support));
    xmin = xmin < 0 ? 0 : xmin;
    long long xmax = M::aa_bound(M::add(center, support));
    xmax = xmax < in_size ? xmax : in_size;
    long long xsize = xmax - xmin;
    xsize = xsize < 0 ? 0 : (xsize > max_taps ? max_taps : xsize);

    A total = zero;
    for (long long j = 0; j < xsize; ++j) {
        A arg = M::aa_arg(M::sub(M::from_ll(xmin + j), center), invscale);
        total = M::add(total, resize_aa_filter(arg, mode));
    }
    const bool normalise = total != zero;
    const long long blocked = xsize > 0 ? M::AA_BLOCK * ((xsize - 1) / M::AA_BLOCK) : 0;
    A acc = zero;
    for (long long j = 0; j < xsize; ++j) {
        A w = resize_aa_filter(M::aa_arg(M::sub(M::from_ll(xmin + j), center), invscale), mode);
        if (normalise) {
            w = M::div(w, total);
        }
        A x = RESIZE_X(xmin + j);
        if (j == 0) {
            acc = M::mul(x, w);
        } else if (j <= blocked) {
            acc = M::add(acc, M::mul(x, w));
        } else {
            acc = M::fma(x, w, acc);
        }
    }
    return acc;
#undef RESIZE_X
}

template <typename I, typename O, typename A>
__device__ __forceinline__ void resize_pass(
    const I* __restrict__ src, O* __restrict__ dst, unsigned long long total,
    unsigned long long dim_c, unsigned long long dim_h, unsigned long long dim_w,
    long long ss0, long long ss1, long long ss2, long long ss3,
    long long ds0, long long ds1, long long ds2, long long ds3,
    unsigned int channels_last, unsigned int axis, unsigned int mode,
    long long in_size, long long out_size) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        unsigned long long n, c, h, w, rest;
        if (channels_last) {
            c = i % dim_c;
            rest = i / dim_c;
            w = rest % dim_w;
            rest /= dim_w;
            h = rest % dim_h;
            n = rest / dim_h;
        } else {
            w = i % dim_w;
            rest = i / dim_w;
            h = rest % dim_h;
            rest /= dim_h;
            c = rest % dim_c;
            n = rest / dim_c;
        }
        const bool along_h = axis == 2u;
        long long o = along_h ? (long long)h : (long long)w;
        long long base = (long long)n * ss0 + (long long)c * ss1 +
                         (along_h ? 0 : (long long)h * ss2) + (along_h ? (long long)w * ss3 : 0);
        long long step = along_h ? ss2 : ss3;
        A v = resize_value<I, A>(src, base, step, mode, o, in_size, out_size);
        long long d = (long long)n * ds0 + (long long)c * ds1 + (long long)h * ds2 +
                      (long long)w * ds3;
        dst[d] = resize_store<O, A>(v);
    }
}

#define NUMR_RESIZE_PARAMS(I, O)                                                      \
    const I* __restrict__ src, O* __restrict__ dst, unsigned long long total,         \
    unsigned long long dim_c, unsigned long long dim_h, unsigned long long dim_w,     \
    long long ss0, long long ss1, long long ss2, long long ss3,                       \
    long long ds0, long long ds1, long long ds2, long long ds3,                       \
    unsigned int channels_last, unsigned int axis, unsigned int mode,                 \
    long long in_size, long long out_size

#define DEFINE_RESIZE_PASS(name, I, O, A)                                             \
__global__ void resize_pass_##name(NUMR_RESIZE_PARAMS(I, O)) {                        \
    resize_pass<I, O, A>(src, dst, total, dim_c, dim_h, dim_w, ss0, ss1, ss2, ss3,    \
                         ds0, ds1, ds2, ds3, channels_last, axis, mode, in_size,      \
                         out_size);                                                   \
}

extern "C" {

DEFINE_RESIZE_PASS(f32_f32, float, float, float)
DEFINE_RESIZE_PASS(f64_f64, double, double, double)
DEFINE_RESIZE_PASS(f16_f32, __half, float, float)
DEFINE_RESIZE_PASS(f32_f16, float, __half, float)
DEFINE_RESIZE_PASS(f16_f16, __half, __half, float)
DEFINE_RESIZE_PASS(bf16_f32, __nv_bfloat16, float, float)
DEFINE_RESIZE_PASS(f32_bf16, float, __nv_bfloat16, float)
DEFINE_RESIZE_PASS(bf16_bf16, __nv_bfloat16, __nv_bfloat16, float)

} // extern "C"
