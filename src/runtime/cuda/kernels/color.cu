// Colour conversion kernels of ColorOps.
//
// This file is PTX module "color" (COLOR_MODULE in color.rs).
//
// Kernel naming:
//   yuv_to_rgb_{p}_{o}     4:2:0 planes of p samples to RGB of o elements
//   rgb_to_yuv_{i}_{p}     RGB of i elements to 4:2:0 planes of p samples
//   rgb24_to_float_{o}     U8 RGB24 to RGB of o elements
//   float_to_rgb24_{i}     RGB of i elements to U8 RGB24
// with p in u8, u16 and i, o in f32, f64, f16, bf16.
//
// Every kernel takes one argument struct by value, a grid-stride loop and
// 64-bit indices. Every view is addressed through its strides in elements,
// which can be negative. An NV12 or P010 chroma plane arrives as a U view and
// a V view with doubled column strides, so one kernel serves every format.
//
// The math matches the CPU kernels in runtime/cpu/kernels/color.rs bit for
// bit. The build passes --use_fast_math, so every arithmetic step uses an
// _rn intrinsic: nvcc neither contracts nor approximates those. fminf, fmaxf
// and rintf are exact.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Matches ColorCoeffs in ops/common/color_plan.rs.
struct ColorCoeffs {
    int y_off;
    int c_off;
    float c_div16;
    float cr_r;
    float cb_b;
    float g_cb;
    float g_cr;
    float kr;
    float kg;
    float kb;
    float y_mul;
    float c_mul;
    float y_lo;
    float y_hi;
    float c_lo;
    float c_hi;
};

// Matches ColorPlaneArg in color.rs.
struct ColorPlane {
    unsigned long long addr;
    long long row_stride;
    long long col_stride;
};

// Matches YuvKernelArgs in color.rs.
struct YuvArgs {
    ColorPlane y;
    ColorPlane u;
    ColorPlane v;
    unsigned long long rgb;
    long long rgb_c;
    long long rgb_h;
    long long rgb_w;
    long long height;
    long long width;
    ColorCoeffs k;
    unsigned int shift;
    unsigned int center;
};

// Matches Rgb24KernelArgs in color.rs.
struct Rgb24Args {
    unsigned long long src;
    unsigned long long dst;
    long long src_strides[3];
    long long dst_strides[3];
    long long height;
    long long width;
};

__device__ __forceinline__ float color_load(float v) { return v; }
__device__ __forceinline__ float color_load(double v) { return __double2float_rn(v); }
__device__ __forceinline__ float color_load(__half v) { return __half2float(v); }
__device__ __forceinline__ float color_load(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ void color_store(float* p, float v) { *p = v; }
__device__ __forceinline__ void color_store(double* p, float v) { *p = (double)v; }
__device__ __forceinline__ void color_store(__half* p, float v) { *p = __float2half_rn(v); }
__device__ __forceinline__ void color_store(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16_rn(v);
}

// min(max(x, 0), 1). A NaN gives 0.
__device__ __forceinline__ float color_clamp01(float x) { return fminf(fmaxf(x, 0.0f), 1.0f); }

__device__ __forceinline__ long long color_at(const ColorPlane& p, long long r, long long c) {
    return r * p.row_stride + c * p.col_stride;
}

template <typename P>
__device__ __forceinline__ int color_value(const ColorPlane& p, long long r, long long c,
                                           unsigned int shift) {
    return (int)((unsigned int)((const P*)p.addr)[color_at(p, r, c)] >> shift);
}

template <typename P>
__device__ __forceinline__ void color_write(const ColorPlane& p, long long r, long long c,
                                            float code, unsigned int shift) {
    ((P*)p.addr)[color_at(p, r, c)] = (P)((unsigned int)code << shift);
}

// Chroma indices and weights in quarters for luma index `pos` along an axis
// of `len` chroma samples.
__device__ __forceinline__ void color_up_taps(long long pos, long long len, bool center,
                                              long long idx[2], int wt[2]) {
    long long k = pos >> 1;
    long long next = k + 1 < len ? k + 1 : len - 1;
    bool even = (pos & 1) == 0;
    if (center) {
        idx[0] = even ? (k > 0 ? k - 1 : 0) : k;
        idx[1] = even ? k : next;
        wt[0] = even ? 1 : 3;
        wt[1] = even ? 3 : 1;
    } else {
        idx[0] = k;
        idx[1] = even ? k : next;
        wt[0] = even ? 4 : 2;
        wt[1] = even ? 0 : 2;
    }
}

__device__ __forceinline__ float color_norm(int code, int off, float div) {
    return __fdiv_rn(__int2float_rn(code - off), div);
}

template <typename P, typename O>
__device__ __forceinline__ void yuv_to_rgb(const YuvArgs& a) {
    const unsigned long long total = (unsigned long long)(a.height * a.width);
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    const ColorCoeffs& k = a.k;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        long long row = (long long)(i / (unsigned long long)a.width);
        long long col = (long long)(i % (unsigned long long)a.width);
        long long rows[2], cols[2];
        int wv[2], wh[2];
        color_up_taps(row, a.height / 2, true, rows, wv);
        color_up_taps(col, a.width / 2, a.center != 0u, cols, wh);
        int cb16 = 0, cr16 = 0;
        for (int p = 0; p < 2; ++p) {
            for (int q = 0; q < 2; ++q) {
                int wt = wv[p] * wh[q];
                cb16 += wt * color_value<P>(a.u, rows[p], cols[q], a.shift);
                cr16 += wt * color_value<P>(a.v, rows[p], cols[q], a.shift);
            }
        }
        float yn = color_norm(color_value<P>(a.y, row, col, a.shift), k.y_off, k.y_mul);
        float cb = color_norm(cb16, 16 * k.c_off, k.c_div16);
        float cr = color_norm(cr16, 16 * k.c_off, k.c_div16);
        float r = __fadd_rn(yn, __fmul_rn(k.cr_r, cr));
        float b = __fadd_rn(yn, __fmul_rn(k.cb_b, cb));
        float g = __fadd_rn(__fadd_rn(yn, __fmul_rn(k.g_cb, cb)), __fmul_rn(k.g_cr, cr));
        O* out = (O*)a.rgb + row * a.rgb_h + col * a.rgb_w;
        color_store(out, color_clamp01(r));
        color_store(out + a.rgb_c, color_clamp01(g));
        color_store(out + 2 * a.rgb_c, color_clamp01(b));
    }
}

// round_ties_even(clamp(v * mul + off, lo, hi)).
__device__ __forceinline__ float color_quantize(float v, float mul, int off, float lo, float hi) {
    return rintf(fminf(fmaxf(__fadd_rn(__fmul_rn(v, mul), __int2float_rn(off)), lo), hi));
}

// Encode pixel (row, col): write its luma, return Cb' and Cr'.
template <typename I, typename P>
__device__ __forceinline__ void color_encode(const YuvArgs& a, long long row, long long col,
                                             bool store_luma, float* cb, float* cr) {
    const ColorCoeffs& k = a.k;
    const I* px = (const I*)a.rgb + row * a.rgb_h + col * a.rgb_w;
    float r = color_clamp01(color_load(px[0]));
    float g = color_clamp01(color_load(px[a.rgb_c]));
    float b = color_clamp01(color_load(px[2 * a.rgb_c]));
    float y = __fadd_rn(__fadd_rn(__fmul_rn(k.kr, r), __fmul_rn(k.kg, g)), __fmul_rn(k.kb, b));
    *cb = __fdiv_rn(__fsub_rn(b, y), k.cb_b);
    *cr = __fdiv_rn(__fsub_rn(r, y), k.cr_r);
    if (store_luma) {
        color_write<P>(a.y, row, col, color_quantize(y, k.y_mul, k.y_off, k.y_lo, k.y_hi),
                       a.shift);
    }
}

// Encode luma row `row` at columns 2j - 1 (clamped), 2j and 2j + 1.
template <typename I, typename P>
__device__ __forceinline__ void color_encode_row(const YuvArgs& a, long long row, long long j,
                                                 float cb[3], float cr[3]) {
    color_encode<I, P>(a, row, 2 * j, true, &cb[1], &cr[1]);
    color_encode<I, P>(a, row, 2 * j + 1, true, &cb[2], &cr[2]);
    if (a.center == 0u) {
        if (j == 0) {
            cb[0] = cb[1];
            cr[0] = cr[1];
        } else {
            color_encode<I, P>(a, row, 2 * j - 1, false, &cb[0], &cr[0]);
        }
    }
}

__device__ __forceinline__ float color_down(const float t[3], const float b[3], bool center) {
    if (center) {
        return __fmul_rn(__fadd_rn(__fadd_rn(t[1], t[2]), __fadd_rn(b[1], b[2])), 0.25f);
    }
    float ht = __fadd_rn(__fadd_rn(t[0], __fmul_rn(2.0f, t[1])), t[2]);
    float hb = __fadd_rn(__fadd_rn(b[0], __fmul_rn(2.0f, b[1])), b[2]);
    return __fmul_rn(__fadd_rn(ht, hb), 0.125f);
}

template <typename I, typename P>
__device__ __forceinline__ void rgb_to_yuv(const YuvArgs& a) {
    const long long cw = a.width / 2;
    const unsigned long long total = (unsigned long long)((a.height / 2) * cw);
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    const ColorCoeffs& k = a.k;
    const bool center = a.center != 0u;
    for (unsigned long long n = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         n < total; n += stride) {
        long long i = (long long)(n / (unsigned long long)cw);
        long long j = (long long)(n % (unsigned long long)cw);
        float cb_top[3] = {0.0f, 0.0f, 0.0f}, cr_top[3] = {0.0f, 0.0f, 0.0f};
        float cb_bot[3] = {0.0f, 0.0f, 0.0f}, cr_bot[3] = {0.0f, 0.0f, 0.0f};
        color_encode_row<I, P>(a, 2 * i, j, cb_top, cr_top);
        color_encode_row<I, P>(a, 2 * i + 1, j, cb_bot, cr_bot);
        float cb_avg = color_down(cb_top, cb_bot, center);
        float cr_avg = color_down(cr_top, cr_bot, center);
        float cb = color_quantize(cb_avg, k.c_mul, k.c_off, k.c_lo, k.c_hi);
        float cr = color_quantize(cr_avg, k.c_mul, k.c_off, k.c_lo, k.c_hi);
        color_write<P>(a.u, i, j, cb, a.shift);
        color_write<P>(a.v, i, j, cr, a.shift);
    }
}

// Decode a linear element index into its source and destination offsets.
__device__ __forceinline__ void rgb24_offsets(const Rgb24Args& a, unsigned long long i,
                                              long long* src, long long* dst) {
    long long c = (long long)(i % 3ull);
    unsigned long long rest = i / 3ull;
    long long col = (long long)(rest % (unsigned long long)a.width);
    long long row = (long long)(rest / (unsigned long long)a.width);
    *src = c * a.src_strides[0] + row * a.src_strides[1] + col * a.src_strides[2];
    *dst = c * a.dst_strides[0] + row * a.dst_strides[1] + col * a.dst_strides[2];
}

template <typename O>
__device__ __forceinline__ void rgb24_to_float(const Rgb24Args& a) {
    const unsigned long long total = (unsigned long long)(a.height * a.width) * 3ull;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        long long s, d;
        rgb24_offsets(a, i, &s, &d);
        float v = __fdiv_rn(__uint2float_rn(((const unsigned char*)a.src)[s]), 255.0f);
        color_store((O*)a.dst + d, v);
    }
}

template <typename I>
__device__ __forceinline__ void float_to_rgb24(const Rgb24Args& a) {
    const unsigned long long total = (unsigned long long)(a.height * a.width) * 3ull;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        long long s, d;
        rgb24_offsets(a, i, &s, &d);
        float x = color_clamp01(color_load(((const I*)a.src)[s]));
        ((unsigned char*)a.dst)[d] = (unsigned char)rintf(__fmul_rn(x, 255.0f));
    }
}

#define DEFINE_YUV_TO_RGB(name, P, O) \
    __global__ void yuv_to_rgb_##name(YuvArgs a) { yuv_to_rgb<P, O>(a); }
#define DEFINE_RGB_TO_YUV(name, I, P) \
    __global__ void rgb_to_yuv_##name(YuvArgs a) { rgb_to_yuv<I, P>(a); }
#define DEFINE_RGB24(name, T)                                                       \
    __global__ void rgb24_to_float_##name(Rgb24Args a) { rgb24_to_float<T>(a); }   \
    __global__ void float_to_rgb24_##name(Rgb24Args a) { float_to_rgb24<T>(a); }

extern "C" {

DEFINE_YUV_TO_RGB(u8_f32, unsigned char, float)
DEFINE_YUV_TO_RGB(u8_f64, unsigned char, double)
DEFINE_YUV_TO_RGB(u8_f16, unsigned char, __half)
DEFINE_YUV_TO_RGB(u8_bf16, unsigned char, __nv_bfloat16)
DEFINE_YUV_TO_RGB(u16_f32, unsigned short, float)
DEFINE_YUV_TO_RGB(u16_f64, unsigned short, double)
DEFINE_YUV_TO_RGB(u16_f16, unsigned short, __half)
DEFINE_YUV_TO_RGB(u16_bf16, unsigned short, __nv_bfloat16)

DEFINE_RGB_TO_YUV(f32_u8, float, unsigned char)
DEFINE_RGB_TO_YUV(f64_u8, double, unsigned char)
DEFINE_RGB_TO_YUV(f16_u8, __half, unsigned char)
DEFINE_RGB_TO_YUV(bf16_u8, __nv_bfloat16, unsigned char)
DEFINE_RGB_TO_YUV(f32_u16, float, unsigned short)
DEFINE_RGB_TO_YUV(f64_u16, double, unsigned short)
DEFINE_RGB_TO_YUV(f16_u16, __half, unsigned short)
DEFINE_RGB_TO_YUV(bf16_u16, __nv_bfloat16, unsigned short)

DEFINE_RGB24(f32, float)
DEFINE_RGB24(f64, double)
DEFINE_RGB24(f16, __half)
DEFINE_RGB24(bf16, __nv_bfloat16)

} // extern "C"
