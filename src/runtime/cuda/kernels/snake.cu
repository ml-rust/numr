// Snake activation CUDA kernels: y = x + sin(alpha * x)^2 / (beta + eps)
// Supports: f32, f64, f16, bf16
//
// x, out and grad are contiguous [outer, channels, inner] buffers; alpha and
// beta hold one value per channel. The flat index walks `inner` fastest, so a
// warp reads consecutive addresses, and the channel of flat element i is
// (i / inner) % channels. alpha[c] and beta[c] are read once per thread.
//
// Half widths compute in F32 through AccumTraits<dtype, acc>: one rounding at
// the store, none in between. F32 and F64 compute natively.
//
// snake_beta_dparams_* reduces the per-channel parameter gradients with one
// block per channel: each thread strides over outer*inner, then a fixed
// shared-memory tree folds the block. No atomics, so a repeated call returns
// identical bits. The channel sums accumulate in double for every dtype, which
// is what the CPU kernel does too.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"
#include "block_reduce.cuh"

#define SNAKE_BLOCK 256

// Single-precision sine with proper argument reduction.
//
// build.rs compiles every kernel with --use_fast_math, which turns sinf into
// a hardware approximation whose absolute error grows with |x|. Snake's
// argument alpha * x is routinely tens of radians, where that approximation
// diverges from the CPU kernel. This is the usual Cody-Waite reduction by pi
// in three parts (each part has few enough bits that q * part is exact for q
// below 2^12, so |x| below about 1.2e4 reduces exactly), then an odd minimax
// polynomial on [-pi/2, pi/2].
__device__ __forceinline__ float snake_sin(float x) {
    const float ONE_OVER_PI = 0.318309886183790671538f;
    const float PI_A = 3.1414794921875f;
    const float PI_B = 0.00011315941810607910156f;
    const float PI_C = 1.9841872589410058936e-09f;
    float q = rintf(x * ONE_OVER_PI);
    float r = fmaf(q, -PI_A, x);
    r = fmaf(q, -PI_B, r);
    r = fmaf(q, -PI_C, r);
    // sin(x) = (-1)^q sin(r)
    int qi = (int)q;
    if (qi & 1) r = -r;
    float s = r * r;
    float u = 2.6083159809786593541503e-06f;
    u = fmaf(u, s, -0.0001981069071916863322258f);
    u = fmaf(u, s, 0.00833307858556509017944336f);
    u = fmaf(u, s, -0.166666597127914428710938f);
    return fmaf(s, u * r, r);
}
__device__ __forceinline__ double snake_sin(double v) { return sin(v); }

#define DEFINE_SNAKE_KERNELS(suffix, dtype, acc) \
__global__ void snake_beta_##suffix( \
    const dtype* x, const dtype* alpha, const dtype* beta, dtype* out, \
    unsigned int n, unsigned int channels, unsigned int inner, double eps \
) { \
    typedef AccumTraits<dtype, acc> AT; \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return; \
    unsigned int c = (idx / inner) % channels; \
    acc a = AT::load(alpha, (int)c); \
    acc inv = ((acc)1) / (AT::load(beta, (int)c) + (acc)eps); \
    acc xv = AT::load(x, (int)idx); \
    acc s = snake_sin(a * xv); \
    AT::store(out, (int)idx, xv + s * s * inv); \
} \
\
__global__ void snake_beta_dx_##suffix( \
    const dtype* grad, const dtype* x, const dtype* alpha, const dtype* beta, dtype* d_x, \
    unsigned int n, unsigned int channels, unsigned int inner, double eps \
) { \
    typedef AccumTraits<dtype, acc> AT; \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return; \
    unsigned int c = (idx / inner) % channels; \
    acc a = AT::load(alpha, (int)c); \
    acc inv = ((acc)1) / (AT::load(beta, (int)c) + (acc)eps); \
    acc xv = AT::load(x, (int)idx); \
    acc g = AT::load(grad, (int)idx); \
    acc s2 = snake_sin(((acc)2) * a * xv); \
    AT::store(d_x, (int)idx, g * (((acc)1) + a * s2 * inv)); \
} \
\
__global__ void snake_beta_dparams_##suffix( \
    const dtype* grad, const dtype* x, const dtype* alpha, const dtype* beta, \
    dtype* d_alpha, dtype* d_beta, \
    unsigned int outer, unsigned int channels, unsigned int inner, double eps \
) { \
    typedef AccumTraits<dtype, acc> AT; \
    __shared__ double s_alpha[SNAKE_BLOCK]; \
    __shared__ double s_beta[SNAKE_BLOCK]; \
    unsigned int c = blockIdx.x; \
    acc a = AT::load(alpha, (int)c); \
    acc inv = ((acc)1) / (AT::load(beta, (int)c) + (acc)eps); \
    acc inv_sq = inv * inv; \
    unsigned int span = outer * inner; \
    double sum_alpha = 0.0; \
    double sum_beta = 0.0; \
    for (unsigned int j = threadIdx.x; j < span; j += blockDim.x) { \
        unsigned int o = j / inner; \
        unsigned int i = j - o * inner; \
        size_t idx = ((size_t)o * channels + c) * inner + i; \
        acc xv = AT::load(x, (int)idx); \
        acc g = AT::load(grad, (int)idx); \
        acc ax = a * xv; \
        acc s = snake_sin(ax); \
        acc s2 = snake_sin(((acc)2) * ax); \
        sum_alpha += (double)(g * xv * s2 * inv); \
        sum_beta -= (double)(g * s * s * inv_sq); \
    } \
    s_alpha[threadIdx.x] = sum_alpha; \
    s_beta[threadIdx.x] = sum_beta; \
    __syncthreads(); \
    double total_alpha = block_sum_reduce<double>(s_alpha); \
    double total_beta = block_sum_reduce<double>(s_beta); \
    if (threadIdx.x == 0) { \
        AT::store(d_alpha, (int)c, (acc)total_alpha); \
        AT::store(d_beta, (int)c, (acc)total_beta); \
    } \
}

extern "C" {

DEFINE_SNAKE_KERNELS(f32, float, float)
DEFINE_SNAKE_KERNELS(f64, double, double)
DEFINE_SNAKE_KERNELS(f16, __half, float)
DEFINE_SNAKE_KERNELS(bf16, __nv_bfloat16, float)

} // extern "C"
