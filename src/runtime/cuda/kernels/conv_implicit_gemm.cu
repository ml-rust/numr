// Implicit-GEMM tensor-core conv2d for F16 and BF16 with fused bias,
// activation and scaled residual. NHWC or NCHW input, OHWI weight; an NCHW
// call gets its OIHW weight relaid to OHWI on the host.
//
// Instantiations only. The body, staging and epilogue live in
// conv_implicit_gemm.cuh and the headers it includes.
//
// Symbols are conv2d_implicit_gemm_{acc}_{tile}_{staging}. {acc} is the
// dtype (f16, bf16) for the F32 accumulator, or f16acc for F16 inputs with
// the F16 accumulator. Block tiles, all with 4 warps:
//   128x64  2x2 warp grid, 64x32 warp tile, F32 accumulator, C_out > 32,
//           8x16 halo patch
//   256x32  4x1 warp grid, 64x32 warp tile, both accumulators, C_out <= 32,
//           8x32 halo patch
//   256x64  4x1 warp grid, 64x64 warp tile, F16 accumulator, C_out > 32,
//           8x32 halo patch
// Staging for NHWC is `halo` (2D output patch, input patch plus halo staged
// once per 32-channel chunk), `vec` (16-byte gathers per tap, tap-major K)
// or `scalar` (16-bit gathers, flat K). Staging for NCHW is `nchw_halo`
// (the halo body, read along W) or `nchw_gather` (the scalar gather with a
// channel stride). The host picks the accumulator, the tile and the staging
// per launch in loader/conv_implicit_gemm_tile.rs and
// loader/conv_implicit_gemm_staging.rs.
//
// F32-accumulator kernels allow one block per SM in __launch_bounds__. The
// F16-accumulator kernels allow two: their registers and static shared
// memory fit two blocks on an SM with 100 KiB of shared memory.
//
// F16 needs sm_75+ (ldmatrix, mma.sync). BF16 needs sm_80+.

#if __CUDA_ARCH__ >= 750

#include "conv_implicit_gemm.cuh"

/* Patch 8x16 with up to 192 halo pixels for 128-row tiles, 8x32 with up to
   352 for 256-row tiles. Both hold a 3x3 stride-1 halo: 10x18 = 180 and
   10x34 = 340 pixels. The NCHW halo stages up to 64 and 104 runs of 4
   pixels: a 3x3 stride-1 halo with any padding takes at most 10 x 6 = 60
   and 10 x 10 = 100. */
typedef ConvTileCfg<128, 64, 2, 2, 16, 6, 8> ConvTile128x64;
typedef ConvTileCfg<256, 32, 4, 1, 32, 11, 13> ConvTile256x32;
typedef ConvTileCfg<256, 64, 4, 1, 32, 11, 13> ConvTile256x64;

/* RUN picks the body: the gather body conv_igemm_run or the halo body
   conv_igemm_run_halo. */
#define CONV_RUN_GATHER(T, CFG, VEC, HALF_ACC, NCHW) conv_igemm_run<T, CFG, VEC, HALF_ACC, NCHW>
#define CONV_RUN_HALO(T, CFG, VEC, HALF_ACC, NCHW) conv_igemm_run_halo<T, CFG, HALF_ACC, NCHW>

// Every pointer except `out` is read-only. `out` can share an allocation
// with `input` or `residual` only as a disjoint view; the host checks that,
// so no element is both read and written.
//
// NHWC entry: element strides (n, h, w) per tensor, channel stride 1.
#define DEFINE_CONV_IGEMM(NAME, T, CFG, RUN, VEC, HALF_ACC, MIN_BLOCKS)      \
extern "C" __global__ void __launch_bounds__(CFG::THREADS, MIN_BLOCKS)       \
conv2d_implicit_gemm_##NAME(                                                 \
    const T* __restrict__ input,                                             \
    const T* __restrict__ weight,                                            \
    const T* __restrict__ bias,                                              \
    const T* __restrict__ prelu,                                             \
    const T* __restrict__ residual,                                          \
    T* __restrict__ out,                                                     \
    int M, int in_h, int in_w, int c_in, int out_h, int out_w, int c_out,    \
    int kernel_h, int kernel_w, int stride_h, int stride_w,                  \
    int pad_top, int pad_left, int dil_h, int dil_w,                         \
    long long in_sn, long long in_sh, long long in_sw,                       \
    long long out_sn, long long out_sh, long long out_sw,                    \
    long long res_sn, long long res_sh, long long res_sw,                    \
    unsigned int act, float slope, int prelu_len, float res_scale) {         \
    const ConvIgemmParams<T> p = {                                           \
        input, weight, bias, prelu, residual, out,                           \
        M, in_h, in_w, c_in, out_h, out_w, c_out,                            \
        kernel_h, kernel_w, stride_h, stride_w,                              \
        pad_top, pad_left, dil_h, dil_w,                                     \
        in_sn, in_sh, in_sw, out_sn, out_sh, out_sw, res_sn, res_sh, res_sw, \
        1ll, 1ll, 1ll,                                                       \
        act, slope, prelu_len, res_scale};                                   \
    RUN(T, CFG, VEC, HALF_ACC, false)(p);                                    \
}

// NCHW entry: element strides (n, c, h, w) per tensor.
#define DEFINE_CONV_IGEMM_NCHW(NAME, T, CFG, RUN, HALF_ACC, MIN_BLOCKS)      \
extern "C" __global__ void __launch_bounds__(CFG::THREADS, MIN_BLOCKS)       \
conv2d_implicit_gemm_##NAME(                                                 \
    const T* __restrict__ input,                                             \
    const T* __restrict__ weight,                                            \
    const T* __restrict__ bias,                                              \
    const T* __restrict__ prelu,                                             \
    const T* __restrict__ residual,                                          \
    T* __restrict__ out,                                                     \
    int M, int in_h, int in_w, int c_in, int out_h, int out_w, int c_out,    \
    int kernel_h, int kernel_w, int stride_h, int stride_w,                  \
    int pad_top, int pad_left, int dil_h, int dil_w,                         \
    long long in_sn, long long in_sc, long long in_sh, long long in_sw,      \
    long long out_sn, long long out_sc, long long out_sh, long long out_sw,  \
    long long res_sn, long long res_sc, long long res_sh, long long res_sw,  \
    unsigned int act, float slope, int prelu_len, float res_scale) {         \
    const ConvIgemmParams<T> p = {                                           \
        input, weight, bias, prelu, residual, out,                           \
        M, in_h, in_w, c_in, out_h, out_w, c_out,                            \
        kernel_h, kernel_w, stride_h, stride_w,                              \
        pad_top, pad_left, dil_h, dil_w,                                     \
        in_sn, in_sh, in_sw, out_sn, out_sh, out_sw, res_sn, res_sh, res_sw, \
        in_sc, out_sc, res_sc,                                               \
        act, slope, prelu_len, res_scale};                                   \
    RUN(T, CFG, false, HALF_ACC, true)(p);                                   \
}

#define DEFINE_CONV_IGEMM_TILE(PRE, TILE, T, CFG, HALF_ACC, MIN_BLOCKS)                          \
    DEFINE_CONV_IGEMM(PRE##_##TILE##_vec, T, CFG, CONV_RUN_GATHER, true, HALF_ACC, MIN_BLOCKS)    \
    DEFINE_CONV_IGEMM(PRE##_##TILE##_scalar, T, CFG, CONV_RUN_GATHER, false, HALF_ACC, MIN_BLOCKS) \
    DEFINE_CONV_IGEMM(PRE##_##TILE##_halo, T, CFG, CONV_RUN_HALO, true, HALF_ACC, MIN_BLOCKS)     \
    DEFINE_CONV_IGEMM_NCHW(PRE##_##TILE##_nchw_gather, T, CFG, CONV_RUN_GATHER, HALF_ACC, MIN_BLOCKS) \
    DEFINE_CONV_IGEMM_NCHW(PRE##_##TILE##_nchw_halo, T, CFG, CONV_RUN_HALO, HALF_ACC, MIN_BLOCKS)

#define DEFINE_CONV_IGEMM_DTYPE(DT, T)                                \
    DEFINE_CONV_IGEMM_TILE(DT, 128x64, T, ConvTile128x64, false, 1)   \
    DEFINE_CONV_IGEMM_TILE(DT, 256x32, T, ConvTile256x32, false, 1)

DEFINE_CONV_IGEMM_DTYPE(f16, __half)

DEFINE_CONV_IGEMM_TILE(f16acc, 256x32, __half, ConvTile256x32, true, 2)
DEFINE_CONV_IGEMM_TILE(f16acc, 256x64, __half, ConvTile256x64, true, 2)

#if __CUDA_ARCH__ >= 800
DEFINE_CONV_IGEMM_DTYPE(bf16, __nv_bfloat16)
#endif  // __CUDA_ARCH__ >= 800

#endif  // __CUDA_ARCH__ >= 750
