// Implicit-GEMM tensor-core conv2d for F16 and BF16, NHWC input, OHWI
// weight, with fused bias, activation and scaled residual.
//
// Instantiations only. The body, staging and epilogue live in
// conv_implicit_gemm.cuh and the headers it includes.
//
// Symbols are conv2d_implicit_gemm_{dtype}_{tile}_{staging}. Block tiles,
// each with a 64x32 warp tile:
//   128x64  4 warps, 2x2 warp grid, for C_out > 32
//   256x32  4 warps, 4x1 warp grid, for C_out <= 32
// Staging is `vec` (16-byte loads, tap-major K) or `scalar` (16-bit
// gathers, flat K). The host picks the tile and the staging per launch in
// loader/conv_implicit_gemm_tile.rs.
//
// F16 needs sm_75+ (ldmatrix, mma.sync). BF16 needs sm_80+.

#if __CUDA_ARCH__ >= 750

#include "conv_implicit_gemm.cuh"

typedef ConvTileCfg<128, 64, 2, 2> ConvTile128x64;
typedef ConvTileCfg<256, 32, 4, 1> ConvTile256x32;

// Every pointer except `out` is read-only. `out` can share an allocation
// with `input` or `residual` only as a disjoint view; the host checks that,
// so no element is both read and written.
#define DEFINE_CONV_IGEMM(NAME, T, CFG, VEC)                                 \
extern "C" __global__ void __launch_bounds__(CFG::THREADS, 1)                \
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
        act, slope, prelu_len, res_scale};                                   \
    conv_igemm_run<T, CFG, VEC>(p);                                          \
}

#define DEFINE_CONV_IGEMM_DTYPE(DT, T)                                       \
    DEFINE_CONV_IGEMM(DT##_128x64_vec, T, ConvTile128x64, true)              \
    DEFINE_CONV_IGEMM(DT##_128x64_scalar, T, ConvTile128x64, false)          \
    DEFINE_CONV_IGEMM(DT##_256x32_vec, T, ConvTile256x32, true)              \
    DEFINE_CONV_IGEMM(DT##_256x32_scalar, T, ConvTile256x32, false)

DEFINE_CONV_IGEMM_DTYPE(f16, __half)

#if __CUDA_ARCH__ >= 800
DEFINE_CONV_IGEMM_DTYPE(bf16, __nv_bfloat16)
#endif  // __CUDA_ARCH__ >= 800

#endif  // __CUDA_ARCH__ >= 750
