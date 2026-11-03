// Implicit-GEMM tensor-core conv2d for F16 and BF16, NHWC input, OHWI
// weight, with fused bias, activation and scaled residual.
//
// Instantiations only. The body, staging and epilogue live in
// conv_implicit_gemm.cuh and the headers it includes.
//
// Symbols are conv2d_implicit_gemm_{dtype}_{tile}. Four block tiles, all 512
// threads:
//   128x128  4x4 warps, 2x2 fragments per warp
//   128x64   4x4 warps, 2x1 fragments per warp
//   64x64    4x4 warps, 1x1 fragment per warp
//   128x32   8x2 warps, 1x1 fragment per warp, for C_out <= 32
// The host picks one per launch in loader/conv_implicit_gemm_tile.rs.
//
// F16 needs sm_70+ (wmma). BF16 needs sm_80+ (mma.sync m16n8k16 bf16).

#if __CUDA_ARCH__ >= 700

#include "conv_implicit_gemm.cuh"

// Every pointer except `out` is read-only. `out` can share an allocation
// with `input` or `residual` only as a disjoint view; the host validates
// that, so no element is both read and written.
#define DEFINE_CONV_IGEMM(NAME, WR, WC, WM, WN, BLOCK_K_VAL, HALF_T, MMA, ZERO_EXPR, LOAD_F32, STORE_FN) \
extern "C" __global__ WMMA_LAUNCH_BOUNDS void conv2d_implicit_gemm_##NAME(    \
    const HALF_T* __restrict__ input,                                         \
    const HALF_T* __restrict__ weight,                                        \
    const HALF_T* __restrict__ bias,                                          \
    const HALF_T* __restrict__ prelu,                                         \
    const HALF_T* __restrict__ residual,                                      \
    HALF_T*       __restrict__ out,                                           \
    int M,                                                                    \
    int in_h,                                                                 \
    int in_w,                                                                 \
    int c_in,                                                                 \
    int out_h,                                                                \
    int out_w,                                                                \
    int c_out,                                                                \
    int kernel_h,                                                             \
    int kernel_w,                                                             \
    int stride_h,                                                             \
    int stride_w,                                                             \
    int pad_top,                                                              \
    int pad_left,                                                             \
    int dil_h,                                                                \
    int dil_w,                                                                \
    long long in_sn,                                                          \
    long long in_sh,                                                          \
    long long in_sw,                                                          \
    long long out_sn,                                                         \
    long long out_sh,                                                         \
    long long out_sw,                                                         \
    long long res_sn,                                                         \
    long long res_sh,                                                         \
    long long res_sw,                                                         \
    unsigned int act,                                                         \
    float slope,                                                              \
    int prelu_len,                                                            \
    float res_scale                                                           \
) {                                                                           \
    CONV_IGEMM_BODY(WR, WC, WM, WN, BLOCK_K_VAL, HALF_T, MMA, ZERO_EXPR,      \
                    LOAD_F32, STORE_FN)                                       \
}

/* Expands a dtype-argument macro before DEFINE_CONV_IGEMM counts its
   arguments. */
#define DEFINE_CONV_IGEMM_EXPAND(...) DEFINE_CONV_IGEMM(__VA_ARGS__)

#define CONV_F16_ARGS __half, WMMA, __float2half(0.0f), __half2float, __float2half

DEFINE_CONV_IGEMM_EXPAND(f16_128x128, 4, 4, 2, 2, WMMA_BLOCK_K_DEFAULT, CONV_F16_ARGS)
DEFINE_CONV_IGEMM_EXPAND(f16_128x64, 4, 4, 2, 1, WMMA_BLOCK_K_DEFAULT, CONV_F16_ARGS)
DEFINE_CONV_IGEMM_EXPAND(f16_64x64, 4, 4, 1, 1, WMMA_BLOCK_K_DEFAULT, CONV_F16_ARGS)
DEFINE_CONV_IGEMM_EXPAND(f16_128x32, 8, 2, 1, 1, WMMA_BLOCK_K_DEFAULT, CONV_F16_ARGS)

#if __CUDA_ARCH__ >= 800

#define CONV_BF16_ARGS __nv_bfloat16, RAW, __float2bfloat16(0.0f), __bfloat162float, __float2bfloat16

DEFINE_CONV_IGEMM_EXPAND(bf16_128x128, 4, 4, 2, 2, WMMA_BLOCK_K_DEFAULT, CONV_BF16_ARGS)
DEFINE_CONV_IGEMM_EXPAND(bf16_128x64, 4, 4, 2, 1, WMMA_BLOCK_K_DEFAULT, CONV_BF16_ARGS)
DEFINE_CONV_IGEMM_EXPAND(bf16_64x64, 4, 4, 1, 1, WMMA_BLOCK_K_DEFAULT, CONV_BF16_ARGS)
DEFINE_CONV_IGEMM_EXPAND(bf16_128x32, 8, 2, 1, 1, WMMA_BLOCK_K_DEFAULT, CONV_BF16_ARGS)

#endif  // __CUDA_ARCH__ >= 800

#endif  // __CUDA_ARCH__ >= 700
