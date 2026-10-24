// Reduction CUDA kernels - Templated implementation using AccumTraits
// Supports: sum, max, min, mean
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2
//
// Uses AccumTraits<T, Acc> for DRY reduction code across all type/accumulation combinations.
// Similar to CPU's Accumulator trait pattern.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"
#include "reduce_global.cuh"
#include "reduce_dim.cuh"
#include "reduce_arg_logical.cuh"

// ============================================================================
// Instantiation macros - eliminates boilerplate in extern "C" wrappers
// ============================================================================

// Global reductions: reduce_sum_SUFFIX, reduce_max_SUFFIX, etc.
#define INSTANTIATE_GLOBAL_REDUCE(T, Acc, SUFFIX) \
__global__ void reduce_sum_##SUFFIX(const T* input, T* output, unsigned int n) { \
    reduce_sum_impl<T, Acc>(input, output, n); \
} \
__global__ void reduce_max_##SUFFIX(const T* input, T* output, unsigned int n) { \
    reduce_max_impl<T, Acc>(input, output, n); \
} \
__global__ void reduce_min_##SUFFIX(const T* input, T* output, unsigned int n) { \
    reduce_min_impl<T, Acc>(input, output, n); \
} \
__global__ void reduce_prod_##SUFFIX(const T* input, T* output, unsigned int n) { \
    reduce_prod_impl<T, Acc>(input, output, n); \
}

// Dim reductions: reduce_sum_dim_SUFFIX, reduce_max_dim_SUFFIX, reduce_min_dim_SUFFIX,
// each with a `_serial` twin for a short reduced axis.
#define INSTANTIATE_DIM_REDUCE_CORE(T, Acc, SUFFIX) \
__global__ void reduce_sum_dim_##SUFFIX( \
    const T* input, T* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    reduce_sum_dim_impl<T, Acc>(input, output, outer_size, reduce_size, inner_size); \
} \
__global__ void reduce_sum_dim_serial_##SUFFIX( \
    const T* input, T* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    reduce_dim_serial_impl<T, Acc, SerialSum<T, Acc>>(input, output, outer_size, reduce_size, inner_size); \
} \
__global__ void reduce_max_dim_serial_##SUFFIX( \
    const T* input, T* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    reduce_dim_serial_impl<T, Acc, SerialMax<T, Acc>>(input, output, outer_size, reduce_size, inner_size); \
} \
__global__ void reduce_min_dim_serial_##SUFFIX( \
    const T* input, T* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    reduce_dim_serial_impl<T, Acc, SerialMin<T, Acc>>(input, output, outer_size, reduce_size, inner_size); \
} \
__global__ void reduce_prod_dim_serial_##SUFFIX( \
    const T* input, T* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    reduce_dim_serial_impl<T, Acc, SerialProd<T, Acc>>(input, output, outer_size, reduce_size, inner_size); \
} \
__global__ void reduce_max_dim_##SUFFIX( \
    const T* input, T* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    reduce_max_dim_impl<T, Acc>(input, output, outer_size, reduce_size, inner_size); \
} \
__global__ void reduce_min_dim_##SUFFIX( \
    const T* input, T* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    reduce_min_dim_impl<T, Acc>(input, output, outer_size, reduce_size, inner_size); \
}

// Argmax/argmin dim: argmax_dim_SUFFIX, argmin_dim_SUFFIX
#define INSTANTIATE_ARG_REDUCE(T, CompareType, SUFFIX) \
__global__ void argmax_dim_##SUFFIX( \
    const T* input, long long* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    arg_dim_impl<T, CompareType, true>(input, output, outer_size, reduce_size, inner_size); \
} \
__global__ void argmin_dim_##SUFFIX( \
    const T* input, long long* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    arg_dim_impl<T, CompareType, false>(input, output, outer_size, reduce_size, inner_size); \
}

// Prod dim: reduce_prod_dim_SUFFIX
#define INSTANTIATE_PROD_DIM(T, Acc, SUFFIX) \
__global__ void reduce_prod_dim_##SUFFIX( \
    const T* input, T* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    reduce_prod_dim_impl<T, Acc>(input, output, outer_size, reduce_size, inner_size); \
}

// Any/all dim: reduce_any_dim_SUFFIX, reduce_all_dim_SUFFIX
#define INSTANTIATE_LOGICAL_DIM(T, CompareType, SUFFIX) \
__global__ void reduce_any_dim_##SUFFIX( \
    const T* input, T* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    reduce_any_dim_impl<T, CompareType>(input, output, outer_size, reduce_size, inner_size); \
} \
__global__ void reduce_all_dim_##SUFFIX( \
    const T* input, T* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    reduce_all_dim_impl<T, CompareType>(input, output, outer_size, reduce_size, inner_size); \
}

// ============================================================================
// extern "C" wrapper kernels for Rust FFI
//
// Accumulation Strategy:
//   Each dtype has multiple accumulation variants selected by suffix:
//   - Native (e.g. f32): Accumulates in the same type. Fastest, but may
//     lose precision for large reductions on low-precision types.
//   - fp32acc (e.g. f16_fp32acc): Accumulates in f32. Default for f16/bf16
//     reductions — good precision/speed tradeoff.
//   - fp64acc (e.g. f16_fp64acc): Accumulates in f64. Maximum precision for
//     scientific computing workloads. Slowest due to f64 throughput on GPU.
//   - bf16acc (e.g. fp8_e4m3_bf16acc): FP8 types accumulate in bf16 as a
//     fast alternative to fp32acc.
//
//   The Rust dispatch layer selects the appropriate suffix at runtime based
//   on the requested precision level.
// ============================================================================

extern "C" {

// --- F32 (native accumulation) ---
INSTANTIATE_GLOBAL_REDUCE(float, float, f32)
INSTANTIATE_DIM_REDUCE_CORE(float, float, f32)
INSTANTIATE_DIM_REDUCE_CORE(float, double, f32_fp64acc)
INSTANTIATE_ARG_REDUCE(float, float, f32)
INSTANTIATE_PROD_DIM(float, float, f32)
INSTANTIATE_PROD_DIM(float, double, f32_fp64acc)
INSTANTIATE_LOGICAL_DIM(float, float, f32)

// --- F64 (native accumulation only) ---
INSTANTIATE_GLOBAL_REDUCE(double, double, f64)
INSTANTIATE_DIM_REDUCE_CORE(double, double, f64)
INSTANTIATE_ARG_REDUCE(double, double, f64)
INSTANTIATE_PROD_DIM(double, double, f64)
INSTANTIATE_LOGICAL_DIM(double, double, f64)

// --- F16 (native, fp32acc, fp64acc) ---
INSTANTIATE_GLOBAL_REDUCE(__half, __half, f16)
INSTANTIATE_GLOBAL_REDUCE(__half, float, f16_fp32acc)
INSTANTIATE_DIM_REDUCE_CORE(__half, __half, f16)
INSTANTIATE_DIM_REDUCE_CORE(__half, float, f16_fp32acc)
INSTANTIATE_DIM_REDUCE_CORE(__half, double, f16_fp64acc)
INSTANTIATE_ARG_REDUCE(__half, float, f16)
INSTANTIATE_PROD_DIM(__half, __half, f16)
INSTANTIATE_PROD_DIM(__half, float, f16_fp32acc)
INSTANTIATE_PROD_DIM(__half, double, f16_fp64acc)
INSTANTIATE_LOGICAL_DIM(__half, float, f16)

// --- BF16 (native, fp32acc, fp64acc) ---
INSTANTIATE_GLOBAL_REDUCE(__nv_bfloat16, __nv_bfloat16, bf16)
INSTANTIATE_GLOBAL_REDUCE(__nv_bfloat16, float, bf16_fp32acc)
INSTANTIATE_DIM_REDUCE_CORE(__nv_bfloat16, __nv_bfloat16, bf16)
INSTANTIATE_DIM_REDUCE_CORE(__nv_bfloat16, float, bf16_fp32acc)
INSTANTIATE_DIM_REDUCE_CORE(__nv_bfloat16, double, bf16_fp64acc)
INSTANTIATE_ARG_REDUCE(__nv_bfloat16, float, bf16)
INSTANTIATE_PROD_DIM(__nv_bfloat16, __nv_bfloat16, bf16)
INSTANTIATE_PROD_DIM(__nv_bfloat16, float, bf16_fp32acc)
INSTANTIATE_PROD_DIM(__nv_bfloat16, double, bf16_fp64acc)
INSTANTIATE_LOGICAL_DIM(__nv_bfloat16, float, bf16)

// --- FP8 E4M3 (fp32 default, bf16acc, fp64acc) ---
INSTANTIATE_GLOBAL_REDUCE(numr_fp8_e4m3, float, fp8_e4m3)
INSTANTIATE_GLOBAL_REDUCE(numr_fp8_e4m3, __nv_bfloat16, fp8_e4m3_bf16acc)
INSTANTIATE_DIM_REDUCE_CORE(numr_fp8_e4m3, float, fp8_e4m3)
INSTANTIATE_DIM_REDUCE_CORE(numr_fp8_e4m3, __nv_bfloat16, fp8_e4m3_bf16acc)
INSTANTIATE_DIM_REDUCE_CORE(numr_fp8_e4m3, double, fp8_e4m3_fp64acc)
INSTANTIATE_ARG_REDUCE(numr_fp8_e4m3, float, fp8_e4m3)
INSTANTIATE_PROD_DIM(numr_fp8_e4m3, float, fp8_e4m3)
INSTANTIATE_PROD_DIM(numr_fp8_e4m3, __nv_bfloat16, fp8_e4m3_bf16acc)
INSTANTIATE_PROD_DIM(numr_fp8_e4m3, double, fp8_e4m3_fp64acc)
INSTANTIATE_LOGICAL_DIM(numr_fp8_e4m3, float, fp8_e4m3)

// --- FP8 E5M2 (fp32 default, bf16acc, fp64acc) ---
INSTANTIATE_GLOBAL_REDUCE(numr_fp8_e5m2, float, fp8_e5m2)
INSTANTIATE_GLOBAL_REDUCE(numr_fp8_e5m2, __nv_bfloat16, fp8_e5m2_bf16acc)
INSTANTIATE_DIM_REDUCE_CORE(numr_fp8_e5m2, float, fp8_e5m2)
INSTANTIATE_DIM_REDUCE_CORE(numr_fp8_e5m2, __nv_bfloat16, fp8_e5m2_bf16acc)
INSTANTIATE_DIM_REDUCE_CORE(numr_fp8_e5m2, double, fp8_e5m2_fp64acc)
INSTANTIATE_ARG_REDUCE(numr_fp8_e5m2, float, fp8_e5m2)
INSTANTIATE_PROD_DIM(numr_fp8_e5m2, float, fp8_e5m2)
INSTANTIATE_PROD_DIM(numr_fp8_e5m2, __nv_bfloat16, fp8_e5m2_bf16acc)
INSTANTIATE_PROD_DIM(numr_fp8_e5m2, double, fp8_e5m2_fp64acc)
INSTANTIATE_LOGICAL_DIM(numr_fp8_e5m2, float, fp8_e5m2)

} // extern "C"
