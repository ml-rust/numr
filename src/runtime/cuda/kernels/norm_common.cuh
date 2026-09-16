// Device helpers shared by the normalization kernels.
//
// Included by norm_rms.cu, norm_layer.cu, norm_group.cu,
// fused_add_rms_norm.cu and fused_add_layer_norm.cu. Each of those files
// holds one templated body per op and instantiates it once per dtype through
// a DEFINE_* macro; everything the bodies share lives here.
//
// Accumulation: `Acc` is `float` for f32, f16, bf16 and both FP8 widths, and
// `double` for f64. `AccumTraits<T, Acc>` (dtype_traits.cuh) does the load
// and store conversions.

#ifndef NUMR_NORM_COMMON_CUH
#define NUMR_NORM_COMMON_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"
#include "block_reduce.cuh"

__device__ __forceinline__ float numr_norm_rsqrt(float x) { return rsqrtf(x); }
__device__ __forceinline__ double numr_norm_rsqrt(double x) { return rsqrt(x); }

// ============================================================================
// Welford merge helpers (layer_norm)
// ============================================================================

// Welford's online algorithm for numerically stable mean+variance.
// Maintains three accumulators per partition:
//   count: number of elements seen
//   mean:  running mean
//   M2:    sum of squared deviations from the running mean
// Merge formula (combining two partitions a, b):
//   delta    = mean_b - mean_a
//   mean_ab  = mean_a + delta * count_b / (count_a + count_b)
//   M2_ab    = M2_a + M2_b + delta^2 * count_a * count_b / (count_a + count_b)
template <typename Acc>
__device__ __forceinline__ void welford_merge(
    Acc count_a, Acc mean_a, Acc M2_a,
    Acc count_b, Acc mean_b, Acc M2_b,
    Acc &count_out, Acc &mean_out, Acc &M2_out
) {
    Acc count = count_a + count_b;
    if (count == (Acc)0) {
        count_out = (Acc)0;
        mean_out = (Acc)0;
        M2_out = (Acc)0;
        return;
    }
    Acc delta = mean_b - mean_a;
    mean_out = mean_a + delta * count_b / count;
    M2_out = M2_a + M2_b + delta * delta * count_a * count_b / count;
    count_out = count;
}

// Warp-level Welford reduction: merges accumulators across 32 warp lanes
// using shuffle instructions. After this call, lane 0 holds the merged
// result for the entire warp.
template <typename Acc>
__device__ __forceinline__ void welford_warp_reduce(Acc &count, Acc &mean, Acc &M2) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        Acc o_count = __shfl_down_sync(0xffffffff, count, offset);
        Acc o_mean  = __shfl_down_sync(0xffffffff, mean, offset);
        Acc o_M2    = __shfl_down_sync(0xffffffff, M2, offset);
        welford_merge<Acc>(count, mean, M2, o_count, o_mean, o_M2, count, mean, M2);
    }
}

// ============================================================================
// atomicAdd in the storage dtype (backward kernels)
// ============================================================================

__device__ __forceinline__ void norm_atomic_add(float* address, float val) {
    atomicAdd(address, val);
}

__device__ __forceinline__ void norm_atomic_add(double* address, double val) {
    atomicAdd(address, val);
}

__device__ __forceinline__ void norm_atomic_add(__half* address, float val) {
    unsigned short int* address_as_us = (unsigned short int*)address;
    unsigned short int old = *address_as_us, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_us, assumed,
            __half_as_ushort(__float2half(__half2float(__ushort_as_half(assumed)) + val)));
    } while (assumed != old);
}

__device__ __forceinline__ void norm_atomic_add(__nv_bfloat16* address, float val) {
    unsigned short int* address_as_us = (unsigned short int*)address;
    unsigned short int old = *address_as_us, assumed;
    do {
        assumed = old;
        __nv_bfloat16 old_val;
        unsigned short int* old_val_ptr = (unsigned short int*)&old_val;
        *old_val_ptr = assumed;
        float new_float = __bfloat162float(old_val) + val;
        __nv_bfloat16 new_val = __float2bfloat16(new_float);
        unsigned short int* new_val_ptr = (unsigned short int*)&new_val;
        old = atomicCAS(address_as_us, assumed, *new_val_ptr);
    } while (assumed != old);
}

// FP8 is one byte: CAS on the containing 4-byte word.
__device__ __forceinline__ void norm_atomic_add(numr_fp8_e4m3* address, float val) {
    unsigned int* base = (unsigned int*)((size_t)address & ~3ULL);
    unsigned int shift = (unsigned int)((size_t)address & 3) * 8;
    unsigned int old_word = *base, assumed;
    do {
        assumed = old_word;
        uint8_t old_byte = (uint8_t)((assumed >> shift) & 0xFF);
        uint8_t new_byte = f32_to_fp8_e4m3(fp8_e4m3_to_f32(old_byte) + val);
        unsigned int new_word = (assumed & ~(0xFFu << shift)) | ((unsigned int)new_byte << shift);
        old_word = atomicCAS(base, assumed, new_word);
    } while (assumed != old_word);
}

__device__ __forceinline__ void norm_atomic_add(numr_fp8_e5m2* address, float val) {
    unsigned int* base = (unsigned int*)((size_t)address & ~3ULL);
    unsigned int shift = (unsigned int)((size_t)address & 3) * 8;
    unsigned int old_word = *base, assumed;
    do {
        assumed = old_word;
        uint8_t old_byte = (uint8_t)((assumed >> shift) & 0xFF);
        uint8_t new_byte = f32_to_fp8_e5m2(fp8_e5m2_to_f32(old_byte) + val);
        unsigned int new_word = (assumed & ~(0xFFu << shift)) | ((unsigned int)new_byte << shift);
        old_word = atomicCAS(base, assumed, new_word);
    } while (assumed != old_word);
}

// Instantiates a DEFINE_*(suffix, dtype, acc) macro once per dtype. The
// suffix is what `kernel_name` in loader/names.rs appends.
#define NORM_DTYPES(DEFINE) \
    DEFINE(f32, float, float) \
    DEFINE(f64, double, double) \
    DEFINE(f16, __half, float) \
    DEFINE(bf16, __nv_bfloat16, float) \
    DEFINE(fp8_e4m3, numr_fp8_e4m3, float) \
    DEFINE(fp8_e5m2, numr_fp8_e5m2, float)

#endif // NUMR_NORM_COMMON_CUH
