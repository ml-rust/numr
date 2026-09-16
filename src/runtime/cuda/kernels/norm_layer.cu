// LayerNorm CUDA kernels: (x - mean) / sqrt(var + eps) * weight + bias
//
// One templated body, instantiated per dtype through DEFINE_LAYER_NORM. The
// accumulator is F32 for f32, f16, bf16 and FP8, F64 for f64.
//
// Single-pass Welford mean+variance with a warp-level merge via
// __shfl_down_sync, then a block-level merge through shared memory.
//
// Shared memory: 3 * ceil(blockDim.x / 32) accumulator elements, laid out
// [count0..countN, mean0..meanN, M2_0..M2_N] with N = number of warps.

#include "norm_common.cuh"

// Each block handles one row (hidden_size elements).
template <typename T, typename Acc>
__device__ __forceinline__ void layer_norm_impl(
    const T* input, const T* weight, const T* bias, T* output,
    unsigned int batch_size, unsigned int hidden_size, Acc eps, Acc* shared
) {
    typedef AccumTraits<T, Acc> AT;
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    const T* row_in = input + row * hidden_size;
    T* row_out = output + row * hidden_size;

    // Shift every element by the row's first value before accumulating.
    // `x - mean` cancels catastrophically when a row sits far from zero relative
    // to its own spread, and that cancellation, not the reduction, is what costs
    // the mantissa. Welford does NOT help: its mean is still a value of the
    // row's own magnitude, so phase 2 cancels just the same. Mean, variance and
    // the normalized value are all invariant under the shift in exact
    // arithmetic, and the CPU kernels shift by the same element.
    Acc ref = AT::load(row_in, 0);

    // Phase 1: single-pass Welford accumulation
    Acc count = (Acc)0, mean = (Acc)0, M2 = (Acc)0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        Acc x = AT::load(row_in, (int)i) - ref;
        count += (Acc)1;
        Acc delta = x - mean;
        mean += delta / count;
        M2 += delta * (x - mean);
    }

    // Warp-level Welford merge
    welford_warp_reduce<Acc>(count, mean, M2);

    // Block-level merge via shared memory (one entry per warp)
    unsigned int warp_id = threadIdx.x / 32;
    unsigned int lane_id = threadIdx.x % 32;
    unsigned int num_warps = (blockDim.x + 31) / 32;

    Acc* s_count = shared;
    Acc* s_mean  = shared + num_warps;
    Acc* s_M2    = shared + 2 * num_warps;

    if (lane_id == 0) {
        s_count[warp_id] = count;
        s_mean[warp_id]  = mean;
        s_M2[warp_id]    = M2;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        Acc r_count = (lane_id < num_warps) ? s_count[lane_id] : (Acc)0;
        Acc r_mean  = (lane_id < num_warps) ? s_mean[lane_id]  : (Acc)0;
        Acc r_M2    = (lane_id < num_warps) ? s_M2[lane_id]    : (Acc)0;

        welford_warp_reduce<Acc>(r_count, r_mean, r_M2);

        if (lane_id == 0) {
            s_mean[0] = r_mean;
            s_M2[0]   = r_M2;
            s_count[0] = r_count;
        }
    }
    __syncthreads();

    Acc shifted_mean = s_mean[0];
    Acc inv_std = numr_norm_rsqrt(s_M2[0] / s_count[0] + eps);

    // Phase 2: normalize and apply affine transform
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        Acc normalized = ((AT::load(row_in, (int)i) - ref) - shifted_mean) * inv_std;
        AT::store(row_out, (int)i, normalized * AT::load(weight, (int)i) + AT::load(bias, (int)i));
    }
}

#define DEFINE_LAYER_NORM(suffix, dtype, acc) \
__global__ void layer_norm_##suffix( \
    const dtype* input, const dtype* weight, const dtype* bias, dtype* output, \
    unsigned int batch_size, unsigned int hidden_size, acc eps \
) { \
    extern __shared__ acc layer_smem_##acc[]; \
    layer_norm_impl<dtype, acc>( \
        input, weight, bias, output, batch_size, hidden_size, eps, layer_smem_##acc); \
}

extern "C" {

NORM_DTYPES(DEFINE_LAYER_NORM)

} // extern "C"
