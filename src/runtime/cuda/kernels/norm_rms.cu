// RMSNorm CUDA kernels: x * rsqrt(mean(x^2) + eps) * weight
//
// One templated body, instantiated per dtype through DEFINE_RMS_NORM. The
// accumulator is F32 for f32, f16, bf16 and FP8, F64 for f64.
//
// Two kernels per dtype, picked by the launcher on row width:
// - rms_norm_regs_*  (rms_norm_regs.cuh): single pass, row held in registers
// - rms_norm_*       (here): two passes over the row, any width
//
// Shared memory: blockDim.x accumulator elements for either kernel.

#include "norm_common.cuh"
#include "rms_norm_regs.cuh"

// Each block handles one row (hidden_size elements).
template <typename T, typename Acc>
__device__ __forceinline__ void rms_norm_impl(
    const T* input, const T* weight, T* output,
    unsigned int batch_size, unsigned int hidden_size, Acc eps, Acc* shared
) {
    typedef AccumTraits<T, Acc> AT;
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    const T* row_in = input + row * hidden_size;
    T* row_out = output + row * hidden_size;

    // Phase 1: sum of squares
    Acc thread_sum = (Acc)0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        Acc val = AT::load(row_in, (int)i);
        thread_sum += val * val;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce<Acc>(shared);

    Acc rms_inv = numr_norm_rsqrt(shared[0] / hidden_size + eps);
    __syncthreads();

    // Phase 2: normalize and apply weight
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        AT::store(row_out, (int)i, AT::load(row_in, (int)i) * rms_inv * AT::load(weight, (int)i));
    }
}

#define DEFINE_RMS_NORM(suffix, dtype, acc) \
__global__ void rms_norm_##suffix( \
    const dtype* input, const dtype* weight, dtype* output, \
    unsigned int batch_size, unsigned int hidden_size, acc eps \
) { \
    extern __shared__ acc rms_smem_##acc[]; \
    rms_norm_impl<dtype, acc>(input, weight, output, batch_size, hidden_size, eps, rms_smem_##acc); \
}

extern "C" {

NORM_DTYPES(DEFINE_RMS_NORM)

} // extern "C"
