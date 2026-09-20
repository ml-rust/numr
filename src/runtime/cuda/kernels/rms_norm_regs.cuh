// Register-cached single-pass RMSNorm kernels, one per norm dtype.
//
// WHY: the two-pass rms_norm kernel in norm_rms.cu reads the row once to accumulate
// the sum of squares and again to scale it, so it moves 2 reads + 1 write where
// 1 read + 1 write suffices. RMSNorm is bandwidth-bound, so the redundant read
// is close to pure cost.
//
// FIX: when a thread's slice of the row fits in registers, the thread keeps its
// elements after the first read, accumulates from the register copies, and
// scales those same copies. The row is read exactly once.
//
// The array is sized by NORM_MAX_REGS_PER_THREAD (a compile-time constant) and
// indexed only by a fully unrolled loop counter. A runtime bound would put the
// array in local memory, which is DRAM traffic again and defeats the point.
//
// Applies only when hidden_size <= NORM_MAX_REGS_PER_THREAD * blockDim.x; the
// launcher keeps the two-pass kernel as the fallback for wider rows.

#ifndef NUMR_RMS_NORM_REGS_CUH
#define NUMR_RMS_NORM_REGS_CUH

#include "norm_common.cuh"

// 16 elements/thread covers hidden_size up to 4096 at the standard 256-thread
// block, which is the common transformer width. Every element held costs
// registers, and registers cost resident blocks, so raising this buys wider
// coverage at the price of fewer blocks in flight. Check ptxas -v (and for
// spills, which would defeat the whole kernel) before changing it.
#define NORM_MAX_REGS_PER_THREAD 32

// Sum of squares reduced across the block, in the same order as the two-pass
// kernel: ascending element index per thread, then the same shared-memory tree.
// The `threadIdx.x + s < blockDim.x` guard and the power-of-two start make the
// tree correct for a block size that is not a power of two; for a power-of-two
// block it reduces to the identical sequence of adds.
template <typename Acc>
__device__ __forceinline__ Acc rms_norm_block_sum(Acc thread_sum, Acc* shared) {
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    unsigned int s = (blockDim.x <= 1) ? 0u : (1u << (31 - __clz(blockDim.x - 1)));
    for (; s > 0; s >>= 1) {
        if (threadIdx.x < s && threadIdx.x + s < blockDim.x) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    return shared[0];
}

template <typename T, typename Acc>
__device__ __forceinline__ void rms_norm_regs_impl(
    const T* input, const T* weight, T* output,
    unsigned int batch_size, unsigned int hidden_size, Acc eps, Acc* shared
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    const T* row_in = input + (size_t)row * (size_t)hidden_size;
    T* row_out = output + (size_t)row * (size_t)hidden_size;

    // Single read of the row. `regs[j]` is written and read under the same
    // bound check, so the slots past the row end are never consumed.
    Acc regs[NORM_MAX_REGS_PER_THREAD];
    Acc thread_sum = (Acc)0;
    // Carrying the index forward instead of recomputing threadIdx.x + j * blockDim.x
    // costs ptxas ~30% fewer registers here, which is resident blocks.
    unsigned int load_idx = threadIdx.x;
    #pragma unroll
    for (int j = 0; j < NORM_MAX_REGS_PER_THREAD; ++j) {
        if (load_idx < hidden_size) {
            Acc val = AccumTraits<T, Acc>::load(row_in, (int)load_idx);
            regs[j] = val;
            thread_sum += val * val;
        }
        load_idx += blockDim.x;
    }

    Acc total = rms_norm_block_sum<Acc>(thread_sum, shared);
    Acc rms_inv = numr_norm_rsqrt(total / hidden_size + eps);
    __syncthreads();

    #pragma unroll
    for (int j = 0; j < NORM_MAX_REGS_PER_THREAD; ++j) {
        unsigned int i = threadIdx.x + (unsigned int)j * blockDim.x;
        if (i < hidden_size) {
            Acc w = AccumTraits<T, Acc>::load(weight, (int)i);
            AccumTraits<T, Acc>::store(row_out, (int)i, regs[j] * rms_inv * w);
        }
    }
}

// Half and FP8 widths accumulate in FP32, matching the two-pass kernels.
#define DEFINE_RMS_NORM_REGS(suffix, dtype, acc) \
__global__ void rms_norm_regs_##suffix( \
    const dtype* input, const dtype* weight, dtype* output, \
    unsigned int batch_size, unsigned int hidden_size, acc eps \
) { \
    extern __shared__ acc rms_regs_smem_##acc[]; \
    rms_norm_regs_impl<dtype, acc>( \
        input, weight, output, batch_size, hidden_size, eps, rms_regs_smem_##acc); \
}

extern "C" {

NORM_DTYPES(DEFINE_RMS_NORM_REGS)

} // extern "C"

#endif // NUMR_RMS_NORM_REGS_CUH
