// Register-cached single-pass RMSNorm kernels, one per norm dtype.
//
// The two-pass rms_norm kernel in norm_rms.cu reads the row once to accumulate
// the sum of squares and again to scale it, so it moves 2 reads + 1 write where
// 1 read + 1 write suffices. RMSNorm is bandwidth-bound, so the redundant read
// is close to pure cost.
//
// When a thread's slice of the row fits in registers, the thread keeps its
// elements after the first read, accumulates from the register copies, and
// scales those same copies. The row is read exactly once. The thread's slice
// of the weight is read in the same pass, so the scale pass waits on no
// memory round trip: it is the sum, the rsqrt and the stores.
//
// Both arrays are sized by NORM_MAX_REGS_PER_THREAD (a compile-time constant)
// and indexed only by a fully unrolled loop counter. A runtime bound would put
// them in local memory, which is DRAM traffic again and defeats the point.
//
// Elements are owned in quads: thread t holds elements [4q, 4q + 4) for
// q = t, t + blockDim.x, ... . The `quad` kernels move each quad as one packed
// access (norm_quad.cuh); the `regs` kernels move it as four scalar accesses
// with the same ownership and the same accumulation order, so both produce
// the same bits for the same row. The launcher picks `quad` when the row
// width is a multiple of four and every base pointer is packed-aligned.
//
// Applies only when hidden_size <= NORM_MAX_REGS_PER_THREAD * blockDim.x; the
// launcher keeps the two-pass kernel as the fallback for wider rows.
//
// Every kernel here needs blockDim.x to be a multiple of 32: the block sum
// shuffles with a full warp mask.

#ifndef NUMR_RMS_NORM_REGS_CUH
#define NUMR_RMS_NORM_REGS_CUH

#include "norm_common.cuh"
#include "norm_quad.cuh"

// Every element held costs two registers (row and weight), and registers
// cost resident blocks, so raising this buys wider coverage at the price of
// fewer blocks in flight. Check ptxas -v for the register count and for
// spills (which would defeat the whole kernel) before changing it. Must be a
// multiple of 4.
#define NORM_MAX_REGS_PER_THREAD 32
#define NORM_QUADS_PER_THREAD (NORM_MAX_REGS_PER_THREAD / 4)

// Explicit fused multiply-add for the sum of squares, so the packed and
// scalar kernels accumulate with the same instruction.
__device__ __forceinline__ float norm_fma(float a, float b, float c) { return fmaf(a, b, c); }
__device__ __forceinline__ double norm_fma(double a, double b, double c) { return fma(a, b, c); }

// Sum of `thread_sum` across the block, returned to every thread.
//
// Butterfly shuffle within each warp, one shared-memory slot per warp, then
// warp 0 folds the per-warp partials with a second butterfly. `warp_sums`
// needs ceil(blockDim.x / 32) slots. blockDim.x must be a multiple of 32.
template <typename Acc>
__device__ __forceinline__ Acc rms_norm_block_sum(Acc thread_sum, Acc* warp_sums) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_xor_sync(0xffffffffu, thread_sum, offset);
    }
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int num_warps = (blockDim.x + 31u) >> 5;
    if (lane == 0) warp_sums[warp] = thread_sum;
    __syncthreads();

    if (warp == 0) {
        Acc total = (lane < num_warps) ? warp_sums[lane] : (Acc)0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_xor_sync(0xffffffffu, total, offset);
        }
        if (lane == 0) warp_sums[0] = total;
    }
    __syncthreads();
    return warp_sums[0];
}

// One quad of `src` into `r[0..4)`. Scalar access skips slots past the row
// end, and every later read of `r` sits under the same bound.
template <typename T, typename Acc, bool PACKED>
__device__ __forceinline__ void rms_norm_load_quad(
    const T* src, unsigned int base, unsigned int hidden_size, Acc* r
) {
    if (PACKED) {
        NormQuad<T, Acc>::load(src + base, r);
    } else {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            unsigned int i = base + (unsigned int)k;
            if (i < hidden_size) r[k] = AccumTraits<T, Acc>::load(src, (int)i);
        }
    }
}

// Squares of one quad folded into `sum`, ascending element index.
template <typename Acc, bool PACKED>
__device__ __forceinline__ void rms_norm_accumulate_quad(
    const Acc* r, unsigned int base, unsigned int hidden_size, Acc& sum
) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (PACKED || base + (unsigned int)k < hidden_size) sum = norm_fma(r[k], r[k], sum);
    }
}

// One quad scaled by `rms_inv` and its weight `w`, written to the output row.
template <typename T, typename Acc, bool PACKED>
__device__ __forceinline__ void rms_norm_store_quad(
    T* row_out, unsigned int base, unsigned int hidden_size,
    const Acc* r, const Acc* w, Acc rms_inv
) {
    if (PACKED) {
        Acc o[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) o[k] = r[k] * rms_inv * w[k];
        NormQuad<T, Acc>::store(row_out + base, o);
    } else {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            unsigned int i = base + (unsigned int)k;
            if (i < hidden_size) {
                AccumTraits<T, Acc>::store(row_out, (int)i, r[k] * rms_inv * w[k]);
            }
        }
    }
}

template <typename T, typename Acc, bool PACKED>
__device__ __forceinline__ void rms_norm_regs_impl(
    const T* input, const T* weight, T* output,
    unsigned int batch_size, unsigned int hidden_size, Acc eps, Acc* warp_sums
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    const T* row_in = input + (size_t)row * (size_t)hidden_size;
    T* row_out = output + (size_t)row * (size_t)hidden_size;
    unsigned int quads = (hidden_size + 3u) >> 2;

    // Single read of the row and of the weight. Each quad slot is written and
    // read under the same bound check, so the slots past the row end are
    // never consumed.
    Acc regs[NORM_MAX_REGS_PER_THREAD];
    Acc wregs[NORM_MAX_REGS_PER_THREAD];
    Acc thread_sum = (Acc)0;
    unsigned int q = threadIdx.x;
    #pragma unroll
    for (int j = 0; j < NORM_QUADS_PER_THREAD; ++j) {
        if (q < quads) {
            unsigned int base = q * 4u;
            rms_norm_load_quad<T, Acc, PACKED>(row_in, base, hidden_size, regs + 4 * j);
            rms_norm_load_quad<T, Acc, PACKED>(weight, base, hidden_size, wregs + 4 * j);
            rms_norm_accumulate_quad<Acc, PACKED>(regs + 4 * j, base, hidden_size, thread_sum);
        }
        q += blockDim.x;
    }

    Acc total = rms_norm_block_sum<Acc>(thread_sum, warp_sums);
    Acc rms_inv = numr_norm_rsqrt(total / hidden_size + eps);

    q = threadIdx.x;
    #pragma unroll
    for (int j = 0; j < NORM_QUADS_PER_THREAD; ++j) {
        if (q < quads) {
            rms_norm_store_quad<T, Acc, PACKED>(
                row_out, q * 4u, hidden_size, regs + 4 * j, wregs + 4 * j, rms_inv);
        }
        q += blockDim.x;
    }
}

// Half and FP8 widths accumulate in FP32, matching the two-pass kernels.
#define DEFINE_RMS_NORM_REGS(suffix, dtype, acc) \
__global__ void rms_norm_regs_##suffix( \
    const dtype* input, const dtype* weight, dtype* output, \
    unsigned int batch_size, unsigned int hidden_size, acc eps \
) { \
    extern __shared__ acc rms_regs_smem_##acc[]; \
    rms_norm_regs_impl<dtype, acc, false>( \
        input, weight, output, batch_size, hidden_size, eps, rms_regs_smem_##acc); \
} \
__global__ void rms_norm_quad_##suffix( \
    const dtype* input, const dtype* weight, dtype* output, \
    unsigned int batch_size, unsigned int hidden_size, acc eps \
) { \
    extern __shared__ acc rms_regs_smem_##acc[]; \
    rms_norm_regs_impl<dtype, acc, true>( \
        input, weight, output, batch_size, hidden_size, eps, rms_regs_smem_##acc); \
}

extern "C" {

NORM_DTYPES(DEFINE_RMS_NORM_REGS)

} // extern "C"

#endif // NUMR_RMS_NORM_REGS_CUH
