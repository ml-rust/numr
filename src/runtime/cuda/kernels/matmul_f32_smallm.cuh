// Small-M FP32 GEMM against a transposed weight: C[M,N] = A[M,K] · Bᵀ with
// B stored [N,K] row-major, one thread per output element.
//
// WHY: the tiled family serves M <= 64 with its 16-row tile
// (`matmul_f32_tiled_bt_16x64x32_4x4`). At the decode shapes, M in 1..=4,
// 12 to 15 of those 16 rows are padding, so most of the FMAs, the shared
// memory traffic and the barriers do nothing. `[1,5120] x [5120,48]ᵀ` takes
// ~0.24 ms there. This kernel launches only the threads that own an output.
//
// BIT IDENTITY: the tiled kernel forms each output as ONE FMA PER k, k
// ascending (`accum[i][j] += reg_a[i] * reg_b[j]`, contracted to `fma.rn`
// by nvcc `-fmad=true`, which `--use_fast_math` implies); the k's past K in
// its last tile are `fma(0, 0, acc)`, which is `acc` for every finite,
// infinite and NaN `acc` (the accumulator starts at +0 and round-to-nearest
// never produces -0 from a sum, so the sign is safe too). This kernel forms
// each output as `acc = fmaf(a[k], b[k], acc)` for k ascending, explicit
// `fmaf` so the contraction does not depend on build flags, with ONE
// accumulator chain: the unroll issues eight independent loads and then
// eight dependent FMAs, never a second partial sum. The bits are therefore
// identical to the tiled kernel's at every M, which is what
// `tests/cuda_matmul_batch_invariance.rs` checks through the public op and
// `tests/cuda_matmul_smallm_bt_parity.rs` checks kernel against kernel.
//
// Grid: (ceil(N/256), M, batch)   Block: (256, 1, 1)
//
// SCOPE: N <= MAX_SMALL_N (loader/matmul_f32_smallm.rs). Adjacent threads
// read B rows K floats apart, so their loads never coalesce; past a few
// hundred rows that traffic costs more than the tile padding saved, and the
// tiled kernel, which stages B through shared memory, is faster.
//
// Thread t of block (bx, m, z) owns C[m][bx*256 + t]. All threads of a block
// read the same A row, so its loads broadcast from L1; each thread streams
// its own B row. When K is a multiple of four and both operands start
// 16-byte aligned, every row start is 16-byte aligned, so the loads are
// float4; otherwise scalar. Load width does not change the FMA sequence.
//
// This header is included by matmul.cu only, so the `extern "C"` entry
// points live here as well: a CUDA module exposes only the kernels compiled
// into its translation unit, and matmul.cu is the `matmul` module.

#ifndef NUMR_MATMUL_F32_SMALLM_CUH
#define NUMR_MATMUL_F32_SMALLM_CUH

#include <stdint.h>

// LOAD SCHEDULING: the chain is one dependent FMA per k, so the kernel's
// floor is about K x 4 cycles of FMA latency per thread, whatever the
// loads do. The loads must not add to that floor: each must be in flight
// long before the FMA that consumes it. So the k loop works in groups of 16
// and stages two groups in registers: group g+1's eight float4 (four per
// operand) are issued before group g's sixteen FMAs run, so at any time up
// to sixteen float4 are outstanding and the memory latency of a group
// overlaps the ~64 cycles of FMA latency of the one before it. Only the
// issue order of loads changes; the FMAs consume a[k], b[k] for k ascending
// on the one accumulator exactly as before.

// Four consecutive FMAs on the one chain from a quad of each operand.
__device__ __forceinline__ float smallm_fma4(float acc, float4 a, float4 b) {
    acc = fmaf(a.x, b.x, acc);
    acc = fmaf(a.y, b.y, acc);
    acc = fmaf(a.z, b.z, acc);
    acc = fmaf(a.w, b.w, acc);
    return acc;
}

// Sixteen elements of each operand starting at `k`, as four float4 each.
// Aligned float4 loads only: the vec path's two-stage pipeline (group g+1's
// eight LDG.128 issued ahead of group g's sixteen FFMA) depends on every
// load here being a plain float4 read, so this has no branch on `vec`.
__device__ __forceinline__ void smallm_load16(
    const float* __restrict__ a_row,
    const float* __restrict__ b_row,
    unsigned int k,
    float4 (&a)[4],
    float4 (&b)[4]
) {
    #pragma unroll
    for (int q = 0; q < 4; q++) {
        a[q] = *reinterpret_cast<const float4*>(a_row + k + 4 * q);
        b[q] = *reinterpret_cast<const float4*>(b_row + k + 4 * q);
    }
}

// One output element: the dot product of `a_row[0..K)` and `b_row[0..K)`
// as one ascending FMA chain.
__device__ __forceinline__ float smallm_dot_f32(
    const float* __restrict__ a_row,
    const float* __restrict__ b_row,
    unsigned int K,
    bool vec
) {
    float acc = 0.f;
    unsigned int k = 0;

    if (vec) {
        float4 a[4], b[4];
        float4 an[4], bn[4];
        if (k + 16 <= K) {
            smallm_load16(a_row, b_row, k, a, b);
        }
        for (; k + 16 <= K; k += 16) {
            // Issue the next group's loads before consuming this group.
            const bool more = k + 32 <= K;
            if (more) {
                smallm_load16(a_row, b_row, k + 16, an, bn);
            }
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                acc = smallm_fma4(acc, a[q], b[q]);
            }
            if (more) {
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    a[q] = an[q];
                    b[q] = bn[q];
                }
            }
        }
        // K is a multiple of four on this path: up to three quads remain.
        for (; k + 4 <= K; k += 4) {
            const float4 a0 = *reinterpret_cast<const float4*>(a_row + k);
            const float4 b0 = *reinterpret_cast<const float4*>(b_row + k);
            acc = smallm_fma4(acc, a0, b0);
        }
    } else {
        // Scalar path: no register-staged pipeline, just a plain quad loop
        // consumed through the same `smallm_fma4` chain as the vec path.
        for (; k + 4 <= K; k += 4) {
            const float4 a0 = make_float4(a_row[k], a_row[k + 1], a_row[k + 2], a_row[k + 3]);
            const float4 b0 = make_float4(b_row[k], b_row[k + 1], b_row[k + 2], b_row[k + 3]);
            acc = smallm_fma4(acc, a0, b0);
        }
    }

    // Ragged tail, still ascending on the same chain.
    for (; k < K; k++) {
        acc = fmaf(a_row[k], b_row[k], acc);
    }
    return acc;
}

// One [M,K] x [N,K]ᵀ product for the block's (m, n) pair. Batch offsets are
// applied by the caller.
__device__ __forceinline__ void matmul_f32_smallm_bt_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int m = blockIdx.y;
    if (m >= M || n >= N) return;

    const float* a_row = A + (size_t)m * K;
    const float* b_row = B + (size_t)n * K;

    // Block-uniform: A's base and K are the same for every thread, and B's
    // row start `B + n*K` is 16-byte aligned for every n once B's base is
    // and K is a multiple of four.
    const bool vec = (K & 3u) == 0u
        && (reinterpret_cast<uintptr_t>(A) & 15u) == 0u
        && (reinterpret_cast<uintptr_t>(B) & 15u) == 0u;

    C[(size_t)m * N + n] = smallm_dot_f32(a_row, b_row, K, vec);
}

// C[M,N] = A[M,K] · B[N,K]ᵀ.
extern "C" __global__ void matmul_f32_smallm_bt(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    matmul_f32_smallm_bt_impl(A, B, C, M, N, K);
}

// C[batch,M,N] = A[batch,M,K] · B[batch,N,K]ᵀ. Batch index is blockIdx.z;
// `a_batch_count` / `b_batch_count` let one operand broadcast over the
// batch, the same `b % count` convention as `matmul_batched_f32_tiled_bt_*`.
extern "C" __global__ void matmul_batched_f32_smallm_bt(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const float* A_batch = A + (size_t)(b % a_batch_count) * ((size_t)M * K);
    const float* B_batch = B + (size_t)(b % b_batch_count) * ((size_t)K * N);
    float* C_batch = C + (size_t)b * ((size_t)M * N);

    matmul_f32_smallm_bt_impl(A_batch, B_batch, C_batch, M, N, K);
}

#endif // NUMR_MATMUL_F32_SMALLM_CUH
