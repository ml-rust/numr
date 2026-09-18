// Compile-time-tiled FP32 GEMM core with a pluggable epilogue.
//
// WHY: the dynamic matmul_f32 / matmul_bias_f32 / gemm_bias_act_f32 kernels
// pass BM/BN/BK/TM/TN as runtime unsigned-int arguments. NVCC cannot unroll
// the micro-kernel loops when the bounds are runtime values, so reg_c[TM][TN]
// spills to local memory and every FMA becomes a pair of local-memory
// loads/stores.
//
// FIX: make tile sizes compile-time C++ template parameters so NVCC can:
//   1. Unroll all inner loops with #pragma unroll (no loop overhead).
//   2. Keep accum[TM][TN], reg_a[TM], reg_b[TN] in registers (no spill).
//   3. Emit back-to-back FMA instructions filling the FMA pipe.
//
// Structure:
//   - Double-buffered shared memory (two ping-pong slots), with the next
//     tile's global loads issued into registers before the current tile's
//     micro-kernel and stored to the idle slot after it.
//   - Float4-vectorised cooperative tile loads, scalar fallback for ragged edges.
//   - Full bounds checks on A/B loads (out-of-range -> 0) and C stores.
//   - Grid: (ceil(N/BN), ceil(M/BM), 1)   Block: (BN/TN, BM/TM, 1)
//
// The epilogue is a compile-time functor parameter, so bias, activation and
// residual variants reuse this one GEMM core instead of copying it. A functor
// carries its own pointers, so the core needs no extra arguments per variant.
//
// This header holds the device code only. Each translation unit that includes
// it declares its own extern "C" entry points, because a CUDA module only
// exposes the kernels compiled into it.

#ifndef NUMR_MATMUL_F32_TILED_CUH
#define NUMR_MATMUL_F32_TILED_CUH

// Storage dtype is a template parameter defaulting to `float`, so every
// existing call site is unchanged and its generated code identical: for
// `float`, AccumTraits load/store are `p[i]` and `p[i] = v`, and the
// float4 path is selected at compile time exactly as before.
//
// Shared memory and the accumulator stay FP32 whatever the storage dtype, so a
// half-precision GEMM here accumulates in single precision rather than in half.
#include <stdint.h>
#include "dtype_traits.cuh"

// Tile loads: `ct_fetch_*` / `ct_store_*` in matmul_f32_tiled_fetch.cuh, a
// register-staged pair so the next tile's global latency overlaps this
// tile's micro-kernel.
#include "matmul_f32_tiled_fetch.cuh"

// ---------------------------------------------------------------------------
// Epilogue functors
// ---------------------------------------------------------------------------
//
// `apply` receives the accumulated value, the flat output index
// (global_row * N + global_col) and the output column, and returns the value
// to store.  It runs only for in-range (row, col) pairs, so an epilogue may
// index bias[col] and residual[idx] without its own bounds check.
//
// MatmulEpilogueNone is the plain-GEMM store and the template default.

struct MatmulEpilogueNone {
    __device__ __forceinline__ float apply(float acc, unsigned int, unsigned int) const {
        return acc;
    }
};

// ---------------------------------------------------------------------------
// Main templated kernel
// ---------------------------------------------------------------------------
//
// Template parameters:
//   BM, BN   - block tile rows/cols (shared memory footprint)
//   BK       - K-depth of tile (controls inner-loop trip count)
//   TM, TN   - per-thread register micro-tile
//   Epilogue - functor applied to each accumulator before the store
//
// Thread block: (BN/TN) × (BM/TM)   [x × y]
// Grid:         (ceil(N/BN), ceil(M/BM), 1)
// Shared memory: 2 × (BM×BK + BK×BN) × 4 bytes  (double-buffered, static)
//
// NOTE: smem uses static 2-D arrays so the index arithmetic is compile-time.
// The double-buffer is realised as two separate static arrays (buf0/buf1) rather
// than a runtime-indexed extern __shared__ pointer so the compiler sees fixed
// strides and can pipeline the loads.
// ---------------------------------------------------------------------------
//   BT       - B is [N × K] row-major (C = A · Bᵀ) instead of [K × N]
template<int BM, int BN, int BK, int TM, int TN, class Epilogue = MatmulEpilogueNone,
         typename T = float, bool BT = false>
__device__ __forceinline__ void matmul_f32_tiled_impl(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    Epilogue epi = Epilogue()
) {
    // Static shared memory — compile-time sizes enable unrolling. The B tile
    // carries four floats of row padding on the transposed path: the
    // transposed scatter in `ct_store_bt` then spreads a warp's stores over
    // the banks, and every row stays 16-byte aligned so the micro-kernel
    // reads its TN-wide B fragment as float4.
    constexpr int BNP = BT ? BN + 4 : BN;
    __shared__ __align__(16) float As0[BM][BK];
    __shared__ __align__(16) float Bs0[BK][BNP];
    __shared__ __align__(16) float As1[BM][BK];
    __shared__ __align__(16) float Bs1[BK][BNP];

    const unsigned int tx = threadIdx.x;      // [0, BN/TN)
    const unsigned int ty = threadIdx.y;      // [0, BM/TM)
    const unsigned int block_row = blockIdx.y * BM;
    const unsigned int block_col = blockIdx.x * BN;
    const unsigned int thread_row = ty * TM;  // start row within block tile
    const unsigned int thread_col = tx * TN;  // start col within block tile

    // Register accumulator — stays in registers because TM/TN are compile-time.
    float accum[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            accum[i][j] = 0.f;
        }
    }

    float reg_a[TM];
    float reg_b[TN];

    const unsigned int num_k_tiles = (K + BK - 1) / BK;
    const unsigned int thread_id  = ty * (BN / TN) + tx;
    constexpr int THREADS = (BM / TM) * (BN / TN);

    // Float4 loads when the storage is 32-bit, the operand base is 16-byte
    // aligned and the contiguous axis is a whole number of quads, so every
    // row start stays aligned; block-uniform, so fetch and store agree. A
    // view into a larger buffer can start at any element, which is what the
    // base test guards: a misaligned 128-bit load faults.
    constexpr bool is_f32 = std::is_same_v<T, float>;
    const bool a_aligned = (reinterpret_cast<uintptr_t>(A) & 15u) == 0u;
    const bool b_aligned = (reinterpret_cast<uintptr_t>(B) & 15u) == 0u;
    const bool vec_a = is_f32 && a_aligned && (BK & 3) == 0 && (K & 3u) == 0u;
    const bool vec_b = BT ? (is_f32 && b_aligned && (BK & 3) == 0 && (K & 3u) == 0u)
                          : (is_f32 && b_aligned && (BN & 3) == 0 && (N & 3u) == 0u
                             && BNP == BN);
    CtTileRegs<BM * BK, THREADS> ra;
    CtTileRegs<BK * BN, THREADS> rb;

    // K == 0 leaves the accumulator at zero and falls straight to the
    // epilogue, so C is still written (the generic kernels do the same).
    // The branch is uniform across the block, so the __syncthreads calls
    // inside stay collective.
    if (num_k_tiles > 0) {

        // Preload tile 0 into buffer 0.
        ct_fetch_a<T, BM, BK, THREADS>(A, block_row, 0u, M, K, thread_id, vec_a, ra);
        if constexpr (BT) {
            ct_fetch_bt<T, BK, BN, THREADS>(B, block_col, 0u, K, N, thread_id, vec_b, rb);
        } else {
            ct_fetch_b<T, BK, BN, BNP, THREADS>(B, block_col, 0u, K, N, thread_id, vec_b, rb);
        }
        ct_store_a<BM, BK, THREADS>(As0, thread_id, vec_a, ra);
        if constexpr (BT) {
            ct_store_bt<BK, BN, BNP, THREADS>(Bs0, thread_id, vec_b, rb);
        } else {
            ct_store_b<BK, BN, BNP, THREADS>(Bs0, thread_id, vec_b, rb);
        }
        __syncthreads();

        for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
            // Select current / next ping-pong buffers.
            float (*As_cur)[BK] = (bk & 1u) ? As1 : As0;
            float (*Bs_cur)[BNP] = (bk & 1u) ? Bs1 : Bs0;
            float (*As_nxt)[BK] = (bk & 1u) ? As0 : As1;
            float (*Bs_nxt)[BNP] = (bk & 1u) ? Bs0 : Bs1;

            // Issue the next tile's global loads now; they land in registers
            // while the micro-kernel below runs, and are stored after it.
            const unsigned int next_k = (bk + 1) * BK;
            const bool has_next = bk + 1 < num_k_tiles;
            if (has_next) {
                ct_fetch_a<T, BM, BK, THREADS>(A, block_row, next_k, M, K, thread_id, vec_a, ra);
                if constexpr (BT) {
                    ct_fetch_bt<T, BK, BN, THREADS>(B, block_col, next_k, K, N, thread_id,
                                                    vec_b, rb);
                } else {
                    ct_fetch_b<T, BK, BN, BNP, THREADS>(B, block_col, next_k, K, N, thread_id,
                                                        vec_b, rb);
                }
            }

            // Micro-kernel: fully unrolled, accumulators stay in registers.
            #pragma unroll
            for (int k = 0; k < BK; k++) {
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    reg_a[i] = As_cur[thread_row + i][k];
                }
                // `thread_col` is a multiple of TN, so a TN that is a whole
                // number of quads reads 16-byte aligned float4s.
                if constexpr (TN % 4 == 0) {
                    #pragma unroll
                    for (int j = 0; j < TN; j += 4) {
                        const float4 q =
                            *reinterpret_cast<const float4*>(&Bs_cur[k][thread_col + j]);
                        reg_b[j] = q.x;
                        reg_b[j + 1] = q.y;
                        reg_b[j + 2] = q.z;
                        reg_b[j + 3] = q.w;
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < TN; j++) {
                        reg_b[j] = Bs_cur[k][thread_col + j];
                    }
                }
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    #pragma unroll
                    for (int j = 0; j < TN; j++) {
                        accum[i][j] += reg_a[i] * reg_b[j];
                    }
                }
            }

            // Publish the next tile: the idle slot was last read at bk-1,
            // behind that iteration's barrier, so the stores race nothing.
            if (has_next) {
                ct_store_a<BM, BK, THREADS>(As_nxt, thread_id, vec_a, ra);
                if constexpr (BT) {
                    ct_store_bt<BK, BN, BNP, THREADS>(Bs_nxt, thread_id, vec_b, rb);
                } else {
                    ct_store_b<BK, BN, BNP, THREADS>(Bs_nxt, thread_id, vec_b, rb);
                }
                __syncthreads();
            }
        }

    }  // num_k_tiles > 0

    // Epilogue: write results with full bounds check.
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    const unsigned int idx = global_row * N + global_col;
                    AccumTraits<T, float>::store(
                        C, (int)idx, epi.apply(accum[i][j], idx, global_col));
                }
            }
        }
    }
}

#endif // NUMR_MATMUL_F32_TILED_CUH
