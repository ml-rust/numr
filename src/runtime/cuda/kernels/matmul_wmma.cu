// WMMA (Tensor-Core) GEMM for F16 and BF16
//
// Uses nvcuda::wmma 16x16x16 fragments with F32 accumulation.
// Shared-memory staging of A and B tiles means global addresses never need
// to satisfy the fragment-alignment requirement (sidesteps the
// CUDA_ERROR_MISALIGNED_ADDRESS class of bug that float4 global loads hit).
//
// Layout per warp tile (WMMA_M=16, WMMA_N=16, WMMA_K=16):
//   Block processes WARP_ROWS*WMMA_M rows x WARP_COLS*WMMA_N columns.
//   Block = WARP_ROWS*WARP_COLS warps  (each warp = 32 threads).
//   Grid  = ceil(N / (WARP_COLS*WMMA_N)) x ceil(M / (WARP_ROWS*WMMA_M)) [x batch]
//
// Caller must guarantee M, N, K are all multiples of 16 before dispatching here.
// FMA fallback handles all other shapes.
//
// sm_80+: double-buffered cp.async K-tile prefetch overlaps global-load latency
// with WMMA compute.  Falls back to plain __syncthreads loads on sm_70/sm_75.

#if __CUDA_ARCH__ >= 700

#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARP_ROWS 4
#define WARP_COLS 4
// NOTE: BLOCK_K=16. BLOCK_K=32 was tried for fewer cp.async cycles on large-K
// projections (~2% p50) but introduced a correctness regression on the real
// workload (reranker 5-query recall@10 1.000→0.400) that the synthetic K-tail
// parity tests (K=48/80) did NOT catch. Reverted; not worth broken recall.
#define BLOCK_K   16

#define BLOCK_TILE_M  (WARP_ROWS * WMMA_M)   // 64
#define BLOCK_TILE_N  (WARP_COLS * WMMA_N)   // 64
#define SMEM_STRIDE_A (BLOCK_K  + 8)          // 24 halves — avoids 32-bank conflict
#define SMEM_STRIDE_B (BLOCK_TILE_N + 8)      // 72 halves

// ---------------------------------------------------------------------------
// cp.async helpers (sm_80+)
// ---------------------------------------------------------------------------

// Copy one __half (2 bytes) from global to shared using cp.async.ca (4-byte).
// We always use 4-byte granularity (cp.async.ca) to avoid any alignment
// requirement beyond the natural 2-byte alignment of __half.
// The instruction copies SRC_BYTES bytes; unused bytes are zeroed when the
// predicate is false (out-of-bounds element), matching the scalar path.

#if __CUDA_ARCH__ >= 800

// Copy 4 bytes (two __half elements) from global to shared.
// pred: true  => real copy;  false => fills shared with zero.
__device__ __forceinline__
void async_copy_2halves(void* dst, const void* src, bool pred) {
    unsigned dst_off = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    if (pred) {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4;\n"
            :: "r"(dst_off), "l"(src)
        );
    } else {
        // Zero the destination — shared mem is NOT automatically zeroed.
        asm volatile(
            "cp.async.ca.shared.global.zero [%0], [%1], 4, 0;\n"
            :: "r"(dst_off), "l"(src)
        );
    }
}

__device__ __forceinline__ void async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most N groups are in flight (i.e., all older groups finish).
__device__ __forceinline__ void async_wait_1() {
    asm volatile("cp.async.wait_group 1;\n" ::);
}

__device__ __forceinline__ void async_wait_0() {
    asm volatile("cp.async.wait_all;\n" ::);
}

#endif // __CUDA_ARCH__ >= 800

// ---------------------------------------------------------------------------
// Kernel body macro — instantiated for F16 (non-batched), F16 (batched),
// BF16 (non-batched), BF16 (batched).
//
// HALF_T  : __half | __nv_bfloat16
// ZERO    : __float2half(0.0f) | __float2bfloat16(0.0f)
// STORE   : __float2half | __float2bfloat16
//
// For the batched variant, the caller sets up A_ptr/B_ptr/C_ptr from the
// batch-slice offset before entering WMMA_KERNEL_BODY.
// ---------------------------------------------------------------------------

#define WMMA_KERNEL_BODY(HALF_T, ZERO_EXPR, STORE_FN, A_ptr, B_ptr, C_ptr) \
{                                                                            \
    const unsigned int warp_id  = threadIdx.x / 32;                         \
    const unsigned int warp_row = warp_id / WARP_COLS;                      \
    const unsigned int warp_col = warp_id % WARP_COLS;                      \
    const unsigned int block_row = blockIdx.y * BLOCK_TILE_M;               \
    const unsigned int block_col = blockIdx.x * BLOCK_TILE_N;               \
    const unsigned int num_threads = blockDim.x;                            \
    const unsigned int tid = threadIdx.x;                                    \
                                                                             \
    /* Double-buffered smem: [2][...] layout, ping-pong index = buf_idx. */ \
    __shared__ HALF_T smem_A[2][BLOCK_TILE_M * SMEM_STRIDE_A];              \
    __shared__ HALF_T smem_B[2][BLOCK_K      * SMEM_STRIDE_B];              \
                                                                             \
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;            \
    fill_fragment(frag_c, 0.0f);                                             \
                                                                             \
    const unsigned int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;          \
                                                                             \
    /* ------------------------------------------------------------------ */ \
    /* sm_80+ path: double-buffered cp.async                               */ \
    /* ------------------------------------------------------------------ */ \
    IF_ASYNC(                                                                \
        /* Prologue: issue cp.async loads for tile 0 into buffer 0. */      \
        {                                                                    \
            const unsigned int k_off = 0;                                   \
            for (unsigned int idx = tid;                                     \
                 idx < BLOCK_TILE_M * BLOCK_K; idx += num_threads) {        \
                unsigned int r = idx / BLOCK_K;                             \
                unsigned int c = idx % BLOCK_K;                             \
                unsigned int gr = block_row + r;                             \
                unsigned int gc = k_off     + c;                             \
                /* Two halves share one cp.async.ca 4-byte copy.          */ \
                /* Stride c is always even here (idx += num_threads=512), */ \
                /* but guard odd c to be safe: only copy on even c.       */ \
                if ((c & 1) == 0) {                                          \
                    bool in_a = (gr < M) && (gc + 1 <= K);                  \
                    bool in_a0 = (gr < M) && (gc < K);                      \
                    /* If both in-bounds, copy 4 bytes; else scalar load. */ \
                    if (in_a) {                                              \
                        async_copy_2halves(                                  \
                            &smem_A[0][r * SMEM_STRIDE_A + c],              \
                            &(A_ptr)[gr * K + gc], true);                    \
                    } else {                                                 \
                        /* Boundary: scalar with zero-pad. */               \
                        smem_A[0][r * SMEM_STRIDE_A + c]     =              \
                            (in_a0) ? (A_ptr)[gr * K + gc]                  \
                                    : ZERO_EXPR;                             \
                        smem_A[0][r * SMEM_STRIDE_A + c + 1] = ZERO_EXPR;  \
                    }                                                        \
                }                                                            \
            }                                                                \
            for (unsigned int idx = tid;                                     \
                 idx < BLOCK_K * BLOCK_TILE_N; idx += num_threads) {        \
                unsigned int r = idx / BLOCK_TILE_N;                        \
                unsigned int c = idx % BLOCK_TILE_N;                        \
                unsigned int gr = k_off     + r;                             \
                unsigned int gc = block_col + c;                             \
                if ((c & 1) == 0) {                                          \
                    bool in_b = (gr < K) && (gc + 1 <= N);                  \
                    bool in_b0 = (gr < K) && (gc < N);                      \
                    if (in_b) {                                              \
                        async_copy_2halves(                                  \
                            &smem_B[0][r * SMEM_STRIDE_B + c],              \
                            &(B_ptr)[gr * N + gc], true);                    \
                    } else {                                                 \
                        smem_B[0][r * SMEM_STRIDE_B + c]     =              \
                            (in_b0) ? (B_ptr)[gr * N + gc]                  \
                                    : ZERO_EXPR;                             \
                        smem_B[0][r * SMEM_STRIDE_B + c + 1] = ZERO_EXPR;  \
                    }                                                        \
                }                                                            \
            }                                                                \
            async_commit();                                                  \
        }                                                                    \
        ,                                                                    \
        /* else: no prologue needed for scalar path */                       \
        {}                                                                   \
    )                                                                        \
                                                                             \
    unsigned int buf_idx = 0;                                               \
                                                                             \
    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {                     \
        const unsigned int k_off = bk * BLOCK_K;                            \
        const unsigned int next_bk = bk + 1;                                \
        const unsigned int next_buf = 1 - buf_idx;                          \
                                                                             \
        IF_ASYNC(                                                            \
            /* sm_80+ path ------------------------------------------- */  \
            {                                                                \
                /* Prefetch tile (bk+1) into next_buf. */                   \
                if (next_bk < num_k_tiles) {                                \
                    unsigned int nk_off = next_bk * BLOCK_K;               \
                    for (unsigned int idx = tid;                             \
                         idx < BLOCK_TILE_M * BLOCK_K; idx += num_threads) {\
                        unsigned int r = idx / BLOCK_K;                     \
                        unsigned int c = idx % BLOCK_K;                     \
                        unsigned int gr = block_row + r;                    \
                        unsigned int gc = nk_off    + c;                    \
                        if ((c & 1) == 0) {                                 \
                            bool in_a = (gr < M) && (gc + 1 <= K);         \
                            bool in_a0 = (gr < M) && (gc < K);             \
                            if (in_a) {                                     \
                                async_copy_2halves(                         \
                                    &smem_A[next_buf][r * SMEM_STRIDE_A + c],\
                                    &(A_ptr)[gr * K + gc], true);           \
                            } else {                                        \
                                smem_A[next_buf][r * SMEM_STRIDE_A + c]   =\
                                    (in_a0) ? (A_ptr)[gr * K + gc]          \
                                            : ZERO_EXPR;                    \
                                smem_A[next_buf][r * SMEM_STRIDE_A + c+1] =\
                                    ZERO_EXPR;                              \
                            }                                               \
                        }                                                   \
                    }                                                       \
                    for (unsigned int idx = tid;                             \
                         idx < BLOCK_K * BLOCK_TILE_N; idx += num_threads) {\
                        unsigned int r = idx / BLOCK_TILE_N;               \
                        unsigned int c = idx % BLOCK_TILE_N;               \
                        unsigned int gr = nk_off    + r;                    \
                        unsigned int gc = block_col + c;                    \
                        if ((c & 1) == 0) {                                 \
                            bool in_b = (gr < K) && (gc + 1 <= N);         \
                            bool in_b0 = (gr < K) && (gc < N);             \
                            if (in_b) {                                     \
                                async_copy_2halves(                         \
                                    &smem_B[next_buf][r * SMEM_STRIDE_B + c],\
                                    &(B_ptr)[gr * N + gc], true);           \
                            } else {                                        \
                                smem_B[next_buf][r * SMEM_STRIDE_B + c]   =\
                                    (in_b0) ? (B_ptr)[gr * N + gc]          \
                                            : ZERO_EXPR;                    \
                                smem_B[next_buf][r * SMEM_STRIDE_B + c+1] =\
                                    ZERO_EXPR;                              \
                            }                                               \
                        }                                                   \
                    }                                                       \
                    async_commit();                                          \
                }                                                            \
                /* Wait for tile bk (buf_idx) to land. */                   \
                async_wait_1();                                              \
                __syncthreads();                                             \
            }                                                                \
            ,                                                                \
            /* sm_70/75 scalar path ----------------------------------- */  \
            {                                                                \
                for (unsigned int idx = tid;                                 \
                     idx < BLOCK_TILE_M * BLOCK_K; idx += num_threads) {    \
                    unsigned int r = idx / BLOCK_K;                         \
                    unsigned int c = idx % BLOCK_K;                         \
                    unsigned int gr = block_row + r;                        \
                    unsigned int gc = k_off     + c;                        \
                    HALF_T val = ZERO_EXPR;                                  \
                    if (gr < M && gc < K) val = (A_ptr)[gr * K + gc];       \
                    smem_A[0][r * SMEM_STRIDE_A + c] = val;                 \
                }                                                            \
                for (unsigned int idx = tid;                                 \
                     idx < BLOCK_K * BLOCK_TILE_N; idx += num_threads) {    \
                    unsigned int r = idx / BLOCK_TILE_N;                    \
                    unsigned int c = idx % BLOCK_TILE_N;                    \
                    unsigned int gr = k_off     + r;                        \
                    unsigned int gc = block_col + c;                        \
                    HALF_T val = ZERO_EXPR;                                  \
                    if (gr < K && gc < N) val = (B_ptr)[gr * N + gc];       \
                    smem_B[0][r * SMEM_STRIDE_B + c] = val;                 \
                }                                                            \
                __syncthreads();                                             \
            }                                                                \
        )                                                                    \
                                                                             \
        /* WMMA compute on tile bk from buffer buf_idx. */                  \
        for (unsigned int k = 0; k < BLOCK_K; k += WMMA_K) {               \
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, HALF_T, row_major> frag_a; \
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, HALF_T, row_major> frag_b; \
            const HALF_T* a_ptr = smem_A[buf_idx]                           \
                + (warp_row * WMMA_M) * SMEM_STRIDE_A + k;                  \
            load_matrix_sync(frag_a, a_ptr, SMEM_STRIDE_A);                 \
            const HALF_T* b_ptr = smem_B[buf_idx]                           \
                + k * SMEM_STRIDE_B + (warp_col * WMMA_N);                  \
            load_matrix_sync(frag_b, b_ptr, SMEM_STRIDE_B);                 \
            mma_sync(frag_c, frag_a, frag_b, frag_c);                       \
        }                                                                    \
                                                                             \
        IF_ASYNC(                                                            \
            /* Barrier BEFORE swapping/overwriting: ensure all warps have   \
               finished reading smem[buf_idx] (load_matrix_sync above)      \
               before the next iteration's prefetch overwrites that buffer. \
               Without this the double-buffer has a WAR race that corrupts  \
               results nondeterministically. */                            \
            { __syncthreads(); buf_idx = next_buf; }                         \
            ,                                                                \
            { __syncthreads(); }                                             \
        )                                                                    \
    }                                                                        \
                                                                             \
    /* sm_80+: drain all remaining async groups before epilogue. */         \
    IF_ASYNC( { async_wait_0(); __syncthreads(); } , {} )                    \
                                                                             \
    /* Store accumulator to global C via float staging buffer. */           \
    __shared__ float smem_out[BLOCK_TILE_M * BLOCK_TILE_N];                 \
    float* out_ptr = smem_out                                                \
        + (warp_row * WMMA_M) * BLOCK_TILE_N + (warp_col * WMMA_N);         \
    store_matrix_sync(out_ptr, frag_c, BLOCK_TILE_N, mem_row_major);        \
    __syncthreads();                                                         \
    for (unsigned int idx = tid; idx < BLOCK_TILE_M * BLOCK_TILE_N;         \
         idx += num_threads) {                                               \
        unsigned int r  = idx / BLOCK_TILE_N;                               \
        unsigned int c  = idx % BLOCK_TILE_N;                               \
        unsigned int gr = block_row + r;                                     \
        unsigned int gc = block_col + c;                                     \
        if (gr < M && gc < N) {                                              \
            (C_ptr)[gr * N + gc] = STORE_FN(smem_out[idx]);                 \
        }                                                                    \
    }                                                                        \
}

// ---------------------------------------------------------------------------
// Dispatch macros: IF_ASYNC selects cp.async or scalar path at compile-time.
// ---------------------------------------------------------------------------

// cp.async double-buffering is DISABLED: it has a data race that corrupts GEMM
// output nondeterministically (observed as reranker recall@10 flipping 0.0<->1.0
// run-to-run). A WAR barrier fix (sync before buf swap) was necessary but NOT
// sufficient — at least one more hazard remains in the async path. The synchronous
// path is deterministic AND equally fast here (the T7 speedup came from the kernel
// restructure, not the async overlap), so we use it on all arches. Do NOT re-enable
// the async path until the race is fully fixed and verified by REPEATED-RUN
// determinism on the real workload (parity tests with clean shapes did not catch it).
#if 0 // __CUDA_ARCH__ >= 800
#  define IF_ASYNC(async_code, scalar_code) async_code
#else
#  define IF_ASYNC(async_code, scalar_code) scalar_code
#endif

// ---------------------------------------------------------------------------
// F16 non-batched
// ---------------------------------------------------------------------------

extern "C" __global__ void matmul_wmma_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    WMMA_KERNEL_BODY(__half, __float2half(0.0f), __float2half, A, B, C)
}

// ---------------------------------------------------------------------------
// F16 batched
// ---------------------------------------------------------------------------

extern "C" __global__ void matmul_wmma_batched_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch) return;
    const __half* A_b = A + (b % a_batch_count) * (M * K);
    const __half* B_b = B + (b % b_batch_count) * (K * N);
    __half*       C_b = C + b * (M * N);
    WMMA_KERNEL_BODY(__half, __float2half(0.0f), __float2half, A_b, B_b, C_b)
}

// ---------------------------------------------------------------------------
// BF16 non-batched
// ---------------------------------------------------------------------------

extern "C" __global__ void matmul_wmma_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16*       __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    WMMA_KERNEL_BODY(__nv_bfloat16, __float2bfloat16(0.0f), __float2bfloat16, A, B, C)
}

// ---------------------------------------------------------------------------
// BF16 batched
// ---------------------------------------------------------------------------

extern "C" __global__ void matmul_wmma_batched_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16*       __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch) return;
    const __nv_bfloat16* A_b = A + (b % a_batch_count) * (M * K);
    const __nv_bfloat16* B_b = B + (b % b_batch_count) * (K * N);
    __nv_bfloat16*       C_b = C + b * (M * N);
    WMMA_KERNEL_BODY(__nv_bfloat16, __float2bfloat16(0.0f), __float2bfloat16, A_b, B_b, C_b)
}

#undef IF_ASYNC

#endif  // __CUDA_ARCH__ >= 700
