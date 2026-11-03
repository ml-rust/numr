// Implicit-GEMM tensor-core conv2d body for NHWC input and OHWI weight.
//
// GEMM view:
//   M = N * H_out * W_out   one row per output pixel, row m <-> (n, ho, wo)
//   N = C_out               one column per output channel
//   K = Kh * Kw * C_in      ordered (kh, kw, ci), the OHWI weight order
// A[m][k] is the input element under tap (kh, kw) of pixel m, or 0 in the
// padding. It is never written to memory: the staging loop gathers it
// straight from the NHWC input into shared memory. B is the OHWI weight read
// as [C_out][K] row-major, so B is stored N x K with K contiguous.
//
// Reused unchanged from the GEMM headers: the 16x16x16 fragment shape, the
// launch bounds, the default BLOCK_K, the two-stage ring, the 128-bit staging
// store (WMMA_VEC_STAGE), the accumulator declarations and the scratch
// stores (NUMR_WMMA_ACC_DECL_*, NUMR_WMMA_SCRATCH_*). The GEMM body macro
// hard-codes a row-major [K][N] B and a flat C index, so this file has its
// own body, K-step and staging.
//
// Shared memory per stage:
//   A: BLOCK_TILE_M x (BLOCK_K + 8) halves, [m][k]
//   B: BLOCK_TILE_N x (BLOCK_K + 8) halves, [n][k]
// Both operands are K-contiguous. F16 reads B as a col_major wmma matrix_b.
// BF16 reads B with ldmatrix without .trans, which hands mma.sync .row.col
// its B operand directly. The weight is never transposed in memory.
//
// A separate table holds (n, ho, wo) for every row of the block tile. It is
// filled once per block, read by every A staging pass, and read again by the
// epilogue to address the strided output and residual.
//
// Staging is synchronous double buffering, as in matmul_wmma.cuh. cp.async
// is not used: see matmul_wmma.cu for the unresolved race.
//
// Results are deterministic: every output element sums its K terms in one
// fixed order, with no split-K.

#ifndef NUMR_CONV_IMPLICIT_GEMM_CUH
#define NUMR_CONV_IMPLICIT_GEMM_CUH

// The `#if __CUDA_ARCH__ >= 700` guard lives in the including .cu file.

#include "matmul_wmma.cuh"

// Threads per block. Every tile keeps 16 warps, the count WMMA_LAUNCH_BOUNDS
// is written for. Tiles differ in the warp grid and the fragments per warp.
#define CONV_IGEMM_THREADS 512

#include "conv_implicit_gemm_stage.cuh"
#include "conv_implicit_gemm_epilogue.cuh"

/* Tile constants, declared inside the kernel body so each instantiation can
   pick its own tile.
     WR x WC : warp grid (WR * WC * 32 == CONV_IGEMM_THREADS)
     WM x WN : 16x16 fragments per warp
   Block tile = (WR * WM * 16) x (WC * WN * 16). */
#define CONV_TILE_CONSTANTS(WR, WC, WM, WN, BLOCK_K_ARG, HALF_T)             \
    constexpr int BLOCK_K = (BLOCK_K_ARG);                                   \
    constexpr int CONV_WARP_COLS = (WC);                                     \
    constexpr int BLOCK_TILE_M = (WR) * (WM) * WMMA_M;                       \
    constexpr int BLOCK_TILE_N = (WC) * (WN) * WMMA_N;                       \
    constexpr int SMEM_STRIDE_K = BLOCK_K + 8;                               \
    constexpr int SMEM_A_ELEMS = BLOCK_TILE_M * SMEM_STRIDE_K;               \
    constexpr int SMEM_B_ELEMS = BLOCK_TILE_N * SMEM_STRIDE_K;               \
    constexpr int K_VECS_PER_ROW = BLOCK_K / WMMA_VEC_HALVES;                \
    constexpr unsigned int SMEM_STAGE_BYTES_C =                              \
        WMMA_STAGES * (SMEM_A_ELEMS + SMEM_B_ELEMS) * sizeof(HALF_T);        \
    constexpr unsigned int SMEM_SCRATCH_BYTES_C =                            \
        (WR) * (WC) * WMMA_M * WMMA_N * sizeof(float);                       \
    constexpr unsigned int SMEM_POOL_BYTES =                                 \
        SMEM_STAGE_BYTES_C > SMEM_SCRATCH_BYTES_C                            \
            ? SMEM_STAGE_BYTES_C : SMEM_SCRATCH_BYTES_C;                     \
    constexpr unsigned int SMEM_ROWS_BYTES = BLOCK_TILE_M * sizeof(int4);    \
    static_assert((WR) * (WC) * 32 == CONV_IGEMM_THREADS,                    \
                  "the warp grid must fill CONV_IGEMM_THREADS");             \
    static_assert(BLOCK_K % WMMA_K == 0, "BLOCK_K must be whole K-steps");   \
    /* A thread stages the same K column in every K-tile only when the      \
       thread count is a multiple of the columns per row. The K cursor      \
       relies on it. */                                                      \
    static_assert(CONV_IGEMM_THREADS % BLOCK_K == 0 &&                       \
                  CONV_IGEMM_THREADS % K_VECS_PER_ROW == 0,                  \
                  "thread count must be a multiple of the staging columns"); \
    /* 128-bit staging stores and ldmatrix need 16-byte aligned rows. wmma  \
       fragment pointers need 32-byte alignment: every fragment starts at a \
       multiple of 16 rows and 16 halves, so 16 rows must span 32 bytes.    */ \
    static_assert(SMEM_STRIDE_K % WMMA_VEC_HALVES == 0 &&                    \
                  (SMEM_STRIDE_K * sizeof(HALF_T)) % 16 == 0,                \
                  "smem rows must be 16-byte aligned");                      \
    static_assert((SMEM_A_ELEMS * sizeof(HALF_T)) % 32 == 0 &&               \
                  (SMEM_B_ELEMS * sizeof(HALF_T)) % 32 == 0,                 \
                  "smem stage bases must be 32-byte aligned");               \
    static_assert(SMEM_POOL_BYTES + SMEM_ROWS_BYTES <= 49152,                \
                  "conv static smem exceeds the 49152 B per-block cap");

/* One K-step of WMMA_K for F16: B is [n][k] in smem, so matrix_b is
   col_major with leading dimension SMEM_STRIDE_K. */
#define CONV_KSTEP_WMMA(WM, WN, HALF_T, SMEM_A, SMEM_B, K_SUB)               \
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, HALF_T, row_major>            \
        frag_a[WM];                                                          \
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, HALF_T, col_major>            \
        frag_b[WN];                                                          \
    for (int wi = 0; wi < (WM); wi++) {                                      \
        const int row = (warp_row * (WM) + wi) * WMMA_M;                     \
        load_matrix_sync(frag_a[wi],                                         \
                         (SMEM_A) + row * SMEM_STRIDE_K + (K_SUB),           \
                         SMEM_STRIDE_K);                                     \
    }                                                                        \
    for (int wj = 0; wj < (WN); wj++) {                                      \
        const int col = (warp_col * (WN) + wj) * WMMA_N;                     \
        load_matrix_sync(frag_b[wj],                                         \
                         (SMEM_B) + col * SMEM_STRIDE_K + (K_SUB),           \
                         SMEM_STRIDE_K);                                     \
    }                                                                        \
    for (int wi = 0; wi < (WM); wi++) {                                      \
        for (int wj = 0; wj < (WN); wj++) {                                  \
            mma_sync(frag_c[wi][wj], frag_a[wi], frag_b[wj],                 \
                     frag_c[wi][wj]);                                        \
        }                                                                    \
    }

/* One K-step of WMMA_K for BF16 (sm_80+). A loads as in the GEMM. B is
   [n][k] in smem, so ldmatrix without .trans gives each lane the
   k-contiguous pairs mma.sync .row.col expects. With the shared lane split
   (row = mma_row, col = mma_col) the four matrices come back as
     d0 = (n 0-7, k 0-7)   d1 = (n 8-15, k 0-7)
     d2 = (n 0-7, k 8-15)  d3 = (n 8-15, k 8-15)
   so n-sub 0 takes {d0, d2} and n-sub 1 takes {d1, d3}. */
#define CONV_KSTEP_RAW(WM, WN, HALF_T, SMEM_A, SMEM_B, K_SUB)                \
    unsigned int frag_a[WM][4];                                              \
    unsigned int frag_b[WN][4];                                              \
    for (int wi = 0; wi < (WM); wi++) {                                      \
        const int row = (warp_row * (WM) + wi) * WMMA_M + mma_row;           \
        NUMR_MMA_LDM_X4(frag_a[wi][0], frag_a[wi][1], frag_a[wi][2],         \
                        frag_a[wi][3],                                       \
                        (SMEM_A) + row * SMEM_STRIDE_K + (K_SUB) + mma_col); \
    }                                                                        \
    for (int wj = 0; wj < (WN); wj++) {                                      \
        const int col = (warp_col * (WN) + wj) * WMMA_N + mma_row;           \
        NUMR_MMA_LDM_X4(frag_b[wj][0], frag_b[wj][1], frag_b[wj][2],         \
                        frag_b[wj][3],                                       \
                        (SMEM_B) + col * SMEM_STRIDE_K + (K_SUB) + mma_col); \
    }                                                                        \
    for (int wi = 0; wi < (WM); wi++) {                                      \
        for (int wj = 0; wj < (WN); wj++) {                                  \
            NUMR_MMA_M16N8K16_BF16(                                          \
                frag_c[wi][wj][0], frag_c[wi][wj][1],                        \
                frag_c[wi][wj][2], frag_c[wi][wj][3],                        \
                frag_a[wi][0], frag_a[wi][1], frag_a[wi][2], frag_a[wi][3],  \
                frag_b[wj][0], frag_b[wj][2]);                               \
            NUMR_MMA_M16N8K16_BF16(                                          \
                frag_c[wi][wj][4], frag_c[wi][wj][5],                        \
                frag_c[wi][wj][6], frag_c[wi][wj][7],                        \
                frag_a[wi][0], frag_a[wi][1], frag_a[wi][2], frag_a[wi][3],  \
                frag_b[wj][1], frag_b[wj][3]);                               \
        }                                                                    \
    }

/* Kernel body. Expands against the kernel parameters declared by
   DEFINE_CONV_IGEMM in conv_implicit_gemm.cu.
     HALF_T   : __half | __nv_bfloat16
     MMA      : WMMA (F16) | RAW (BF16)
     ZERO_EXPR: the HALF_T zero
     LOAD_F32 : HALF_T -> float
     STORE_FN : float -> HALF_T */
#define CONV_IGEMM_BODY(WR, WC, WM, WN, BLOCK_K_ARG, HALF_T, MMA, ZERO_EXPR, LOAD_F32, STORE_FN) \
{                                                                            \
    CONV_TILE_CONSTANTS(WR, WC, WM, WN, BLOCK_K_ARG, HALF_T)                 \
                                                                             \
    const int tid = threadIdx.x;                                             \
    const int warp_id = tid / 32;                                            \
    const int warp_row = warp_id / CONV_WARP_COLS;                           \
    const int warp_col = warp_id % CONV_WARP_COLS;                           \
    const int block_row = (int)(blockIdx.x * BLOCK_TILE_M);                  \
    const int block_col = (int)(blockIdx.y * BLOCK_TILE_N);                  \
    const int K = kernel_h * kernel_w * c_in;                                \
                                                                             \
    /* Staging ring and epilogue scratch share one array. The scratch is    \
       written only after the barrier that follows the K-loop. */           \
    __shared__ __align__(128) char smem_raw[SMEM_POOL_BYTES];                \
    /* (n, ho, wo, unused) per block row. n is -1 past the last row. */     \
    __shared__ int4 smem_rows[BLOCK_TILE_M];                                 \
    HALF_T* const smem_A0 = reinterpret_cast<HALF_T*>(smem_raw);             \
    HALF_T* const smem_B0 = smem_A0 + SMEM_A_ELEMS;                          \
    HALF_T* const smem_A1 = smem_B0 + SMEM_B_ELEMS;                          \
    HALF_T* const smem_B1 = smem_A1 + SMEM_A_ELEMS;                          \
    float* const smem_scratch = reinterpret_cast<float*>(smem_raw);          \
    static_assert(WMMA_STAGES == 2, "the stage selection below names two"); \
                                                                             \
    CONV_FILL_ROW_TABLE(smem_rows)                                           \
    __syncthreads();                                                         \
                                                                             \
    NUMR_WMMA_ACC_DECL_##MMA(WM, WN)                                         \
                                                                             \
    /* Staging paths. Both tests are block-uniform. The 128-bit A path      \
       needs every 8-element K group to be 8 contiguous, 16-byte aligned    \
       channels of one tap. The 128-bit B path needs 16-byte aligned rows. */ \
    const bool a_vec = (c_in % WMMA_VEC_HALVES) == 0 &&                      \
        CONV_ALIGNED16(input) && (in_sn % WMMA_VEC_HALVES) == 0 &&           \
        (in_sh % WMMA_VEC_HALVES) == 0 && (in_sw % WMMA_VEC_HALVES) == 0;    \
    const bool b_vec = (K % WMMA_VEC_HALVES) == 0 && CONV_ALIGNED16(weight); \
    /* The K column this thread stages. It is the same in every K-tile. */  \
    const int a_col = a_vec ? (tid % K_VECS_PER_ROW) * WMMA_VEC_HALVES       \
                            : tid % BLOCK_K;                                 \
    const int b_col = b_vec ? (tid % K_VECS_PER_ROW) * WMMA_VEC_HALVES       \
                            : tid % BLOCK_K;                                 \
    ConvKCursor a_k = conv_k_cursor_init(a_col, c_in, kernel_w);             \
                                                                             \
    const int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;                     \
    if (num_k_tiles > 0) {                                                   \
        CONV_STAGE_A(HALF_T, ZERO_EXPR, smem_A0)                             \
        CONV_STAGE_B(HALF_T, ZERO_EXPR, smem_B0, 0)                          \
        conv_k_cursor_advance(a_k, BLOCK_K, c_in, kernel_w);                 \
        __syncthreads();                                                     \
                                                                             \
        for (int bk = 0; bk < num_k_tiles; bk++) {                           \
            const int stage_cur = bk % WMMA_STAGES;                          \
            const HALF_T* smem_A_cur = (stage_cur == 0) ? smem_A0 : smem_A1; \
            const HALF_T* smem_B_cur = (stage_cur == 0) ? smem_B0 : smem_B1; \
            HALF_T* smem_A_nxt = (stage_cur == 0) ? smem_A1 : smem_A0;       \
            HALF_T* smem_B_nxt = (stage_cur == 0) ? smem_B1 : smem_B0;       \
                                                                             \
            /* Loads for tile bk+1 go into the stage last read at bk-1,     \
               before that iteration's closing barrier. They are issued     \
               ahead of the mma work so their latency overlaps it. */       \
            if (bk + 1 < num_k_tiles) {                                      \
                CONV_STAGE_A(HALF_T, ZERO_EXPR, smem_A_nxt)                  \
                CONV_STAGE_B(HALF_T, ZERO_EXPR, smem_B_nxt,                  \
                             (bk + 1) * BLOCK_K)                             \
                conv_k_cursor_advance(a_k, BLOCK_K, c_in, kernel_w);         \
            }                                                                \
                                                                             \
            for (int k = 0; k < BLOCK_K; k += WMMA_K) {                      \
                CONV_KSTEP_##MMA(WM, WN, HALF_T, smem_A_cur, smem_B_cur, k)  \
            }                                                                \
                                                                             \
            if (bk + 1 < num_k_tiles) {                                      \
                __syncthreads();                                             \
            }                                                                \
        }                                                                    \
    }                                                                        \
                                                                             \
    /* Orders every fragment read of the staging buffers before the first  \
       scratch write, since the scratch aliases them. */                    \
    __syncthreads();                                                         \
                                                                             \
    /* Epilogue: F32 fragment -> per-warp scratch -> bias, activation,      \
       residual in F32 -> one narrowing store at the output's strides. */   \
    float* const warp_scratch = smem_scratch + warp_id * (WMMA_M * WMMA_N);  \
    const int lane = tid % 32;                                               \
    /* Unrolled so frag_c is indexed by constants and stays in registers. */ \
    _Pragma("unroll")                                                        \
    for (int wi = 0; wi < (WM); wi++) {                                      \
        _Pragma("unroll")                                                    \
        for (int wj = 0; wj < (WN); wj++) {                                  \
            const int local_r0 = (warp_row * (WM) + wi) * WMMA_M;            \
            const int gc = block_col + (warp_col * (WN) + wj) * WMMA_N;      \
            NUMR_WMMA_SCRATCH_##MMA(warp_scratch, wi, wj)                    \
            __syncwarp();                                                    \
            for (int idx = lane; idx < WMMA_M * WMMA_N; idx += 32) {         \
                const int r = idx / WMMA_N;                                  \
                const int co = gc + idx % WMMA_N;                            \
                const int4 pix = smem_rows[local_r0 + r];                    \
                if (pix.x >= 0 && co < c_out) {                              \
                    CONV_EPILOGUE_STORE(LOAD_F32, STORE_FN,                  \
                                        warp_scratch[idx], pix, co)          \
                }                                                            \
            }                                                                \
            __syncwarp();                                                    \
        }                                                                    \
    }                                                                        \
}

#endif  // NUMR_CONV_IMPLICIT_GEMM_CUH
