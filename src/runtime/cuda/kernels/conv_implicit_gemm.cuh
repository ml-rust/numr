// Implicit-GEMM tensor-core conv2d body for NHWC input and OHWI weight.
//
// GEMM view:
//   M = N * H_out * W_out   one row per output pixel, row m <-> (n, ho, wo)
//   N = C_out               one column per output channel
//   K = Kh * Kw * C_in      the OHWI weight order
// A[m][k] is the input element under one tap of pixel m, or 0 in the
// padding. It is never written to memory: the loaders gather it straight
// from the NHWC input. B is the OHWI weight read as [C_out][K] row-major.
//
// Block tile BM x BN x 32, warp tile 64 x 32. Per k16 step a warp issues 4
// A ldmatrix.x4, 2 B ldmatrix.x4 and 16 mma.sync m16n8k16 (see
// conv_implicit_gemm_mma.cuh for sm_75). Both operands sit in shared memory
// K-contiguous, as rows of 32 halves with the XOR swizzle of
// conv_stage_swizzle. B is [n][k], so ldmatrix without .trans hands
// mma.sync .row.col its B operand directly.
//
// Pipeline: two stages and a register prefetch. Global loads for K-tile
// k + 1 issue into registers before the mma work of tile k, and land in the
// other stage after it. One barrier per K-tile. cp.async is not used, so
// sm_75 runs the same pipeline.
//
// Shared memory, all static:
//   stages     2 x (BM + BN) x 32 halves; the epilogue reuses it as the
//              [BM][BN] output tile
//   row table  BM x {i64 out offset, i64 residual offset}
//   col table  2 x 32 int4, scalar loader only
//
// Results are deterministic: every output element sums its K terms in one
// fixed order, with no split-K.

#ifndef NUMR_CONV_IMPLICIT_GEMM_CUH
#define NUMR_CONV_IMPLICIT_GEMM_CUH

// The `#if __CUDA_ARCH__ >= 750` guard lives in the including .cu file.

#include "conv_implicit_gemm_mma.cuh"

/* Kernel arguments. `out` can share an allocation with `input` or
   `residual` only as a disjoint view; the host checks that, so no element is
   both read and written, and the read-only cache path is safe. */
template <typename T>
struct ConvIgemmParams {
    const T* input;
    const T* weight;
    const T* bias;
    const T* prelu;
    const T* residual;
    T* out;
    int M;
    int in_h;
    int in_w;
    int c_in;
    int out_h;
    int out_w;
    int c_out;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_top;
    int pad_left;
    int dil_h;
    int dil_w;
    long long in_sn;
    long long in_sh;
    long long in_sw;
    long long out_sn;
    long long out_sh;
    long long out_sw;
    long long res_sn;
    long long res_sh;
    long long res_sw;
    unsigned int act;
    float slope;
    int prelu_len;
    float res_scale;
};

/* Block tile BM x BN on a WARPS_M x WARPS_N warp grid. */
template <int BM_, int BN_, int WARPS_M_, int WARPS_N_>
struct ConvTileCfg {
    static constexpr int BM = BM_;
    static constexpr int BN = BN_;
    static constexpr int BK = 32;
    static constexpr int WARPS_M = WARPS_M_;
    static constexpr int WARPS_N = WARPS_N_;
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
    static constexpr int WARP_M = BM / WARPS_M;
    static constexpr int WARP_N = BN / WARPS_N;
    static constexpr int FRAG_M = WARP_M / 16;
    static constexpr int FRAG_N = WARP_N / 8;
    /* Four threads share a row, one 16-byte chunk each. One pass of all
       threads covers ROWS_PER_PASS rows. */
    static constexpr int ROWS_PER_PASS = THREADS / 4;
    static constexpr int A_ITERS = BM / ROWS_PER_PASS;
    static constexpr int B_ITERS = BN / ROWS_PER_PASS;
    static constexpr int STAGE_A_BYTES = BM * BK * 2;
    static constexpr int STAGE_B_BYTES = BN * BK * 2;
    static constexpr int STAGE_BYTES = STAGE_A_BYTES + STAGE_B_BYTES;
    static constexpr int POOL_BYTES = 2 * STAGE_BYTES;
    static constexpr int ROW_TABLE_BYTES = BM * 16;
    static constexpr int COL_TABLE_BYTES = 2 * BK * 16;

    static_assert(WARP_M == 64 && WARP_N == 32, "the mma loop is written for a 64x32 warp tile");
    static_assert(BM % ROWS_PER_PASS == 0 && BN % ROWS_PER_PASS == 0 && B_ITERS >= 1,
                  "every thread stages whole A and B rows");
    static_assert(BM * BN * 2 <= POOL_BYTES, "the output tile must fit the stage buffers");
    static_assert((BM * (BN / 8)) % THREADS == 0, "the vector store covers the tile evenly");
    static_assert(POOL_BYTES + ROW_TABLE_BYTES + COL_TABLE_BYTES <= 49152,
                  "conv static shared memory exceeds the 48 KiB per-block cap");
};

#include "conv_implicit_gemm_stage.cuh"
#include "conv_implicit_gemm_epilogue.cuh"

template <typename T, typename Cfg, bool VEC> struct ConvLoaderFor;
template <typename T, typename Cfg> struct ConvLoaderFor<T, Cfg, true> {
    using type = ConvLoaderVec<T, Cfg>;
};
template <typename T, typename Cfg> struct ConvLoaderFor<T, Cfg, false> {
    using type = ConvLoaderScalar<T, Cfg>;
};

/* Shared-window byte offsets of this lane's ldmatrix rows at k16 step 0 and
   1, from the start of a stage's A or B buffer. Fragment i of the warp tile
   adds i * 16 rows; 16 is a multiple of the swizzle period, so the chunk
   XOR is the same for every fragment. */
template <typename Cfg>
struct ConvLdmOffsets {
    unsigned int a[2];
    unsigned int b[2];

    __device__ __forceinline__ ConvLdmOffsets(int lane, int warp_m, int warp_n) {
        const int a_row = warp_m * Cfg::WARP_M + (lane & 15);
        const int b_row = warp_n * Cfg::WARP_N + (lane & 7) + ((lane >> 4) << 3);
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            /* A matrices: (rows 0-7, k lo) (rows 8-15, k lo) (rows 0-7, k hi)
               (rows 8-15, k hi), the m16n8k16 a0..a3 order. */
            const int a_chunk = conv_stage_swizzle(a_row, ks * 2 + (lane >> 4));
            a[ks] = static_cast<unsigned int>((a_row * Cfg::BK + a_chunk * 8) * 2);
            /* B matrices: (n 0-7, k lo) (n 0-7, k hi) (n 8-15, k lo)
               (n 8-15, k hi), so regs {0, 1} are n-fragment 0 and {2, 3}
               n-fragment 1. */
            const int b_chunk = conv_stage_swizzle(b_row, ks * 2 + ((lane >> 3) & 1));
            b[ks] = static_cast<unsigned int>((b_row * Cfg::BK + b_chunk * 8) * 2);
            /* Opaque copies: the compiler keeps these in registers instead
               of recomputing them in every K-tile. */
            asm volatile("mov.b32 %0, %0;" : "+r"(a[ks]));
            asm volatile("mov.b32 %0, %0;" : "+r"(b[ks]));
        }
    }
};

/* The mma work of one K-tile on the stage at shared address `stage`. */
template <typename T, typename Cfg>
__device__ __forceinline__ void conv_mma_tile(float (&acc)[Cfg::FRAG_M][Cfg::FRAG_N][4], unsigned int stage,
                                              const ConvLdmOffsets<Cfg>& off) {
    constexpr unsigned int FRAG_ROW_BYTES = 16 * Cfg::BK * 2;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
        unsigned int a[Cfg::FRAG_M][4];
        unsigned int b[Cfg::FRAG_N / 2][4];
#pragma unroll
        for (int i = 0; i < Cfg::FRAG_M; ++i) {
            conv_ldmatrix_x4(a[i], stage + off.a[ks] + i * FRAG_ROW_BYTES);
        }
#pragma unroll
        for (int jj = 0; jj < Cfg::FRAG_N / 2; ++jj) {
            conv_ldmatrix_x4(b[jj], stage + Cfg::STAGE_A_BYTES + off.b[ks] + jj * FRAG_ROW_BYTES);
        }
#pragma unroll
        for (int i = 0; i < Cfg::FRAG_M; ++i) {
#pragma unroll
            for (int j = 0; j < Cfg::FRAG_N; ++j) {
                ConvHalf<T>::mma_k16(acc[i][j], a[i], b[j >> 1][(j & 1) * 2], b[j >> 1][(j & 1) * 2 + 1]);
            }
        }
    }
}

template <typename T, typename Cfg, bool VEC>
__device__ __forceinline__ void conv_igemm_run(const ConvIgemmParams<T>& p) {
    using Loader = typename ConvLoaderFor<T, Cfg, VEC>::type;

    __shared__ __align__(128) unsigned char smem_pool[Cfg::POOL_BYTES];
    __shared__ __align__(16) longlong2 smem_rows[Cfg::BM];
    __shared__ __align__(16) int4 smem_cols[VEC ? 1 : 2 * Cfg::BK];

    /* Read once through volatile asm, so the compiler keeps the values
       derived from it in registers instead of recomputing them from the
       special register inside the K loop. */
    int tid;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = warp / Cfg::WARPS_N;
    const int warp_n = warp % Cfg::WARPS_N;
    const long long block_row = static_cast<long long>(blockIdx.x) * Cfg::BM;
    const int block_col = static_cast<int>(blockIdx.y) * Cfg::BN;

    conv_fill_row_table<T, Cfg>(p, smem_rows, tid, block_row);

    Loader ld;
    ld.init(p, tid, block_row, block_col);
    const int num_k_tiles = Loader::num_k_tiles(p);
    const ConvLdmOffsets<Cfg> ldm(lane, warp_m, warp_n);
    const unsigned int pool = conv_smem_addr(smem_pool);

    float acc[Cfg::FRAG_M][Cfg::FRAG_N][4];
#pragma unroll
    for (int i = 0; i < Cfg::FRAG_M; ++i) {
#pragma unroll
        for (int j = 0; j < Cfg::FRAG_N; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
        }
    }

    /* Column tables of K-tiles 0 and 1. Table t % 2 serves K-tile t. */
    if (!VEC) {
        ld.write_cols(p, smem_cols, tid);
        ld.write_cols(p, smem_cols + Cfg::BK, tid);
        __syncthreads();
    }
    ld.load(p, smem_cols);
    ld.store(smem_pool);
    __syncthreads();

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const bool more = kt + 1 < num_k_tiles;
        const int cur = kt & 1;
        if (more) {
            ld.advance(p);
            ld.load(p, smem_cols + (cur ^ 1) * Cfg::BK);
        }
        conv_mma_tile<T, Cfg>(acc, pool + cur * Cfg::STAGE_BYTES, ldm);
        if (more) {
            ld.store(smem_pool + (cur ^ 1) * Cfg::STAGE_BYTES);
        }
        /* Table `cur` last served the load of K-tile kt, before the
           previous barrier. It now takes K-tile kt + 2. */
        if (!VEC) ld.write_cols(p, smem_cols + cur * Cfg::BK, tid);
        /* Publishes the stage and table written above, and orders every
           ldmatrix of stage `cur` before its next overwrite. After the last
           K-tile it frees the stages for the epilogue's output tile. */
        __syncthreads();
    }

    conv_igemm_epilogue<T, Cfg>(p, acc, smem_pool, smem_rows, tid, warp_m, warp_n, block_col);
}

#endif  // NUMR_CONV_IMPLICIT_GEMM_CUH
