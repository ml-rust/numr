// Implicit-GEMM tensor-core conv2d body for NHWC input and OHWI weight.
//
// GEMM view:
//   M = N * H_out * W_out   one row per output pixel
//   N = C_out               one column per output channel
//   K = Kh * Kw * C_in      the OHWI weight order
// A[m][k] is the input element under one tap of pixel m, or 0 in the
// padding. It is never written to memory: the loaders gather it straight
// from the NHWC input. B is the OHWI weight read as [C_out][K] row-major.
//
// Block tile BM x BN x 32, warp tile 64 x WN with WN = 32 or 64. Per k16
// step a warp issues 4 A ldmatrix.x4, WN / 16 B ldmatrix.x4 and WN / 2
// mma.sync m16n8k16 (see conv_implicit_gemm_mma.cuh for sm_75). The
// accumulator is F32, or F16 for F16 inputs when the host asks for it. A
// 64x64 warp tile needs the F16 accumulator: its F32 accumulators take
// 128 registers. After the K loop an F16 accumulator widens to F32, so the
// epilogue always runs in F32. Both operands sit in shared memory
// K-contiguous, as rows of 32 halves with the XOR swizzle of
// conv_stage_swizzle. B is [n][k], so ldmatrix without .trans hands
// mma.sync .row.col its B operand directly.
//
// Two bodies:
//   conv_igemm_run       gather, in conv_implicit_gemm_gather.cuh. Row m of
//                        a block is GEMM row blockIdx.x * BM + m. K runs
//                        tap-major, and every K-tile gathers its A rows from
//                        global memory.
//   conv_igemm_run_halo  halo, in conv_implicit_gemm_halo.cuh. A block owns
//                        a PATCH_H x PATCH_W patch of output pixels of one
//                        image. K runs channel-chunk-major, and each
//                        32-channel chunk of the input patch plus its halo
//                        is staged once and serves every tap.
//
// Gather pipeline: two stages and one register set. Per K-tile t:
//   ldmatrix k16 step 1 of t      mma k16 step 0 of t
//   STS of tile t + 1 into the other stage
//   barrier
//   ldmatrix k16 step 0 of t + 1  global loads of tile t + 2
//   mma k16 step 1 of t
// The fragments of the next k16 step load while the current one runs, also
// across the barrier. The scalar gather keeps its loads at the top of the
// K-tile instead, before any fragment is live: its packed 16-bit loads hold
// many registers in flight. The stage addresses toggle by XOR. cp.async is
// not used, so sm_75 runs the same pipeline.
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

/* Block tile BM x BN on a WARPS_M x WARPS_N warp grid. The halo body
   covers a PATCH_H x PATCH_W patch of output pixels with BM = PATCH_H *
   PATCH_W, and stages at most HALO_ITERS * ROWS_PER_PASS halo pixels. */
template <int BM_, int BN_, int WARPS_M_, int WARPS_N_, int PATCH_W_, int HALO_ITERS_>
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
    static constexpr int OUT_TILE_BYTES = BM * BN * 2;

    static constexpr int PATCH_W = PATCH_W_;
    static constexpr int PATCH_H = BM / PATCH_W;
    static constexpr int HALO_ITERS = HALO_ITERS_;
    static constexpr int HALO_PIX = HALO_ITERS * ROWS_PER_PASS;
    static constexpr int HALO_BYTES = HALO_PIX * BK * 2;
    /* One i64 input byte offset per halo pixel. */
    static constexpr int HALO_TABLE_BYTES = HALO_PIX * 8;
    static constexpr int HALO_MAIN_BYTES = HALO_BYTES + 2 * STAGE_B_BYTES + HALO_TABLE_BYTES;
    /* Halo, two B stages, then the halo offset table. The epilogue reuses
       the start as the output tile. */
    static constexpr int HALO_POOL_BYTES = HALO_MAIN_BYTES > OUT_TILE_BYTES ? HALO_MAIN_BYTES : OUT_TILE_BYTES;

    static_assert(WARP_M == 64 && (WARP_N == 32 || WARP_N == 64),
                  "the mma loop is written for a 64x32 or 64x64 warp tile");
    static_assert(BM % ROWS_PER_PASS == 0 && BN % ROWS_PER_PASS == 0 && B_ITERS >= 1,
                  "every thread stages whole A and B rows");
    static_assert(OUT_TILE_BYTES <= POOL_BYTES, "the output tile must fit the stage buffers");
    static_assert((BM * (BN / 8)) % THREADS == 0, "the vector store covers the tile evenly");
    static_assert(POOL_BYTES + ROW_TABLE_BYTES + COL_TABLE_BYTES <= 49152,
                  "gather static shared memory exceeds the 48 KiB per-block cap");
    static_assert(PATCH_W % 16 == 0 && BM % PATCH_W == 0,
                  "a 16-row mma fragment must lie in one patch row");
    static_assert(HALO_POOL_BYTES + ROW_TABLE_BYTES <= 49152,
                  "halo static shared memory exceeds the 48 KiB per-block cap");
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

/* Shared-window byte offsets of this lane's B ldmatrix rows at k16 step 0
   and 1, from the start of a B tile of BN rows of 32 halves. Fragment pair
   jj adds jj * 16 rows; 16 is a multiple of the swizzle period, so the
   chunk XOR is the same for every pair.
   B matrices: (n 0-7, k lo) (n 0-7, k hi) (n 8-15, k lo) (n 8-15, k hi),
   so regs {0, 1} are n-fragment 0 and {2, 3} n-fragment 1. */
template <typename Cfg>
__device__ __forceinline__ void conv_b_ldm_offsets(unsigned int (&b)[2], int lane, int warp_n) {
    const int b_row = warp_n * Cfg::WARP_N + (lane & 7) + ((lane >> 4) << 3);
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
        const int b_chunk = conv_stage_swizzle(b_row, ks * 2 + ((lane >> 3) & 1));
        b[ks] = static_cast<unsigned int>((b_row * Cfg::BK + b_chunk * 8) * 2);
        /* Opaque copy: the compiler keeps it in a register instead of
           recomputing it in every K-tile. */
        asm volatile("mov.b32 %0, %0;" : "+r"(b[ks]));
    }
}

/* Gather-body ldmatrix offsets of this lane at k16 step 0 and 1, from the
   start of a stage. Fragment i of the warp tile adds i * 16 rows; 16 is a
   multiple of the swizzle period, so the chunk XOR is the same for every
   fragment. */
template <typename Cfg>
struct ConvLdmOffsets {
    unsigned int a[2];
    unsigned int b[2];

    __device__ __forceinline__ ConvLdmOffsets(int lane, int warp_m, int warp_n) {
        const int a_row = warp_m * Cfg::WARP_M + (lane & 15);
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            /* A matrices: (rows 0-7, k lo) (rows 8-15, k lo) (rows 0-7, k hi)
               (rows 8-15, k hi), the m16n8k16 a0..a3 order. */
            const int a_chunk = conv_stage_swizzle(a_row, ks * 2 + (lane >> 4));
            a[ks] = static_cast<unsigned int>((a_row * Cfg::BK + a_chunk * 8) * 2);
            asm volatile("mov.b32 %0, %0;" : "+r"(a[ks]));
        }
        /* B sits after A in a gather stage. */
        conv_b_ldm_offsets<Cfg>(b, lane, warp_n);
        b[0] += Cfg::STAGE_A_BYTES;
        b[1] += Cfg::STAGE_A_BYTES;
    }
};

/* ldmatrix fragments of one k16 step of the warp tile. */
template <typename Cfg>
struct ConvFrags {
    unsigned int a[Cfg::FRAG_M][4];
    unsigned int b[Cfg::FRAG_N / 2][4];
};

/* B fragments of one k16 step. `addr` is the shared-window address of the
   B tile plus this lane's offset for the step. */
template <typename Cfg>
__device__ __forceinline__ void conv_load_b_frags(ConvFrags<Cfg>& f, unsigned int addr) {
    constexpr unsigned int FRAG_ROW_BYTES = 16 * Cfg::BK * 2;
#pragma unroll
    for (int jj = 0; jj < Cfg::FRAG_N / 2; ++jj) {
        conv_ldmatrix_x4(f.b[jj], addr + jj * FRAG_ROW_BYTES);
    }
}

/* Gather-body fragments of k16 step KS from the stage at shared-window
   address `stage`. */
template <typename Cfg, int KS>
__device__ __forceinline__ void conv_load_frags(ConvFrags<Cfg>& f, unsigned int stage,
                                                const ConvLdmOffsets<Cfg>& off) {
    constexpr unsigned int FRAG_ROW_BYTES = 16 * Cfg::BK * 2;
#pragma unroll
    for (int i = 0; i < Cfg::FRAG_M; ++i) {
        conv_ldmatrix_x4(f.a[i], stage + off.a[KS] + i * FRAG_ROW_BYTES);
    }
    conv_load_b_frags<Cfg>(f, stage + off.b[KS]);
}

/* The mma work of one k16 step. */
template <typename T, typename Cfg, bool HALF_ACC>
__device__ __forceinline__ void conv_mma_k16(
    typename ConvAccum<HALF_ACC>::type (&acc)[Cfg::FRAG_M][Cfg::FRAG_N][ConvAccum<HALF_ACC>::REGS],
    const ConvFrags<Cfg>& f) {
#pragma unroll
    for (int i = 0; i < Cfg::FRAG_M; ++i) {
#pragma unroll
        for (int j = 0; j < Cfg::FRAG_N; ++j) {
            ConvHalf<T>::mma_k16(acc[i][j], f.a[i], f.b[j >> 1][(j & 1) * 2], f.b[j >> 1][(j & 1) * 2 + 1]);
        }
    }
}

/* F32 copy of an F16 accumulator, in the F32 fragment layout the epilogue
   reads. */
template <typename Cfg>
__device__ __forceinline__ void conv_widen_acc(const unsigned int (&acc)[Cfg::FRAG_M][Cfg::FRAG_N][2],
                                               float (&wide)[Cfg::FRAG_M][Cfg::FRAG_N][4]) {
#pragma unroll
    for (int i = 0; i < Cfg::FRAG_M; ++i) {
#pragma unroll
        for (int j = 0; j < Cfg::FRAG_N; ++j) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float2 v = ConvHalf<__half>::acc_to_float2(acc[i][j][h]);
                wide[i][j][h * 2] = v.x;
                wide[i][j][h * 2 + 1] = v.y;
            }
        }
    }
}

/* Zero the accumulator. Zero bits are +0.0 in F32 and in F16. */
template <typename Cfg, bool HALF_ACC>
__device__ __forceinline__ void conv_zero_acc(
    typename ConvAccum<HALF_ACC>::type (&acc)[Cfg::FRAG_M][Cfg::FRAG_N][ConvAccum<HALF_ACC>::REGS]) {
#pragma unroll
    for (int i = 0; i < Cfg::FRAG_M; ++i) {
#pragma unroll
        for (int j = 0; j < Cfg::FRAG_N; ++j) {
#pragma unroll
            for (int e = 0; e < ConvAccum<HALF_ACC>::REGS; ++e) acc[i][j][e] = 0;
        }
    }
}

/* Widen when needed, then run the epilogue. The caller's last barrier
   orders every ldmatrix before the epilogue's tile stores. */
template <typename T, typename Cfg, bool HALF_ACC>
__device__ __forceinline__ void conv_finish(
    const ConvIgemmParams<T>& p,
    typename ConvAccum<HALF_ACC>::type (&acc)[Cfg::FRAG_M][Cfg::FRAG_N][ConvAccum<HALF_ACC>::REGS],
    void* pool, const longlong2* rows, int tid, int warp_m, int warp_n, int block_col) {
    if constexpr (HALF_ACC) {
        float wide[Cfg::FRAG_M][Cfg::FRAG_N][4];
        conv_widen_acc<Cfg>(acc, wide);
        conv_igemm_epilogue<T, Cfg>(p, wide, pool, rows, tid, warp_m, warp_n, block_col);
    } else {
        conv_igemm_epilogue<T, Cfg>(p, acc, pool, rows, tid, warp_m, warp_n, block_col);
    }
}

template <typename T, typename Cfg, bool HALF_ACC>
__device__ __forceinline__ void conv_check_acc() {
    static_assert(!HALF_ACC || ConvHalf<T>::HALF_ACCUMULATE, "half accumulation is for __half inputs only");
    static_assert(HALF_ACC || Cfg::WARP_N == 32, "a 64x64 warp tile needs the half accumulator");
}

#include "conv_implicit_gemm_gather.cuh"
#include "conv_implicit_gemm_halo.cuh"

#endif  // NUMR_CONV_IMPLICIT_GEMM_CUH
