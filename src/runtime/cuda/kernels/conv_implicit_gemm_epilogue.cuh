// Epilogue of the implicit-GEMM conv2d in conv_implicit_gemm.cuh.
//
// Per output element, in F32 registers, in this order:
//   y = acc + bias[co]                      (when bias is not null)
//   y = act(y)
//   y = residual[pixel, co] + scale * y     (when residual is not null)
// then one rounding to the storage dtype. This matches
// `FusedConvOps::conv2d_fused`.
//
// Bias and PReLU slopes load once per thread column. Each operand test and
// the activation switch run once per thread, outside the element loops. The
// residual loads in the accumulator fragment layout.
//
// NHWC: rounded values go to a swizzled [BM][BN] tile in the stage buffers,
// then leave by rows: one 16-byte store per 8 channels when the output is
// 16-byte aligned, its N/H/W strides are multiples of 8, and the block's N
// tile is whole. Otherwise half2 stores where a channel pair is in range and
// 4-byte aligned, and one-element stores for the rest. Channel slices of a
// wider output take either path.
//
// NCHW: the residual and the store are in
// conv_implicit_gemm_epilogue_nchw.cuh. Bias, activation and the F32 math
// are the same code for both layouts.
//
// Activation codes are this kernel's own. They are not the GEMM epilogue
// codes of gemm_activation.cuh. The host maps them in
// loader/conv_implicit_gemm.rs (`ConvEpilogueActivation`).

#ifndef NUMR_CONV_IMPLICIT_GEMM_EPILOGUE_CUH
#define NUMR_CONV_IMPLICIT_GEMM_EPILOGUE_CUH

#include "conv_implicit_gemm_epilogue_nchw.cuh"

#define CONV_ACT_NONE  0u
#define CONV_ACT_RELU  1u
#define CONV_ACT_LEAKY 2u
#define CONV_ACT_PRELU 3u

/* 16-byte chunk of logical chunk `chunk` in row `row` of the [BM][BN]
   output tile. A warp's fragment store writes 8 rows of one chunk; the XOR
   puts them in 8 distinct bank groups. */
template <int CHUNKS>
__device__ __forceinline__ int conv_out_swizzle(int row, int chunk) {
    static_assert(CHUNKS == 4 || CHUNKS == 8, "the output tile is 32 or 64 channels wide");
    return CHUNKS == 8 ? (chunk ^ (row & 7)) : (chunk ^ ((row >> 1) & 3));
}

/* Out and residual offsets of one block row, at column 0. `x` is -1 past
   the last row. */
template <typename T, typename Cfg>
__device__ __forceinline__ void conv_fill_row_table(const ConvIgemmParams<T>& p, longlong2* rows, int tid,
                                                    long long block_row) {
    const int pixels = p.out_h * p.out_w;
    for (int r = tid; r < Cfg::BM; r += Cfg::THREADS) {
        const long long m = block_row + r;
        longlong2 e = make_longlong2(-1ll, 0ll);
        if (m < p.M) {
            int n, ho, wo;
            conv_decode_pixel(static_cast<int>(m), p.out_w, pixels, n, ho, wo);
            e.x = n * p.out_sn + ho * p.out_sh + wo * p.out_sw;
            e.y = n * p.res_sn + ho * p.res_sh + wo * p.res_sw;
        }
        rows[r] = e;
    }
}

#define CONV_EPI_FOR_EACH(BODY)                                              \
    _Pragma("unroll") for (int i = 0; i < FM; ++i)                           \
    _Pragma("unroll") for (int j = 0; j < FN; ++j)                           \
    _Pragma("unroll") for (int e = 0; e < 4; ++e) { BODY }

/* acc[i][j][e] holds row warp_m0 + i * 16 + g + (e / 2) * 8 and column
   warp_n0 + j * 8 + 2t + e % 2, with g = lane / 4 and t = lane % 4.
   The stage buffers must be free: the caller's barrier after the last
   ldmatrix orders every read of them before the first tile store here.
   `runs_in_rows` is read by the NCHW store only. */
template <typename T, typename Cfg, bool NCHW>
__device__ __forceinline__ void conv_igemm_epilogue(const ConvIgemmParams<T>& p,
                                                    float (&acc)[Cfg::FRAG_M][Cfg::FRAG_N][4],
                                                    void* pool, const longlong2* rows, int tid,
                                                    int warp_m, int warp_n, int block_col,
                                                    bool runs_in_rows) {
    using H = ConvHalf<T>;
    using T2 = typename H::T2;
    constexpr int FM = Cfg::FRAG_M;
    constexpr int FN = Cfg::FRAG_N;
    constexpr int CHUNKS = Cfg::BN / 8;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int row0 = warp_m * Cfg::WARP_M + g;
    const int col0 = warp_n * Cfg::WARP_N + 2 * t;

    if (p.bias != nullptr) {
        float b[FN][2];
#pragma unroll
        for (int j = 0; j < FN; ++j) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int co = block_col + col0 + j * 8 + h;
                b[j][h] = co < p.c_out ? H::to_float(__ldg(p.bias + co)) : 0.0f;
            }
        }
        CONV_EPI_FOR_EACH(acc[i][j][e] += b[j][e & 1];)
    }

    switch (p.act) {
        case CONV_ACT_RELU:
            CONV_EPI_FOR_EACH(acc[i][j][e] = fmaxf(acc[i][j][e], 0.0f);)
            break;
        case CONV_ACT_LEAKY: {
            const float s = p.slope;
            CONV_EPI_FOR_EACH(
                const float y = acc[i][j][e];
                acc[i][j][e] = y > 0.0f ? y : s * y;)
            break;
        }
        case CONV_ACT_PRELU: {
            float s[FN][2];
            const float shared = p.prelu_len == 1 ? H::to_float(__ldg(p.prelu)) : 0.0f;
#pragma unroll
            for (int j = 0; j < FN; ++j) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int co = block_col + col0 + j * 8 + h;
                    s[j][h] = p.prelu_len == 1 ? shared
                            : (co < p.c_out ? H::to_float(__ldg(p.prelu + co)) : 0.0f);
                }
            }
            CONV_EPI_FOR_EACH(
                const float y = acc[i][j][e];
                acc[i][j][e] = y > 0.0f ? y : s[j][e & 1] * y;)
            break;
        }
        default:
            break;
    }

    if constexpr (NCHW) {
        if (p.residual != nullptr) conv_epi_residual_nchw<T, Cfg>(p, acc, rows, row0, col0, block_col);
        conv_epi_store_nchw<T, Cfg>(p, acc, pool, rows, tid, row0, col0, block_col, runs_in_rows);
        return;
    }

    if (p.residual != nullptr) {
        const float scale = p.res_scale;
        const bool pair_ok = (reinterpret_cast<unsigned long long>(p.residual) & 3ull) == 0ull
            && ((p.res_sn | p.res_sh | p.res_sw) & 1ll) == 0ll;
#pragma unroll
        for (int i = 0; i < FM; ++i) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const longlong2 rt = rows[row0 + i * 16 + hh * 8];
                if (rt.x < 0) continue;
#pragma unroll
                for (int j = 0; j < FN; ++j) {
                    const int co = block_col + col0 + j * 8;
                    float r0 = 0.0f;
                    float r1 = 0.0f;
                    if (pair_ok && co + 1 < p.c_out) {
                        const float2 r = H::to_float2(
                            __ldg(reinterpret_cast<const T2*>(p.residual + (rt.y + co))));
                        r0 = r.x;
                        r1 = r.y;
                    } else {
                        if (co < p.c_out) r0 = H::to_float(__ldg(p.residual + (rt.y + co)));
                        if (co + 1 < p.c_out) r1 = H::to_float(__ldg(p.residual + (rt.y + co + 1)));
                    }
                    acc[i][j][hh * 2] = r0 + scale * acc[i][j][hh * 2];
                    acc[i][j][hh * 2 + 1] = r1 + scale * acc[i][j][hh * 2 + 1];
                }
            }
        }
    }

    T* const tile = reinterpret_cast<T*>(pool);
#pragma unroll
    for (int i = 0; i < FM; ++i) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int row = row0 + i * 16 + hh * 8;
#pragma unroll
            for (int j = 0; j < FN; ++j) {
                const int col = col0 + j * 8;
                const int chunk = conv_out_swizzle<CHUNKS>(row, col >> 3);
                *reinterpret_cast<T2*>(tile + row * Cfg::BN + chunk * 8 + (col & 7)) =
                    H::pack(acc[i][j][hh * 2], acc[i][j][hh * 2 + 1]);
            }
        }
    }
    __syncthreads();

    const bool vec_ok = (reinterpret_cast<unsigned long long>(p.out) & 15ull) == 0ull
        && ((p.out_sn | p.out_sh | p.out_sw) & 7ll) == 0ll
        && block_col + Cfg::BN <= p.c_out;
    if (vec_ok) {
#pragma unroll
        for (int it = 0; it < Cfg::BM * CHUNKS / Cfg::THREADS; ++it) {
            const int idx = tid + it * Cfg::THREADS;
            const int row = idx / CHUNKS;
            const int c = idx % CHUNKS;
            const long long o = rows[row].x;
            if (o >= 0) {
                const int4 v = reinterpret_cast<const int4*>(tile)[row * CHUNKS + conv_out_swizzle<CHUNKS>(row, c)];
                *reinterpret_cast<int4*>(p.out + (o + block_col + c * 8)) = v;
            }
        }
        return;
    }

    const bool pair_ok = (reinterpret_cast<unsigned long long>(p.out) & 3ull) == 0ull
        && ((p.out_sn | p.out_sh | p.out_sw) & 1ll) == 0ll;
    constexpr int PAIRS = Cfg::BN / 2;
#pragma unroll 4
    for (int idx = tid; idx < Cfg::BM * PAIRS; idx += Cfg::THREADS) {
        const int row = idx / PAIRS;
        const int col = (idx % PAIRS) * 2;
        const long long o = rows[row].x;
        const int co = block_col + col;
        if (o < 0 || co >= p.c_out) continue;
        const T2 v = *reinterpret_cast<const T2*>(
            tile + row * Cfg::BN + conv_out_swizzle<CHUNKS>(row, col >> 3) * 8 + (col & 7));
        if (pair_ok && co + 1 < p.c_out) {
            *reinterpret_cast<T2*>(p.out + (o + co)) = v;
        } else {
            p.out[o + co] = v.x;
            if (co + 1 < p.c_out) p.out[o + co + 1] = v.y;
        }
    }
}

#undef CONV_EPI_FOR_EACH

#endif  // NUMR_CONV_IMPLICIT_GEMM_EPILOGUE_CUH
