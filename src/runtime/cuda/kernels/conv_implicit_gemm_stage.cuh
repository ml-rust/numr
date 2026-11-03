// Shared-memory staging for the implicit-GEMM conv2d in
// conv_implicit_gemm.cuh: the per-block row table, the K cursor, and the
// A (implicit im2col) and B (OHWI weight) tile loads.
//
// The macros expand against the tile constants of CONV_TILE_CONSTANTS and
// the kernel parameters of DEFINE_CONV_IGEMM. Each operand has a 128-bit
// path and a one-element path. Both write +0.0 for every position outside
// the input, past K, or past C_out, so the two paths stage the same bits.

#ifndef NUMR_CONV_IMPLICIT_GEMM_STAGE_CUH
#define NUMR_CONV_IMPLICIT_GEMM_STAGE_CUH

#define CONV_ALIGNED16(PTR) \
    (((reinterpret_cast<unsigned long long>(PTR)) & 15ull) == 0ull)

/* Position of one K index in (kh, kw, ci) order. `k` can pass K; the
   staging loops test it before use. */
struct ConvKCursor {
    int k;
    int ci;
    int kw;
    int kh;
};

__device__ __forceinline__ ConvKCursor conv_k_cursor_init(int k, int c_in, int kernel_w) {
    ConvKCursor cur;
    cur.k = k;
    const int tap = k / c_in;
    cur.ci = k - tap * c_in;
    cur.kh = tap / kernel_w;
    cur.kw = tap - cur.kh * kernel_w;
    return cur;
}

/* Move the cursor `step` positions along K. A division runs only when the
   step crosses a tap boundary. */
__device__ __forceinline__ void conv_k_cursor_advance(ConvKCursor& cur, int step, int c_in, int kernel_w) {
    cur.k += step;
    cur.ci += step;
    if (cur.ci >= c_in) {
        const int taps = cur.ci / c_in;
        cur.ci -= taps * c_in;
        cur.kw += taps;
        if (cur.kw >= kernel_w) {
            const int rows = cur.kw / kernel_w;
            cur.kw -= rows * kernel_w;
            cur.kh += rows;
        }
    }
}

/* Fill the (n, ho, wo) table for the block's rows. Two divisions per row,
   once per block, instead of once per staged element. */
#define CONV_FILL_ROW_TABLE(ROWS)                                            \
    for (int r = tid; r < BLOCK_TILE_M; r += CONV_IGEMM_THREADS) {           \
        const int m = block_row + r;                                         \
        int4 info = make_int4(-1, 0, 0, 0);                                  \
        if (m < M) {                                                         \
            const int pixels = out_h * out_w;                                \
            const int n = m / pixels;                                        \
            const int p = m - n * pixels;                                    \
            const int ho = p / out_w;                                        \
            info = make_int4(n, ho, p - ho * out_w, 0);                      \
        }                                                                    \
        (ROWS)[r] = info;                                                    \
    }

/* Input offset of tap (dh, dw) for table entry `pix`, or -1 when the tap
   falls in the padding or the row is past M. `dh` and `dw` hold
   kh * dil_h - pad_top and kw * dil_w - pad_left. 64-bit: an image of many
   channels and pixels passes 2^31 elements. */
__device__ __forceinline__ long long conv_input_offset(
    int4 pix, int dh, int dw, int stride_h, int stride_w, int in_h, int in_w,
    long long in_sn, long long in_sh, long long in_sw) {
    const int hi = pix.y * stride_h + dh;
    const int wi = pix.z * stride_w + dw;
    if (pix.x < 0 || hi < 0 || hi >= in_h || wi < 0 || wi >= in_w) {
        return -1ll;
    }
    return (long long)pix.x * in_sn + (long long)hi * in_sh + (long long)wi * in_sw;
}

#define CONV_INPUT_OFFSET(PIX, DH, DW)                                       \
    conv_input_offset((PIX), (DH), (DW), stride_h, stride_w, in_h, in_w,     \
                      in_sn, in_sh, in_sw)

/* A tile [BLOCK_TILE_M][BLOCK_K] at the K position of cursor a_k. Each
   thread stages column a_col of rows tid / COLS, tid / COLS + THREADS / COLS,
   ... where COLS is the staging columns per row. */
#define CONV_STAGE_A(HALF_T, ZERO_EXPR, DST_A)                               \
{                                                                            \
    const bool conv_k_ok = a_k.k < K;                                        \
    const int conv_dh = a_k.kh * dil_h - pad_top;                            \
    const int conv_dw = a_k.kw * dil_w - pad_left;                           \
    if (a_vec) {                                                             \
        for (int r = tid / K_VECS_PER_ROW; r < BLOCK_TILE_M;                 \
             r += CONV_IGEMM_THREADS / K_VECS_PER_ROW) {                     \
            const long long off = conv_k_ok                                  \
                ? CONV_INPUT_OFFSET(smem_rows[r], conv_dh, conv_dw) : -1ll;  \
            const bool in_bounds = off >= 0;                                 \
            WMMA_VEC_STAGE(                                                  \
                reinterpret_cast<int4*>(                                     \
                    (DST_A) + r * SMEM_STRIDE_K + a_col),                    \
                reinterpret_cast<const int4*>(                               \
                    input + (in_bounds ? off + a_k.ci : 0ll)),               \
                in_bounds);                                                  \
        }                                                                    \
    } else {                                                                 \
        for (int r = tid / BLOCK_K; r < BLOCK_TILE_M;                        \
             r += CONV_IGEMM_THREADS / BLOCK_K) {                            \
            const long long off = conv_k_ok                                  \
                ? CONV_INPUT_OFFSET(smem_rows[r], conv_dh, conv_dw) : -1ll;  \
            HALF_T val = ZERO_EXPR;                                          \
            if (off >= 0) val = input[off + a_k.ci];                         \
            (DST_A)[r * SMEM_STRIDE_K + a_col] = val;                        \
        }                                                                    \
    }                                                                        \
}

/* B tile [BLOCK_TILE_N][BLOCK_K] from the OHWI weight, read as [C_out][K]
   row-major and stored K-contiguous. */
#define CONV_STAGE_B(HALF_T, ZERO_EXPR, DST_B, K_OFF)                        \
{                                                                            \
    const int conv_kk = (K_OFF) + b_col;                                     \
    if (b_vec) {                                                             \
        for (int r = tid / K_VECS_PER_ROW; r < BLOCK_TILE_N;                 \
             r += CONV_IGEMM_THREADS / K_VECS_PER_ROW) {                     \
            const int co = block_col + r;                                    \
            const bool in_bounds = co < c_out && conv_kk < K;                \
            WMMA_VEC_STAGE(                                                  \
                reinterpret_cast<int4*>(                                     \
                    (DST_B) + r * SMEM_STRIDE_K + b_col),                    \
                reinterpret_cast<const int4*>(                               \
                    weight + (in_bounds                                      \
                        ? (long long)co * K + conv_kk : 0ll)),               \
                in_bounds);                                                  \
        }                                                                    \
    } else {                                                                 \
        for (int r = tid / BLOCK_K; r < BLOCK_TILE_N;                        \
             r += CONV_IGEMM_THREADS / BLOCK_K) {                            \
            const int co = block_col + r;                                    \
            HALF_T val = ZERO_EXPR;                                          \
            if (co < c_out && conv_kk < K) {                                 \
                val = weight[(long long)co * K + conv_kk];                   \
            }                                                                \
            (DST_B)[r * SMEM_STRIDE_K + b_col] = val;                        \
        }                                                                    \
    }                                                                        \
}

#endif  // NUMR_CONV_IMPLICIT_GEMM_STAGE_CUH
