// NCHW residual and store of the implicit-GEMM conv2d epilogue in
// conv_implicit_gemm_epilogue.cuh.
//
// The output is [n][co][oh][ow]. Element (block row r, channel co) sits at
// rows[r].x + co * out_sc, and its residual at rows[r].y + co * res_sc. The
// row table holds the N/H/W part of both offsets, so any non-negative
// strides work.
//
// Residual: one 16-bit load per element in the accumulator fragment layout.
// The 8 lanes of one t read 8 consecutive pixels of one channel.
//
// Store: rounded values go to a channel-major [BN][BM] tile, then leave
// along pixels for each channel:
//   vector  one 16-byte store per run of 8 block rows. Needs a 16-byte
//           aligned output, W stride 1, N/C/H strides that are multiples of
//           8, and `runs_in_rows`: each aligned run of 8 block rows lies in
//           one output row and starts on a multiple of 8 pixels. A run cut
//           by the image edge stores its valid pixels one by one.
//   scalar  one 16-bit store per element, lanes on consecutive pixels.
// The halo body always passes `runs_in_rows`: a run never leaves its patch
// row. The gather body passes it when W_out is a multiple of 8.

#ifndef NUMR_CONV_IMPLICIT_GEMM_EPILOGUE_NCHW_CUH
#define NUMR_CONV_IMPLICIT_GEMM_EPILOGUE_NCHW_CUH

/* Element index of (column col, block row row) in the channel-major
   [BN][BM] tile. Rows group into 16-byte chunks of 8 pixels; the XOR puts
   the 4 channel pairs one fragment store writes in 4 distinct chunks, and
   keeps each aligned group of 8 chunks inside one 128-byte window. */
template <typename Cfg>
__device__ __forceinline__ int conv_nchw_tile_index(int col, int row) {
    static_assert(Cfg::BM % 32 == 0, "the chunk XOR needs 4 chunks per aligned group");
    return col * Cfg::BM + ((((row >> 3) ^ ((col >> 1) & 3))) << 3) + (row & 7);
}

/* y = residual + scale * y for every element of the warp tile. */
template <typename T, typename Cfg>
__device__ __forceinline__ void conv_epi_residual_nchw(const ConvIgemmParams<T>& p,
                                                       float (&acc)[Cfg::FRAG_M][Cfg::FRAG_N][4],
                                                       const longlong2* rows, int row0, int col0,
                                                       int block_col) {
    using H = ConvHalf<T>;
    const float scale = p.res_scale;
#pragma unroll
    for (int i = 0; i < Cfg::FRAG_M; ++i) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const longlong2 rt = rows[row0 + i * 16 + hh * 8];
            if (rt.x < 0) continue;
#pragma unroll
            for (int j = 0; j < Cfg::FRAG_N; ++j) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int co = block_col + col0 + j * 8 + e;
                    const float r = co < p.c_out ? H::to_float(__ldg(p.residual + (rt.y + co * p.res_sc))) : 0.0f;
                    acc[i][j][hh * 2 + e] = r + scale * acc[i][j][hh * 2 + e];
                }
            }
        }
    }
}

/* Low (e even) or high (e odd) 16 bits of word e / 2 of `v`. */
__device__ __forceinline__ unsigned short conv_int4_half(const int4& v, int e) {
    const int w = e < 2 ? v.x : e < 4 ? v.y : e < 6 ? v.z : v.w;
    return static_cast<unsigned short>(static_cast<unsigned int>(w) >> ((e & 1) * 16));
}

/* Round the warp tile into the channel-major tile, then store it. */
template <typename T, typename Cfg>
__device__ __forceinline__ void conv_epi_store_nchw(const ConvIgemmParams<T>& p,
                                                    const float (&acc)[Cfg::FRAG_M][Cfg::FRAG_N][4],
                                                    void* pool, const longlong2* rows, int tid, int row0,
                                                    int col0, int block_col, bool runs_in_rows) {
    using H = ConvHalf<T>;
    T* const tile = reinterpret_cast<T*>(pool);
#pragma unroll
    for (int i = 0; i < Cfg::FRAG_M; ++i) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
            for (int j = 0; j < Cfg::FRAG_N; ++j) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int row = row0 + i * 16 + hh * 8;
                    const int col = col0 + j * 8 + e;
                    tile[conv_nchw_tile_index<Cfg>(col, row)] = H::from_float(acc[i][j][hh * 2 + e]);
                }
            }
        }
    }
    __syncthreads();

    constexpr int RUNS = Cfg::BM / 8;
    const bool vec_ok = runs_in_rows && (reinterpret_cast<unsigned long long>(p.out) & 15ull) == 0ull
        && p.out_sw == 1ll && ((p.out_sn | p.out_sc | p.out_sh) & 7ll) == 0ll;
    if (vec_ok) {
        unsigned short* const out = reinterpret_cast<unsigned short*>(p.out);
#pragma unroll
        for (int it = 0; it < Cfg::BN * RUNS / Cfg::THREADS; ++it) {
            const int idx = tid + it * Cfg::THREADS;
            const int k = idx % RUNS;
            const int c = idx / RUNS;
            const int co = block_col + c;
            const long long o0 = rows[k * 8].x;
            /* Validity only falls along a run, so an invalid first pixel
               ends it. */
            if (co >= p.c_out || o0 < 0) continue;
            const int4 v = reinterpret_cast<const int4*>(tile)[c * RUNS + (k ^ ((c >> 1) & 3))];
            const long long o = o0 + co * p.out_sc;
            if (rows[k * 8 + 7].x >= 0) {
                *reinterpret_cast<int4*>(out + o) = v;
            } else {
                out[o] = conv_int4_half(v, 0);
#pragma unroll
                for (int e = 1; e < 8; ++e) {
                    if (rows[k * 8 + e].x >= 0) out[o + e] = conv_int4_half(v, e);
                }
            }
        }
        return;
    }

#pragma unroll 4
    for (int idx = tid; idx < Cfg::BM * Cfg::BN; idx += Cfg::THREADS) {
        const int r = idx % Cfg::BM;
        const int c = idx / Cfg::BM;
        const int co = block_col + c;
        const long long o = rows[r].x;
        if (o < 0 || co >= p.c_out) continue;
        p.out[o + co * p.out_sc] = tile[conv_nchw_tile_index<Cfg>(c, r)];
    }
}

#endif  // NUMR_CONV_IMPLICIT_GEMM_EPILOGUE_NCHW_CUH
