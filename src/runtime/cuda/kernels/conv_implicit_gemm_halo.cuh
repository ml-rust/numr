// Halo body of the implicit-GEMM conv2d in conv_implicit_gemm.cuh.
//
// A block owns a PATCH_H x PATCH_W patch of output pixels of one image:
// GEMM row r of the block is pixel (ho0 + r / PATCH_W, wo0 + r % PATCH_W).
// Grid x enumerates patches in (n, patch row, patch column) order. Rows
// past the image edge compute on halo values and are masked at the store.
//
// K runs channel-chunk-major: for each 32-channel chunk, every tap
// (kh, kw) in OHWI order. The A operand of one chunk is its input patch
// plus halo:
//   halo_h = (PATCH_H - 1) * stride_h + (Kh - 1) * dil_h + 1
//   halo_w = (PATCH_W - 1) * stride_w + (Kw - 1) * dil_w + 1
// pixels of 32 halves, staged into shared memory once and read by every
// tap. Each A ldmatrix lane addresses the halo pixel under its own output
// pixel and the current tap:
//   (lr * stride_h + kh * dil_h) * halo_w + lc * stride_w + kw * dil_w
// Halo pixel q is a 64-byte row with the conv_stage_swizzle XOR of row q,
// at the STS and at the ldmatrix alike. Pixels outside the image, channels
// past C_in and slots past halo_h * halo_w hold +0.0.
//
// The host routes an NHWC shape here only when halo_h * halo_w <= HALO_PIX
// and the vector staging conditions hold (C_in % 8 == 0, 16-byte aligned
// input and weight, input N/H/W strides that are multiples of 8). See
// loader/conv_implicit_gemm_staging.rs. The NCHW A loader, with its own
// conditions, is in conv_implicit_gemm_halo_nchw.cuh. Both loaders fill
// the same halo layout, and the body below runs either.
//
// Pipeline:
//   B   one [BN][32] tile per K-tile in two stages, with one register set,
//       as in the gather body.
//   A   one halo buffer. The halo of chunk c + 1 loads into registers at
//       the start of chunk c, so its global latency overlaps every tap of
//       chunk c. After the barrier of the last tap of chunk c, each warp
//       issues the k16 step 1 mma of that tap, then stores the halo, and a
//       second barrier publishes it. The mma work runs while the stores and
//       the barrier wait.
// The fragments of the next k16 step load while the current one runs, as
// in the gather body. A chunk of T taps costs T + 1 barriers.
//
// Shared memory, all static:
//   halo         HALO_PIX x 32 halves
//   B stages     2 x BN x 32 halves
//   offset table HALO_PIX x i64: input byte offset of each halo pixel from
//                image n, or -1 outside the image or past the halo. Only
//                the chunk start differs between chunks, so a halo reload
//                is one table read and one add per owned pixel. The NCHW
//                loader keeps its run table here instead.
//   row table    BM x {i64 out offset, i64 residual offset}
// The epilogue reuses the halo, B stages and offset table as the [BM][BN]
// output tile.

#ifndef NUMR_CONV_IMPLICIT_GEMM_HALO_CUH
#define NUMR_CONV_IMPLICIT_GEMM_HALO_CUH

/* Halo registers of one chunk: owned halo pixels tid / 4 + i * ROWS_PER_PASS,
   chunk tid % 4. */
template <typename Cfg>
struct ConvHaloRegs {
    int4 v[Cfg::HALO_ITERS];
};

/* Out and residual offsets of one block row of a patch. `x` is -1 past the
   image edge. */
template <typename T, typename Cfg>
__device__ __forceinline__ void conv_fill_patch_rows(const ConvIgemmParams<T>& p, longlong2* rows, int tid,
                                                     int n, int ho0, int wo0) {
    for (int r = tid; r < Cfg::BM; r += Cfg::THREADS) {
        const int ho = ho0 + r / Cfg::PATCH_W;
        const int wo = wo0 + r % Cfg::PATCH_W;
        longlong2 e = make_longlong2(-1ll, 0ll);
        if (ho < p.out_h && wo < p.out_w) {
            e.x = n * p.out_sn + ho * p.out_sh + wo * p.out_sw;
            e.y = n * p.res_sn + ho * p.res_sh + wo * p.res_sw;
        }
        rows[r] = e;
    }
}

#include "conv_implicit_gemm_halo_b.cuh"

/* NHWC A loader of the halo body. Each thread owns chunk tid % 4 of halo
   pixels tid / 4 + i * ROWS_PER_PASS: one 16-byte load per pixel. */
template <typename T, typename Cfg>
struct ConvHaloLoader {
    using Regs = ConvHaloRegs<Cfg>;
    ConvHaloB<T, Cfg> wb;
    /* Byte address of image n of the input. */
    unsigned long long in_base;
    /* Input row and column of halo pixel (0, 0). Negative in the padding. */
    int hi0;
    int wi0;
    int halo_w;
    int halo_pix;
    /* First owned halo pixel, tid / 4. */
    int q0;
    int chunk8;
    unsigned int sts_off;
    /* First channel of the chunk the halo registers load. */
    int halo_c0;

    __device__ __forceinline__ void init(const ConvIgemmParams<T>& p, int tid, int n, int ho0, int wo0,
                                         int block_col) {
        wb.init(p, tid, block_col);
        chunk8 = (tid & 3) * 8;
        q0 = tid >> 2;
        sts_off = conv_store_offset<Cfg>(tid);
        in_base = reinterpret_cast<unsigned long long>(p.input)
                + static_cast<unsigned long long>(n * p.in_sn * 2);
        hi0 = ho0 * p.stride_h - p.pad_top;
        wi0 = wo0 * p.stride_w - p.pad_left;
        halo_w = (Cfg::PATCH_W - 1) * p.stride_w + (p.kernel_w - 1) * p.dil_w + 1;
        const int halo_h = (Cfg::PATCH_H - 1) * p.stride_h + (p.kernel_h - 1) * p.dil_h + 1;
        halo_pix = halo_h * halo_w;
        halo_c0 = 0;
    }

    /* Offset-table entries of the owned halo pixels. Thread tid % 4 == 0 of
       each pixel writes; the caller's barrier publishes them to the other
       three. */
    __device__ __forceinline__ void fill_table(const ConvIgemmParams<T>& p, void* table_bytes, int) const {
        if (chunk8 != 0) return;
        long long* const table = static_cast<long long*>(table_bytes);
#pragma unroll
        for (int i = 0; i < Cfg::HALO_ITERS; ++i) {
            const int q = q0 + i * Cfg::ROWS_PER_PASS;
            const int hr = q / halo_w;
            const int hi = hi0 + hr;
            const int wi = wi0 + (q - hr * halo_w);
            const bool ok = q < halo_pix && hi >= 0 && hi < p.in_h && wi >= 0 && wi < p.in_w;
            table[q] = ok ? (static_cast<long long>(hi) * p.in_sh + static_cast<long long>(wi) * p.in_sw) * 2
                          : -1ll;
        }
    }

    /* Owned halo pixels of the chunk at halo_c0 into `r`. `table_bytes` is
       the published offset table. */
    __device__ __forceinline__ void load_halo(const ConvIgemmParams<T>& p, const void* table_bytes,
                                              Regs& r) const {
        const long long* const table = static_cast<const long long*>(table_bytes);
        const bool c_ok = halo_c0 + chunk8 < p.c_in;
        const unsigned long long base = in_base + static_cast<unsigned long long>((halo_c0 + chunk8) * 2);
#pragma unroll
        for (int i = 0; i < Cfg::HALO_ITERS; ++i) {
            const long long off = table[q0 + i * Cfg::ROWS_PER_PASS];
            const bool ok = c_ok && off >= 0;
            r.v[i] = conv_ldg_128(base + static_cast<unsigned long long>(off), ok ? 1u : 0u);
        }
    }

    /* `r` to the halo buffer at shared-window address `halo`. */
    __device__ __forceinline__ void store_halo(unsigned int halo, const void*, const Regs& r) const {
#pragma unroll
        for (int i = 0; i < Cfg::HALO_ITERS; ++i) {
            conv_sts_128(halo + sts_off + i * Cfg::ROWS_PER_PASS * 64, r.v[i]);
        }
    }

    __device__ __forceinline__ void next_halo_chunk() { halo_c0 += Cfg::BK; }
};

#include "conv_implicit_gemm_halo_nchw.cuh"

template <typename T, typename Cfg, bool NCHW> struct ConvHaloLoaderFor;
template <typename T, typename Cfg> struct ConvHaloLoaderFor<T, Cfg, false> {
    using type = ConvHaloLoader<T, Cfg>;
};
template <typename T, typename Cfg> struct ConvHaloLoaderFor<T, Cfg, true> {
    using type = ConvHaloLoaderNchw<T, Cfg>;
};

/* Halo-body ldmatrix state of one lane. */
template <typename Cfg>
struct ConvHaloLdm {
    /* Halo pixel of fragment i's row for this lane at tap (0, 0). */
    int hp0[Cfg::FRAG_M];
    /* This lane's A k-chunk at k16 step 0, times 16 bytes. */
    unsigned int a_k;
    unsigned int b[2];

    template <typename P>
    __device__ __forceinline__ ConvHaloLdm(const P& p, int lane, int warp_m, int warp_n, int halo_w) {
#pragma unroll
        for (int i = 0; i < Cfg::FRAG_M; ++i) {
            const int row = warp_m * Cfg::WARP_M + i * 16 + (lane & 15);
            const int lr = row / Cfg::PATCH_W;
            const int lc = row % Cfg::PATCH_W;
            hp0[i] = lr * p.stride_h * halo_w + lc * p.stride_w;
            asm volatile("mov.b32 %0, %0;" : "+r"(hp0[i]));
        }
        a_k = static_cast<unsigned int>((lane >> 4) << 4);
        conv_b_ldm_offsets<Cfg>(b, lane, warp_n);
    }
};

/* Halo-body fragments of k16 step KS at the tap whose halo-pixel offset is
   `tap_off`. `halo` and `bstage` are shared-window addresses. */
template <typename Cfg, int KS>
__device__ __forceinline__ void conv_load_halo_frags(ConvFrags<Cfg>& f, unsigned int halo, int tap_off,
                                                     unsigned int bstage, const ConvHaloLdm<Cfg>& l) {
#pragma unroll
    for (int i = 0; i < Cfg::FRAG_M; ++i) {
        const unsigned int hp = static_cast<unsigned int>(l.hp0[i] + tap_off);
        const unsigned int swz = ((hp >> 1) & 3u) << 4;
        conv_ldmatrix_x4(f.a[i], halo + hp * 64u + ((KS * 32u + l.a_k) ^ swz));
    }
    conv_load_b_frags<Cfg>(f, bstage + l.b[KS]);
}

/* Halo body. See the top of this file. */
template <typename T, typename Cfg, bool HALF_ACC, bool NCHW>
__device__ __forceinline__ void conv_igemm_run_halo(const ConvIgemmParams<T>& p) {
    conv_check_acc<T, Cfg, HALF_ACC>();
    using Acc = ConvAccum<HALF_ACC>;
    using Loader = typename ConvHaloLoaderFor<T, Cfg, NCHW>::type;

    __shared__ __align__(128) unsigned char smem_pool[Cfg::HALO_POOL_BYTES];
    __shared__ __align__(16) longlong2 smem_rows[Cfg::BM];

    int tid;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = warp / Cfg::WARPS_N;
    const int warp_n = warp % Cfg::WARPS_N;
    const int block_col = static_cast<int>(blockIdx.y) * Cfg::BN;

    const int tiles_w = (p.out_w + Cfg::PATCH_W - 1) / Cfg::PATCH_W;
    const int tiles_h = (p.out_h + Cfg::PATCH_H - 1) / Cfg::PATCH_H;
    const int patch = static_cast<int>(blockIdx.x);
    const int per_image = tiles_h * tiles_w;
    const int n = patch / per_image;
    const int in_image = patch - n * per_image;
    const int ph = in_image / tiles_w;
    const int ho0 = ph * Cfg::PATCH_H;
    const int wo0 = (in_image - ph * tiles_w) * Cfg::PATCH_W;

    conv_fill_patch_rows<T, Cfg>(p, smem_rows, tid, n, ho0, wo0);

    Loader ld;
    ld.init(p, tid, n, ho0, wo0, block_col);
    const ConvHaloLdm<Cfg> ldm(p, lane, warp_m, warp_n, ld.halo_w);
    const int chunks = (p.c_in + Cfg::BK - 1) / Cfg::BK;
    const int num_k_tiles = ld.wb.taps * chunks;
    const int row_step = p.dil_h * ld.halo_w;

    const unsigned int halo = conv_smem_addr(smem_pool);
    const unsigned int bbase = halo + Cfg::HALO_BYTES;
    void* const table = smem_pool + Cfg::HALO_BYTES + 2 * Cfg::STAGE_B_BYTES;
    ld.fill_table(p, table, tid);
    /* Publishes the offset table. */
    __syncthreads();
    /* B stage the current K-tile reads from and the next one lands in. */
    unsigned int rd = 0u;
    unsigned int wr = Cfg::STAGE_B_BYTES;

    typename Acc::type acc[Cfg::FRAG_M][Cfg::FRAG_N][Acc::REGS];
    conv_zero_acc<Cfg, HALF_ACC>(acc);

    typename Loader::Regs hregs;
    ConvBRegs<Cfg> bregs;
    ld.load_halo(p, table, hregs);
    ld.store_halo(halo, table, hregs);
    ld.wb.load(p, bregs);
    ld.wb.store(bbase, bregs);
    if (num_k_tiles > 1) {
        ld.wb.advance();
        ld.wb.load(p, bregs);
    }
    if (chunks > 1) {
        ld.next_halo_chunk();
        ld.load_halo(p, table, hregs);
    }
    __syncthreads();

    ConvFrags<Cfg> f0;
    ConvFrags<Cfg> f1;
    conv_load_halo_frags<Cfg, 0>(f0, halo, 0, bbase, ldm);

    /* Tap (kh, kw) of the current K-tile and its halo-pixel offset. */
    int kh = 0;
    int kw = 0;
    int tap_off = 0;
    /* Chunk in the halo buffer. */
    int chunk = 0;
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const bool more = kt + 1 < num_k_tiles;
        int nkw = kw + 1;
        int nkh = kh;
        int ntap_off = tap_off + p.dil_w;
        bool chunk_end = false;
        if (nkw == p.kernel_w) {
            nkw = 0;
            ++nkh;
            ntap_off = nkh * row_step;
            if (nkh == p.kernel_h) {
                nkh = 0;
                ntap_off = 0;
                chunk_end = true;
            }
        }

        conv_load_halo_frags<Cfg, 1>(f1, halo, tap_off, bbase + rd, ldm);
        conv_mma_k16<T, Cfg, HALF_ACC>(acc, f0);
        /* The other B stage was last read before the previous barrier. */
        if (more) ld.wb.store(bbase + wr, bregs);
        /* Publishes the B stage written above, and orders every ldmatrix of
           B stage `rd` and, at a chunk end, of the halo before their next
           overwrite. After the last K-tile it frees shared memory for the
           epilogue's output tile. */
        __syncthreads();
        if (chunk_end && more) {
            /* Chunk end. The halo is free once every warp passed the barrier
               above. The mma of k16 step 1 issues first, so it runs while
               the halo stores and the second barrier wait. */
            conv_mma_k16<T, Cfg, HALF_ACC>(acc, f1);
            ld.store_halo(halo, table, hregs);
            /* Publishes the halo of the next chunk. */
            __syncthreads();
            conv_load_halo_frags<Cfg, 0>(f0, halo, 0, bbase + wr, ldm);
            ++chunk;
            if (chunk + 1 < chunks) {
                ld.next_halo_chunk();
                ld.load_halo(p, table, hregs);
            }
            if (kt + 2 < num_k_tiles) {
                ld.wb.advance();
                ld.wb.load(p, bregs);
            }
        } else {
            if (more) conv_load_halo_frags<Cfg, 0>(f0, halo, ntap_off, bbase + wr, ldm);
            if (kt + 2 < num_k_tiles) {
                ld.wb.advance();
                ld.wb.load(p, bregs);
            }
            conv_mma_k16<T, Cfg, HALF_ACC>(acc, f1);
        }
        kh = nkh;
        kw = nkw;
        tap_off = ntap_off;
        rd ^= Cfg::STAGE_B_BYTES;
        wr ^= Cfg::STAGE_B_BYTES;
    }

    /* A run of 8 block rows never leaves its patch row. */
    conv_finish<T, Cfg, HALF_ACC, NCHW>(p, acc, smem_pool, smem_rows, tid, warp_m, warp_n, block_col, true);
}

#endif  // NUMR_CONV_IMPLICIT_GEMM_HALO_CUH
