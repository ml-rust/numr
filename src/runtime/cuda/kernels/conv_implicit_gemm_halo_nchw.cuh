// NCHW A loader of the halo body in conv_implicit_gemm_halo.cuh.
//
// The halo layout in shared memory is the NHWC one: halo pixel q is a
// 64-byte row of 32 channels with the conv_stage_swizzle XOR of row q. Only
// the global reads differ. NCHW holds one channel's halo rows contiguous
// along W, so the loader reads along W and transposes on the way into
// shared memory.
//
// Runs. Each halo row splits into runs of 4 pixels that start on a multiple
// of 4 input columns:
//   mis      = wi0 & 3, the same for every block (wo0 is a multiple of 16)
//   row_runs = (mis + halo_w + 3) / 4
// Run (hr, k) covers input columns wi0 - mis + 4k .. + 3 of halo row hr.
// Its halo pixels are q0 + j, q0 = hr * halo_w + 4k - mis, for the j with
// 0 <= 4k - mis + j < halo_w.
//
// Work. 16 threads share a run, one channel pair each: thread tid owns
// channel pair tid % 16 (channels 2cp and 2cp + 1 of the chunk) of runs
// tid / 16 + i * NCHW_RUN_STEP. Per run and chunk a thread
//   loads  one 8-byte load per channel: 4 pixels of one channel row,
//          aligned. A run cut by the right image edge loads its pixels one
//          by one. A run fully outside the image loads nothing.
//   stores 4 PRMT + 4 STS.32, one per pixel: the two channels of the pixel
//          packed as one half2, low channel in the low half.
// The two half-warps of one store hold adjacent runs, 4 pixels apart. The
// upper half-warp walks its pixels in the order 1, 0, 3, 2, so the two
// pixels of one STS.32 have opposite parity and fill all 32 banks.
//
// Run table, one int4 per run, written once per block:
//   x, y  input element offset of the run from image n and channel 0, as
//         the low and high words of an i64. Unused when the run is empty.
//   z     q0, the halo pixel of the run's first slot (can be negative).
//   w     bits 0-3: slot j is a halo pixel. Bits 4-7: slot j is inside
//         the image.
// Slots outside the image store +0.0, as do channels past C_in. Slots that
// are not halo pixels store nothing. Halo pixels past halo_h * halo_w are
// never written and never read: every ldmatrix row lies inside the halo.
//
// The host routes NCHW here only when C_in % 8 == 0, the input is 8-byte
// aligned, its N/C/H strides are multiples of 4, its W stride is 1, the
// weight is 16-byte aligned, halo_h * halo_w <= HALO_PIX and
// halo_h * row_runs <= NCHW_RUNS. See loader/conv_implicit_gemm_staging.rs.

#ifndef NUMR_CONV_IMPLICIT_GEMM_HALO_NCHW_CUH
#define NUMR_CONV_IMPLICIT_GEMM_HALO_NCHW_CUH

/* 8 bytes at global address `addr` through the read-only path, or zeros
   when `pred` is 0. */
__device__ __forceinline__ uint2 conv_ldg_64(unsigned long long addr, unsigned int pred) {
    uint2 v;
    asm("{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %3, 0;\n\t"
        "mov.b32 %0, 0;\n\tmov.b32 %1, 0;\n\t"
        "@p ld.global.nc.v2.b32 {%0, %1}, [%2];\n\t}"
        : "=r"(v.x), "=r"(v.y)
        : "l"(addr), "r"(pred));
    return v;
}

/* The 4 elements at `addr` packed as two half2 words: element j where bit
   j of `mask` is set, zero elsewhere. One 16-bit load per element. */
__device__ __forceinline__ uint2 conv_ldg_16x4(unsigned long long addr, unsigned int mask) {
    const unsigned int e0 = conv_ldg_16(addr, mask & 1u);
    const unsigned int e1 = conv_ldg_16(addr + 2ull, mask & 2u);
    const unsigned int e2 = conv_ldg_16(addr + 4ull, mask & 4u);
    const unsigned int e3 = conv_ldg_16(addr + 6ull, mask & 8u);
    return make_uint2(e0 | (e1 << 16), e2 | (e3 << 16));
}

/* 4 bytes to shared-window address `addr`. Volatile with a memory clobber,
   as conv_sts_128. */
__device__ __forceinline__ void conv_sts_32(unsigned int addr, unsigned int v) {
    asm volatile("st.shared.b32 [%0], %1;" : : "r"(addr), "r"(v) : "memory");
}

/* Halo registers of one chunk: per owned run, 4 pixels of channel 2cp in
   `lo` and of channel 2cp + 1 in `hi`. Word x holds pixels 0 and 1, word y
   pixels 2 and 3, the lower pixel in the low half. */
template <typename Cfg>
struct ConvHaloNchwRegs {
    uint2 lo[Cfg::NCHW_ITERS];
    uint2 hi[Cfg::NCHW_ITERS];
};

template <typename T, typename Cfg>
struct ConvHaloLoaderNchw {
    using Regs = ConvHaloNchwRegs<Cfg>;
    ConvHaloB<T, Cfg> wb;
    /* Byte address of image n of the input. */
    unsigned long long in_base;
    /* Channel stride in bytes. */
    unsigned long long sc_bytes;
    /* Input row and column of halo pixel (0, 0). Negative in the padding. */
    int hi0;
    int wi0;
    int halo_w;
    int halo_h;
    int mis;
    int row_runs;
    /* Owned channel pair, tid % 16, and first owned run, tid / 16. */
    int cp;
    int run0;
    /* Byte 4 * cp of a halo pixel row, before the swizzle. */
    unsigned int sts_word;
    /* 1 in the upper half-warp, which walks its pixels in the order 1, 0,
       3, 2. sel[s] is the PRMT selector of store slot parity s. */
    unsigned int flip;
    unsigned int sel[2];
    /* First channel of the chunk the halo registers load. */
    int halo_c0;

    __device__ __forceinline__ void init(const ConvIgemmParams<T>& p, int tid, int n, int ho0, int wo0,
                                         int block_col) {
        wb.init(p, tid, block_col);
        cp = tid & 15;
        run0 = tid >> 4;
        sts_word = static_cast<unsigned int>(cp * 4);
        flip = static_cast<unsigned int>(run0 & 1);
        /* 0x5410 packs the low halves of (lo, hi), 0x7632 the high halves. */
        sel[0] = flip != 0u ? 0x7632u : 0x5410u;
        sel[1] = flip != 0u ? 0x5410u : 0x7632u;
        in_base = reinterpret_cast<unsigned long long>(p.input)
                + static_cast<unsigned long long>(n * p.in_sn * 2);
        sc_bytes = static_cast<unsigned long long>(p.in_sc * 2);
        hi0 = ho0 * p.stride_h - p.pad_top;
        wi0 = wo0 * p.stride_w - p.pad_left;
        halo_w = (Cfg::PATCH_W - 1) * p.stride_w + (p.kernel_w - 1) * p.dil_w + 1;
        halo_h = (Cfg::PATCH_H - 1) * p.stride_h + (p.kernel_h - 1) * p.dil_h + 1;
        /* Two's complement: wi0 & 3 is wi0 mod 4 for negative wi0 too. */
        mis = wi0 & 3;
        row_runs = (mis + halo_w + 3) >> 2;
        halo_c0 = 0;
    }

    /* Every run-table entry. The caller's barrier publishes them. */
    __device__ __forceinline__ void fill_table(const ConvIgemmParams<T>& p, void* table_bytes, int tid) const {
        int4* const table = static_cast<int4*>(table_bytes);
        const int runs = halo_h * row_runs;
        for (int rr = tid; rr < Cfg::NCHW_RUNS; rr += Cfg::THREADS) {
            int4 e = make_int4(0, 0, 0, 0);
            if (rr < runs) {
                const int hr = rr / row_runs;
                const int hc0 = (rr - hr * row_runs) * 4 - mis;
                const int a = wi0 + hc0;
                const int hi = hi0 + hr;
                const bool row_ok = hi >= 0 && hi < p.in_h;
                unsigned int halo_bits = 0u;
                unsigned int img_bits = 0u;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (hc0 + j >= 0 && hc0 + j < halo_w) halo_bits |= 1u << j;
                    if (row_ok && a + j >= 0 && a + j < p.in_w) img_bits |= 1u << j;
                }
                const long long off = img_bits != 0u ? static_cast<long long>(hi) * p.in_sh + a : 0ll;
                e = make_int4(static_cast<int>(off & 0xffffffffll), static_cast<int>(off >> 32),
                              hr * halo_w + hc0, static_cast<int>(halo_bits | (img_bits << 4)));
            }
            table[rr] = e;
        }
    }

    /* Owned runs of the chunk at halo_c0 into `r`. `table_bytes` is the
       published run table. */
    __device__ __forceinline__ void load_halo(const ConvIgemmParams<T>& p, const void* table_bytes,
                                              Regs& r) const {
        const int4* const table = static_cast<const int4*>(table_bytes);
        const int c = halo_c0 + 2 * cp;
        const unsigned int ok_lo = c < p.c_in ? 15u : 0u;
        const unsigned int ok_hi = c + 1 < p.c_in ? 15u : 0u;
        const unsigned long long base = in_base + static_cast<unsigned long long>(c * p.in_sc * 2);
#pragma unroll
        for (int i = 0; i < Cfg::NCHW_ITERS; ++i) {
            const int4 e = table[run0 + i * Cfg::NCHW_RUN_STEP];
            const unsigned int img = (static_cast<unsigned int>(e.w) >> 4) & 15u;
            const long long off = (static_cast<long long>(e.y) << 32) | static_cast<unsigned int>(e.x);
            const unsigned long long a = base + static_cast<unsigned long long>(off * 2);
            if (img == 15u || img == 0u) {
                r.lo[i] = conv_ldg_64(a, img & ok_lo);
                r.hi[i] = conv_ldg_64(a + sc_bytes, img & ok_hi);
            } else {
                r.lo[i] = conv_ldg_16x4(a, img & ok_lo);
                r.hi[i] = conv_ldg_16x4(a + sc_bytes, img & ok_hi);
            }
        }
    }

    /* `r` to the halo buffer at shared-window address `halo`. */
    __device__ __forceinline__ void store_halo(unsigned int halo, const void* table_bytes, const Regs& r) const {
        const int4* const table = static_cast<const int4*>(table_bytes);
#pragma unroll
        for (int i = 0; i < Cfg::NCHW_ITERS; ++i) {
            const int4 e = table[run0 + i * Cfg::NCHW_RUN_STEP];
            const unsigned int halo_bits = static_cast<unsigned int>(e.w);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                /* Slot j ^ flip shares word j / 2 with slot j. */
                const unsigned int slot = static_cast<unsigned int>(j) ^ flip;
                const unsigned int lo = j < 2 ? r.lo[i].x : r.lo[i].y;
                const unsigned int hi = j < 2 ? r.hi[i].x : r.hi[i].y;
                const unsigned int v = __byte_perm(lo, hi, sel[j & 1]);
                const unsigned int q = static_cast<unsigned int>(e.z) + slot;
                if ((halo_bits >> slot) & 1u) {
                    conv_sts_32(halo + q * 64u + (sts_word ^ ((q << 3) & 0x30u)), v);
                }
            }
        }
    }

    __device__ __forceinline__ void next_halo_chunk() { halo_c0 += Cfg::BK; }
};

#endif  // NUMR_CONV_IMPLICIT_GEMM_HALO_NCHW_CUH
