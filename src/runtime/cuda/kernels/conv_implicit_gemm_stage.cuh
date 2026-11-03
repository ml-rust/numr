// Operand staging of the implicit-GEMM conv2d in conv_implicit_gemm.cuh.
//
// Each thread owns one 16-byte chunk column of the K-tile (chunk = tid % 4)
// and rows tid / 4 + i * (THREADS / 4) of the A and B tiles. A K-tile moves
// in three steps:
//   load    predicated global loads into a caller-owned register set,
//           zeros where masked
//   store   one STS.128 per owned row into the swizzled stage buffer
//   advance move the K cursor to the next K-tile
// The body keeps up to two register sets in flight, so the global latency
// of a K-tile overlaps the tensor-core work of the tiles before it.
//
// Two gather loaders, picked on the host per launch. The halo loader of the
// halo body lives in conv_implicit_gemm_halo.cuh.
//
// ConvLoaderVec: C_in % 8 == 0, a 16-byte aligned input with N/H/W strides
// that are multiples of 8, and a 16-byte aligned weight. K runs tap-major:
// for each (kh, kw), ceil(C_in / 32) K-tiles of 32 channels. Per row, the
// prologue keeps a signed 64-bit base address and two 32-bit masks whose bit
// kh (kw) is set when that tap row (column) lands inside the image. Per
// K-tile the address is one 64-bit add per row, with no division.
//
// ConvLoaderScalar: every other NHWC shape, and the NCHW gather. K runs
// flat in (kh, kw, ci) order, so C_in = 3 pads no K. A per-K-tile column
// table in shared memory holds (kh * dil_h, kw * dil_w, input byte offset)
// for the 32 columns; the offset includes ci * in_sc, so the same loader
// reads NHWC (in_sc = 1) and NCHW. Each thread gathers 8 predicated 16-bit
// loads per row into one 16-byte chunk. In NCHW the 8 threads of one chunk
// hold 8 consecutive rows, so each load reads consecutive pixels of one
// channel.
//
// Both loaders store +0.0 for every position outside the image, past K, past
// C_in, past M or past C_out, in A and in B. Shared memory is never
// implicitly zero, and a stage keeps the previous K-tile's values.
//
// Addresses are 64-bit integers until a load is valid: a predicated-off load
// never forms or reads its address. Offsets are signed: a row's base is
// below the image when its first tap lies in the padding.

#ifndef NUMR_CONV_IMPLICIT_GEMM_STAGE_CUH
#define NUMR_CONV_IMPLICIT_GEMM_STAGE_CUH

/* 16 bytes at global address `addr` through the read-only path, or zeros
   when `pred` is 0. */
__device__ __forceinline__ int4 conv_ldg_128(unsigned long long addr, unsigned int pred) {
    int4 v;
    asm("{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %5, 0;\n\t"
        "mov.b32 %0, 0;\n\tmov.b32 %1, 0;\n\tmov.b32 %2, 0;\n\tmov.b32 %3, 0;\n\t"
        "@p ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];\n\t}"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(addr), "r"(pred));
    return v;
}

/* 2 bytes at global address `addr` through the read-only path, or zero
   when `pred` is 0. */
__device__ __forceinline__ unsigned int conv_ldg_16(unsigned long long addr, unsigned int pred) {
    unsigned short v;
    asm("{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "mov.b16 %0, 0;\n\t"
        "@p ld.global.nc.b16 %0, [%1];\n\t}"
        : "=h"(v)
        : "l"(addr), "r"(pred));
    return v;
}

/* Output pixel (n, ho, wo) of GEMM row m. */
__device__ __forceinline__ void conv_decode_pixel(int m, int out_w, int pixels, int& n, int& ho, int& wo) {
    n = m / pixels;
    const int rem = m - n * pixels;
    ho = rem / out_w;
    wo = rem - ho * out_w;
}

/* Byte offset of this thread's first owned row and chunk in a stage's A or
   B buffer. Owned rows are ROWS_PER_PASS apart, a multiple of the swizzle
   period, so row i adds i * ROWS_PER_PASS * 64 bytes. */
template <typename Cfg>
__device__ __forceinline__ unsigned int conv_store_offset(int tid) {
    const int r = tid >> 2;
    return static_cast<unsigned int>((r * 4 + conv_stage_swizzle(r, tid & 3)) * 16);
}

/* 16 bytes to shared-window address `addr`. Volatile with a memory
   clobber: the store keeps its place between the barriers that order it
   against the ldmatrix reads of the same stage. */
__device__ __forceinline__ void conv_sts_128(unsigned int addr, int4 v) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :
                 : "r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

/* One register set: the owned A rows and B rows of one K-tile. */
template <typename Cfg>
struct ConvStageRegs {
    int4 a[Cfg::A_ITERS];
    int4 b[Cfg::B_ITERS];
};

/* One STS.128 per owned A row and B row. `addr` is the shared-window
   address of the stage base plus this thread's `conv_store_offset`. Row
   offsets are compile-time, so each store takes an immediate offset. */
template <typename Cfg>
__device__ __forceinline__ void conv_stage_store(unsigned int addr, const ConvStageRegs<Cfg>& r) {
    static_assert(Cfg::ROWS_PER_PASS % 8 == 0, "owned rows must share one swizzle phase");
#pragma unroll
    for (int i = 0; i < Cfg::A_ITERS; ++i) {
        conv_sts_128(addr + i * Cfg::ROWS_PER_PASS * 64, r.a[i]);
    }
#pragma unroll
    for (int j = 0; j < Cfg::B_ITERS; ++j) {
        conv_sts_128(addr + Cfg::STAGE_A_BYTES + j * Cfg::ROWS_PER_PASS * 64, r.b[j]);
    }
}

template <typename T, typename Cfg>
struct ConvLoaderVec {
    /* Byte address of (row base + chunk * 8) in the input. */
    unsigned long long a_addr[Cfg::A_ITERS];
    unsigned int a_mask_h[Cfg::A_ITERS];
    unsigned int a_mask_w[Cfg::A_ITERS];
    /* Bit i: owned row i reads inside the image at the current tap. */
    unsigned int tap_ok;
    /* Byte address of (co * K + chunk * 8) in the weight. */
    unsigned long long b_addr[Cfg::B_ITERS];
    /* Bit j: owned B row j is below C_out. */
    unsigned int b_ok;
    unsigned int sts_off;
    int chunk8;
    int c0;
    int kh;
    int kw;
    /* Input elements from a row base to the current tap. */
    long long tap_off;
    /* Weight elements from a B row base to the current tap. */
    int b_tap;

    static __device__ __forceinline__ int num_k_tiles(const ConvIgemmParams<T>& p) {
        return p.kernel_h * p.kernel_w * ((p.c_in + Cfg::BK - 1) / Cfg::BK);
    }

    __device__ __forceinline__ void update_tap_ok() {
        unsigned int bits = 0u;
#pragma unroll
        for (int i = 0; i < Cfg::A_ITERS; ++i) {
            bits |= ((a_mask_h[i] >> kh) & (a_mask_w[i] >> kw) & 1u) << i;
        }
        tap_ok = bits;
    }

    __device__ __forceinline__ void init(const ConvIgemmParams<T>& p, int tid, long long block_row, int block_col) {
        chunk8 = (tid & 3) * 8;
        sts_off = conv_store_offset<Cfg>(tid);
        const int pixels = p.out_h * p.out_w;
        const unsigned long long in_addr = reinterpret_cast<unsigned long long>(p.input);
#pragma unroll
        for (int i = 0; i < Cfg::A_ITERS; ++i) {
            const long long m = block_row + (tid >> 2) + i * Cfg::ROWS_PER_PASS;
            long long base = 0;
            unsigned int mask_h = 0u;
            unsigned int mask_w = 0u;
            if (m < p.M) {
                int n, ho, wo;
                conv_decode_pixel(static_cast<int>(m), p.out_w, pixels, n, ho, wo);
                const long long hi0 = static_cast<long long>(ho) * p.stride_h - p.pad_top;
                const long long wi0 = static_cast<long long>(wo) * p.stride_w - p.pad_left;
                base = n * p.in_sn + hi0 * p.in_sh + wi0 * p.in_sw + chunk8;
                for (int t = 0; t < p.kernel_h; ++t) {
                    const long long hi = hi0 + static_cast<long long>(t) * p.dil_h;
                    if (hi >= 0 && hi < p.in_h) mask_h |= 1u << t;
                }
                for (int t = 0; t < p.kernel_w; ++t) {
                    const long long wi = wi0 + static_cast<long long>(t) * p.dil_w;
                    if (wi >= 0 && wi < p.in_w) mask_w |= 1u << t;
                }
            }
            a_addr[i] = in_addr + static_cast<unsigned long long>(base * 2);
            a_mask_h[i] = mask_h;
            a_mask_w[i] = mask_w;
        }
        const long long K = static_cast<long long>(p.kernel_h) * p.kernel_w * p.c_in;
        const unsigned long long w_addr = reinterpret_cast<unsigned long long>(p.weight);
        b_ok = 0u;
#pragma unroll
        for (int j = 0; j < Cfg::B_ITERS; ++j) {
            const int co = block_col + (tid >> 2) + j * Cfg::ROWS_PER_PASS;
            const bool ok = co < p.c_out;
            b_addr[j] = w_addr + static_cast<unsigned long long>((ok ? co * K + chunk8 : 0ll) * 2);
            b_ok |= (ok ? 1u : 0u) << j;
        }
        c0 = 0;
        kh = 0;
        kw = 0;
        tap_off = 0;
        b_tap = 0;
        update_tap_ok();
    }

    /* The vector loader keeps its K position in registers. */
    __device__ __forceinline__ void write_cols(const ConvIgemmParams<T>&, int4*, int) {}

    __device__ __forceinline__ void load(const ConvIgemmParams<T>& p, const int4*, ConvStageRegs<Cfg>& r) const {
        const bool c_ok = c0 + chunk8 < p.c_in;
        const unsigned int a_bits = c_ok ? tap_ok : 0u;
        const unsigned int b_bits = c_ok ? b_ok : 0u;
        const unsigned long long a_koff = static_cast<unsigned long long>((tap_off + c0) * 2);
        const unsigned long long b_koff = static_cast<unsigned long long>(b_tap + c0) * 2ull;
#pragma unroll
        for (int i = 0; i < Cfg::A_ITERS; ++i) {
            r.a[i] = conv_ldg_128(a_addr[i] + a_koff, (a_bits >> i) & 1u);
        }
#pragma unroll
        for (int j = 0; j < Cfg::B_ITERS; ++j) {
            r.b[j] = conv_ldg_128(b_addr[j] + b_koff, (b_bits >> j) & 1u);
        }
    }

    __device__ __forceinline__ void advance(const ConvIgemmParams<T>& p) {
        c0 += Cfg::BK;
        if (c0 >= p.c_in) {
            c0 = 0;
            b_tap += p.c_in;
            ++kw;
            tap_off += static_cast<long long>(p.dil_w) * p.in_sw;
            if (kw == p.kernel_w) {
                kw = 0;
                ++kh;
                tap_off = static_cast<long long>(kh) * p.dil_h * p.in_sh;
            }
            update_tap_ok();
        }
    }

};

template <typename T, typename Cfg>
struct ConvLoaderScalar {
    /* Byte address of the row base in the input. */
    unsigned long long a_addr[Cfg::A_ITERS];
    int a_hi0[Cfg::A_ITERS];
    int a_wi0[Cfg::A_ITERS];
    /* Bit i: owned row i is below M. */
    unsigned int a_valid;
    /* Byte address of (co * K + chunk * 8) in the weight. */
    unsigned long long b_addr[Cfg::B_ITERS];
    /* Bit j: owned B row j is below C_out. */
    unsigned int b_ok;
    unsigned int sts_off;
    int chunk8;
    int k0;
    int K;
    /* Column-table cursor of threads tid < BK: K column col_k, at
       (col_kh, col_kw, col_ci). Each write_cols moves it BK columns on,
       with BK = (step_kh * Kw + step_kw) * C_in + step_ci. */
    int col_k;
    int col_ci;
    int col_kw;
    int col_kh;
    int step_ci;
    int step_kw;
    int step_kh;

    static __device__ __forceinline__ int num_k_tiles(const ConvIgemmParams<T>& p) {
        const int k = p.kernel_h * p.kernel_w * p.c_in;
        return (k + Cfg::BK - 1) / Cfg::BK;
    }

    __device__ __forceinline__ void init(const ConvIgemmParams<T>& p, int tid, long long block_row, int block_col) {
        chunk8 = (tid & 3) * 8;
        sts_off = conv_store_offset<Cfg>(tid);
        K = p.kernel_h * p.kernel_w * p.c_in;
        k0 = 0;
        const int pixels = p.out_h * p.out_w;
        const unsigned long long in_addr = reinterpret_cast<unsigned long long>(p.input);
        a_valid = 0u;
#pragma unroll
        for (int i = 0; i < Cfg::A_ITERS; ++i) {
            const long long m = block_row + (tid >> 2) + i * Cfg::ROWS_PER_PASS;
            long long base = 0;
            int hi0 = 0;
            int wi0 = 0;
            if (m < p.M) {
                int n, ho, wo;
                conv_decode_pixel(static_cast<int>(m), p.out_w, pixels, n, ho, wo);
                hi0 = ho * p.stride_h - p.pad_top;
                wi0 = wo * p.stride_w - p.pad_left;
                base = n * p.in_sn + static_cast<long long>(hi0) * p.in_sh
                     + static_cast<long long>(wi0) * p.in_sw;
                a_valid |= 1u << i;
            }
            a_addr[i] = in_addr + static_cast<unsigned long long>(base * 2);
            a_hi0[i] = hi0;
            a_wi0[i] = wi0;
        }
        const unsigned long long w_addr = reinterpret_cast<unsigned long long>(p.weight);
        b_ok = 0u;
#pragma unroll
        for (int j = 0; j < Cfg::B_ITERS; ++j) {
            const int co = block_col + (tid >> 2) + j * Cfg::ROWS_PER_PASS;
            const bool ok = co < p.c_out;
            const long long off = ok ? static_cast<long long>(co) * K + chunk8 : 0ll;
            b_addr[j] = w_addr + static_cast<unsigned long long>(off * 2);
            b_ok |= (ok ? 1u : 0u) << j;
        }
        /* Divisions run once here. write_cols advances without them. */
        col_k = tid;
        const int tap = tid / p.c_in;
        col_ci = tid - tap * p.c_in;
        col_kh = tap / p.kernel_w;
        col_kw = tap - col_kh * p.kernel_w;
        const int q = Cfg::BK / p.c_in;
        step_ci = Cfg::BK - q * p.c_in;
        step_kh = q / p.kernel_w;
        step_kw = q - step_kh * p.kernel_w;
    }

    /* Threads tid < BK write their column of the next table into `table`,
       then step their cursor one K-tile on. A column past K gets a zero
       entry, which load never reads. */
    __device__ __forceinline__ void write_cols(const ConvIgemmParams<T>& p, int4* table, int tid) {
        if (tid >= Cfg::BK) return;
        int4 e = make_int4(0, 0, 0, 0);
        if (col_k < K) {
            const int dh = col_kh * p.dil_h;
            const int dw = col_kw * p.dil_w;
            const long long off = (static_cast<long long>(dh) * p.in_sh
                                 + static_cast<long long>(dw) * p.in_sw + col_ci * p.in_sc) * 2;
            e = make_int4(dh, dw, static_cast<int>(off & 0xffffffffll), static_cast<int>(off >> 32));
        }
        table[tid] = e;
        col_k += Cfg::BK;
        col_ci += step_ci;
        const int carry = col_ci >= p.c_in ? 1 : 0;
        col_ci -= carry * p.c_in;
        /* step_kw + carry <= Kw, so one wrap is enough. */
        col_kw += step_kw + carry;
        if (col_kw >= p.kernel_w) {
            col_kw -= p.kernel_w;
            ++col_kh;
        }
        col_kh += step_kh;
    }

    /* `table` holds the column entries of the K-tile at k0. */
    __device__ __forceinline__ void load(const ConvIgemmParams<T>& p, const int4* table, ConvStageRegs<Cfg>& r) const {
        const int kb = k0 + chunk8;
        int4 col[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) col[j] = table[chunk8 + j];
#pragma unroll
        for (int i = 0; i < Cfg::A_ITERS; ++i) {
            unsigned int w[4] = {0u, 0u, 0u, 0u};
            const bool row_ok = ((a_valid >> i) & 1u) != 0u;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                /* Unsigned sums: the bounds test is exact for any int
                   operands, with no signed overflow. */
                const unsigned int hi = static_cast<unsigned int>(a_hi0[i]) + static_cast<unsigned int>(col[j].x);
                const unsigned int wi = static_cast<unsigned int>(a_wi0[i]) + static_cast<unsigned int>(col[j].y);
                const bool ok = row_ok && kb + j < K && hi < static_cast<unsigned int>(p.in_h)
                             && wi < static_cast<unsigned int>(p.in_w);
                const unsigned long long off =
                    (static_cast<unsigned long long>(static_cast<unsigned int>(col[j].w)) << 32)
                    | static_cast<unsigned int>(col[j].z);
                w[j >> 1] |= conv_ldg_16(a_addr[i] + off, ok ? 1u : 0u) << ((j & 1) * 16);
            }
            r.a[i] = make_int4(static_cast<int>(w[0]), static_cast<int>(w[1]),
                                  static_cast<int>(w[2]), static_cast<int>(w[3]));
        }
        const unsigned long long b_koff = static_cast<unsigned long long>(k0) * 2ull;
#pragma unroll
        for (int jb = 0; jb < Cfg::B_ITERS; ++jb) {
            unsigned int w[4] = {0u, 0u, 0u, 0u};
            const bool row_ok = ((b_ok >> jb) & 1u) != 0u;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned int ok = (row_ok && kb + j < K) ? 1u : 0u;
                w[j >> 1] |= conv_ldg_16(b_addr[jb] + b_koff + j * 2, ok) << ((j & 1) * 16);
            }
            r.b[jb] = make_int4(static_cast<int>(w[0]), static_cast<int>(w[1]),
                                   static_cast<int>(w[2]), static_cast<int>(w[3]));
        }
    }

    __device__ __forceinline__ void advance(const ConvIgemmParams<T>&) { k0 += Cfg::BK; }

};

#endif  // NUMR_CONV_IMPLICIT_GEMM_STAGE_CUH
