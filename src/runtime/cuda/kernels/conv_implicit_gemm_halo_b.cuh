// B staging of the halo body in conv_implicit_gemm_halo.cuh, shared by the
// NHWC and NCHW A loaders. B is the OHWI weight in both layouts.

#ifndef NUMR_CONV_IMPLICIT_GEMM_HALO_B_CUH
#define NUMR_CONV_IMPLICIT_GEMM_HALO_B_CUH

/* B registers of one K-tile. */
template <typename Cfg>
struct ConvBRegs {
    int4 b[Cfg::B_ITERS];
};

/* B staging of the halo body, shared by both A loaders. Each thread owns
   chunk tid % 4 of B rows tid / 4 + j * ROWS_PER_PASS. */
template <typename T, typename Cfg>
struct ConvHaloB {
    /* Byte address of (co * K + chunk * 8) in the weight. */
    unsigned long long b_addr[Cfg::B_ITERS];
    /* Bit j: owned B row j is below C_out. */
    unsigned int b_ok;
    unsigned int sts_off;
    int chunk8;
    int taps;
    /* Channel chunk and tap of the B tile the B registers load. */
    int b_c0;
    int b_tap;

    __device__ __forceinline__ void init(const ConvIgemmParams<T>& p, int tid, int block_col) {
        chunk8 = (tid & 3) * 8;
        sts_off = conv_store_offset<Cfg>(tid);
        taps = p.kernel_h * p.kernel_w;
        const long long K = static_cast<long long>(taps) * p.c_in;
        const unsigned long long w_addr = reinterpret_cast<unsigned long long>(p.weight);
        b_ok = 0u;
#pragma unroll
        for (int j = 0; j < Cfg::B_ITERS; ++j) {
            const int co = block_col + (tid >> 2) + j * Cfg::ROWS_PER_PASS;
            const bool ok = co < p.c_out;
            b_addr[j] = w_addr + static_cast<unsigned long long>((ok ? co * K + chunk8 : 0ll) * 2);
            b_ok |= (ok ? 1u : 0u) << j;
        }
        b_c0 = 0;
        b_tap = 0;
    }

    /* Owned B rows of the K-tile at (b_c0, b_tap) into `r`. */
    __device__ __forceinline__ void load(const ConvIgemmParams<T>& p, ConvBRegs<Cfg>& r) const {
        const unsigned int bits = b_c0 + chunk8 < p.c_in ? b_ok : 0u;
        const unsigned long long koff =
            static_cast<unsigned long long>(static_cast<long long>(b_tap) * p.c_in + b_c0) * 2ull;
#pragma unroll
        for (int j = 0; j < Cfg::B_ITERS; ++j) {
            r.b[j] = conv_ldg_128(b_addr[j] + koff, (bits >> j) & 1u);
        }
    }

    /* `r` to the B stage at shared-window address `stage`. */
    __device__ __forceinline__ void store(unsigned int stage, const ConvBRegs<Cfg>& r) const {
#pragma unroll
        for (int j = 0; j < Cfg::B_ITERS; ++j) {
            conv_sts_128(stage + sts_off + j * Cfg::ROWS_PER_PASS * 64, r.b[j]);
        }
    }

    /* Move the B cursor one K-tile on: the next tap, then the next chunk. */
    __device__ __forceinline__ void advance() {
        if (++b_tap == taps) {
            b_tap = 0;
            b_c0 += Cfg::BK;
        }
    }
};

#endif  // NUMR_CONV_IMPLICIT_GEMM_HALO_B_CUH
