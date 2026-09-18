// Register-staged tile loads for the compile-time-tiled FP32 GEMM core.
//
// `ct_fetch_*` issues one tile's global loads into a per-thread register image
// and `ct_store_*` writes that image to shared memory. The core calls fetch
// for tile bk+1 before tile bk's micro-kernel and store after it, so the
// global latency overlaps the FMAs instead of stalling the block at every
// K-step. The image is the same on the float4 path and the scalar fallback;
// `vec` picks the path and is block-uniform, so fetch and store agree on it.
//
// Bounds: positions past the M/N/K edge read as +0.0, which leaves the k-order
// accumulation of an in-range element unchanged (fma(0, 0, acc) == acc for
// every acc the core can hold, since acc never becomes -0.0 from +0.0).
//
// Included only from matmul_f32_tiled.cuh.

#ifndef NUMR_MATMUL_F32_TILED_FETCH_CUH
#define NUMR_MATMUL_F32_TILED_FETCH_CUH

// One thread's share of a tile of ELEMS floats over THREADS threads, in either
// path. Only one array is live in a given launch; the compiler allocates the
// wider live range, not both.
template<int ELEMS, int THREADS>
struct CtTileRegs {
    static constexpr int VEC_SLOTS = (ELEMS / 4 + THREADS - 1) / THREADS;
    static constexpr int SCA_SLOTS = (ELEMS + THREADS - 1) / THREADS;
    float4 v[VEC_SLOTS];
    float s[SCA_SLOTS];
};

// Reads 4 consecutive elements of a row starting at `col`, zero past `limit`.
template<typename T>
__device__ __forceinline__ float4 ct_quad(
    const T* __restrict__ p, unsigned int base, unsigned int col, unsigned int limit
) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (std::is_same_v<T, float>) {
        if (col + 3 < limit) {
            return *reinterpret_cast<const float4*>(&p[base + col]);
        }
    }
    if (col < limit) {
        v.x = AccumTraits<T, float>::load(p, (int)(base + col));
        v.y = (col + 1 < limit) ? AccumTraits<T, float>::load(p, (int)(base + col + 1)) : 0.f;
        v.z = (col + 2 < limit) ? AccumTraits<T, float>::load(p, (int)(base + col + 2)) : 0.f;
        v.w = (col + 3 < limit) ? AccumTraits<T, float>::load(p, (int)(base + col + 3)) : 0.f;
    }
    return v;
}

// A tile [BM x BK] of the [M x K] operand. Float4 along k when `vec`.
template<typename T, int BM, int BK, int THREADS>
__device__ __forceinline__ void ct_fetch_a(
    const T* __restrict__ A, unsigned int block_row, unsigned int k_offset, unsigned int M,
    unsigned int K, unsigned int tid, bool vec, CtTileRegs<BM * BK, THREADS>& r
) {
    using R = CtTileRegs<BM * BK, THREADS>;
    if (vec) {
#pragma unroll
        for (int u = 0; u < R::VEC_SLOTS; ++u) {
            const unsigned int vi = tid + u * THREADS;
            r.v[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (vi < (unsigned int)(BM * BK / 4)) {
                const unsigned int row = (vi * 4) / BK;
                const unsigned int col = (vi * 4) % BK;
                const unsigned int gr = block_row + row;
                if (gr < M) {
                    r.v[u] = ct_quad<T>(A, gr * K, k_offset + col, K);
                }
            }
        }
    } else {
#pragma unroll
        for (int u = 0; u < R::SCA_SLOTS; ++u) {
            const unsigned int idx = tid + u * THREADS;
            r.s[u] = 0.f;
            if (idx < (unsigned int)(BM * BK)) {
                const unsigned int gr = block_row + idx / BK;
                const unsigned int gc = k_offset + idx % BK;
                if (gr < M && gc < K) {
                    r.s[u] = AccumTraits<T, float>::load(A, (int)(gr * K + gc));
                }
            }
        }
    }
}

template<int BM, int BK, int THREADS>
__device__ __forceinline__ void ct_store_a(
    float smem_A[BM][BK], unsigned int tid, bool vec, const CtTileRegs<BM * BK, THREADS>& r
) {
    using R = CtTileRegs<BM * BK, THREADS>;
    if (vec) {
#pragma unroll
        for (int u = 0; u < R::VEC_SLOTS; ++u) {
            const unsigned int vi = tid + u * THREADS;
            if (vi < (unsigned int)(BM * BK / 4)) {
                *reinterpret_cast<float4*>(&smem_A[(vi * 4) / BK][(vi * 4) % BK]) = r.v[u];
            }
        }
    } else {
#pragma unroll
        for (int u = 0; u < R::SCA_SLOTS; ++u) {
            const unsigned int idx = tid + u * THREADS;
            if (idx < (unsigned int)(BM * BK)) {
                smem_A[idx / BK][idx % BK] = r.s[u];
            }
        }
    }
}

// B tile [BK x BN] of the [K x N] operand. Float4 along n when `vec`; the
// float4 smem store needs the unpadded row stride, so `vec` implies BNP == BN.
template<typename T, int BK, int BN, int BNP, int THREADS>
__device__ __forceinline__ void ct_fetch_b(
    const T* __restrict__ B, unsigned int block_col, unsigned int k_offset, unsigned int K,
    unsigned int N, unsigned int tid, bool vec, CtTileRegs<BK * BN, THREADS>& r
) {
    using R = CtTileRegs<BK * BN, THREADS>;
    if (vec) {
#pragma unroll
        for (int u = 0; u < R::VEC_SLOTS; ++u) {
            const unsigned int vi = tid + u * THREADS;
            r.v[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (vi < (unsigned int)(BK * BN / 4)) {
                const unsigned int gr = k_offset + (vi * 4) / BN;
                const unsigned int col = block_col + (vi * 4) % BN;
                if (gr < K) {
                    r.v[u] = ct_quad<T>(B, gr * N, col, N);
                }
            }
        }
    } else {
#pragma unroll
        for (int u = 0; u < R::SCA_SLOTS; ++u) {
            const unsigned int idx = tid + u * THREADS;
            r.s[u] = 0.f;
            if (idx < (unsigned int)(BK * BN)) {
                const unsigned int gr = k_offset + idx / BN;
                const unsigned int gc = block_col + idx % BN;
                if (gr < K && gc < N) {
                    r.s[u] = AccumTraits<T, float>::load(B, (int)(gr * N + gc));
                }
            }
        }
    }
}

template<int BK, int BN, int BNP, int THREADS>
__device__ __forceinline__ void ct_store_b(
    float smem_B[BK][BNP], unsigned int tid, bool vec, const CtTileRegs<BK * BN, THREADS>& r
) {
    using R = CtTileRegs<BK * BN, THREADS>;
    if (vec) {
#pragma unroll
        for (int u = 0; u < R::VEC_SLOTS; ++u) {
            const unsigned int vi = tid + u * THREADS;
            if (vi < (unsigned int)(BK * BN / 4)) {
                *reinterpret_cast<float4*>(&smem_B[(vi * 4) / BN][(vi * 4) % BN]) = r.v[u];
            }
        }
    } else {
#pragma unroll
        for (int u = 0; u < R::SCA_SLOTS; ++u) {
            const unsigned int idx = tid + u * THREADS;
            if (idx < (unsigned int)(BK * BN)) {
                smem_B[idx / BN][idx % BN] = r.s[u];
            }
        }
    }
}

// Transposed-B tile: B is [N x K] row-major and the tile lands in smem as
// [BK][BN] so the micro-kernel is shared with the plain path. Global reads
// run along k, the contiguous axis, float4 when `vec`; the smem scatter is
// transposed, one scalar store per element, into rows padded by four floats
// (the core's BNP) so the stores spread over the banks and the rows stay
// float4-aligned for the micro-kernel.
template<typename T, int BK, int BN, int THREADS>
__device__ __forceinline__ void ct_fetch_bt(
    const T* __restrict__ B, unsigned int block_col, unsigned int k_offset, unsigned int K,
    unsigned int N, unsigned int tid, bool vec, CtTileRegs<BK * BN, THREADS>& r
) {
    using R = CtTileRegs<BK * BN, THREADS>;
    if (vec) {
        constexpr unsigned int quads_per_row = BK / 4;
#pragma unroll
        for (int u = 0; u < R::VEC_SLOTS; ++u) {
            const unsigned int vi = tid + u * THREADS;
            r.v[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (vi < (unsigned int)(BK * BN / 4)) {
                const unsigned int gn = block_col + vi / quads_per_row;
                const unsigned int k = k_offset + (vi % quads_per_row) * 4;
                if (gn < N) {
                    r.v[u] = ct_quad<T>(B, gn * K, k, K);
                }
            }
        }
    } else {
#pragma unroll
        for (int u = 0; u < R::SCA_SLOTS; ++u) {
            const unsigned int idx = tid + u * THREADS;
            r.s[u] = 0.f;
            if (idx < (unsigned int)(BK * BN)) {
                // k fastest, so consecutive threads read consecutive addresses.
                const unsigned int gn = block_col + idx / BK;
                const unsigned int gk = k_offset + idx % BK;
                if (gn < N && gk < K) {
                    r.s[u] = AccumTraits<T, float>::load(B, (int)(gn * K + gk));
                }
            }
        }
    }
}

template<int BK, int BN, int BNP, int THREADS>
__device__ __forceinline__ void ct_store_bt(
    float smem_B[BK][BNP], unsigned int tid, bool vec, const CtTileRegs<BK * BN, THREADS>& r
) {
    using R = CtTileRegs<BK * BN, THREADS>;
    if (vec) {
        constexpr unsigned int quads_per_row = BK / 4;
#pragma unroll
        for (int u = 0; u < R::VEC_SLOTS; ++u) {
            const unsigned int vi = tid + u * THREADS;
            if (vi < (unsigned int)(BK * BN / 4)) {
                const unsigned int n = vi / quads_per_row;
                const unsigned int k = (vi % quads_per_row) * 4;
                smem_B[k][n] = r.v[u].x;
                smem_B[k + 1][n] = r.v[u].y;
                smem_B[k + 2][n] = r.v[u].z;
                smem_B[k + 3][n] = r.v[u].w;
            }
        }
    } else {
#pragma unroll
        for (int u = 0; u < R::SCA_SLOTS; ++u) {
            const unsigned int idx = tid + u * THREADS;
            if (idx < (unsigned int)(BK * BN)) {
                smem_B[idx % BK][idx / BK] = r.s[u];
            }
        }
    }
}

#endif  // NUMR_MATMUL_F32_TILED_FETCH_CUH
