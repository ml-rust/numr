// Small-M FP32 GEMM against a transposed weight: C[M,N] = A[M,K] · Bᵀ with
// B stored [N,K] row-major, one thread per output element, each block
// staging SMALLM_ROWS rows of B through shared memory with all its threads.
//
// WHY: the tiled family serves M <= 64 with its 16-row tile
// (`matmul_f32_tiled_bt_16x64x32_4x4`). At the decode shapes, M in 1..=4,
// 12 to 15 of those 16 rows are padding, so most of the FMAs, the shared
// memory traffic and the barriers do nothing. `[1,5120] x [5120,48]ᵀ` takes
// ~0.24 ms there. This kernel launches only the threads that own an output,
// and uses the threads that own none to keep the weight stream in flight.
//
// BIT IDENTITY: the tiled kernel forms each output as ONE FMA PER k, k
// ascending (`accum[i][j] += reg_a[i] * reg_b[j]`, contracted to `fma.rn`
// by nvcc `-fmad=true`, which `--use_fast_math` implies); the k's past K in
// its last tile are `fma(0, 0, acc)`, which is `acc` for every finite,
// infinite and NaN `acc` (the accumulator starts at +0 and round-to-nearest
// never produces -0 from a sum, so the sign is safe too). This kernel forms
// each output as `acc = fmaf(a[k], b[k], acc)` for k ascending, explicit
// `fmaf` so the contraction does not depend on build flags, with ONE
// accumulator chain: chunks of k are consumed in ascending order and the
// slots past K in the last chunk hold 0 in both operands, so they are the
// same `fma(0, 0, acc)` the tiled kernel runs; never a second partial sum.
// The bits are therefore identical to the tiled kernel's at every M, which
// is what `tests/cuda_matmul_batch_invariance.rs` checks through the public
// op and `tests/cuda_matmul_smallm_bt_parity.rs` checks kernel against
// kernel.
//
// Grid: (ceil(N/SMALLM_ROWS), M, batch)   Block: (256, 1, 1)
// Shared memory: SMALLM_SMEM_FLOATS x 4 bytes, static.
//
// SCOPE: N <= MAX_SMALL_N (loader/matmul_f32_smallm.rs). The kernel reads
// each B element once per A row; past a few hundred rows the tiled kernel,
// which reuses each staged B element across its 16 A rows, is faster.
//
// Thread t of block (bx, m, z) owns C[m][bx*SMALLM_ROWS + t] for t < rows,
// where rows = min(SMALLM_ROWS, N - bx*SMALLM_ROWS). Every thread of the
// block, owner or not, loads its share of each chunk.
//
// This header is included by matmul.cu only, so the `extern "C"` entry
// points live here as well: a CUDA module exposes only the kernels compiled
// into its translation unit, and matmul.cu is the `matmul` module.

#ifndef NUMR_MATMUL_F32_SMALLM_CUH
#define NUMR_MATMUL_F32_SMALLM_CUH

#include <stdint.h>

// LOAD SCHEDULING: the chain is one dependent FMA per k, so the kernel's
// floor is about K x 4 cycles of FMA latency per owning thread, whatever
// the loads do. The loads must not add to that floor. One block per 256
// rows puts the whole weight stream through one SM's load pipeline, so the
// block is SMALLM_ROWS rows wide instead: N = 48 runs six blocks and
// N = 256 thirty-two, each on its own SM, each streaming its own
// SMALLM_ROWS x K floats of B. A is re-read by every block; it is M x K
// floats and stays in L2.
//
// Within a block the stream is chunked: SMALLM_ROWS rows x SMALLM_KC
// floats, staged in a double-buffered shared tile. All 256 threads load a
// chunk, thread t taking the 16-byte pieces t, t+256, ... of the row-major
// tile, so a warp's loads cover 512 contiguous bytes of one row and
// coalesce fully. Chunk c+1's SMALLM_PIECES float4 per thread are issued
// into registers before chunk c's FMAs run from one shared buffer, then
// stored into the other buffer, with one `__syncthreads()` per chunk. The
// FMA phase of a chunk (SMALLM_KC x ~4 cycles) is longer than a load round
// trip, so one chunk of look-ahead keeps the loads off the chain's floor;
// a deeper register ring would only cost registers. Plain loads, no
// `cp.async`, so the kernel has no arch gate.
//
// Shared memory: 2 buffers x (A chunk SMALLM_KC + B tile
// SMALLM_ROWS x (SMALLM_KC + 4)) floats = 2 x (512 + 4128) = 9280 floats
// = 37,120 bytes, under the 48 KB static limit on every arch. The grid is
// at most 32 x M blocks of 256 threads, so one block per SM is all the
// occupancy the launch can use.
//
// BANKS: the B tile's row stride is SMALLM_KC + 4 floats, so a row start
// stays 16-byte aligned and stride/4 = 129 is odd. A 128-bit shared load is
// served in four phases of eight consecutive lanes; lane t of the FMA phase
// reads the 16-byte word t * 129 + q, and with 129 odd the eight words of a
// phase land in eight distinct 4-bank groups, so the reads are conflict
// free. The A chunk is one row every lane reads at the same address, which
// broadcasts. Stores follow the load mapping, so eight consecutive lanes
// store eight consecutive words of one row.

constexpr int SMALLM_THREADS = 256;
// Rows of B per block; the loader's `SMALLM_ROWS_PER_BLOCK`.
constexpr int SMALLM_ROWS = 8;
// Floats of k per chunk.
constexpr int SMALLM_KC = 512;
// Float4 pieces per row per chunk.
constexpr int SMALLM_Q = SMALLM_KC / 4;
// B tile row stride in floats.
constexpr int SMALLM_STRIDE = SMALLM_KC + 4;
// Float4 pieces of the B tile each thread loads per chunk.
constexpr int SMALLM_PIECES = SMALLM_ROWS * SMALLM_Q / SMALLM_THREADS;
static_assert(SMALLM_PIECES * SMALLM_THREADS == SMALLM_ROWS * SMALLM_Q,
              "the B chunk must split evenly over the block");
static_assert(SMALLM_Q <= SMALLM_THREADS, "the A chunk must fit one piece per thread");
// One buffer: the A chunk, then the B tile.
constexpr int SMALLM_A_FLOATS = SMALLM_KC;
constexpr int SMALLM_B_FLOATS = SMALLM_ROWS * SMALLM_STRIDE;
constexpr int SMALLM_BUF_FLOATS = SMALLM_A_FLOATS + SMALLM_B_FLOATS;
constexpr int SMALLM_SMEM_FLOATS = 2 * SMALLM_BUF_FLOATS;

// Four consecutive FMAs on the one chain from a quad of each operand.
__device__ __forceinline__ float smallm_fma4(float acc, float4 a, float4 b) {
    acc = fmaf(a.x, b.x, acc);
    acc = fmaf(a.y, b.y, acc);
    acc = fmaf(a.z, b.z, acc);
    acc = fmaf(a.w, b.w, acc);
    return acc;
}

// `src[k0..k0+4)`, zero past K. On the `vec` path K is a multiple of four
// and `k0` is too, so a piece that starts inside the row ends inside it and
// one aligned float4 load serves; otherwise four scalar loads, each masked.
__device__ __forceinline__ float4 smallm_piece(
    const float* __restrict__ src,
    unsigned int k0,
    unsigned int K,
    bool vec
) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (k0 >= K) return v;
    if (vec) return *reinterpret_cast<const float4*>(src + k0);
    v.x = src[k0];
    if (k0 + 1 < K) v.y = src[k0 + 1];
    if (k0 + 2 < K) v.z = src[k0 + 2];
    if (k0 + 3 < K) v.w = src[k0 + 3];
    return v;
}

// One chunk of both operands held in registers between issue and stage:
// the A piece of thread t < SMALLM_Q and the B pieces t + 256 j.
struct SmallmChunk {
    float4 a;
    float4 b[SMALLM_PIECES];
};

// Issue the global loads of chunk `k0..k0+SMALLM_KC` into `r`. `rows` and
// `K` are block-uniform; a piece past the tile is neither loaded nor stored.
__device__ __forceinline__ void smallm_issue(
    SmallmChunk& r,
    const float* __restrict__ a_row,
    const float* __restrict__ b_tile,
    unsigned int rows,
    unsigned int K,
    unsigned int k0,
    bool vec
) {
    const unsigned int t = threadIdx.x;
    const unsigned int pieces = rows * SMALLM_Q;
    r.a = (t < SMALLM_Q) ? smallm_piece(a_row, k0 + 4 * t, K, vec)
                         : make_float4(0.f, 0.f, 0.f, 0.f);
    #pragma unroll
    for (int j = 0; j < SMALLM_PIECES; j++) {
        const unsigned int p = t + j * SMALLM_THREADS;
        if (p < pieces) {
            const unsigned int row = p / SMALLM_Q;
            const unsigned int q = p - row * SMALLM_Q;
            r.b[j] = smallm_piece(b_tile + (size_t)row * K, k0 + 4 * q, K, vec);
        }
    }
}

// Store a chunk from registers into the shared buffer at `buf`, same piece
// mapping as `smallm_issue`.
__device__ __forceinline__ void smallm_stage(
    const SmallmChunk& r,
    float* __restrict__ buf,
    unsigned int rows
) {
    const unsigned int t = threadIdx.x;
    const unsigned int pieces = rows * SMALLM_Q;
    float* Bs = buf + SMALLM_A_FLOATS;
    if (t < SMALLM_Q) {
        *reinterpret_cast<float4*>(buf + 4 * t) = r.a;
    }
    #pragma unroll
    for (int j = 0; j < SMALLM_PIECES; j++) {
        const unsigned int p = t + j * SMALLM_THREADS;
        if (p < pieces) {
            const unsigned int row = p / SMALLM_Q;
            const unsigned int q = p - row * SMALLM_Q;
            *reinterpret_cast<float4*>(Bs + row * SMALLM_STRIDE + 4 * q) = r.b[j];
        }
    }
}

// SMALLM_KC FMAs of the staged chunk at `buf` on the one chain, k
// ascending, for the row thread `t` owns. Called by owning threads only.
__device__ __forceinline__ float smallm_consume(
    float acc,
    const float* __restrict__ buf,
    unsigned int t
) {
    const float* b = buf + SMALLM_A_FLOATS + t * SMALLM_STRIDE;
    #pragma unroll 16
    for (int q = 0; q < SMALLM_Q; q++) {
        const float4 a4 = *reinterpret_cast<const float4*>(buf + 4 * q);
        const float4 b4 = *reinterpret_cast<const float4*>(b + 4 * q);
        acc = smallm_fma4(acc, a4, b4);
    }
    return acc;
}

// The dot product of `a_row[0..K)` with each of `rows` rows of `b_tile`,
// thread t returning row t's. Every thread of the block calls this: the
// loads, stores and barriers are block-wide, and only the FMA phase is
// gated on ownership. Chunk c is consumed from buffer c & 1 while chunk
// c+1 is in flight in registers; it is then stored into the other buffer,
// which every thread finished reading before the barrier that ended chunk
// c-1, so one barrier per chunk orders both hazards.
__device__ __forceinline__ float smallm_tile(
    const float* __restrict__ a_row,
    const float* __restrict__ b_tile,
    unsigned int rows,
    unsigned int K,
    bool vec,
    float* __restrict__ smem
) {
    const unsigned int t = threadIdx.x;
    const bool owner = t < rows;
    const unsigned int chunks = (K + SMALLM_KC - 1) / SMALLM_KC;
    float acc = 0.f;
    if (chunks == 0) return acc;

    SmallmChunk r;
    smallm_issue(r, a_row, b_tile, rows, K, 0, vec);
    smallm_stage(r, smem, rows);
    __syncthreads();

    for (unsigned int c = 0; c < chunks; c++) {
        float* cur = smem + (c & 1u) * SMALLM_BUF_FLOATS;
        float* nxt = smem + ((c + 1u) & 1u) * SMALLM_BUF_FLOATS;
        const bool more = c + 1 < chunks;
        if (more) smallm_issue(r, a_row, b_tile, rows, K, (c + 1) * SMALLM_KC, vec);
        if (owner) acc = smallm_consume(acc, cur, t);
        if (!more) break;
        smallm_stage(r, nxt, rows);
        __syncthreads();
    }
    return acc;
}

// One [M,K] x [N,K]ᵀ product for the block's (m, n-tile) pair. Batch
// offsets are applied by the caller. No early return before the tile: the
// barriers inside need every thread of the block.
__device__ __forceinline__ void matmul_f32_smallm_bt_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    __shared__ __align__(16) float smem[SMALLM_SMEM_FLOATS];

    const unsigned int t = threadIdx.x;
    const unsigned int m = blockIdx.y;
    const unsigned int n0 = blockIdx.x * SMALLM_ROWS;
    if (m >= M || n0 >= N) return;
    const unsigned int rows = min((unsigned int)SMALLM_ROWS, N - n0);

    const float* a_row = A + (size_t)m * K;
    const float* b_tile = B + (size_t)n0 * K;

    // Block-uniform: A's base and K are the same for every thread, and B's
    // row start `B + n*K` is 16-byte aligned for every n once B's base is
    // and K is a multiple of four.
    const bool vec = (K & 3u) == 0u
        && (reinterpret_cast<uintptr_t>(A) & 15u) == 0u
        && (reinterpret_cast<uintptr_t>(B) & 15u) == 0u;

    const float acc = smallm_tile(a_row, b_tile, rows, K, vec, smem);

    if (t < rows) {
        C[(size_t)m * N + n0 + t] = acc;
    }
}

// C[M,N] = A[M,K] · B[N,K]ᵀ.
extern "C" __global__ void matmul_f32_smallm_bt(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    matmul_f32_smallm_bt_impl(A, B, C, M, N, K);
}

// C[batch,M,N] = A[batch,M,K] · B[batch,N,K]ᵀ. Batch index is blockIdx.z;
// `a_batch_count` / `b_batch_count` let one operand broadcast over the
// batch, the same `b % count` convention as `matmul_batched_f32_tiled_bt_*`.
extern "C" __global__ void matmul_batched_f32_smallm_bt(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const float* A_batch = A + (size_t)(b % a_batch_count) * ((size_t)M * K);
    const float* B_batch = B + (size_t)(b % b_batch_count) * ((size_t)K * N);
    float* C_batch = C + (size_t)b * ((size_t)M * N);

    matmul_f32_smallm_bt_impl(A_batch, B_batch, C_batch, M, N, K);
}

#endif // NUMR_MATMUL_F32_SMALLM_CUH
