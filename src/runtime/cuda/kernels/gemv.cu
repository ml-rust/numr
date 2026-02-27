// GEMV (General Matrix-Vector Multiply) CUDA Kernels
// Optimized for C[M,N] = A[M,K] @ B[K,N] when M is small (M <= 16)
//
// For LLM inference decode: M=1, K=2048-8192, N=2048-8192
//
// Strategy: Each thread computes one output element C[m, col].
// - A vector is broadcast (all threads in a warp read same a[k], hits L1 cache)
// - B reads are coalesced: consecutive threads read consecutive columns
// - K-loop is unrolled 4x for instruction-level parallelism
//
// Launch config: grid=(ceil(N/COLS_PER_BLOCK), M, batch), block=(256, 1, 1)

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define COLS_PER_BLOCK 256

// ============================================================================
// GEMV kernel for BF16 (compute in F32, store BF16)
// This is the primary kernel for LLM inference (models stored in BF16)
// ============================================================================

extern "C" __global__ void gemv_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int m = blockIdx.y;
    const unsigned int col = blockIdx.x * COLS_PER_BLOCK + threadIdx.x;
    const unsigned int batch = blockIdx.z;

    if (col >= N) return;

    const __nv_bfloat16* a_row = A + batch * M * K + m * K;
    const __nv_bfloat16* b_base = B + batch * K * N + col;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // Unroll 4x for ILP
    unsigned int k = 0;
    const unsigned int K4 = K & ~3u;
    for (; k < K4; k += 4) {
        float a0 = __bfloat162float(a_row[k]);
        float a1 = __bfloat162float(a_row[k + 1]);
        float a2 = __bfloat162float(a_row[k + 2]);
        float a3 = __bfloat162float(a_row[k + 3]);
        acc0 += a0 * __bfloat162float(b_base[k * N]);
        acc1 += a1 * __bfloat162float(b_base[(k + 1) * N]);
        acc2 += a2 * __bfloat162float(b_base[(k + 2) * N]);
        acc3 += a3 * __bfloat162float(b_base[(k + 3) * N]);
    }
    // Handle remainder
    for (; k < K; k++) {
        acc0 += __bfloat162float(a_row[k]) * __bfloat162float(b_base[k * N]);
    }

    C[batch * M * N + m * N + col] = __float2bfloat16(acc0 + acc1 + acc2 + acc3);
}

// ============================================================================
// GEMV kernel for F32
// ============================================================================

extern "C" __global__ void gemv_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int m = blockIdx.y;
    const unsigned int col = blockIdx.x * COLS_PER_BLOCK + threadIdx.x;
    const unsigned int batch = blockIdx.z;

    if (col >= N) return;

    const float* a_row = A + batch * M * K + m * K;
    const float* b_base = B + batch * K * N + col;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    unsigned int k = 0;
    const unsigned int K4 = K & ~3u;
    for (; k < K4; k += 4) {
        acc0 += a_row[k]     * b_base[k * N];
        acc1 += a_row[k + 1] * b_base[(k + 1) * N];
        acc2 += a_row[k + 2] * b_base[(k + 2) * N];
        acc3 += a_row[k + 3] * b_base[(k + 3) * N];
    }
    for (; k < K; k++) {
        acc0 += a_row[k] * b_base[k * N];
    }

    C[batch * M * N + m * N + col] = acc0 + acc1 + acc2 + acc3;
}

// ============================================================================
// GEMV kernel for F16 (compute in F32, store F16)
// ============================================================================

extern "C" __global__ void gemv_f16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int m = blockIdx.y;
    const unsigned int col = blockIdx.x * COLS_PER_BLOCK + threadIdx.x;
    const unsigned int batch = blockIdx.z;

    if (col >= N) return;

    const half* a_row = A + batch * M * K + m * K;
    const half* b_base = B + batch * K * N + col;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    unsigned int k = 0;
    const unsigned int K4 = K & ~3u;
    for (; k < K4; k += 4) {
        acc0 += __half2float(a_row[k])     * __half2float(b_base[k * N]);
        acc1 += __half2float(a_row[k + 1]) * __half2float(b_base[(k + 1) * N]);
        acc2 += __half2float(a_row[k + 2]) * __half2float(b_base[(k + 2) * N]);
        acc3 += __half2float(a_row[k + 3]) * __half2float(b_base[(k + 3) * N]);
    }
    for (; k < K; k++) {
        acc0 += __half2float(a_row[k]) * __half2float(b_base[k * N]);
    }

    C[batch * M * N + m * N + col] = __float2half(acc0 + acc1 + acc2 + acc3);
}

// ============================================================================
// GEMV kernel for F64
// ============================================================================

extern "C" __global__ void gemv_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int m = blockIdx.y;
    const unsigned int col = blockIdx.x * COLS_PER_BLOCK + threadIdx.x;
    const unsigned int batch = blockIdx.z;

    if (col >= N) return;

    const double* a_row = A + batch * M * K + m * K;
    const double* b_base = B + batch * K * N + col;

    double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;

    unsigned int k = 0;
    const unsigned int K4 = K & ~3u;
    for (; k < K4; k += 4) {
        acc0 += a_row[k]     * b_base[k * N];
        acc1 += a_row[k + 1] * b_base[(k + 1) * N];
        acc2 += a_row[k + 2] * b_base[(k + 2) * N];
        acc3 += a_row[k + 3] * b_base[(k + 3) * N];
    }
    for (; k < K; k++) {
        acc0 += a_row[k] * b_base[k * N];
    }

    C[batch * M * N + m * N + col] = acc0 + acc1 + acc2 + acc3;
}
