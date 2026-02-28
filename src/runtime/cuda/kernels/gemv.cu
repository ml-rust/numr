// GEMV (General Matrix-Vector Multiply) CUDA Kernels
// C[M,N] = A[M,K] @ B[K,N] for small M (M <= 16, typically M=1 for LLM decode)
//
// Two kernel families:
//
// 1. gemv_* : B is [K,N] row-major (non-transposed)
//    - One thread per output column, iterates K
//    - Coalesced B reads: consecutive threads read B[k*N + col], B[k*N + col+1]
//    - Grid: (ceil(N/256), M, batch), block: (256, 1, 1)
//
// 2. gemv_bt_* : B is [N,K] row-major (transposed weight, the common case for nn.Linear)
//    - Warp-cooperative: each warp reduces one output column along K
//    - Coalesced B reads: lanes read B[col*K + lane], B[col*K + lane+1] (stride-1)
//    - Grid: (ceil(N/WARPS_PER_BLOCK), M, batch), block: (256, 1, 1)
//
// The bt (B-transposed) variant avoids a 500MB contiguous copy when Linear
// computes y = x @ W^T by passing the raw [N,K] weight pointer directly.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ============================================================================
// Non-transposed B: one thread per output, iterate K
// B layout: [K, N] row-major — B[k,n] = B_data[k*N + n]
// ============================================================================

extern "C" __global__ void gemv_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    if (col >= N) return;

    const __nv_bfloat16* a_row = A + batch * M * K + m * K;
    const __nv_bfloat16* b_base = B + batch * K * N;

    float acc = 0.0f;
    for (unsigned int k = 0; k < K; k++) {
        acc += __bfloat162float(a_row[k]) * __bfloat162float(b_base[k * N + col]);
    }

    C[batch * M * N + m * N + col] = __float2bfloat16(acc);
}

extern "C" __global__ void gemv_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    if (col >= N) return;

    const float* a_row = A + batch * M * K + m * K;
    const float* b_base = B + batch * K * N;

    float acc = 0.0f;
    for (unsigned int k = 0; k < K; k++) {
        acc += a_row[k] * b_base[k * N + col];
    }

    C[batch * M * N + m * N + col] = acc;
}

extern "C" __global__ void gemv_f16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    if (col >= N) return;

    const half* a_row = A + batch * M * K + m * K;
    const half* b_base = B + batch * K * N;

    float acc = 0.0f;
    for (unsigned int k = 0; k < K; k++) {
        acc += __half2float(a_row[k]) * __half2float(b_base[k * N + col]);
    }

    C[batch * M * N + m * N + col] = __float2half(acc);
}

extern "C" __global__ void gemv_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    if (col >= N) return;

    const double* a_row = A + batch * M * K + m * K;
    const double* b_base = B + batch * K * N;

    double acc = 0.0;
    for (unsigned int k = 0; k < K; k++) {
        acc += a_row[k] * b_base[k * N + col];
    }

    C[batch * M * N + m * N + col] = acc;
}

// ============================================================================
// Transposed B: warp-cooperative K-reduction
// B layout: [N, K] row-major (weight matrix) — B_logical[k,n] = B_data[n*K + k]
//
// Each warp handles one output column. Lanes cooperatively reduce along K.
// B_data[col*K + lane_id] reads are stride-1 (coalesced within each warp).
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)

extern "C" __global__ void gemv_bt_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,   // stored [N, K] row-major
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    if (col >= N) return;

    const __nv_bfloat16* a_row = A + batch * M * K + m * K;
    const __nv_bfloat16* b_row = B + batch * N * K + col * K;  // B[col, 0..K]

    float acc = 0.0f;
    for (unsigned int k = lane_id; k < K; k += WARP_SIZE) {
        acc += __bfloat162float(a_row[k]) * __bfloat162float(b_row[k]);
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        C[batch * M * N + m * N + col] = __float2bfloat16(acc);
    }
}

extern "C" __global__ void gemv_bt_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    if (col >= N) return;

    const float* a_row = A + batch * M * K + m * K;
    const float* b_row = B + batch * N * K + col * K;

    float acc = 0.0f;
    for (unsigned int k = lane_id; k < K; k += WARP_SIZE) {
        acc += a_row[k] * b_row[k];
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        C[batch * M * N + m * N + col] = acc;
    }
}

extern "C" __global__ void gemv_bt_f16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    if (col >= N) return;

    const half* a_row = A + batch * M * K + m * K;
    const half* b_row = B + batch * N * K + col * K;

    float acc = 0.0f;
    for (unsigned int k = lane_id; k < K; k += WARP_SIZE) {
        acc += __half2float(a_row[k]) * __half2float(b_row[k]);
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        C[batch * M * N + m * N + col] = __float2half(acc);
    }
}

extern "C" __global__ void gemv_bt_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    if (col >= N) return;

    const double* a_row = A + batch * M * K + m * K;
    const double* b_row = B + batch * N * K + col * K;

    double acc = 0.0;
    for (unsigned int k = lane_id; k < K; k += WARP_SIZE) {
        acc += a_row[k] * b_row[k];
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane_id == 0) {
        C[batch * M * N + m * N + col] = acc;
    }
}
