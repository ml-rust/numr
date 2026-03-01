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

// ============================================================================
// Multi-Row Transposed B with Vectorized Loads
//
// Each warp computes ROWS_PER_WARP output columns. Activation vector loaded
// once, reused across rows. Vectorized loads (float4 = 16 bytes per load)
// saturate memory bus — 8x fewer transactions for bf16/f16, 4x for f32.
//
// Runtime alignment check: if K is divisible by VEC elements AND pointers are
// 16-byte aligned, use float4 loads. Otherwise fall back to scalar.
// ============================================================================

#define ROWS_PER_WARP 2

// Helper: check if a pointer is aligned to N bytes
#define IS_ALIGNED(ptr, n) (((unsigned long long)(ptr)) % (n) == 0)

// --- BF16: float4 = 8 bf16 values per load ---

extern "C" __global__ void gemv_bt_mr_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col_base = (blockIdx.x * WARPS_PER_BLOCK + warp_id) * ROWS_PER_WARP;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;

    const __nv_bfloat16* a_row = A + batch * M * K + m * K;

    float acc[ROWS_PER_WARP] = {0.0f, 0.0f};

    // float4 = 16 bytes = 8 bf16. Use vectorized path if K is multiple of 8
    // and both A and B rows are 16-byte aligned.
    const unsigned int VEC = 8;
    const bool can_vec = (K % VEC == 0) && IS_ALIGNED(a_row, 16);

    if (can_vec) {
        const unsigned int K_vec = K / VEC;
        const float4* a_vec = reinterpret_cast<const float4*>(a_row);

        for (unsigned int vi = lane_id; vi < K_vec; vi += WARP_SIZE) {
            float4 av = a_vec[vi];
            const __nv_bfloat16* a8 = reinterpret_cast<const __nv_bfloat16*>(&av);

            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                if (col_base + r < N) {
                    const float4* b_vec = reinterpret_cast<const float4*>(
                        B + batch * N * K + (col_base + r) * K);
                    float4 bv = b_vec[vi];
                    const __nv_bfloat16* b8 = reinterpret_cast<const __nv_bfloat16*>(&bv);

                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        acc[r] += __bfloat162float(a8[j]) * __bfloat162float(b8[j]);
                    }
                }
            }
        }
    } else {
        for (unsigned int k = lane_id; k < K; k += WARP_SIZE) {
            float a_val = __bfloat162float(a_row[k]);
            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                if (col_base + r < N) {
                    acc[r] += a_val * __bfloat162float(
                        B[batch * N * K + (col_base + r) * K + k]);
                }
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            acc[r] += __shfl_down_sync(0xFFFFFFFF, acc[r], off);
        if (lane_id == 0 && col_base + r < N)
            C[batch * M * N + m * N + col_base + r] = __float2bfloat16(acc[r]);
    }
}

// --- F32: float4 = 4 f32 values per load ---

extern "C" __global__ void gemv_bt_mr_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col_base = (blockIdx.x * WARPS_PER_BLOCK + warp_id) * ROWS_PER_WARP;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;

    const float* a_row = A + batch * M * K + m * K;

    float acc[ROWS_PER_WARP] = {0.0f, 0.0f};

    const unsigned int VEC = 4;
    const bool can_vec = (K % VEC == 0) && IS_ALIGNED(a_row, 16);

    if (can_vec) {
        const unsigned int K_vec = K / VEC;
        const float4* a_vec = reinterpret_cast<const float4*>(a_row);

        for (unsigned int vi = lane_id; vi < K_vec; vi += WARP_SIZE) {
            float4 av = a_vec[vi];

            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                if (col_base + r < N) {
                    const float4* b_vec = reinterpret_cast<const float4*>(
                        B + batch * N * K + (col_base + r) * K);
                    float4 bv = b_vec[vi];
                    acc[r] += av.x * bv.x + av.y * bv.y + av.z * bv.z + av.w * bv.w;
                }
            }
        }
    } else {
        for (unsigned int k = lane_id; k < K; k += WARP_SIZE) {
            float a_val = a_row[k];
            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                if (col_base + r < N) {
                    acc[r] += a_val * B[batch * N * K + (col_base + r) * K + k];
                }
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            acc[r] += __shfl_down_sync(0xFFFFFFFF, acc[r], off);
        if (lane_id == 0 && col_base + r < N)
            C[batch * M * N + m * N + col_base + r] = acc[r];
    }
}

// --- F16: float4 = 8 half values per load ---

extern "C" __global__ void gemv_bt_mr_f16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col_base = (blockIdx.x * WARPS_PER_BLOCK + warp_id) * ROWS_PER_WARP;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;

    const half* a_row = A + batch * M * K + m * K;

    float acc[ROWS_PER_WARP] = {0.0f, 0.0f};

    const unsigned int VEC = 8;
    const bool can_vec = (K % VEC == 0) && IS_ALIGNED(a_row, 16);

    if (can_vec) {
        const unsigned int K_vec = K / VEC;
        const float4* a_vec = reinterpret_cast<const float4*>(a_row);

        for (unsigned int vi = lane_id; vi < K_vec; vi += WARP_SIZE) {
            float4 av = a_vec[vi];
            const half* a8 = reinterpret_cast<const half*>(&av);

            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                if (col_base + r < N) {
                    const float4* b_vec = reinterpret_cast<const float4*>(
                        B + batch * N * K + (col_base + r) * K);
                    float4 bv = b_vec[vi];
                    const half* b8 = reinterpret_cast<const half*>(&bv);

                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        acc[r] += __half2float(a8[j]) * __half2float(b8[j]);
                    }
                }
            }
        }
    } else {
        for (unsigned int k = lane_id; k < K; k += WARP_SIZE) {
            float a_val = __half2float(a_row[k]);
            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                if (col_base + r < N) {
                    acc[r] += a_val * __half2float(
                        B[batch * N * K + (col_base + r) * K + k]);
                }
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            acc[r] += __shfl_down_sync(0xFFFFFFFF, acc[r], off);
        if (lane_id == 0 && col_base + r < N)
            C[batch * M * N + m * N + col_base + r] = __float2half(acc[r]);
    }
}

// --- F64: double2 = 2 f64 values per load ---

extern "C" __global__ void gemv_bt_mr_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col_base = (blockIdx.x * WARPS_PER_BLOCK + warp_id) * ROWS_PER_WARP;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;

    const double* a_row = A + batch * M * K + m * K;

    double acc[ROWS_PER_WARP] = {0.0, 0.0};

    const unsigned int VEC = 2;
    const bool can_vec = (K % VEC == 0) && IS_ALIGNED(a_row, 16);

    if (can_vec) {
        const unsigned int K_vec = K / VEC;
        const double2* a_vec = reinterpret_cast<const double2*>(a_row);

        for (unsigned int vi = lane_id; vi < K_vec; vi += WARP_SIZE) {
            double2 av = a_vec[vi];

            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                if (col_base + r < N) {
                    const double2* b_vec = reinterpret_cast<const double2*>(
                        B + batch * N * K + (col_base + r) * K);
                    double2 bv = b_vec[vi];
                    acc[r] += av.x * bv.x + av.y * bv.y;
                }
            }
        }
    } else {
        for (unsigned int k = lane_id; k < K; k += WARP_SIZE) {
            double a_val = a_row[k];
            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                if (col_base + r < N) {
                    acc[r] += a_val * B[batch * N * K + (col_base + r) * K + k];
                }
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            acc[r] += __shfl_down_sync(0xFFFFFFFF, acc[r], off);
        if (lane_id == 0 && col_base + r < N)
            C[batch * M * N + m * N + col_base + r] = acc[r];
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
