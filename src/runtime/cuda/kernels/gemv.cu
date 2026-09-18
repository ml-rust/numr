// GEMV (General Matrix-Vector Multiply) CUDA Kernels
// C[M,N] = A[M,K] @ B^T for small M (M <= 16), B stored [N,K] row-major (the
// transposed weight nn.Linear multiplies by).
//
// gemv_bt_mr_* : warp-cooperative K-reduction, two output columns per warp.
//    - Lanes read B[col*K + 4*lane ..] as float4 (stride-1 across the warp)
//    - Each lane sums its k-strided share, then a shuffle tree joins the lanes
//    - Grid: (ceil(N/16), M, batch), block: (256, 1, 1)
//
// Reached only by F16/BF16 (`ops/helpers/matmul.rs`): the lane-strided sum
// and the shuffle tree give a different float sequence from the tiled GEMM,
// so a row's result here differs from the same row past the M cutoff. F32
// takes its tiled transposed-B kernel at every M instead. The bt layout
// avoids a contiguous copy of the weight by reading the raw [N,K] pointer.

#include <cuda_fp16.h>
#include <cuda_bf16.h>


// B layout: [N, K] row-major (weight matrix) — B_logical[k,n] = B_data[n*K + k]

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)

// ============================================================================
// Multi-Row Transposed B with Vectorized Loads
//
// Each warp computes ROWS_PER_WARP output columns. Activation vector loaded
// once, reused across rows. Vectorized loads (float4 = 16 bytes per load)
// saturate memory bus — 8x fewer transactions than scalar half loads.
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
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col_base = (blockIdx.x * WARPS_PER_BLOCK + warp_id) * ROWS_PER_WARP;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int a_batch = batch % a_batch_count;
    const unsigned int b_batch = batch % b_batch_count;

    const __nv_bfloat16* a_row = A + a_batch * M * K + m * K;

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
                        B + b_batch * N * K + (col_base + r) * K);
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
                        B[b_batch * N * K + (col_base + r) * K + k]);
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

// --- F16: float4 = 8 half values per load ---

extern "C" __global__ void gemv_bt_mr_f16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int col_base = (blockIdx.x * WARPS_PER_BLOCK + warp_id) * ROWS_PER_WARP;
    const unsigned int m = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int a_batch = batch % a_batch_count;
    const unsigned int b_batch = batch % b_batch_count;

    const half* a_row = A + a_batch * M * K + m * K;

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
                        B + b_batch * N * K + (col_base + r) * K);
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
                        B[b_batch * N * K + (col_base + r) * K + k]);
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
