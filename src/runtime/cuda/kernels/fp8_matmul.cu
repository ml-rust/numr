// FP8 Matrix Multiplication CUDA Kernels
//
// Computes: C = scale_a * scale_b * (A_fp8 @ B_fp8)
// where A,B are FP8 tensors, accumulation is in FP32, output is F32/F16/BF16.
//
// Variants:
//   - E4M3 x E4M3 -> F32/F16/BF16 (forward pass)
//   - E5M2 x E4M3 -> F32/F16/BF16 (backward pass: gradients x weights)
//   - Batched versions of both
//
// Algorithm: tiled GEMM with shared memory (F32 accumulation), FP8 loads via conversion.

#include "dtype_traits.cuh"

// Tile sizes for FP8 GEMM
// FP8 elements are 1 byte, so we can fit more in shared memory
#define FP8_TILE_M 64
#define FP8_TILE_N 64
#define FP8_TILE_K 32
#define FP8_THREAD_M 4
#define FP8_THREAD_N 4

// ============================================================================
// Helper: store result with dtype conversion and scaling
// ============================================================================

__device__ __forceinline__ void store_f32(float* out, unsigned int idx, float val) {
    out[idx] = val;
}

__device__ __forceinline__ void store_f16(__half* out, unsigned int idx, float val) {
    out[idx] = __float2half(val);
}

__device__ __forceinline__ void store_bf16(__nv_bfloat16* out, unsigned int idx, float val) {
    out[idx] = __float2bfloat16(val);
}

// ============================================================================
// FP8 E4M3 x E4M3 -> output dtype (tiled GEMM with F32 accumulation)
// ============================================================================

template<typename OutT, void (*store_fn)(OutT*, unsigned int, float)>
__device__ void fp8_matmul_e4m3_kernel(
    const numr_fp8_e4m3* __restrict__ A,
    const numr_fp8_e4m3* __restrict__ B,
    OutT* __restrict__ C,
    float scale_a,
    float scale_b,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    // Shared memory for tiles (store as f32 after conversion)
    __shared__ float As[FP8_TILE_M][FP8_TILE_K];
    __shared__ float Bs[FP8_TILE_K][FP8_TILE_N];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = FP8_TILE_N / FP8_THREAD_N;
    const unsigned int threads_y = FP8_TILE_M / FP8_THREAD_M;

    const unsigned int block_row = blockIdx.y * FP8_TILE_M;
    const unsigned int block_col = blockIdx.x * FP8_TILE_N;
    const unsigned int thread_row = ty * FP8_THREAD_M;
    const unsigned int thread_col = tx * FP8_THREAD_N;

    // Register accumulators (F32)
    float reg_c[FP8_THREAD_M][FP8_THREAD_N];
    #pragma unroll
    for (int i = 0; i < FP8_THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < FP8_THREAD_N; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = threads_x * threads_y;
    const unsigned int num_k_tiles = (K + FP8_TILE_K - 1) / FP8_TILE_K;
    const float combined_scale = scale_a * scale_b;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * FP8_TILE_K;

        // Cooperative load A tile, convert FP8 -> F32
        unsigned int a_elems = FP8_TILE_M * FP8_TILE_K;
        for (unsigned int idx = thread_id; idx < a_elems; idx += num_threads) {
            unsigned int r = idx / FP8_TILE_K;
            unsigned int c = idx % FP8_TILE_K;
            unsigned int gr = block_row + r;
            unsigned int gc = k_offset + c;
            if (gr < M && gc < K) {
                As[r][c] = fp8_e4m3_to_f32(A[gr * K + gc].data);
            } else {
                As[r][c] = 0.0f;
            }
        }

        // Cooperative load B tile, convert FP8 -> F32
        unsigned int b_elems = FP8_TILE_K * FP8_TILE_N;
        for (unsigned int idx = thread_id; idx < b_elems; idx += num_threads) {
            unsigned int r = idx / FP8_TILE_N;
            unsigned int c = idx % FP8_TILE_N;
            unsigned int gr = k_offset + r;
            unsigned int gc = block_col + c;
            if (gr < K && gc < N) {
                Bs[r][c] = fp8_e4m3_to_f32(B[gr * N + gc].data);
            } else {
                Bs[r][c] = 0.0f;
            }
        }

        __syncthreads();

        // Compute partial products
        #pragma unroll
        for (unsigned int kk = 0; kk < FP8_TILE_K; kk++) {
            float reg_a[FP8_THREAD_M];
            float reg_b[FP8_THREAD_N];

            #pragma unroll
            for (int i = 0; i < FP8_THREAD_M; i++) {
                reg_a[i] = As[thread_row + i][kk];
            }
            #pragma unroll
            for (int j = 0; j < FP8_THREAD_N; j++) {
                reg_b[j] = Bs[kk][thread_col + j];
            }

            #pragma unroll
            for (int i = 0; i < FP8_THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < FP8_THREAD_N; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // Write output with scaling and dtype conversion
    #pragma unroll
    for (int i = 0; i < FP8_THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < FP8_THREAD_N; j++) {
            unsigned int gr = block_row + thread_row + i;
            unsigned int gc = block_col + thread_col + j;
            if (gr < M && gc < N) {
                store_fn(C, gr * N + gc, reg_c[i][j] * combined_scale);
            }
        }
    }
}

// ============================================================================
// FP8 E5M2 x E4M3 -> output dtype (backward pass)
// ============================================================================

template<typename OutT, void (*store_fn)(OutT*, unsigned int, float)>
__device__ void fp8_matmul_e5m2_kernel(
    const numr_fp8_e5m2* __restrict__ A,
    const numr_fp8_e4m3* __restrict__ B,
    OutT* __restrict__ C,
    float scale_a,
    float scale_b,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    __shared__ float As[FP8_TILE_M][FP8_TILE_K];
    __shared__ float Bs[FP8_TILE_K][FP8_TILE_N];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = FP8_TILE_N / FP8_THREAD_N;
    const unsigned int threads_y = FP8_TILE_M / FP8_THREAD_M;

    const unsigned int block_row = blockIdx.y * FP8_TILE_M;
    const unsigned int block_col = blockIdx.x * FP8_TILE_N;
    const unsigned int thread_row = ty * FP8_THREAD_M;
    const unsigned int thread_col = tx * FP8_THREAD_N;

    float reg_c[FP8_THREAD_M][FP8_THREAD_N];
    #pragma unroll
    for (int i = 0; i < FP8_THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < FP8_THREAD_N; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = threads_x * threads_y;
    const unsigned int num_k_tiles = (K + FP8_TILE_K - 1) / FP8_TILE_K;
    const float combined_scale = scale_a * scale_b;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * FP8_TILE_K;

        // Load A (E5M2) -> F32
        unsigned int a_elems = FP8_TILE_M * FP8_TILE_K;
        for (unsigned int idx = thread_id; idx < a_elems; idx += num_threads) {
            unsigned int r = idx / FP8_TILE_K;
            unsigned int c = idx % FP8_TILE_K;
            unsigned int gr = block_row + r;
            unsigned int gc = k_offset + c;
            if (gr < M && gc < K) {
                As[r][c] = fp8_e5m2_to_f32(A[gr * K + gc].data);
            } else {
                As[r][c] = 0.0f;
            }
        }

        // Load B (E4M3) -> F32
        unsigned int b_elems = FP8_TILE_K * FP8_TILE_N;
        for (unsigned int idx = thread_id; idx < b_elems; idx += num_threads) {
            unsigned int r = idx / FP8_TILE_N;
            unsigned int c = idx % FP8_TILE_N;
            unsigned int gr = k_offset + r;
            unsigned int gc = block_col + c;
            if (gr < K && gc < N) {
                Bs[r][c] = fp8_e4m3_to_f32(B[gr * N + gc].data);
            } else {
                Bs[r][c] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (unsigned int kk = 0; kk < FP8_TILE_K; kk++) {
            float reg_a[FP8_THREAD_M];
            float reg_b[FP8_THREAD_N];

            #pragma unroll
            for (int i = 0; i < FP8_THREAD_M; i++) {
                reg_a[i] = As[thread_row + i][kk];
            }
            #pragma unroll
            for (int j = 0; j < FP8_THREAD_N; j++) {
                reg_b[j] = Bs[kk][thread_col + j];
            }

            #pragma unroll
            for (int i = 0; i < FP8_THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < FP8_THREAD_N; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < FP8_THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < FP8_THREAD_N; j++) {
            unsigned int gr = block_row + thread_row + i;
            unsigned int gc = block_col + thread_col + j;
            if (gr < M && gc < N) {
                store_fn(C, gr * N + gc, reg_c[i][j] * combined_scale);
            }
        }
    }
}

// ============================================================================
// Batched variants
// ============================================================================

template<typename OutT, void (*store_fn)(OutT*, unsigned int, float)>
__device__ void fp8_matmul_e4m3_batched_kernel(
    const numr_fp8_e4m3* __restrict__ A,
    const numr_fp8_e4m3* __restrict__ B,
    OutT* __restrict__ C,
    float scale_a,
    float scale_b,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int batch_idx = blockIdx.z;
    if (batch_idx >= batch) return;

    const numr_fp8_e4m3* A_batch = A + batch_idx * M * K;
    const numr_fp8_e4m3* B_batch = B + batch_idx * K * N;
    OutT* C_batch = C + batch_idx * M * N;

    __shared__ float As[FP8_TILE_M][FP8_TILE_K];
    __shared__ float Bs[FP8_TILE_K][FP8_TILE_N];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = FP8_TILE_N / FP8_THREAD_N;
    const unsigned int threads_y = FP8_TILE_M / FP8_THREAD_M;

    const unsigned int block_row = blockIdx.y * FP8_TILE_M;
    const unsigned int block_col = blockIdx.x * FP8_TILE_N;
    const unsigned int thread_row = ty * FP8_THREAD_M;
    const unsigned int thread_col = tx * FP8_THREAD_N;

    float reg_c[FP8_THREAD_M][FP8_THREAD_N];
    #pragma unroll
    for (int i = 0; i < FP8_THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < FP8_THREAD_N; j++)
            reg_c[i][j] = 0.0f;

    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = threads_x * threads_y;
    const unsigned int num_k_tiles = (K + FP8_TILE_K - 1) / FP8_TILE_K;
    const float combined_scale = scale_a * scale_b;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * FP8_TILE_K;

        unsigned int a_elems = FP8_TILE_M * FP8_TILE_K;
        for (unsigned int idx = thread_id; idx < a_elems; idx += num_threads) {
            unsigned int r = idx / FP8_TILE_K;
            unsigned int c = idx % FP8_TILE_K;
            unsigned int gr = block_row + r;
            unsigned int gc = k_offset + c;
            As[r][c] = (gr < M && gc < K) ? fp8_e4m3_to_f32(A_batch[gr * K + gc].data) : 0.0f;
        }

        unsigned int b_elems = FP8_TILE_K * FP8_TILE_N;
        for (unsigned int idx = thread_id; idx < b_elems; idx += num_threads) {
            unsigned int r = idx / FP8_TILE_N;
            unsigned int c = idx % FP8_TILE_N;
            unsigned int gr = k_offset + r;
            unsigned int gc = block_col + c;
            Bs[r][c] = (gr < K && gc < N) ? fp8_e4m3_to_f32(B_batch[gr * N + gc].data) : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (unsigned int kk = 0; kk < FP8_TILE_K; kk++) {
            float ra[FP8_THREAD_M], rb[FP8_THREAD_N];
            #pragma unroll
            for (int i = 0; i < FP8_THREAD_M; i++) ra[i] = As[thread_row + i][kk];
            #pragma unroll
            for (int j = 0; j < FP8_THREAD_N; j++) rb[j] = Bs[kk][thread_col + j];
            #pragma unroll
            for (int i = 0; i < FP8_THREAD_M; i++)
                #pragma unroll
                for (int j = 0; j < FP8_THREAD_N; j++)
                    reg_c[i][j] += ra[i] * rb[j];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < FP8_THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < FP8_THREAD_N; j++) {
            unsigned int gr = block_row + thread_row + i;
            unsigned int gc = block_col + thread_col + j;
            if (gr < M && gc < N)
                store_fn(C_batch, gr * N + gc, reg_c[i][j] * combined_scale);
        }
}

template<typename OutT, void (*store_fn)(OutT*, unsigned int, float)>
__device__ void fp8_matmul_e5m2_batched_kernel(
    const numr_fp8_e5m2* __restrict__ A,
    const numr_fp8_e4m3* __restrict__ B,
    OutT* __restrict__ C,
    float scale_a,
    float scale_b,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int batch_idx = blockIdx.z;
    if (batch_idx >= batch) return;

    const numr_fp8_e5m2* A_batch = A + batch_idx * M * K;
    const numr_fp8_e4m3* B_batch = B + batch_idx * K * N;
    OutT* C_batch = C + batch_idx * M * N;

    __shared__ float As[FP8_TILE_M][FP8_TILE_K];
    __shared__ float Bs[FP8_TILE_K][FP8_TILE_N];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = FP8_TILE_N / FP8_THREAD_N;
    const unsigned int threads_y = FP8_TILE_M / FP8_THREAD_M;

    const unsigned int block_row = blockIdx.y * FP8_TILE_M;
    const unsigned int block_col = blockIdx.x * FP8_TILE_N;
    const unsigned int thread_row = ty * FP8_THREAD_M;
    const unsigned int thread_col = tx * FP8_THREAD_N;

    float reg_c[FP8_THREAD_M][FP8_THREAD_N];
    #pragma unroll
    for (int i = 0; i < FP8_THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < FP8_THREAD_N; j++)
            reg_c[i][j] = 0.0f;

    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = threads_x * threads_y;
    const unsigned int num_k_tiles = (K + FP8_TILE_K - 1) / FP8_TILE_K;
    const float combined_scale = scale_a * scale_b;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * FP8_TILE_K;

        unsigned int a_elems = FP8_TILE_M * FP8_TILE_K;
        for (unsigned int idx = thread_id; idx < a_elems; idx += num_threads) {
            unsigned int r = idx / FP8_TILE_K;
            unsigned int c = idx % FP8_TILE_K;
            unsigned int gr = block_row + r;
            unsigned int gc = k_offset + c;
            As[r][c] = (gr < M && gc < K) ? fp8_e5m2_to_f32(A_batch[gr * K + gc].data) : 0.0f;
        }

        unsigned int b_elems = FP8_TILE_K * FP8_TILE_N;
        for (unsigned int idx = thread_id; idx < b_elems; idx += num_threads) {
            unsigned int r = idx / FP8_TILE_N;
            unsigned int c = idx % FP8_TILE_N;
            unsigned int gr = k_offset + r;
            unsigned int gc = block_col + c;
            Bs[r][c] = (gr < K && gc < N) ? fp8_e4m3_to_f32(B_batch[gr * N + gc].data) : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (unsigned int kk = 0; kk < FP8_TILE_K; kk++) {
            float ra[FP8_THREAD_M], rb[FP8_THREAD_N];
            #pragma unroll
            for (int i = 0; i < FP8_THREAD_M; i++) ra[i] = As[thread_row + i][kk];
            #pragma unroll
            for (int j = 0; j < FP8_THREAD_N; j++) rb[j] = Bs[kk][thread_col + j];
            #pragma unroll
            for (int i = 0; i < FP8_THREAD_M; i++)
                #pragma unroll
                for (int j = 0; j < FP8_THREAD_N; j++)
                    reg_c[i][j] += ra[i] * rb[j];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < FP8_THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < FP8_THREAD_N; j++) {
            unsigned int gr = block_row + thread_row + i;
            unsigned int gc = block_col + thread_col + j;
            if (gr < M && gc < N)
                store_fn(C_batch, gr * N + gc, reg_c[i][j] * combined_scale);
        }
}

// ============================================================================
// Extern "C" entry points
// ============================================================================

extern "C" {

// --- E4M3 x E4M3 -> F32 ---
__global__ void fp8_matmul_e4m3_f32(
    const numr_fp8_e4m3* A, const numr_fp8_e4m3* B, float* C,
    float scale_a, float scale_b, unsigned int M, unsigned int N, unsigned int K
) { fp8_matmul_e4m3_kernel<float, store_f32>(A, B, C, scale_a, scale_b, M, N, K); }

// --- E4M3 x E4M3 -> F16 ---
__global__ void fp8_matmul_e4m3_f16(
    const numr_fp8_e4m3* A, const numr_fp8_e4m3* B, __half* C,
    float scale_a, float scale_b, unsigned int M, unsigned int N, unsigned int K
) { fp8_matmul_e4m3_kernel<__half, store_f16>(A, B, C, scale_a, scale_b, M, N, K); }

// --- E4M3 x E4M3 -> BF16 ---
__global__ void fp8_matmul_e4m3_bf16(
    const numr_fp8_e4m3* A, const numr_fp8_e4m3* B, __nv_bfloat16* C,
    float scale_a, float scale_b, unsigned int M, unsigned int N, unsigned int K
) { fp8_matmul_e4m3_kernel<__nv_bfloat16, store_bf16>(A, B, C, scale_a, scale_b, M, N, K); }

// --- E5M2 x E4M3 -> F32 ---
__global__ void fp8_matmul_e5m2_f32(
    const numr_fp8_e5m2* A, const numr_fp8_e4m3* B, float* C,
    float scale_a, float scale_b, unsigned int M, unsigned int N, unsigned int K
) { fp8_matmul_e5m2_kernel<float, store_f32>(A, B, C, scale_a, scale_b, M, N, K); }

// --- E5M2 x E4M3 -> F16 ---
__global__ void fp8_matmul_e5m2_f16(
    const numr_fp8_e5m2* A, const numr_fp8_e4m3* B, __half* C,
    float scale_a, float scale_b, unsigned int M, unsigned int N, unsigned int K
) { fp8_matmul_e5m2_kernel<__half, store_f16>(A, B, C, scale_a, scale_b, M, N, K); }

// --- E5M2 x E4M3 -> BF16 ---
__global__ void fp8_matmul_e5m2_bf16(
    const numr_fp8_e5m2* A, const numr_fp8_e4m3* B, __nv_bfloat16* C,
    float scale_a, float scale_b, unsigned int M, unsigned int N, unsigned int K
) { fp8_matmul_e5m2_kernel<__nv_bfloat16, store_bf16>(A, B, C, scale_a, scale_b, M, N, K); }

// --- Batched E4M3 x E4M3 ---
__global__ void fp8_matmul_e4m3_batched_f32(
    const numr_fp8_e4m3* A, const numr_fp8_e4m3* B, float* C,
    float scale_a, float scale_b, unsigned int batch, unsigned int M, unsigned int N, unsigned int K
) { fp8_matmul_e4m3_batched_kernel<float, store_f32>(A, B, C, scale_a, scale_b, batch, M, N, K); }

__global__ void fp8_matmul_e4m3_batched_f16(
    const numr_fp8_e4m3* A, const numr_fp8_e4m3* B, __half* C,
    float scale_a, float scale_b, unsigned int batch, unsigned int M, unsigned int N, unsigned int K
) { fp8_matmul_e4m3_batched_kernel<__half, store_f16>(A, B, C, scale_a, scale_b, batch, M, N, K); }

__global__ void fp8_matmul_e4m3_batched_bf16(
    const numr_fp8_e4m3* A, const numr_fp8_e4m3* B, __nv_bfloat16* C,
    float scale_a, float scale_b, unsigned int batch, unsigned int M, unsigned int N, unsigned int K
) { fp8_matmul_e4m3_batched_kernel<__nv_bfloat16, store_bf16>(A, B, C, scale_a, scale_b, batch, M, N, K); }

// --- Batched E5M2 x E4M3 ---
__global__ void fp8_matmul_e5m2_batched_f32(
    const numr_fp8_e5m2* A, const numr_fp8_e4m3* B, float* C,
    float scale_a, float scale_b, unsigned int batch, unsigned int M, unsigned int N, unsigned int K
) { fp8_matmul_e5m2_batched_kernel<float, store_f32>(A, B, C, scale_a, scale_b, batch, M, N, K); }

__global__ void fp8_matmul_e5m2_batched_f16(
    const numr_fp8_e5m2* A, const numr_fp8_e4m3* B, __half* C,
    float scale_a, float scale_b, unsigned int batch, unsigned int M, unsigned int N, unsigned int K
) { fp8_matmul_e5m2_batched_kernel<__half, store_f16>(A, B, C, scale_a, scale_b, batch, M, N, K); }

__global__ void fp8_matmul_e5m2_batched_bf16(
    const numr_fp8_e5m2* A, const numr_fp8_e4m3* B, __nv_bfloat16* C,
    float scale_a, float scale_b, unsigned int batch, unsigned int M, unsigned int N, unsigned int K
) { fp8_matmul_e5m2_batched_kernel<__nv_bfloat16, store_bf16>(A, B, C, scale_a, scale_b, batch, M, N, K); }

} // extern "C"
