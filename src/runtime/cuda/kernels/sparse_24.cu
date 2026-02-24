// 2:4 Structured Sparsity CUDA kernels
// Operations: prune to 2:4, decompress to dense, sparse matmul
//
// Metadata format: 4 bits per group of 4, bitmask with exactly 2 bits set.
// 8 groups packed per U32 (8 × 4 = 32 bits).

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Prune to 2:4: For each group of 4 elements, keep the 2 with largest magnitude
// ============================================================================

template<typename T>
__device__ float to_abs_float(T val) {
    return fabsf(static_cast<float>(val));
}

__device__ float to_abs_float(__half val) {
    return fabsf(__half2float(val));
}

__device__ float to_abs_float(__nv_bfloat16 val) {
    return fabsf(__bfloat162float(val));
}

// One thread per group of 4 elements
template<typename T>
__device__ void prune_to_24_impl(
    const T* __restrict__ dense,      // [M, K]
    T* __restrict__ compressed,        // [M, K/2]
    unsigned int* __restrict__ metadata, // [M, meta_cols]
    unsigned int M,
    unsigned int K
) {
    unsigned int num_groups_per_row = K / 4;
    unsigned int meta_cols = (num_groups_per_row + 7) / 8;
    unsigned int half_k = K / 2;

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_groups = M * num_groups_per_row;
    if (tid >= total_groups) return;

    unsigned int row = tid / num_groups_per_row;
    unsigned int g = tid % num_groups_per_row;
    unsigned int base = row * K + g * 4;

    // Load 4 values
    T vals[4];
    vals[0] = dense[base];
    vals[1] = dense[base + 1];
    vals[2] = dense[base + 2];
    vals[3] = dense[base + 3];

    // Compute magnitudes
    float mags[4];
    mags[0] = to_abs_float(vals[0]);
    mags[1] = to_abs_float(vals[1]);
    mags[2] = to_abs_float(vals[2]);
    mags[3] = to_abs_float(vals[3]);

    // Find top-2 by magnitude (stable: prefer earlier indices on tie)
    // Simple selection network for 4 elements
    int idx0 = 0, idx1 = 1;
    float m0 = mags[0], m1 = mags[1];

    // Ensure m0 >= m1
    if (m1 > m0) { int t = idx0; idx0 = idx1; idx1 = t; float ft = m0; m0 = m1; m1 = ft; }

    // Compare with index 2
    if (mags[2] > m1) {
        idx1 = 2; m1 = mags[2];
        if (m1 > m0) { int t = idx0; idx0 = idx1; idx1 = t; float ft = m0; m0 = m1; m1 = ft; }
    }

    // Compare with index 3
    if (mags[3] > m1) {
        idx1 = 3; m1 = mags[3];
        if (m1 > m0) { int t = idx0; idx0 = idx1; idx1 = t; }
    }

    // Sort kept indices so lower index comes first
    int first = min(idx0, idx1);
    int second = max(idx0, idx1);

    // Write compressed values (2 per group)
    unsigned int out_base = row * half_k + g * 2;
    compressed[out_base] = vals[first];
    compressed[out_base + 1] = vals[second];

    // Build 4-bit bitmask
    unsigned int mask = (1u << first) | (1u << second);

    // Pack into metadata (atomic OR since multiple threads may write to same U32)
    unsigned int word_idx = g / 8;
    unsigned int nibble_idx = g % 8;
    unsigned int meta_offset = row * meta_cols + word_idx;
    atomicOr(&metadata[meta_offset], mask << (nibble_idx * 4));
}

// ============================================================================
// Decompress: Reconstruct dense matrix from 2:4 compressed format
// ============================================================================

template<typename T>
__device__ void decompress_24_impl(
    const T* __restrict__ compressed,        // [M, K/2]
    const unsigned int* __restrict__ metadata, // [M, meta_cols]
    T* __restrict__ dense,                    // [M, K]
    unsigned int M,
    unsigned int K
) {
    unsigned int num_groups_per_row = K / 4;
    unsigned int meta_cols = (num_groups_per_row + 7) / 8;
    unsigned int half_k = K / 2;

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_groups = M * num_groups_per_row;
    if (tid >= total_groups) return;

    unsigned int row = tid / num_groups_per_row;
    unsigned int g = tid % num_groups_per_row;

    // Read metadata
    unsigned int word_idx = g / 8;
    unsigned int nibble_idx = g % 8;
    unsigned int word = metadata[row * meta_cols + word_idx];
    unsigned int mask = (word >> (nibble_idx * 4)) & 0xF;

    // Read 2 compressed values
    unsigned int in_base = row * half_k + g * 2;
    T v0 = compressed[in_base];
    T v1 = compressed[in_base + 1];

    // Write to dense (zero all 4 first, then fill kept positions)
    unsigned int out_base = row * K + g * 4;
    T zero = static_cast<T>(0);
    dense[out_base] = zero;
    dense[out_base + 1] = zero;
    dense[out_base + 2] = zero;
    dense[out_base + 3] = zero;

    // Place values at their positions
    int val_idx = 0;
    for (int bit = 0; bit < 4; bit++) {
        if (mask & (1u << bit)) {
            dense[out_base + bit] = (val_idx == 0) ? v0 : v1;
            val_idx++;
        }
    }
}

// ============================================================================
// Sparse 2:4 MatMul: C = A @ B^T where B is in 2:4 compressed format
// A: [N, K] dense, B: [M, K] compressed as [M, K/2] + metadata → C: [N, M]
//
// Each thread computes one element of C by decompressing B on the fly.
// Tiled with shared memory for better performance.
// ============================================================================

#define TILE_SIZE 16

template<typename T, typename AccT>
__device__ void sparse_24_matmul_impl(
    const T* __restrict__ A,               // [N, K] dense input
    const T* __restrict__ B_compressed,    // [M, K/2] compressed weights
    const unsigned int* __restrict__ B_metadata, // [M, meta_cols]
    T* __restrict__ C,                     // [N, M] output
    unsigned int N,
    unsigned int M,
    unsigned int K
) {
    unsigned int num_groups = K / 4;
    unsigned int meta_cols = (num_groups + 7) / 8;
    unsigned int half_k = K / 2;

    unsigned int row = blockIdx.y * TILE_SIZE + threadIdx.y; // output row (N dim)
    unsigned int col = blockIdx.x * TILE_SIZE + threadIdx.x; // output col (M dim)

    if (row >= N || col >= M) return;

    AccT sum = static_cast<AccT>(0);

    // For each group of 4 in K dimension
    for (unsigned int g = 0; g < num_groups; g++) {
        // Read A values (dense, 4 consecutive)
        unsigned int a_base = row * K + g * 4;
        AccT a0 = static_cast<AccT>(A[a_base]);
        AccT a1 = static_cast<AccT>(A[a_base + 1]);
        AccT a2 = static_cast<AccT>(A[a_base + 2]);
        AccT a3 = static_cast<AccT>(A[a_base + 3]);

        // Read B compressed values (2 per group)
        unsigned int b_base = col * half_k + g * 2;
        AccT b0 = static_cast<AccT>(B_compressed[b_base]);
        AccT b1 = static_cast<AccT>(B_compressed[b_base + 1]);

        // Read B metadata
        unsigned int word_idx = g / 8;
        unsigned int nibble_idx = g % 8;
        unsigned int word = B_metadata[col * meta_cols + word_idx];
        unsigned int mask = (word >> (nibble_idx * 4)) & 0xF;

        // Decompress and accumulate on the fly
        AccT a_vals[4] = {a0, a1, a2, a3};
        int val_idx = 0;
        for (int bit = 0; bit < 4; bit++) {
            if (mask & (1u << bit)) {
                AccT b_val = (val_idx == 0) ? b0 : b1;
                sum += a_vals[bit] * b_val;
                val_idx++;
            }
        }
    }

    C[row * M + col] = static_cast<T>(sum);
}

// ============================================================================
// F16/BF16 specialization: decompress kernel (same logic, no special accumulation needed)
// ============================================================================

// For F16 decompress, the template works directly since we just copy values.
// For F16 matmul, we accumulate in F32.

// ============================================================================
// Extern "C" instantiations
// ============================================================================

extern "C" {

// --- Prune ---
__global__ void sparse_24_prune_f32(const float* d, float* c, unsigned int* m, unsigned int M, unsigned int K) {
    prune_to_24_impl<float>(d, c, m, M, K);
}
__global__ void sparse_24_prune_f64(const double* d, double* c, unsigned int* m, unsigned int M, unsigned int K) {
    prune_to_24_impl<double>(d, c, m, M, K);
}
__global__ void sparse_24_prune_f16(const __half* d, __half* c, unsigned int* m, unsigned int M, unsigned int K) {
    prune_to_24_impl<__half>(d, c, m, M, K);
}
__global__ void sparse_24_prune_bf16(const __nv_bfloat16* d, __nv_bfloat16* c, unsigned int* m, unsigned int M, unsigned int K) {
    prune_to_24_impl<__nv_bfloat16>(d, c, m, M, K);
}

// --- Decompress ---
__global__ void sparse_24_decompress_f32(const float* c, const unsigned int* m, float* d, unsigned int M, unsigned int K) {
    decompress_24_impl<float>(c, m, d, M, K);
}
__global__ void sparse_24_decompress_f64(const double* c, const unsigned int* m, double* d, unsigned int M, unsigned int K) {
    decompress_24_impl<double>(c, m, d, M, K);
}
__global__ void sparse_24_decompress_f16(const __half* c, const unsigned int* m, __half* d, unsigned int M, unsigned int K) {
    decompress_24_impl<__half>(c, m, d, M, K);
}
__global__ void sparse_24_decompress_bf16(const __nv_bfloat16* c, const unsigned int* m, __nv_bfloat16* d, unsigned int M, unsigned int K) {
    decompress_24_impl<__nv_bfloat16>(c, m, d, M, K);
}

// --- Matmul (accumulate in appropriate precision) ---
__global__ void sparse_24_matmul_f32(const float* A, const float* Bc, const unsigned int* Bm, float* C, unsigned int N, unsigned int M, unsigned int K) {
    sparse_24_matmul_impl<float, float>(A, Bc, Bm, C, N, M, K);
}
__global__ void sparse_24_matmul_f64(const double* A, const double* Bc, const unsigned int* Bm, double* C, unsigned int N, unsigned int M, unsigned int K) {
    sparse_24_matmul_impl<double, double>(A, Bc, Bm, C, N, M, K);
}
__global__ void sparse_24_matmul_f16(const __half* A, const __half* Bc, const unsigned int* Bm, __half* C, unsigned int N, unsigned int M, unsigned int K) {
    sparse_24_matmul_impl<__half, float>(A, Bc, Bm, C, N, M, K);
}
__global__ void sparse_24_matmul_bf16(const __nv_bfloat16* A, const __nv_bfloat16* Bc, const unsigned int* Bm, __nv_bfloat16* C, unsigned int N, unsigned int M, unsigned int K) {
    sparse_24_matmul_impl<__nv_bfloat16, float>(A, Bc, Bm, C, N, M, K);
}

} // extern "C"
