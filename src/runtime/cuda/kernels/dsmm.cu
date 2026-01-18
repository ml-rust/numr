//! Dense × Sparse Matrix Multiplication (SpMM / DSMM)
//!
//! Computes C = A @ B where:
//! - A is dense [M, K] (row-major)
//! - B is sparse CSC [K, N]
//! - C is dense [M, N] (row-major)
//!
//! # Algorithm: Column-Parallel (Baseline Implementation)
//!
//! This implements a **column-parallel baseline** algorithm that matches the CPU
//! implementation for backend parity.
//!
//! ## Core Algorithm (matches CPU):
//!
//! ```
//! For each column j of B (CSC format):
//!   For each non-zero B[k, j] in column j:
//!     C[:, j] += A[:, k] * B[k, j]
//! ```
//!
//! ## Why CSC Format?
//!
//! - CSC (Column Sparse Column) stores columns efficiently
//! - Natural fit for A @ B where B's columns are accessed
//! - No race conditions (each thread owns a column)
//!
//! ## Current Implementation (Baseline):
//!
//! - One block per column (simple, matches CPU structure)
//! - Shared memory caching for sparse column data (up to 1024 nnz)
//! - Threads cooperate to process rows within each column
//!
//! ## Future Optimization: Merge-Path Load Balancing
//!
//! The baseline implementation may have suboptimal performance when column sparsity
//! varies significantly (some columns with 1 nnz, others with 10,000+ nnz).
//!
//! Planned optimization uses merge-path partitioning:
//! 1. Flatten all (column, nnz_in_column) work into 1D stream
//! 2. Use binary search to evenly partition across thread blocks
//! 3. Each block processes ~equal amount of non-zeros
//!
//! This would eliminate warp divergence and improve GPU utilization.
//!
//! Reference: "Merge Path - Parallel Merging Made Simple" (NVIDIA, 2012)
//!
//! Note: Baseline implementation is correct and provides good performance for
//! matrices with uniform column sparsity. Merge-path optimization will be added
//! when profiling indicates it's needed.

#include <cuda_runtime.h>
#include <cuda_fp16.h>       // __half
#include <cuda_bf16.h>       // __nv_bfloat16

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ============================================================================
// Column-Parallel DSMM (without merge-path for now - simple baseline)
// ============================================================================

/// Simple column-parallel dsmm kernel (baseline version)
///
/// Each block processes one column of the output.
/// Each thread processes multiple rows of the output.
///
/// This baseline matches the CPU algorithm structure for backend parity.
/// Performance is good for matrices with uniform column sparsity.
///
/// Future: Add merge-path load balancing for matrices with highly variable
/// column sparsity (will be implemented when profiling indicates need).
template<typename T>
__global__ void dsmm_column_parallel_kernel(
    const T* __restrict__ A,                      // Dense [M, K] row-major
    const long long* __restrict__ col_ptrs,       // CSC column pointers [N+1]
    const long long* __restrict__ row_indices,    // CSC row indices [nnz]
    const T* __restrict__ values,                 // CSC values [nnz]
    T* __restrict__ C,                            // Dense output [M, N] row-major
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    // Column index (each block handles one column)
    int col = blockIdx.x;
    if (col >= N) return;

    // Get non-zeros in this column
    long long col_start = col_ptrs[col];
    long long col_end = col_ptrs[col + 1];
    long long nnz_in_col = col_end - col_start;

    if (nnz_in_col == 0) return;  // Empty column

    // Each thread processes a subset of rows in the output
    // This loop is vectorizable on SIMD backends
    for (unsigned int row = threadIdx.x; row < M; row += blockDim.x) {
        T sum = static_cast<T>(0);

        // For each non-zero B[k, col] in this column
        for (long long idx = col_start; idx < col_end; idx++) {
            long long k = row_indices[idx];  // row in B (column in A)
            T b_val = values[idx];
            T a_val = A[row * K + k];  // A[row, k]
            sum += a_val * b_val;
        }

        // Write result (no race condition - each thread owns a unique C[row, col])
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Optimized Version: Shared Memory + Warp Reduction
// ============================================================================

/// Optimized dsmm with shared memory for the sparse column
///
/// Loads sparse column into shared memory once per block.
/// Uses warp-level parallelism for better performance.
template<typename T>
__global__ void dsmm_column_parallel_optimized_kernel(
    const T* __restrict__ A,
    const long long* __restrict__ col_ptrs,
    const long long* __restrict__ row_indices,
    const T* __restrict__ values,
    T* __restrict__ C,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    int col = blockIdx.x;
    if (col >= N) return;

    // Shared memory for column data
    __shared__ long long shared_row_indices[1024];  // Max 1024 nnz per block
    __shared__ T shared_values[1024];

    long long col_start = col_ptrs[col];
    long long col_end = col_ptrs[col + 1];
    long long nnz_in_col = col_end - col_start;

    if (nnz_in_col == 0) return;

    // Cooperatively load column data into shared memory
    for (long long i = threadIdx.x; i < nnz_in_col; i += blockDim.x) {
        long long global_idx = col_start + i;
        if (i < 1024) {  // Shared memory limit
            shared_row_indices[i] = row_indices[global_idx];
            shared_values[i] = values[global_idx];
        }
    }
    __syncthreads();

    // Process rows
    long long nnz_to_process = min(nnz_in_col, (long long)1024);
    for (unsigned int row = threadIdx.x; row < M; row += blockDim.x) {
        T sum = static_cast<T>(0);

        // Process non-zeros from shared memory (faster access)
        for (long long i = 0; i < nnz_to_process; i++) {
            long long k = shared_row_indices[i];
            T b_val = shared_values[i];
            T a_val = A[row * K + k];
            sum += a_val * b_val;
        }

        // Process remaining nnz from global memory (if any)
        for (long long i = nnz_to_process; i < nnz_in_col; i++) {
            long long global_idx = col_start + i;
            long long k = row_indices[global_idx];
            T b_val = values[global_idx];
            T a_val = A[row * K + k];
            sum += a_val * b_val;
        }

        C[row * N + col] = sum;
    }
}

// ============================================================================
// Extern "C" Wrappers for FFI
// ============================================================================

extern "C" {

/// F32 dsmm kernel
__global__ void dsmm_csc_f32(
    const float* A,
    const long long* col_ptrs,
    const long long* row_indices,
    const float* values,
    float* C,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    // Column index (each block handles one column)
    int col = blockIdx.x;
    if (col >= N) return;

    // Shared memory for column data
    __shared__ long long shared_row_indices[1024];
    __shared__ float shared_values[1024];

    long long col_start = col_ptrs[col];
    long long col_end = col_ptrs[col + 1];
    long long nnz_in_col = col_end - col_start;

    if (nnz_in_col == 0) return;

    // Cooperatively load column data into shared memory
    for (long long i = threadIdx.x; i < nnz_in_col; i += blockDim.x) {
        long long global_idx = col_start + i;
        if (i < 1024) {
            shared_row_indices[i] = row_indices[global_idx];
            shared_values[i] = values[global_idx];
        }
    }
    __syncthreads();

    // Process rows
    long long nnz_to_process = min(nnz_in_col, (long long)1024);
    for (unsigned int row = threadIdx.x; row < M; row += blockDim.x) {
        float sum = 0.0f;

        // Process non-zeros from shared memory
        for (long long i = 0; i < nnz_to_process; i++) {
            long long k = shared_row_indices[i];
            float b_val = shared_values[i];
            float a_val = A[row * K + k];
            sum += a_val * b_val;
        }

        // Process remaining nnz from global memory (if any)
        for (long long i = nnz_to_process; i < nnz_in_col; i++) {
            long long global_idx = col_start + i;
            long long k = row_indices[global_idx];
            float b_val = values[global_idx];
            float a_val = A[row * K + k];
            sum += a_val * b_val;
        }

        C[row * N + col] = sum;
    }
}

/// F64 dsmm kernel
__global__ void dsmm_csc_f64(
    const double* A,
    const long long* col_ptrs,
    const long long* row_indices,
    const double* values,
    double* C,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    int col = blockIdx.x;
    if (col >= N) return;

    __shared__ long long shared_row_indices[1024];
    __shared__ double shared_values[1024];

    long long col_start = col_ptrs[col];
    long long col_end = col_ptrs[col + 1];
    long long nnz_in_col = col_end - col_start;

    if (nnz_in_col == 0) return;

    for (long long i = threadIdx.x; i < nnz_in_col; i += blockDim.x) {
        long long global_idx = col_start + i;
        if (i < 1024) {
            shared_row_indices[i] = row_indices[global_idx];
            shared_values[i] = values[global_idx];
        }
    }
    __syncthreads();

    long long nnz_to_process = min(nnz_in_col, (long long)1024);
    for (unsigned int row = threadIdx.x; row < M; row += blockDim.x) {
        double sum = 0.0;

        for (long long i = 0; i < nnz_to_process; i++) {
            long long k = shared_row_indices[i];
            double b_val = shared_values[i];
            double a_val = A[row * K + k];
            sum += a_val * b_val;
        }

        for (long long i = nnz_to_process; i < nnz_in_col; i++) {
            long long global_idx = col_start + i;
            long long k = row_indices[global_idx];
            double b_val = values[global_idx];
            double a_val = A[row * K + k];
            sum += a_val * b_val;
        }

        C[row * N + col] = sum;
    }
}

/// F16 dsmm kernel
__global__ void dsmm_csc_f16(
    const __half* A,
    const long long* col_ptrs,
    const long long* row_indices,
    const __half* values,
    __half* C,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    int col = blockIdx.x;
    if (col >= N) return;

    __shared__ long long shared_row_indices[1024];
    __shared__ __half shared_values[1024];

    long long col_start = col_ptrs[col];
    long long col_end = col_ptrs[col + 1];
    long long nnz_in_col = col_end - col_start;

    if (nnz_in_col == 0) return;

    for (long long i = threadIdx.x; i < nnz_in_col; i += blockDim.x) {
        long long global_idx = col_start + i;
        if (i < 1024) {
            shared_row_indices[i] = row_indices[global_idx];
            shared_values[i] = values[global_idx];
        }
    }
    __syncthreads();

    long long nnz_to_process = min(nnz_in_col, (long long)1024);
    for (unsigned int row = threadIdx.x; row < M; row += blockDim.x) {
        float sum = 0.0f;  // Use F32 accumulator

        for (long long i = 0; i < nnz_to_process; i++) {
            long long k = shared_row_indices[i];
            float b_val = __half2float(shared_values[i]);
            float a_val = __half2float(A[row * K + k]);
            sum += a_val * b_val;
        }

        for (long long i = nnz_to_process; i < nnz_in_col; i++) {
            long long global_idx = col_start + i;
            long long k = row_indices[global_idx];
            float b_val = __half2float(values[global_idx]);
            float a_val = __half2float(A[row * K + k]);
            sum += a_val * b_val;
        }

        C[row * N + col] = __float2half(sum);
    }
}

/// BF16 dsmm kernel
__global__ void dsmm_csc_bf16(
    const __nv_bfloat16* A,
    const long long* col_ptrs,
    const long long* row_indices,
    const __nv_bfloat16* values,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    int col = blockIdx.x;
    if (col >= N) return;

    __shared__ long long shared_row_indices[1024];
    __shared__ __nv_bfloat16 shared_values[1024];

    long long col_start = col_ptrs[col];
    long long col_end = col_ptrs[col + 1];
    long long nnz_in_col = col_end - col_start;

    if (nnz_in_col == 0) return;

    for (long long i = threadIdx.x; i < nnz_in_col; i += blockDim.x) {
        long long global_idx = col_start + i;
        if (i < 1024) {
            shared_row_indices[i] = row_indices[global_idx];
            shared_values[i] = values[global_idx];
        }
    }
    __syncthreads();

    long long nnz_to_process = min(nnz_in_col, (long long)1024);
    for (unsigned int row = threadIdx.x; row < M; row += blockDim.x) {
        float sum = 0.0f;  // Use F32 accumulator

        for (long long i = 0; i < nnz_to_process; i++) {
            long long k = shared_row_indices[i];
            float b_val = __bfloat162float(shared_values[i]);
            float a_val = __bfloat162float(A[row * K + k]);
            sum += a_val * b_val;
        }

        for (long long i = nnz_to_process; i < nnz_in_col; i++) {
            long long global_idx = col_start + i;
            long long k = row_indices[global_idx];
            float b_val = __bfloat162float(values[global_idx]);
            float a_val = __bfloat162float(A[row * K + k]);
            sum += a_val * b_val;
        }

        C[row * N + col] = __float2bfloat16(sum);
    }
}

}  // extern "C"
