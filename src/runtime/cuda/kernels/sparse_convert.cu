//! Sparse matrix format conversion kernels
//!
//! This file implements GPU-native format conversions between COO, CSR, and CSC formats.
//!
//! Key operations:
//! - COO → CSR: Radix sort by (row, col) + prefix sum
//! - COO → CSC: Radix sort by (col, row) + prefix sum
//! - CSR ↔ CSC: Direct transpose (scatter to transposed positions)
//! - CSR/CSC → COO: Parallel pointer expansion
//!
//! All kernels avoid GPU→CPU→GPU transfers and operate entirely on device memory.

#include <cuda_runtime.h>
#include <stdint.h>
#include <cub/cub.cuh>

// ============================================================================
// CSR/CSC → COO: Pointer Expansion
// ============================================================================

/// Expand compressed row/column pointers to explicit indices (device function)
///
/// For CSR: row_ptrs[i] to row_ptrs[i+1] contains indices for row i
/// Output: row_indices[j] = i for all j in [row_ptrs[i], row_ptrs[i+1])
__device__ void expand_ptrs_impl(
    unsigned int idx,
    const int64_t* ptrs,
    int64_t* indices_out,
    unsigned int n_rows_or_cols
) {
    if (idx >= n_rows_or_cols) return;

    int64_t start = ptrs[idx];
    int64_t end = ptrs[idx + 1];

    // Write this row/col index to all positions it owns
    for (int64_t pos = start; pos < end; pos++) {
        indices_out[pos] = idx;
    }
}

extern "C" {

__global__ void expand_ptrs_i64(
    const int64_t* ptrs,
    int64_t* indices_out,
    unsigned int n_rows_or_cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    expand_ptrs_impl(idx, ptrs, indices_out, n_rows_or_cols);
}

} // extern "C"

// ============================================================================
// COO → CSR/CSC: Sort + Build Pointers
// ============================================================================

// Note: build_ptrs_from_sorted not currently used (COO→CSR/CSC needs radix sort first)

// ============================================================================
// Histogram: Count NNZ per Column/Row
// ============================================================================

extern "C" {

/// Count non-zeros per column from CSR format
__global__ void histogram_csr_columns_i64(
    const int64_t* row_ptrs,
    const int64_t* col_indices,
    int64_t* col_counts,
    unsigned int n_rows
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_rows) return;

    int64_t start = row_ptrs[row];
    int64_t end = row_ptrs[row + 1];

    // Count this row's contribution to each column
    for (int64_t idx = start; idx < end; idx++) {
        int64_t col = col_indices[idx];
        atomicAdd((unsigned long long*)&col_counts[col], 1ULL);
    }
}

/// Count non-zeros per row from CSC format
__global__ void histogram_csc_rows_i64(
    const int64_t* col_ptrs,
    const int64_t* row_indices,
    int64_t* row_counts,
    unsigned int n_cols
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= n_cols) return;

    int64_t start = col_ptrs[col];
    int64_t end = col_ptrs[col + 1];

    // Count this column's contribution to each row
    for (int64_t idx = start; idx < end; idx++) {
        int64_t row = row_indices[idx];
        atomicAdd((unsigned long long*)&row_counts[row], 1ULL);
    }
}

} // extern "C"

// ============================================================================
// CSR ↔ CSC: Direct Transpose
// ============================================================================

/// CSR → CSC transpose device function
template<typename T>
__device__ void csr_to_csc_transpose_impl(
    unsigned int row,
    const int64_t* csr_row_ptrs,
    const int64_t* csr_col_indices,
    const T* csr_values,
    int64_t* csc_col_ptrs,
    int64_t* csc_row_indices,
    T* csc_values,
    unsigned int n_rows
) {
    if (row >= n_rows) return;

    int64_t row_start = csr_row_ptrs[row];
    int64_t row_end = csr_row_ptrs[row + 1];

    for (int64_t idx = row_start; idx < row_end; idx++) {
        int64_t col = csr_col_indices[idx];
        T value = csr_values[idx];

        // Use atomicAdd to get unique position in column
        int64_t csc_pos = atomicAdd((unsigned long long*)&csc_col_ptrs[col], 1ULL);

        // Write to CSC
        csc_row_indices[csc_pos] = row;
        csc_values[csc_pos] = value;
    }
}

/// CSC → CSR transpose device function
template<typename T>
__device__ void csc_to_csr_transpose_impl(
    unsigned int col,
    const int64_t* csc_col_ptrs,
    const int64_t* csc_row_indices,
    const T* csc_values,
    int64_t* csr_row_ptrs,
    int64_t* csr_col_indices,
    T* csr_values,
    unsigned int n_cols
) {
    if (col >= n_cols) return;

    int64_t col_start = csc_col_ptrs[col];
    int64_t col_end = csc_col_ptrs[col + 1];

    for (int64_t idx = col_start; idx < col_end; idx++) {
        int64_t row = csc_row_indices[idx];
        T value = csc_values[idx];

        // Use atomicAdd to get unique position in row
        int64_t csr_pos = atomicAdd((unsigned long long*)&csr_row_ptrs[row], 1ULL);

        // Write to CSR
        csr_col_indices[csr_pos] = col;
        csr_values[csr_pos] = value;
    }
}

extern "C" {

__global__ void csr_to_csc_transpose_f32(
    const int64_t* csr_row_ptrs,
    const int64_t* csr_col_indices,
    const float* csr_values,
    int64_t* csc_col_ptrs,
    int64_t* csc_row_indices,
    float* csc_values,
    unsigned int n_rows,
    unsigned int n_cols
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    csr_to_csc_transpose_impl<float>(
        row, csr_row_ptrs, csr_col_indices, csr_values,
        csc_col_ptrs, csc_row_indices, csc_values, n_rows
    );
}

__global__ void csr_to_csc_transpose_f64(
    const int64_t* csr_row_ptrs,
    const int64_t* csr_col_indices,
    const double* csr_values,
    int64_t* csc_col_ptrs,
    int64_t* csc_row_indices,
    double* csc_values,
    unsigned int n_rows,
    unsigned int n_cols
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    csr_to_csc_transpose_impl<double>(
        row, csr_row_ptrs, csr_col_indices, csr_values,
        csc_col_ptrs, csc_row_indices, csc_values, n_rows
    );
}

__global__ void csc_to_csr_transpose_f32(
    const int64_t* csc_col_ptrs,
    const int64_t* csc_row_indices,
    const float* csc_values,
    int64_t* csr_row_ptrs,
    int64_t* csr_col_indices,
    float* csr_values,
    unsigned int n_rows,
    unsigned int n_cols
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    csc_to_csr_transpose_impl<float>(
        col, csc_col_ptrs, csc_row_indices, csc_values,
        csr_row_ptrs, csr_col_indices, csr_values, n_cols
    );
}

__global__ void csc_to_csr_transpose_f64(
    const int64_t* csc_col_ptrs,
    const int64_t* csc_row_indices,
    const double* csc_values,
    int64_t* csr_row_ptrs,
    int64_t* csr_col_indices,
    double* csr_values,
    unsigned int n_rows,
    unsigned int n_cols
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    csc_to_csr_transpose_impl<double>(
        col, csc_col_ptrs, csc_row_indices, csc_values,
        csr_row_ptrs, csr_col_indices, csr_values, n_cols
    );
}

} // extern "C"

// ============================================================================
// Scalar Operations
// ============================================================================

extern "C" {

__global__ void sparse_scale_f32(
    const float* values_in,
    float* values_out,
    float scalar,
    unsigned int nnz
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        values_out[idx] = values_in[idx] * scalar;
    }
}

__global__ void sparse_scale_f64(
    const double* values_in,
    double* values_out,
    double scalar,
    unsigned int nnz
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        values_out[idx] = values_in[idx] * scalar;
    }
}

} // extern "C"

// ============================================================================
// Reduction Operations
// ============================================================================

extern "C" {

__global__ void sparse_sum_f32(
    const float* values,
    float* result,
    unsigned int nnz
) {
    __shared__ float shared[256];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // Load data
    float val = (idx < nnz) ? values[idx] : 0.0f;
    shared[tid] = val;
    __syncthreads();

    // Block-level reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // First thread of each block writes partial sum
    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

__global__ void sparse_sum_f64(
    const double* values,
    double* result,
    unsigned int nnz
) {
    __shared__ double shared[256];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // Load data
    double val = (idx < nnz) ? values[idx] : 0.0;
    shared[tid] = val;
    __syncthreads();

    // Block-level reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // First thread of each block writes partial sum
    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

} // extern "C"

// ============================================================================
// Row/Column-wise Reductions
// ============================================================================

extern "C" {

__global__ void sparse_sum_rows_csr_f32(
    const int64_t* row_ptrs,
    const float* values,
    float* row_sums,
    unsigned int n_rows
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_rows) return;

    int64_t start = row_ptrs[row];
    int64_t end = row_ptrs[row + 1];

    float sum = 0.0f;
    for (int64_t idx = start; idx < end; idx++) {
        sum += values[idx];
    }

    row_sums[row] = sum;
}

__global__ void sparse_sum_rows_csr_f64(
    const int64_t* row_ptrs,
    const double* values,
    double* row_sums,
    unsigned int n_rows
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_rows) return;

    int64_t start = row_ptrs[row];
    int64_t end = row_ptrs[row + 1];

    double sum = 0.0;
    for (int64_t idx = start; idx < end; idx++) {
        sum += values[idx];
    }

    row_sums[row] = sum;
}

} // extern "C"

extern "C" {

__global__ void sparse_nnz_per_row_csr_i64(
    const int64_t* row_ptrs,
    int64_t* row_nnz,
    unsigned int n_rows
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_rows) return;

    row_nnz[row] = row_ptrs[row + 1] - row_ptrs[row];
}

} // extern "C"

// ============================================================================
// COO → CSR/CSC: Build Pointers from Sorted Indices
// ============================================================================

extern "C" {

/// Build row_ptrs/col_ptrs from sorted row/column indices
///
/// After sorting COO entries by row (for CSR) or column (for CSC),
/// this kernel builds the compressed pointer array.
///
/// Algorithm:
/// - row_ptrs[0] = 0 (first row starts at position 0)
/// - For each position i, if row_indices[i] != row_indices[i-1], record transition
/// - row_ptrs[row+1] = position where row ends
/// - row_ptrs[n_rows] = nnz (total entries)
__global__ void build_ptrs_from_sorted_indices_i64(
    const int64_t* sorted_indices,  // Sorted row or column indices
    int64_t* ptrs_out,              // Output row_ptrs or col_ptrs
    unsigned int nnz,               // Total number of non-zeros
    unsigned int n_rows_or_cols     // Number of rows (for CSR) or columns (for CSC)
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize first element
    if (idx == 0) {
        ptrs_out[0] = 0;
        if (nnz == 0) {
            // Empty matrix - set all pointers to 0
            for (unsigned int i = 1; i <= n_rows_or_cols; i++) {
                ptrs_out[i] = 0;
            }
            return;
        }
    }

    // Process transitions
    if (idx > 0 && idx < nnz) {
        int64_t prev_idx_val = sorted_indices[idx - 1];
        int64_t curr_idx_val = sorted_indices[idx];

        // If we transition to a new row/column
        if (curr_idx_val != prev_idx_val) {
            // Fill in pointers for all rows/cols between prev and curr
            for (int64_t r = prev_idx_val + 1; r <= curr_idx_val; r++) {
                ptrs_out[r] = idx;
            }
        }
    }

    // Last thread handles final row/column and fills remaining
    if (idx == nnz - 1 || (idx == 0 && nnz > 0)) {
        int64_t last_idx_val = sorted_indices[nnz - 1];
        // Fill from last row/col to end
        for (int64_t r = last_idx_val + 1; r <= (int64_t)n_rows_or_cols; r++) {
            ptrs_out[r] = nnz;
        }
    }
}

} // extern "C"
