// Sparse utility CUDA kernels
// GPU-native implementations to eliminate CPU transfers
//
// Operations:
// - filter_csr_values: Filter values below threshold and rebuild CSR
// - csr_sum_rows: Sum each row (CSR → dense vector)
// - csc_sum_cols: Sum each column (CSC → dense vector)
// - csr_nnz_per_row: Count non-zeros per row (row_ptrs difference)
// - csc_nnz_per_col: Count non-zeros per column (col_ptrs difference)
// - csr_to_dense: Expand CSR to dense matrix
// - dense_to_coo: Convert dense to COO format (2-pass)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// CSR Filtering (templates must be outside extern "C")
// ============================================================================

// Pass 1: Count how many values per row survive filtering
template<typename T>
__device__ __inline__ void filter_csr_count_impl(
    const long long* row_ptrs,
    const T* values,
    int* row_counts,
    unsigned int nrows,
    T threshold
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        long long row_start = row_ptrs[row];
        long long row_end = row_ptrs[row + 1];

        int count = 0;
        for (long long i = row_start; i < row_end; i++) {
            T val = values[i];
            // Use abs for comparison (cast 0 to T for F16/BF16 compatibility)
            T zero = T(0);
            T abs_val = val >= zero ? val : -val;
            if (abs_val >= threshold) {
                count++;
            }
        }

        row_counts[row] = count;
    }
}

// Pass 2: Copy filtered values and indices
template<typename T>
__device__ __inline__ void filter_csr_compute_impl(
    const long long* row_ptrs,
    const long long* col_indices,
    const T* values,
    const long long* out_row_ptrs,
    long long* out_col_indices,
    T* out_values,
    unsigned int nrows,
    T threshold
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        long long in_row_start = row_ptrs[row];
        long long in_row_end = row_ptrs[row + 1];
        long long out_row_start = out_row_ptrs[row];

        long long out_idx = out_row_start;
        for (long long i = in_row_start; i < in_row_end; i++) {
            T val = values[i];
            T zero = T(0);
            T abs_val = val >= zero ? val : -val;
            if (abs_val >= threshold) {
                out_col_indices[out_idx] = col_indices[i];
                out_values[out_idx] = val;
                out_idx++;
            }
        }
    }
}

// ============================================================================
// Row/Column Sums
// ============================================================================

template<typename T>
__device__ __inline__ void csr_sum_rows_impl(
    const long long* row_ptrs,
    const T* values,
    T* out,
    unsigned int nrows
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        long long row_start = row_ptrs[row];
        long long row_end = row_ptrs[row + 1];

        T sum = 0;
        for (long long i = row_start; i < row_end; i++) {
            sum += values[i];
        }

        out[row] = sum;
    }
}

template<typename T>
__device__ __inline__ void csc_sum_cols_impl(
    const long long* col_ptrs,
    const T* values,
    T* out,
    unsigned int ncols
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ncols) {
        long long col_start = col_ptrs[col];
        long long col_end = col_ptrs[col + 1];

        T sum = 0;
        for (long long i = col_start; i < col_end; i++) {
            sum += values[i];
        }

        out[col] = sum;
    }
}

// ============================================================================
// Sparse to Dense Conversion
// ============================================================================

template<typename T>
__device__ __inline__ void csr_to_dense_impl(
    const long long* row_ptrs,
    const long long* col_indices,
    const T* values,
    T* out,
    unsigned int nrows,
    unsigned int ncols
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        // Initialize row to zero
        for (unsigned int col = 0; col < ncols; col++) {
            out[row * ncols + col] = 0;
        }

        // Fill in non-zero values
        long long row_start = row_ptrs[row];
        long long row_end = row_ptrs[row + 1];
        for (long long i = row_start; i < row_end; i++) {
            unsigned int col = col_indices[i];
            out[row * ncols + col] = values[i];
        }
    }
}

// ============================================================================
// Dense to COO Conversion (2-pass)
// ============================================================================

// Pass 1: Count non-zeros (returns count per row)
template<typename T>
__device__ __inline__ void dense_to_coo_count_impl(
    const T* input,
    int* row_counts,
    unsigned int nrows,
    unsigned int ncols,
    T threshold
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        int count = 0;
        for (unsigned int col = 0; col < ncols; col++) {
            T val = input[row * ncols + col];
            T zero = T(0);
            T abs_val = val >= zero ? val : -val;
            if (abs_val >= threshold) {
                count++;
            }
        }
        row_counts[row] = count;
    }
}

// Pass 2: Extract COO data
template<typename T>
__device__ __inline__ void dense_to_coo_extract_impl(
    const T* input,
    const long long* offsets,  // From exclusive scan of row_counts
    long long* row_indices,
    long long* col_indices,
    T* values,
    unsigned int nrows,
    unsigned int ncols,
    T threshold
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        long long out_idx = offsets[row];

        for (unsigned int col = 0; col < ncols; col++) {
            T val = input[row * ncols + col];
            T zero = T(0);
            T abs_val = val >= zero ? val : -val;
            if (abs_val >= threshold) {
                row_indices[out_idx] = row;
                col_indices[out_idx] = col;
                values[out_idx] = val;
                out_idx++;
            }
        }
    }
}

// ============================================================================
// extern "C" wrappers for Rust FFI
// ============================================================================

extern "C" {

// CSR filter count wrappers
__global__ void filter_csr_count_f32(const long long* row_ptrs, const float* values, int* row_counts, unsigned int nrows, float threshold) {
    filter_csr_count_impl<float>(row_ptrs, values, row_counts, nrows, threshold);
}

__global__ void filter_csr_count_f64(const long long* row_ptrs, const double* values, int* row_counts, unsigned int nrows, double threshold) {
    filter_csr_count_impl<double>(row_ptrs, values, row_counts, nrows, threshold);
}

__global__ void filter_csr_count_f16(const long long* row_ptrs, const __half* values, int* row_counts, unsigned int nrows, __half threshold) {
    filter_csr_count_impl<__half>(row_ptrs, values, row_counts, nrows, threshold);
}

__global__ void filter_csr_count_bf16(const long long* row_ptrs, const __nv_bfloat16* values, int* row_counts, unsigned int nrows, __nv_bfloat16 threshold) {
    filter_csr_count_impl<__nv_bfloat16>(row_ptrs, values, row_counts, nrows, threshold);
}

// CSR filter compute wrappers
__global__ void filter_csr_compute_f32(const long long* row_ptrs, const long long* col_indices, const float* values, const long long* out_row_ptrs, long long* out_col_indices, float* out_values, unsigned int nrows, float threshold) {
    filter_csr_compute_impl<float>(row_ptrs, col_indices, values, out_row_ptrs, out_col_indices, out_values, nrows, threshold);
}

__global__ void filter_csr_compute_f64(const long long* row_ptrs, const long long* col_indices, const double* values, const long long* out_row_ptrs, long long* out_col_indices, double* out_values, unsigned int nrows, double threshold) {
    filter_csr_compute_impl<double>(row_ptrs, col_indices, values, out_row_ptrs, out_col_indices, out_values, nrows, threshold);
}

__global__ void filter_csr_compute_f16(const long long* row_ptrs, const long long* col_indices, const __half* values, const long long* out_row_ptrs, long long* out_col_indices, __half* out_values, unsigned int nrows, __half threshold) {
    filter_csr_compute_impl<__half>(row_ptrs, col_indices, values, out_row_ptrs, out_col_indices, out_values, nrows, threshold);
}

__global__ void filter_csr_compute_bf16(const long long* row_ptrs, const long long* col_indices, const __nv_bfloat16* values, const long long* out_row_ptrs, long long* out_col_indices, __nv_bfloat16* out_values, unsigned int nrows, __nv_bfloat16 threshold) {
    filter_csr_compute_impl<__nv_bfloat16>(row_ptrs, col_indices, values, out_row_ptrs, out_col_indices, out_values, nrows, threshold);
}

// CSR sum rows wrappers
__global__ void csr_sum_rows_f32(const long long* row_ptrs, const float* values, float* out, unsigned int nrows) {
    csr_sum_rows_impl<float>(row_ptrs, values, out, nrows);
}

__global__ void csr_sum_rows_f64(const long long* row_ptrs, const double* values, double* out, unsigned int nrows) {
    csr_sum_rows_impl<double>(row_ptrs, values, out, nrows);
}

__global__ void csr_sum_rows_f16(const long long* row_ptrs, const __half* values, __half* out, unsigned int nrows) {
    csr_sum_rows_impl<__half>(row_ptrs, values, out, nrows);
}

__global__ void csr_sum_rows_bf16(const long long* row_ptrs, const __nv_bfloat16* values, __nv_bfloat16* out, unsigned int nrows) {
    csr_sum_rows_impl<__nv_bfloat16>(row_ptrs, values, out, nrows);
}

// CSC sum cols wrappers
__global__ void csc_sum_cols_f32(const long long* col_ptrs, const float* values, float* out, unsigned int ncols) {
    csc_sum_cols_impl<float>(col_ptrs, values, out, ncols);
}

__global__ void csc_sum_cols_f64(const long long* col_ptrs, const double* values, double* out, unsigned int ncols) {
    csc_sum_cols_impl<double>(col_ptrs, values, out, ncols);
}

__global__ void csc_sum_cols_f16(const long long* col_ptrs, const __half* values, __half* out, unsigned int ncols) {
    csc_sum_cols_impl<__half>(col_ptrs, values, out, ncols);
}

__global__ void csc_sum_cols_bf16(const long long* col_ptrs, const __nv_bfloat16* values, __nv_bfloat16* out, unsigned int ncols) {
    csc_sum_cols_impl<__nv_bfloat16>(col_ptrs, values, out, ncols);
}

// NNZ per row/col kernels (no template needed)
__global__ void csr_nnz_per_row_kernel(
    const long long* row_ptrs,
    long long* out,
    unsigned int nrows
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        out[row] = row_ptrs[row + 1] - row_ptrs[row];
    }
}

__global__ void csc_nnz_per_col_kernel(
    const long long* col_ptrs,
    long long* out,
    unsigned int ncols
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ncols) {
        out[col] = col_ptrs[col + 1] - col_ptrs[col];
    }
}

// CSR to dense wrappers
__global__ void csr_to_dense_f32(const long long* row_ptrs, const long long* col_indices, const float* values, float* out, unsigned int nrows, unsigned int ncols) {
    csr_to_dense_impl<float>(row_ptrs, col_indices, values, out, nrows, ncols);
}

__global__ void csr_to_dense_f64(const long long* row_ptrs, const long long* col_indices, const double* values, double* out, unsigned int nrows, unsigned int ncols) {
    csr_to_dense_impl<double>(row_ptrs, col_indices, values, out, nrows, ncols);
}

__global__ void csr_to_dense_f16(const long long* row_ptrs, const long long* col_indices, const __half* values, __half* out, unsigned int nrows, unsigned int ncols) {
    csr_to_dense_impl<__half>(row_ptrs, col_indices, values, out, nrows, ncols);
}

__global__ void csr_to_dense_bf16(const long long* row_ptrs, const long long* col_indices, const __nv_bfloat16* values, __nv_bfloat16* out, unsigned int nrows, unsigned int ncols) {
    csr_to_dense_impl<__nv_bfloat16>(row_ptrs, col_indices, values, out, nrows, ncols);
}

// Dense to COO count wrappers
__global__ void dense_to_coo_count_f32(const float* input, int* row_counts, unsigned int nrows, unsigned int ncols, float threshold) {
    dense_to_coo_count_impl<float>(input, row_counts, nrows, ncols, threshold);
}

__global__ void dense_to_coo_count_f64(const double* input, int* row_counts, unsigned int nrows, unsigned int ncols, double threshold) {
    dense_to_coo_count_impl<double>(input, row_counts, nrows, ncols, threshold);
}

__global__ void dense_to_coo_count_f16(const __half* input, int* row_counts, unsigned int nrows, unsigned int ncols, __half threshold) {
    dense_to_coo_count_impl<__half>(input, row_counts, nrows, ncols, threshold);
}

__global__ void dense_to_coo_count_bf16(const __nv_bfloat16* input, int* row_counts, unsigned int nrows, unsigned int ncols, __nv_bfloat16 threshold) {
    dense_to_coo_count_impl<__nv_bfloat16>(input, row_counts, nrows, ncols, threshold);
}

// Dense to COO extract wrappers
__global__ void dense_to_coo_extract_f32(const float* input, const long long* offsets, long long* row_indices, long long* col_indices, float* values, unsigned int nrows, unsigned int ncols, float threshold) {
    dense_to_coo_extract_impl<float>(input, offsets, row_indices, col_indices, values, nrows, ncols, threshold);
}

__global__ void dense_to_coo_extract_f64(const double* input, const long long* offsets, long long* row_indices, long long* col_indices, double* values, unsigned int nrows, unsigned int ncols, double threshold) {
    dense_to_coo_extract_impl<double>(input, offsets, row_indices, col_indices, values, nrows, ncols, threshold);
}

__global__ void dense_to_coo_extract_f16(const __half* input, const long long* offsets, long long* row_indices, long long* col_indices, __half* values, unsigned int nrows, unsigned int ncols, __half threshold) {
    dense_to_coo_extract_impl<__half>(input, offsets, row_indices, col_indices, values, nrows, ncols, threshold);
}

__global__ void dense_to_coo_extract_bf16(const __nv_bfloat16* input, const long long* offsets, long long* row_indices, long long* col_indices, __nv_bfloat16* values, unsigned int nrows, unsigned int ncols, __nv_bfloat16 threshold) {
    dense_to_coo_extract_impl<__nv_bfloat16>(input, offsets, row_indices, col_indices, values, nrows, ncols, threshold);
}

} // extern "C"
