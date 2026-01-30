// Basic matrix operations CUDA kernels
// Includes: trace, diag, diagflat, identity, copy, scatter, extract, max_abs, count_above, transpose
//
// Uses C++ templates to eliminate f32/f64 duplication.

#include "dtype_traits.cuh"

// ============================================================================
// Trace - Sum of diagonal elements (parallel reduction)
// ============================================================================

template<typename T>
__device__ void trace_impl(const T* __restrict__ a, T* __restrict__ out,
                           unsigned int n, unsigned int stride) {
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    T val = dtype_traits<T>::zero();
    if (i < n) {
        val = a[i * stride + i];
    }
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        dtype_traits<T>::atomic_add(out, sdata[0]);
    }
}

// ============================================================================
// Diag - Extract diagonal elements
// ============================================================================

template<typename T>
__device__ void diag_impl(const T* __restrict__ a, T* __restrict__ out,
                          unsigned int min_dim, unsigned int n_cols) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < min_dim) {
        out[i] = a[i * n_cols + i];
    }
}

// ============================================================================
// Diagflat - Create diagonal matrix from vector
// ============================================================================

template<typename T>
__device__ void diagflat_impl(const T* __restrict__ diag, T* __restrict__ out,
                              unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * n;
    if (idx < total) {
        unsigned int row = idx / n;
        unsigned int col = idx % n;
        out[idx] = (row == col) ? diag[row] : dtype_traits<T>::zero();
    }
}

// ============================================================================
// Create Identity Matrix
// ============================================================================

template<typename T>
__device__ void create_identity_impl(T* __restrict__ out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * n;
    if (idx < total) {
        unsigned int row = idx / n;
        unsigned int col = idx % n;
        out[idx] = (row == col) ? dtype_traits<T>::one() : dtype_traits<T>::zero();
    }
}

// ============================================================================
// Matrix Copy
// ============================================================================

template<typename T>
__device__ void matrix_copy_impl(const T* __restrict__ src,
                                 T* __restrict__ dst, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// ============================================================================
// Scatter Column - write vector to matrix column
// ============================================================================

template<typename T>
__device__ void scatter_column_impl(const T* __restrict__ vec,
                                    T* __restrict__ matrix,
                                    unsigned int n, unsigned int col) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        matrix[row * n + col] = vec[row];
    }
}

// ============================================================================
// Extract Column
// ============================================================================

template<typename T>
__device__ void extract_column_impl(const T* __restrict__ matrix,
                                    T* __restrict__ col_out,
                                    unsigned int m, unsigned int n_cols,
                                    unsigned int col) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        col_out[row] = matrix[row * n_cols + col];
    }
}

// ============================================================================
// Count Above Threshold - for matrix_rank
// ============================================================================

template<typename T>
__device__ void count_above_threshold_impl(const T* __restrict__ values,
                                           unsigned int* __restrict__ count,
                                           unsigned int n, T threshold) {
    extern __shared__ unsigned int scount[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int local_count = 0;
    if (i < n && dtype_traits<T>::abs(values[i]) > threshold) {
        local_count = 1;
    }
    scount[tid] = local_count;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            scount[tid] += scount[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(count, scount[0]);
    }
}

// ============================================================================
// Max Absolute Value
// ============================================================================

template<typename T>
__device__ void max_abs_impl(const T* __restrict__ values,
                             T* __restrict__ max_val, unsigned int n) {
    extern __shared__ char shared_mem_max[];
    T* smax = reinterpret_cast<T*>(shared_mem_max);

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    T local_max = dtype_traits<T>::zero();
    if (i < n) {
        local_max = dtype_traits<T>::abs(values[i]);
    }
    smax[tid] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smax[tid] = dtype_traits<T>::max(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        dtype_traits<T>::atomic_max(max_val, smax[0]);
    }
}

// ============================================================================
// Matrix Transpose - Optimized with shared memory tiling
// ============================================================================

#define TILE_DIM 32
#define BLOCK_ROWS 8

template<typename T>
__device__ void transpose_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int rows,
    unsigned int cols
) {
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];

    unsigned int x = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
        }
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && (y + j) < cols) {
            output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

#undef TILE_DIM
#undef BLOCK_ROWS

// ============================================================================
// Extern "C" wrappers for PTX export
// ============================================================================

extern "C" {

__global__ void trace_f32(const float* a, float* out, unsigned int n, unsigned int stride) {
    trace_impl<float>(a, out, n, stride);
}

__global__ void trace_f64(const double* a, double* out, unsigned int n, unsigned int stride) {
    trace_impl<double>(a, out, n, stride);
}

__global__ void diag_f32(const float* a, float* out, unsigned int min_dim, unsigned int n_cols) {
    diag_impl<float>(a, out, min_dim, n_cols);
}

__global__ void diag_f64(const double* a, double* out, unsigned int min_dim, unsigned int n_cols) {
    diag_impl<double>(a, out, min_dim, n_cols);
}

__global__ void diagflat_f32(const float* diag, float* out, unsigned int n) {
    diagflat_impl<float>(diag, out, n);
}

__global__ void diagflat_f64(const double* diag, double* out, unsigned int n) {
    diagflat_impl<double>(diag, out, n);
}

__global__ void create_identity_f32(float* out, unsigned int n) {
    create_identity_impl<float>(out, n);
}

__global__ void create_identity_f64(double* out, unsigned int n) {
    create_identity_impl<double>(out, n);
}

__global__ void matrix_copy_f32(const float* src, float* dst, unsigned int n) {
    matrix_copy_impl<float>(src, dst, n);
}

__global__ void matrix_copy_f64(const double* src, double* dst, unsigned int n) {
    matrix_copy_impl<double>(src, dst, n);
}

__global__ void scatter_column_f32(const float* vec, float* matrix, unsigned int n, unsigned int col) {
    scatter_column_impl<float>(vec, matrix, n, col);
}

__global__ void scatter_column_f64(const double* vec, double* matrix, unsigned int n, unsigned int col) {
    scatter_column_impl<double>(vec, matrix, n, col);
}

__global__ void extract_column_f32(const float* matrix, float* col_out, unsigned int m, unsigned int n_cols, unsigned int col) {
    extract_column_impl<float>(matrix, col_out, m, n_cols, col);
}

__global__ void extract_column_f64(const double* matrix, double* col_out, unsigned int m, unsigned int n_cols, unsigned int col) {
    extract_column_impl<double>(matrix, col_out, m, n_cols, col);
}

__global__ void count_above_threshold_f32(const float* values, unsigned int* count, unsigned int n, float threshold) {
    count_above_threshold_impl<float>(values, count, n, threshold);
}

__global__ void count_above_threshold_f64(const double* values, unsigned int* count, unsigned int n, double threshold) {
    count_above_threshold_impl<double>(values, count, n, threshold);
}

__global__ void max_abs_f32(const float* values, float* max_val, unsigned int n) {
    max_abs_impl<float>(values, max_val, n);
}

__global__ void max_abs_f64(const double* values, double* max_val, unsigned int n) {
    max_abs_impl<double>(values, max_val, n);
}

__global__ void transpose_f32(const float* input, float* output, unsigned int rows, unsigned int cols) {
    transpose_impl<float>(input, output, rows, cols);
}

__global__ void transpose_f64(const double* input, double* output, unsigned int rows, unsigned int cols) {
    transpose_impl<double>(input, output, rows, cols);
}

// ============================================================================
// F16 (__half) Wrappers
// ============================================================================

__global__ void trace_f16(const __half* a, __half* out, unsigned int n, unsigned int stride) {
    trace_impl<__half>(a, out, n, stride);
}

__global__ void diag_f16(const __half* a, __half* out, unsigned int min_dim, unsigned int n_cols) {
    diag_impl<__half>(a, out, min_dim, n_cols);
}

__global__ void diagflat_f16(const __half* diag, __half* out, unsigned int n) {
    diagflat_impl<__half>(diag, out, n);
}

__global__ void create_identity_f16(__half* out, unsigned int n) {
    create_identity_impl<__half>(out, n);
}

__global__ void matrix_copy_f16(const __half* src, __half* dst, unsigned int n) {
    matrix_copy_impl<__half>(src, dst, n);
}

__global__ void scatter_column_f16(const __half* vec, __half* matrix, unsigned int n, unsigned int col) {
    scatter_column_impl<__half>(vec, matrix, n, col);
}

__global__ void extract_column_f16(const __half* matrix, __half* col_out, unsigned int m, unsigned int n_cols, unsigned int col) {
    extract_column_impl<__half>(matrix, col_out, m, n_cols, col);
}

__global__ void count_above_threshold_f16(const __half* values, unsigned int* count, unsigned int n, __half threshold) {
    count_above_threshold_impl<__half>(values, count, n, threshold);
}

__global__ void max_abs_f16(const __half* values, __half* max_val, unsigned int n) {
    max_abs_impl<__half>(values, max_val, n);
}

__global__ void transpose_f16(const __half* input, __half* output, unsigned int rows, unsigned int cols) {
    transpose_impl<__half>(input, output, rows, cols);
}

// ============================================================================
// BF16 (__nv_bfloat16) Wrappers
// ============================================================================

__global__ void trace_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n, unsigned int stride) {
    trace_impl<__nv_bfloat16>(a, out, n, stride);
}

__global__ void diag_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int min_dim, unsigned int n_cols) {
    diag_impl<__nv_bfloat16>(a, out, min_dim, n_cols);
}

__global__ void diagflat_bf16(const __nv_bfloat16* diag, __nv_bfloat16* out, unsigned int n) {
    diagflat_impl<__nv_bfloat16>(diag, out, n);
}

__global__ void create_identity_bf16(__nv_bfloat16* out, unsigned int n) {
    create_identity_impl<__nv_bfloat16>(out, n);
}

__global__ void matrix_copy_bf16(const __nv_bfloat16* src, __nv_bfloat16* dst, unsigned int n) {
    matrix_copy_impl<__nv_bfloat16>(src, dst, n);
}

__global__ void scatter_column_bf16(const __nv_bfloat16* vec, __nv_bfloat16* matrix, unsigned int n, unsigned int col) {
    scatter_column_impl<__nv_bfloat16>(vec, matrix, n, col);
}

__global__ void extract_column_bf16(const __nv_bfloat16* matrix, __nv_bfloat16* col_out, unsigned int m, unsigned int n_cols, unsigned int col) {
    extract_column_impl<__nv_bfloat16>(matrix, col_out, m, n_cols, col);
}

__global__ void count_above_threshold_bf16(const __nv_bfloat16* values, unsigned int* count, unsigned int n, __nv_bfloat16 threshold) {
    count_above_threshold_impl<__nv_bfloat16>(values, count, n, threshold);
}

__global__ void max_abs_bf16(const __nv_bfloat16* values, __nv_bfloat16* max_val, unsigned int n) {
    max_abs_impl<__nv_bfloat16>(values, max_val, n);
}

__global__ void transpose_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int rows, unsigned int cols) {
    transpose_impl<__nv_bfloat16>(input, output, rows, cols);
}

} // extern "C"
