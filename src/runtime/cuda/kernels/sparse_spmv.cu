// Sparse matrix operations CUDA kernels
// SpMV (Sparse Matrix-Vector multiplication) and SpMM (Sparse Matrix-Matrix multiplication)
// Formats: CSR (Compressed Sparse Row)
// Types: f32, f64, f16, bf16

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// SpMV: Row-per-thread kernel (good for moderate sparsity)
// ============================================================================

template<typename T>
__device__ T spmv_row(
    int row,
    const int* row_ptrs,
    const int* col_indices,
    const T* values,
    const T* x
) {
    T sum = static_cast<T>(0);
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    for (int j = start; j < end; j++) {
        sum += values[j] * x[col_indices[j]];
    }
    return sum;
}

// Specialization for F16 (accumulate in F32)
template<>
__device__ __half spmv_row<__half>(
    int row,
    const int* row_ptrs,
    const int* col_indices,
    const __half* values,
    const __half* x
) {
    float sum = 0.0f;
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    for (int j = start; j < end; j++) {
        float val = __half2float(values[j]);
        float x_val = __half2float(x[col_indices[j]]);
        sum += val * x_val;
    }
    return __float2half(sum);
}

// Specialization for BF16 (accumulate in F32)
template<>
__device__ __nv_bfloat16 spmv_row<__nv_bfloat16>(
    int row,
    const int* row_ptrs,
    const int* col_indices,
    const __nv_bfloat16* values,
    const __nv_bfloat16* x
) {
    float sum = 0.0f;
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    for (int j = start; j < end; j++) {
        float val = __bfloat162float(values[j]);
        float x_val = __bfloat162float(x[col_indices[j]]);
        sum += val * x_val;
    }
    return __float2bfloat16(sum);
}

template<typename T>
__global__ void csr_spmv_kernel(
    const int* row_ptrs,
    const int* col_indices,
    const T* values,
    const T* x,
    T* y,
    int nrows
) {
    // Grid-stride loop for better scaling
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < nrows;
         row += blockDim.x * gridDim.x) {
        y[row] = spmv_row<T>(row, row_ptrs, col_indices, values, x);
    }
}

// ============================================================================
// SpMV: Warp-level reduction kernel (good for very sparse matrices)
// ============================================================================

template<typename T>
__device__ T warp_reduce_sum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Specialization for F16
template<>
__device__ __half warp_reduce_sum<__half>(__half val) {
    float f_val = __half2float(val);
    for (int offset = 16; offset > 0; offset /= 2) {
        f_val += __shfl_down_sync(0xffffffff, f_val, offset);
    }
    return __float2half(f_val);
}

// Specialization for BF16
template<>
__device__ __nv_bfloat16 warp_reduce_sum<__nv_bfloat16>(__nv_bfloat16 val) {
    float f_val = __bfloat162float(val);
    for (int offset = 16; offset > 0; offset /= 2) {
        f_val += __shfl_down_sync(0xffffffff, f_val, offset);
    }
    return __float2bfloat16(f_val);
}

template<typename T>
__global__ void csr_spmv_warp_kernel(
    const int* row_ptrs,
    const int* col_indices,
    const T* values,
    const T* x,
    T* y,
    int nrows
) {
    int row = blockIdx.x;
    if (row >= nrows) return;

    int lane = threadIdx.x % 32;
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    T sum = static_cast<T>(0);
    for (int j = start + lane; j < end; j += 32) {
        sum += values[j] * x[col_indices[j]];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = sum;
}

// ============================================================================
// SpMM: Sparse Matrix-Dense Matrix multiplication (batched SpMV)
// ============================================================================

template<typename T>
__global__ void csr_spmm_kernel(
    const int* row_ptrs,
    const int* col_indices,
    const T* values,
    const T* B,        // Dense matrix [K, N]
    T* C,              // Output [M, N]
    int nrows,
    int ncols_B
) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= nrows || col >= ncols_B) return;

    T sum = static_cast<T>(0);
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    for (int j = start; j < end; j++) {
        int k = col_indices[j];
        sum += values[j] * B[k * ncols_B + col];
    }

    C[row * ncols_B + col] = sum;
}

// Specialization for F16 (accumulate in F32)
template<>
__global__ void csr_spmm_kernel<__half>(
    const int* row_ptrs,
    const int* col_indices,
    const __half* values,
    const __half* B,
    __half* C,
    int nrows,
    int ncols_B
) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= nrows || col >= ncols_B) return;

    float sum = 0.0f;
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    for (int j = start; j < end; j++) {
        int k = col_indices[j];
        float val = __half2float(values[j]);
        float b_val = __half2float(B[k * ncols_B + col]);
        sum += val * b_val;
    }

    C[row * ncols_B + col] = __float2half(sum);
}

// Specialization for BF16 (accumulate in F32)
template<>
__global__ void csr_spmm_kernel<__nv_bfloat16>(
    const int* row_ptrs,
    const int* col_indices,
    const __nv_bfloat16* values,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int nrows,
    int ncols_B
) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= nrows || col >= ncols_B) return;

    float sum = 0.0f;
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    for (int j = start; j < end; j++) {
        int k = col_indices[j];
        float val = __bfloat162float(values[j]);
        float b_val = __bfloat162float(B[k * ncols_B + col]);
        sum += val * b_val;
    }

    C[row * ncols_B + col] = __float2bfloat16(sum);
}

// ============================================================================
// Extern "C" wrapper kernels for Rust FFI
// ============================================================================

extern "C" {

// SpMV row-per-thread kernels
__global__ void csr_spmv_f32(
    const int* row_ptrs,
    const int* col_indices,
    const float* values,
    const float* x,
    float* y,
    int nrows
) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < nrows;
         row += blockDim.x * gridDim.x) {
        y[row] = spmv_row<float>(row, row_ptrs, col_indices, values, x);
    }
}

__global__ void csr_spmv_f64(
    const int* row_ptrs,
    const int* col_indices,
    const double* values,
    const double* x,
    double* y,
    int nrows
) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < nrows;
         row += blockDim.x * gridDim.x) {
        y[row] = spmv_row<double>(row, row_ptrs, col_indices, values, x);
    }
}

__global__ void csr_spmv_f16(
    const int* row_ptrs,
    const int* col_indices,
    const __half* values,
    const __half* x,
    __half* y,
    int nrows
) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < nrows;
         row += blockDim.x * gridDim.x) {
        y[row] = spmv_row<__half>(row, row_ptrs, col_indices, values, x);
    }
}

__global__ void csr_spmv_bf16(
    const int* row_ptrs,
    const int* col_indices,
    const __nv_bfloat16* values,
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int nrows
) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < nrows;
         row += blockDim.x * gridDim.x) {
        y[row] = spmv_row<__nv_bfloat16>(row, row_ptrs, col_indices, values, x);
    }
}

// SpMV warp-level kernels
__global__ void csr_spmv_warp_f32(
    const int* row_ptrs,
    const int* col_indices,
    const float* values,
    const float* x,
    float* y,
    int nrows
) {
    int row = blockIdx.x;
    if (row >= nrows) return;

    int lane = threadIdx.x % 32;
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    float sum = 0.0f;
    for (int j = start + lane; j < end; j += 32) {
        sum += values[j] * x[col_indices[j]];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = sum;
}

__global__ void csr_spmv_warp_f64(
    const int* row_ptrs,
    const int* col_indices,
    const double* values,
    const double* x,
    double* y,
    int nrows
) {
    int row = blockIdx.x;
    if (row >= nrows) return;

    int lane = threadIdx.x % 32;
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    double sum = 0.0;
    for (int j = start + lane; j < end; j += 32) {
        sum += values[j] * x[col_indices[j]];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = sum;
}

__global__ void csr_spmv_warp_f16(
    const int* row_ptrs,
    const int* col_indices,
    const __half* values,
    const __half* x,
    __half* y,
    int nrows
) {
    int row = blockIdx.x;
    if (row >= nrows) return;

    int lane = threadIdx.x % 32;
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    __half sum = __float2half(0.0f);
    for (int j = start + lane; j < end; j += 32) {
        sum = __hadd(sum, __hmul(values[j], x[col_indices[j]]));
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = sum;
}

__global__ void csr_spmv_warp_bf16(
    const int* row_ptrs,
    const int* col_indices,
    const __nv_bfloat16* values,
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int nrows
) {
    int row = blockIdx.x;
    if (row >= nrows) return;

    int lane = threadIdx.x % 32;
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    __nv_bfloat16 sum = __float2bfloat16(0.0f);
    for (int j = start + lane; j < end; j += 32) {
        sum = __hadd(sum, __hmul(values[j], x[col_indices[j]]));
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = sum;
}

// SpMM kernels
__global__ void csr_spmm_f32(
    const int* row_ptrs,
    const int* col_indices,
    const float* values,
    const float* B,
    float* C,
    int nrows,
    int ncols_B
) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= nrows || col >= ncols_B) return;

    float sum = 0.0f;
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    for (int j = start; j < end; j++) {
        int k = col_indices[j];
        sum += values[j] * B[k * ncols_B + col];
    }

    C[row * ncols_B + col] = sum;
}

__global__ void csr_spmm_f64(
    const int* row_ptrs,
    const int* col_indices,
    const double* values,
    const double* B,
    double* C,
    int nrows,
    int ncols_B
) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= nrows || col >= ncols_B) return;

    double sum = 0.0;
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    for (int j = start; j < end; j++) {
        int k = col_indices[j];
        sum += values[j] * B[k * ncols_B + col];
    }

    C[row * ncols_B + col] = sum;
}

__global__ void csr_spmm_f16(
    const int* row_ptrs,
    const int* col_indices,
    const __half* values,
    const __half* B,
    __half* C,
    int nrows,
    int ncols_B
) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= nrows || col >= ncols_B) return;

    float sum = 0.0f;
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    for (int j = start; j < end; j++) {
        int k = col_indices[j];
        float val = __half2float(values[j]);
        float b_val = __half2float(B[k * ncols_B + col]);
        sum += val * b_val;
    }

    C[row * ncols_B + col] = __float2half(sum);
}

__global__ void csr_spmm_bf16(
    const int* row_ptrs,
    const int* col_indices,
    const __nv_bfloat16* values,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int nrows,
    int ncols_B
) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= nrows || col >= ncols_B) return;

    float sum = 0.0f;
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    for (int j = start; j < end; j++) {
        int k = col_indices[j];
        float val = __bfloat162float(values[j]);
        float b_val = __bfloat162float(B[k * ncols_B + col]);
        sum += val * b_val;
    }

    C[row * ncols_B + col] = __float2bfloat16(sum);
}

} // extern "C"
