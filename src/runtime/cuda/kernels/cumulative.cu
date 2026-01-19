//! Cumulative operations CUDA kernels
//!
//! Implements cumulative sum (cumsum), cumulative product (cumprod),
//! and log-sum-exp (logsumexp) operations.
//!
//! Uses block-level parallel scan for efficiency with large arrays.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Constants
// ============================================================================

#define BLOCK_SIZE 256

// ============================================================================
// Cumulative Sum (Inclusive Scan)
// ============================================================================

// Simple sequential cumsum for small arrays or when scan dimension is last
template<typename T>
__global__ void cumsum_simple_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;

    unsigned int base = outer_idx * scan_size;
    T acc = T(0);
    for (unsigned int i = 0; i < scan_size; i++) {
        acc = acc + input[base + i];
        output[base + i] = acc;
    }
}

// Strided cumsum for non-last dimension
template<typename T>
__global__ void cumsum_strided_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;

    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;

    T acc = T(0);
    for (unsigned int s = 0; s < scan_size; s++) {
        unsigned int offset = outer_idx * scan_size * inner_size + s * inner_size + inner_idx;
        acc = acc + input[offset];
        output[offset] = acc;
    }
}

// ============================================================================
// Cumulative Product (Inclusive Scan)
// ============================================================================

template<typename T>
__global__ void cumprod_simple_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;

    unsigned int base = outer_idx * scan_size;
    T acc = T(1);
    for (unsigned int i = 0; i < scan_size; i++) {
        acc = acc * input[base + i];
        output[base + i] = acc;
    }
}

template<typename T>
__global__ void cumprod_strided_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int scan_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;

    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;

    T acc = T(1);
    for (unsigned int s = 0; s < scan_size; s++) {
        unsigned int offset = outer_idx * scan_size * inner_size + s * inner_size + inner_idx;
        acc = acc * input[offset];
        output[offset] = acc;
    }
}

// ============================================================================
// Log-Sum-Exp (Numerically Stable Reduction)
// ============================================================================

// logsumexp = max(x) + log(sum(exp(x - max(x))))
// This is a reduction operation, not a scan

template<typename T>
__global__ void logsumexp_simple_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int reduce_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;

    unsigned int base = outer_idx * reduce_size;

    // Step 1: Find max
    T max_val = input[base];
    for (unsigned int i = 1; i < reduce_size; i++) {
        T val = input[base + i];
        if (val > max_val) max_val = val;
    }

    // Step 2: Compute sum(exp(x - max))
    T sum = T(0);
    for (unsigned int i = 0; i < reduce_size; i++) {
        sum = sum + exp(float(input[base + i] - max_val));
    }

    // Step 3: Result = max + log(sum)
    output[outer_idx] = max_val + T(log(float(sum)));
}

template<typename T>
__global__ void logsumexp_strided_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    unsigned int reduce_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;

    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;

    // Step 1: Find max along reduce dimension
    unsigned int first_offset = outer_idx * reduce_size * inner_size + inner_idx;
    T max_val = input[first_offset];
    for (unsigned int r = 1; r < reduce_size; r++) {
        unsigned int offset = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
        T val = input[offset];
        if (val > max_val) max_val = val;
    }

    // Step 2: Compute sum(exp(x - max))
    T sum = T(0);
    for (unsigned int r = 0; r < reduce_size; r++) {
        unsigned int offset = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
        sum = sum + exp(float(input[offset] - max_val));
    }

    // Step 3: Write result
    output[outer_idx * inner_size + inner_idx] = max_val + T(log(float(sum)));
}

// ============================================================================
// F64 specializations (use double math)
// ============================================================================

__global__ void logsumexp_simple_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    unsigned int reduce_size,
    unsigned int outer_size
) {
    unsigned int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outer_idx >= outer_size) return;

    unsigned int base = outer_idx * reduce_size;

    double max_val = input[base];
    for (unsigned int i = 1; i < reduce_size; i++) {
        double val = input[base + i];
        if (val > max_val) max_val = val;
    }

    double sum = 0.0;
    for (unsigned int i = 0; i < reduce_size; i++) {
        sum += exp(input[base + i] - max_val);
    }

    output[outer_idx] = max_val + log(sum);
}

__global__ void logsumexp_strided_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    unsigned int reduce_size,
    unsigned int outer_size,
    unsigned int inner_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inner = outer_size * inner_size;
    if (idx >= total_inner) return;

    unsigned int outer_idx = idx / inner_size;
    unsigned int inner_idx = idx % inner_size;

    unsigned int first_offset = outer_idx * reduce_size * inner_size + inner_idx;
    double max_val = input[first_offset];
    for (unsigned int r = 1; r < reduce_size; r++) {
        unsigned int offset = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
        double val = input[offset];
        if (val > max_val) max_val = val;
    }

    double sum = 0.0;
    for (unsigned int r = 0; r < reduce_size; r++) {
        unsigned int offset = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;
        sum += exp(input[offset] - max_val);
    }

    output[outer_idx * inner_size + inner_idx] = max_val + log(sum);
}

// ============================================================================
// Extern "C" Wrapper Kernels
// ============================================================================

extern "C" {

// ===== Cumulative Sum =====

__global__ void cumsum_f32(const float* in, float* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_kernel(in, out, scan_size, outer_size);
}

__global__ void cumsum_f64(const double* in, double* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_kernel(in, out, scan_size, outer_size);
}

__global__ void cumsum_i32(const int* in, int* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_kernel(in, out, scan_size, outer_size);
}

__global__ void cumsum_i64(const long long* in, long long* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_kernel(in, out, scan_size, outer_size);
}

__global__ void cumsum_u32(const unsigned int* in, unsigned int* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_kernel(in, out, scan_size, outer_size);
}

__global__ void cumsum_u64(const unsigned long long* in, unsigned long long* out, unsigned int scan_size, unsigned int outer_size) {
    cumsum_simple_kernel(in, out, scan_size, outer_size);
}

// Strided versions
__global__ void cumsum_strided_f32(const float* in, float* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_kernel(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumsum_strided_f64(const double* in, double* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_kernel(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumsum_strided_i32(const int* in, int* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_kernel(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumsum_strided_i64(const long long* in, long long* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_kernel(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumsum_strided_u32(const unsigned int* in, unsigned int* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_kernel(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumsum_strided_u64(const unsigned long long* in, unsigned long long* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumsum_strided_kernel(in, out, scan_size, outer_size, inner_size);
}

// ===== Cumulative Product =====

__global__ void cumprod_f32(const float* in, float* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_kernel(in, out, scan_size, outer_size);
}

__global__ void cumprod_f64(const double* in, double* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_kernel(in, out, scan_size, outer_size);
}

__global__ void cumprod_i32(const int* in, int* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_kernel(in, out, scan_size, outer_size);
}

__global__ void cumprod_i64(const long long* in, long long* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_kernel(in, out, scan_size, outer_size);
}

__global__ void cumprod_u32(const unsigned int* in, unsigned int* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_kernel(in, out, scan_size, outer_size);
}

__global__ void cumprod_u64(const unsigned long long* in, unsigned long long* out, unsigned int scan_size, unsigned int outer_size) {
    cumprod_simple_kernel(in, out, scan_size, outer_size);
}

// Strided versions
__global__ void cumprod_strided_f32(const float* in, float* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_kernel(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumprod_strided_f64(const double* in, double* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_kernel(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumprod_strided_i32(const int* in, int* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_kernel(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumprod_strided_i64(const long long* in, long long* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_kernel(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumprod_strided_u32(const unsigned int* in, unsigned int* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_kernel(in, out, scan_size, outer_size, inner_size);
}

__global__ void cumprod_strided_u64(const unsigned long long* in, unsigned long long* out, unsigned int scan_size, unsigned int outer_size, unsigned int inner_size) {
    cumprod_strided_kernel(in, out, scan_size, outer_size, inner_size);
}

// ===== Log-Sum-Exp =====

__global__ void logsumexp_f32(const float* in, float* out, unsigned int reduce_size, unsigned int outer_size) {
    logsumexp_simple_kernel(in, out, reduce_size, outer_size);
}

__global__ void logsumexp_f64(const double* in, double* out, unsigned int reduce_size, unsigned int outer_size) {
    logsumexp_simple_f64(in, out, reduce_size, outer_size);
}

// Strided versions
__global__ void logsumexp_strided_f32(const float* in, float* out, unsigned int reduce_size, unsigned int outer_size, unsigned int inner_size) {
    logsumexp_strided_kernel(in, out, reduce_size, outer_size, inner_size);
}

__global__ void logsumexp_strided_f64(const double* in, double* out, unsigned int reduce_size, unsigned int outer_size, unsigned int inner_size) {
    logsumexp_strided_f64(in, out, reduce_size, outer_size, inner_size);
}

} // extern "C"
