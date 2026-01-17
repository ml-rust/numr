// Ternary CUDA kernels
// Supports: where (conditional select)
// where(cond, x, y) = cond ? x : y
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2, i32, i64
// Condition is always u8 (bool: 0 = false, non-zero = true)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Where Template (must be outside extern "C")
// ============================================================================

template<typename T>
__device__ __forceinline__ T where_impl(unsigned char cond, T x, T y) {
    return cond ? x : y;
}

// ============================================================================
// Where Broadcast Template
// Handles broadcasting for cond, x, and y tensors
// ============================================================================

template<typename T>
__device__ void where_broadcast_impl(
    const unsigned char* cond, const T* x, const T* y, T* out,
    const unsigned int* cond_strides, const unsigned int* x_strides, const unsigned int* y_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int idx
) {
    // Compute offsets for each input based on strides
    unsigned int remaining = idx;
    unsigned int cond_offset = 0;
    unsigned int x_offset = 0;
    unsigned int y_offset = 0;

    for (int d = ndim - 1; d >= 0; d--) {
        unsigned int coord = remaining % shape[d];
        remaining /= shape[d];
        cond_offset += coord * cond_strides[d];
        x_offset += coord * x_strides[d];
        y_offset += coord * y_strides[d];
    }

    out[idx] = where_impl<T>(cond[cond_offset], x[x_offset], y[y_offset]);
}

extern "C" {

// ============================================================================
// Where Operations (element-wise, same shape)
// ============================================================================

__global__ void where_f32(
    const unsigned char* cond, const float* x, const float* y,
    float* out, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = where_impl<float>(cond[idx], x[idx], y[idx]);
    }
}

__global__ void where_f64(
    const unsigned char* cond, const double* x, const double* y,
    double* out, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = where_impl<double>(cond[idx], x[idx], y[idx]);
    }
}

__global__ void where_f16(
    const unsigned char* cond, const __half* x, const __half* y,
    __half* out, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = where_impl<__half>(cond[idx], x[idx], y[idx]);
    }
}

__global__ void where_bf16(
    const unsigned char* cond, const __nv_bfloat16* x, const __nv_bfloat16* y,
    __nv_bfloat16* out, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = where_impl<__nv_bfloat16>(cond[idx], x[idx], y[idx]);
    }
}

__global__ void where_i32(
    const unsigned char* cond, const int* x, const int* y,
    int* out, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = where_impl<int>(cond[idx], x[idx], y[idx]);
    }
}

__global__ void where_i64(
    const unsigned char* cond, const long long* x, const long long* y,
    long long* out, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = where_impl<long long>(cond[idx], x[idx], y[idx]);
    }
}

__global__ void where_fp8_e4m3(
    const unsigned char* cond, const numr_fp8_e4m3* x, const numr_fp8_e4m3* y,
    numr_fp8_e4m3* out, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = where_impl<numr_fp8_e4m3>(cond[idx], x[idx], y[idx]);
    }
}

__global__ void where_fp8_e5m2(
    const unsigned char* cond, const numr_fp8_e5m2* x, const numr_fp8_e5m2* y,
    numr_fp8_e5m2* out, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = where_impl<numr_fp8_e5m2>(cond[idx], x[idx], y[idx]);
    }
}

// ============================================================================
// Where Broadcast Operations (different shapes with broadcasting)
// ============================================================================

__global__ void where_broadcast_f32(
    const unsigned char* cond, const float* x, const float* y, float* out,
    const unsigned int* cond_strides, const unsigned int* x_strides, const unsigned int* y_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        where_broadcast_impl<float>(cond, x, y, out, cond_strides, x_strides, y_strides, shape, ndim, idx);
    }
}

__global__ void where_broadcast_f64(
    const unsigned char* cond, const double* x, const double* y, double* out,
    const unsigned int* cond_strides, const unsigned int* x_strides, const unsigned int* y_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        where_broadcast_impl<double>(cond, x, y, out, cond_strides, x_strides, y_strides, shape, ndim, idx);
    }
}

__global__ void where_broadcast_f16(
    const unsigned char* cond, const __half* x, const __half* y, __half* out,
    const unsigned int* cond_strides, const unsigned int* x_strides, const unsigned int* y_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        where_broadcast_impl<__half>(cond, x, y, out, cond_strides, x_strides, y_strides, shape, ndim, idx);
    }
}

__global__ void where_broadcast_bf16(
    const unsigned char* cond, const __nv_bfloat16* x, const __nv_bfloat16* y, __nv_bfloat16* out,
    const unsigned int* cond_strides, const unsigned int* x_strides, const unsigned int* y_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        where_broadcast_impl<__nv_bfloat16>(cond, x, y, out, cond_strides, x_strides, y_strides, shape, ndim, idx);
    }
}

__global__ void where_broadcast_i32(
    const unsigned char* cond, const int* x, const int* y, int* out,
    const unsigned int* cond_strides, const unsigned int* x_strides, const unsigned int* y_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        where_broadcast_impl<int>(cond, x, y, out, cond_strides, x_strides, y_strides, shape, ndim, idx);
    }
}

__global__ void where_broadcast_i64(
    const unsigned char* cond, const long long* x, const long long* y, long long* out,
    const unsigned int* cond_strides, const unsigned int* x_strides, const unsigned int* y_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        where_broadcast_impl<long long>(cond, x, y, out, cond_strides, x_strides, y_strides, shape, ndim, idx);
    }
}

__global__ void where_broadcast_fp8_e4m3(
    const unsigned char* cond, const numr_fp8_e4m3* x, const numr_fp8_e4m3* y, numr_fp8_e4m3* out,
    const unsigned int* cond_strides, const unsigned int* x_strides, const unsigned int* y_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        where_broadcast_impl<numr_fp8_e4m3>(cond, x, y, out, cond_strides, x_strides, y_strides, shape, ndim, idx);
    }
}

__global__ void where_broadcast_fp8_e5m2(
    const unsigned char* cond, const numr_fp8_e5m2* x, const numr_fp8_e5m2* y, numr_fp8_e5m2* out,
    const unsigned int* cond_strides, const unsigned int* x_strides, const unsigned int* y_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        where_broadcast_impl<numr_fp8_e5m2>(cond, x, y, out, cond_strides, x_strides, y_strides, shape, ndim, idx);
    }
}

} // extern "C"
