// Binary element-wise CUDA kernels
// Supports: add, sub, mul, div, pow, max, min, logical_and, logical_or, logical_xor
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2, i32, i64, u8 (bool for logical ops)
// Includes broadcast variants for all arithmetic types

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Broadcast Helper Device Functions (Templated)
// ============================================================================

template<typename T>
__device__ __forceinline__ T broadcast_add(T a, T b) { return a + b; }

template<typename T>
__device__ __forceinline__ T broadcast_sub(T a, T b) { return a - b; }

template<typename T>
__device__ __forceinline__ T broadcast_mul(T a, T b) { return a * b; }

template<typename T>
__device__ __forceinline__ T broadcast_div(T a, T b) { return a / b; }

template<typename T>
__device__ __forceinline__ T broadcast_pow(T a, T b) { return powf((float)a, (float)b); }

template<typename T>
__device__ __forceinline__ T broadcast_max(T a, T b) { return a > b ? a : b; }

template<typename T>
__device__ __forceinline__ T broadcast_min(T a, T b) { return a < b ? a : b; }

// Specializations for half precision
template<>
__device__ __forceinline__ __half broadcast_pow(__half a, __half b) {
    return __float2half(powf(__half2float(a), __half2float(b)));
}

// Specializations for bfloat16
template<>
__device__ __forceinline__ __nv_bfloat16 broadcast_pow(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16(powf(__bfloat162float(a), __bfloat162float(b)));
}

// Specializations for integers
template<>
__device__ __forceinline__ int32_t broadcast_pow(int32_t a, int32_t b) {
    return (int32_t)powf((float)a, (float)b);
}

template<>
__device__ __forceinline__ int64_t broadcast_pow(int64_t a, int64_t b) {
    return (int64_t)powf((float)a, (float)b);
}

// Specializations for FP8E4M3 (compute in F32)
template<>
__device__ __forceinline__ numr_fp8_e4m3 broadcast_add(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3(fa + fb));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 broadcast_sub(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3(fa - fb));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 broadcast_mul(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3(fa * fb));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 broadcast_div(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3(fa / fb));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 broadcast_pow(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3(powf(fa, fb)));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 broadcast_max(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3(fmaxf(fa, fb)));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 broadcast_min(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3(fminf(fa, fb)));
}

// Specializations for FP8E5M2 (compute in F32)
template<>
__device__ __forceinline__ numr_fp8_e5m2 broadcast_add(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2(fa + fb));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 broadcast_sub(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2(fa - fb));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 broadcast_mul(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2(fa * fb));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 broadcast_div(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2(fa / fb));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 broadcast_pow(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2(powf(fa, fb)));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 broadcast_max(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2(fmaxf(fa, fb)));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 broadcast_min(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2(fminf(fa, fb)));
}

// Generic broadcast kernel template
template<typename T, typename OpFunc>
__device__ void broadcast_kernel_impl(
    const T* a, const T* b, T* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n,
    OpFunc op
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Convert linear index to multi-dimensional indices
    unsigned int remaining = idx;
    unsigned int a_offset = 0, b_offset = 0;

    for (int d = ndim - 1; d >= 0; d--) {
        unsigned int coord = remaining % shape[d];
        remaining /= shape[d];
        a_offset += coord * a_strides[d];
        b_offset += coord * b_strides[d];
    }

    out[idx] = op(a[a_offset], b[b_offset]);
}

extern "C" {

// ============================================================================
// F32 Binary Operations
// ============================================================================

__global__ void add_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void div_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void pow_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = powf(a[idx], b[idx]);
    }
}

__global__ void max_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaxf(a[idx], b[idx]);
    }
}

__global__ void min_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fminf(a[idx], b[idx]);
    }
}

__global__ void atan2_f32(const float* y, const float* x, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = atan2f(y[idx], x[idx]);
    }
}

// ============================================================================
// F16 Binary Operations (half precision)
// Uses half2 vectorization for 2x throughput where possible
// ============================================================================

__global__ void add_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hadd(a[idx], b[idx]);
    }
}

__global__ void sub_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hsub(a[idx], b[idx]);
    }
}

__global__ void mul_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hmul(a[idx], b[idx]);
    }
}

__global__ void div_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hdiv(a[idx], b[idx]);
    }
}

__global__ void pow_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use FP32 for pow computation (more accurate)
        float af = __half2float(a[idx]);
        float bf = __half2float(b[idx]);
        out[idx] = __float2half(powf(af, bf));
    }
}

__global__ void max_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hgt(a[idx], b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void min_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hlt(a[idx], b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void atan2_f16(const __half* y, const __half* x, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(atan2f(__half2float(y[idx]), __half2float(x[idx])));
    }
}

// ============================================================================
// BF16 Binary Operations (bfloat16)
// Note: BF16 arithmetic requires SM 8.0+ (Ampere)
// ============================================================================

__global__ void add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hadd(a[idx], b[idx]);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
        #endif
    }
}

__global__ void sub_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hsub(a[idx], b[idx]);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) - __bfloat162float(b[idx]));
        #endif
    }
}

__global__ void mul_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hmul(a[idx], b[idx]);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) * __bfloat162float(b[idx]));
        #endif
    }
}

__global__ void div_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hdiv(a[idx], b[idx]);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) / __bfloat162float(b[idx]));
        #endif
    }
}

__global__ void pow_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use FP32 for pow computation (more accurate)
        float af = __bfloat162float(a[idx]);
        float bf = __bfloat162float(b[idx]);
        out[idx] = __float2bfloat16(powf(af, bf));
    }
}

__global__ void max_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hgt(a[idx], b[idx]) ? a[idx] : b[idx];
        #else
        out[idx] = (__bfloat162float(a[idx]) > __bfloat162float(b[idx])) ? a[idx] : b[idx];
        #endif
    }
}

__global__ void min_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hlt(a[idx], b[idx]) ? a[idx] : b[idx];
        #else
        out[idx] = (__bfloat162float(a[idx]) < __bfloat162float(b[idx])) ? a[idx] : b[idx];
        #endif
    }
}

__global__ void atan2_bf16(const __nv_bfloat16* y, const __nv_bfloat16* x, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(atan2f(__bfloat162float(y[idx]), __bfloat162float(x[idx])));
    }
}

// ============================================================================
// F64 Binary Operations
// ============================================================================

__global__ void add_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void div_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void pow_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = pow(a[idx], b[idx]);
    }
}

__global__ void max_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmax(a[idx], b[idx]);
    }
}

__global__ void min_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmin(a[idx], b[idx]);
    }
}

__global__ void atan2_f64(const double* y, const double* x, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = atan2(y[idx], x[idx]);
    }
}

// ============================================================================
// I32 Binary Operations
// ============================================================================

__global__ void add_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void div_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void max_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void min_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? a[idx] : b[idx];
    }
}

// ============================================================================
// I64 Binary Operations
// ============================================================================

__global__ void add_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void div_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void max_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void min_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? a[idx] : b[idx];
    }
}

// ============================================================================
// Broadcasting Binary Operations (strided access)
// ============================================================================

// ============================================================================
// Broadcast Kernels - Templated for all dtypes
// Thin wrappers call templated device functions
// ============================================================================

// F32 broadcast kernels
__global__ void add_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_add<float>);
}

__global__ void sub_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_sub<float>);
}

__global__ void mul_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_mul<float>);
}

__global__ void div_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_div<float>);
}

__global__ void pow_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_pow<float>);
}

__global__ void max_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_max<float>);
}

__global__ void min_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_min<float>);
}

// F64 broadcast kernels
__global__ void add_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_add<double>);
}

__global__ void sub_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_sub<double>);
}

__global__ void mul_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_mul<double>);
}

__global__ void div_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_div<double>);
}

__global__ void pow_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_pow<double>);
}

__global__ void max_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_max<double>);
}

__global__ void min_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_min<double>);
}

// F16 broadcast kernels
__global__ void add_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_add<__half>);
}

__global__ void sub_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_sub<__half>);
}

__global__ void mul_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_mul<__half>);
}

__global__ void div_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_div<__half>);
}

__global__ void pow_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_pow<__half>);
}

__global__ void max_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_max<__half>);
}

__global__ void min_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_min<__half>);
}

// BF16 broadcast kernels
__global__ void add_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_add<__nv_bfloat16>);
}

__global__ void sub_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_sub<__nv_bfloat16>);
}

__global__ void mul_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_mul<__nv_bfloat16>);
}

__global__ void div_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_div<__nv_bfloat16>);
}

__global__ void pow_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_pow<__nv_bfloat16>);
}

__global__ void max_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_max<__nv_bfloat16>);
}

__global__ void min_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_min<__nv_bfloat16>);
}

// I32 broadcast kernels
__global__ void add_broadcast_i32(
    const int32_t* a, const int32_t* b, int32_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int32_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_add<int32_t>);
}

__global__ void sub_broadcast_i32(
    const int32_t* a, const int32_t* b, int32_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int32_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_sub<int32_t>);
}

__global__ void mul_broadcast_i32(
    const int32_t* a, const int32_t* b, int32_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int32_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_mul<int32_t>);
}

__global__ void div_broadcast_i32(
    const int32_t* a, const int32_t* b, int32_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int32_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_div<int32_t>);
}

__global__ void pow_broadcast_i32(
    const int32_t* a, const int32_t* b, int32_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int32_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_pow<int32_t>);
}

__global__ void max_broadcast_i32(
    const int32_t* a, const int32_t* b, int32_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int32_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_max<int32_t>);
}

__global__ void min_broadcast_i32(
    const int32_t* a, const int32_t* b, int32_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int32_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_min<int32_t>);
}

// I64 broadcast kernels
__global__ void add_broadcast_i64(
    const int64_t* a, const int64_t* b, int64_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int64_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_add<int64_t>);
}

__global__ void sub_broadcast_i64(
    const int64_t* a, const int64_t* b, int64_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int64_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_sub<int64_t>);
}

__global__ void mul_broadcast_i64(
    const int64_t* a, const int64_t* b, int64_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int64_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_mul<int64_t>);
}

__global__ void div_broadcast_i64(
    const int64_t* a, const int64_t* b, int64_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int64_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_div<int64_t>);
}

__global__ void pow_broadcast_i64(
    const int64_t* a, const int64_t* b, int64_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int64_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_pow<int64_t>);
}

__global__ void max_broadcast_i64(
    const int64_t* a, const int64_t* b, int64_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int64_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_max<int64_t>);
}

__global__ void min_broadcast_i64(
    const int64_t* a, const int64_t* b, int64_t* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<int64_t>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_min<int64_t>);
}

// FP8E4M3 broadcast kernels
__global__ void add_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_add<numr_fp8_e4m3>);
}

__global__ void sub_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_sub<numr_fp8_e4m3>);
}

__global__ void mul_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_mul<numr_fp8_e4m3>);
}

__global__ void div_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_div<numr_fp8_e4m3>);
}

__global__ void pow_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_pow<numr_fp8_e4m3>);
}

__global__ void max_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_max<numr_fp8_e4m3>);
}

__global__ void min_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_min<numr_fp8_e4m3>);
}

// FP8E5M2 broadcast kernels
__global__ void add_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_add<numr_fp8_e5m2>);
}

__global__ void sub_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_sub<numr_fp8_e5m2>);
}

__global__ void mul_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_mul<numr_fp8_e5m2>);
}

__global__ void div_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_div<numr_fp8_e5m2>);
}

__global__ void pow_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_pow<numr_fp8_e5m2>);
}

__global__ void max_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_max<numr_fp8_e5m2>);
}

__global__ void min_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, broadcast_min<numr_fp8_e5m2>);
}

// ============================================================================
// FP8 E4M3 Binary Operations
// All computation done in F32, stored back as FP8
// Uses Hopper PTX intrinsics on SM 8.9+, software emulation on SM 8.0+
// ============================================================================

__global__ void add_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fa + fb));
    }
}

__global__ void sub_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fa - fb));
    }
}

__global__ void mul_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fa * fb));
    }
}

__global__ void div_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fa / fb));
    }
}

__global__ void pow_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(powf(fa, fb)));
    }
}

__global__ void max_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fmaxf(fa, fb)));
    }
}

__global__ void min_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fminf(fa, fb)));
    }
}

__global__ void atan2_fp8_e4m3(const numr_fp8_e4m3* y, const numr_fp8_e4m3* x, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fy = fp8_e4m3_to_f32(y[idx].data);
        float fx = fp8_e4m3_to_f32(x[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(atan2f(fy, fx)));
    }
}

// ============================================================================
// FP8 E5M2 Binary Operations
// ============================================================================

__global__ void add_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fa + fb));
    }
}

__global__ void sub_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fa - fb));
    }
}

__global__ void mul_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fa * fb));
    }
}

__global__ void div_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fa / fb));
    }
}

__global__ void pow_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(powf(fa, fb)));
    }
}

__global__ void max_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fmaxf(fa, fb)));
    }
}

__global__ void min_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fminf(fa, fb)));
    }
}

__global__ void atan2_fp8_e5m2(const numr_fp8_e5m2* y, const numr_fp8_e5m2* x, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fy = fp8_e5m2_to_f32(y[idx].data);
        float fx = fp8_e5m2_to_f32(x[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(atan2f(fy, fx)));
    }
}

// ============================================================================
// Logical AND - Boolean AND operation (input and output are u8)
// ============================================================================

__global__ void logical_and_u8(const unsigned char* a, const unsigned char* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] && b[idx]) ? 1 : 0;
    }
}

// ============================================================================
// Logical OR - Boolean OR operation (input and output are u8)
// ============================================================================

__global__ void logical_or_u8(const unsigned char* a, const unsigned char* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] || b[idx]) ? 1 : 0;
    }
}

// ============================================================================
// Logical XOR - Boolean XOR operation (input and output are u8)
// ============================================================================

__global__ void logical_xor_u8(const unsigned char* a, const unsigned char* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = ((a[idx] != 0) != (b[idx] != 0)) ? 1 : 0;
    }
}

// ============================================================================
// Complex64 (float2) Binary Operations
// ============================================================================

__global__ void add_c64(const float2* a, const float2* b, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_add(a[idx], b[idx]);
    }
}

__global__ void sub_c64(const float2* a, const float2* b, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_sub(a[idx], b[idx]);
    }
}

__global__ void mul_c64(const float2* a, const float2* b, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_mul(a[idx], b[idx]);
    }
}

__global__ void div_c64(const float2* a, const float2* b, float2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex64_div(a[idx], b[idx]);
    }
}

// ============================================================================
// Complex128 (double2) Binary Operations
// ============================================================================

__global__ void add_c128(const double2* a, const double2* b, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_add(a[idx], b[idx]);
    }
}

__global__ void sub_c128(const double2* a, const double2* b, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_sub(a[idx], b[idx]);
    }
}

__global__ void mul_c128(const double2* a, const double2* b, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_mul(a[idx], b[idx]);
    }
}

__global__ void div_c128(const double2* a, const double2* b, double2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = complex128_div(a[idx], b[idx]);
    }
}

} // extern "C"
