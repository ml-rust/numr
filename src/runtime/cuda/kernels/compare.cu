// Comparison CUDA kernels
// Supports: eq, ne, lt, le, gt, ge
// Types: f32, f64, f16, bf16, i32, i64
// Output: same type as input (1 for true, 0 for false)
//
// NOTE: Same-dtype output is intentional - it allows using comparison results
// directly in arithmetic operations (e.g., mask * tensor) without dtype conversion.
// Includes broadcast variants for all types

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Broadcast Helper Device Functions (Templated)
// ============================================================================

template<typename T>
__device__ __forceinline__ T compare_eq(T a, T b) { return (a == b) ? (T)1 : (T)0; }

template<typename T>
__device__ __forceinline__ T compare_ne(T a, T b) { return (a != b) ? (T)1 : (T)0; }

template<typename T>
__device__ __forceinline__ T compare_lt(T a, T b) { return (a < b) ? (T)1 : (T)0; }

template<typename T>
__device__ __forceinline__ T compare_le(T a, T b) { return (a <= b) ? (T)1 : (T)0; }

template<typename T>
__device__ __forceinline__ T compare_gt(T a, T b) { return (a > b) ? (T)1 : (T)0; }

template<typename T>
__device__ __forceinline__ T compare_ge(T a, T b) { return (a >= b) ? (T)1 : (T)0; }

// Specializations for float
template<>
__device__ __forceinline__ float compare_eq(float a, float b) { return (a == b) ? 1.0f : 0.0f; }

template<>
__device__ __forceinline__ float compare_ne(float a, float b) { return (a != b) ? 1.0f : 0.0f; }

template<>
__device__ __forceinline__ float compare_lt(float a, float b) { return (a < b) ? 1.0f : 0.0f; }

template<>
__device__ __forceinline__ float compare_le(float a, float b) { return (a <= b) ? 1.0f : 0.0f; }

template<>
__device__ __forceinline__ float compare_gt(float a, float b) { return (a > b) ? 1.0f : 0.0f; }

template<>
__device__ __forceinline__ float compare_ge(float a, float b) { return (a >= b) ? 1.0f : 0.0f; }

// Specializations for double
template<>
__device__ __forceinline__ double compare_eq(double a, double b) { return (a == b) ? 1.0 : 0.0; }

template<>
__device__ __forceinline__ double compare_ne(double a, double b) { return (a != b) ? 1.0 : 0.0; }

template<>
__device__ __forceinline__ double compare_lt(double a, double b) { return (a < b) ? 1.0 : 0.0; }

template<>
__device__ __forceinline__ double compare_le(double a, double b) { return (a <= b) ? 1.0 : 0.0; }

template<>
__device__ __forceinline__ double compare_gt(double a, double b) { return (a > b) ? 1.0 : 0.0; }

template<>
__device__ __forceinline__ double compare_ge(double a, double b) { return (a >= b) ? 1.0 : 0.0; }

// Specializations for half
template<>
__device__ __forceinline__ __half compare_eq(__half a, __half b) { return __heq(a, b) ? __float2half(1.0f) : __float2half(0.0f); }

template<>
__device__ __forceinline__ __half compare_ne(__half a, __half b) { return __hne(a, b) ? __float2half(1.0f) : __float2half(0.0f); }

template<>
__device__ __forceinline__ __half compare_lt(__half a, __half b) { return __hlt(a, b) ? __float2half(1.0f) : __float2half(0.0f); }

template<>
__device__ __forceinline__ __half compare_le(__half a, __half b) { return __hle(a, b) ? __float2half(1.0f) : __float2half(0.0f); }

template<>
__device__ __forceinline__ __half compare_gt(__half a, __half b) { return __hgt(a, b) ? __float2half(1.0f) : __float2half(0.0f); }

template<>
__device__ __forceinline__ __half compare_ge(__half a, __half b) { return __hge(a, b) ? __float2half(1.0f) : __float2half(0.0f); }

// Specializations for bfloat16
template<>
__device__ __forceinline__ __nv_bfloat16 compare_eq(__nv_bfloat16 a, __nv_bfloat16 b) {
    #if __CUDA_ARCH__ >= 800
    return __heq(a, b) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
    #else
    return (__bfloat162float(a) == __bfloat162float(b)) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
    #endif
}

template<>
__device__ __forceinline__ __nv_bfloat16 compare_ne(__nv_bfloat16 a, __nv_bfloat16 b) {
    #if __CUDA_ARCH__ >= 800
    return __hne(a, b) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
    #else
    return (__bfloat162float(a) != __bfloat162float(b)) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
    #endif
}

template<>
__device__ __forceinline__ __nv_bfloat16 compare_lt(__nv_bfloat16 a, __nv_bfloat16 b) {
    #if __CUDA_ARCH__ >= 800
    return __hlt(a, b) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
    #else
    return (__bfloat162float(a) < __bfloat162float(b)) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
    #endif
}

template<>
__device__ __forceinline__ __nv_bfloat16 compare_le(__nv_bfloat16 a, __nv_bfloat16 b) {
    #if __CUDA_ARCH__ >= 800
    return __hle(a, b) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
    #else
    return (__bfloat162float(a) <= __bfloat162float(b)) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
    #endif
}

template<>
__device__ __forceinline__ __nv_bfloat16 compare_gt(__nv_bfloat16 a, __nv_bfloat16 b) {
    #if __CUDA_ARCH__ >= 800
    return __hgt(a, b) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
    #else
    return (__bfloat162float(a) > __bfloat162float(b)) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
    #endif
}

template<>
__device__ __forceinline__ __nv_bfloat16 compare_ge(__nv_bfloat16 a, __nv_bfloat16 b) {
    #if __CUDA_ARCH__ >= 800
    return __hge(a, b) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
    #else
    return (__bfloat162float(a) >= __bfloat162float(b)) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
    #endif
}

// Specializations for FP8E4M3 (compare in F32)
template<>
__device__ __forceinline__ numr_fp8_e4m3 compare_eq(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3((fa == fb) ? 1.0f : 0.0f));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 compare_ne(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3((fa != fb) ? 1.0f : 0.0f));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 compare_lt(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3((fa < fb) ? 1.0f : 0.0f));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 compare_le(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3((fa <= fb) ? 1.0f : 0.0f));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 compare_gt(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3((fa > fb) ? 1.0f : 0.0f));
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 compare_ge(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    float fa = fp8_e4m3_to_f32(a.data);
    float fb = fp8_e4m3_to_f32(b.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3((fa >= fb) ? 1.0f : 0.0f));
}

// Specializations for FP8E5M2 (compare in F32)
template<>
__device__ __forceinline__ numr_fp8_e5m2 compare_eq(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2((fa == fb) ? 1.0f : 0.0f));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 compare_ne(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2((fa != fb) ? 1.0f : 0.0f));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 compare_lt(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2((fa < fb) ? 1.0f : 0.0f));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 compare_le(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2((fa <= fb) ? 1.0f : 0.0f));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 compare_gt(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2((fa > fb) ? 1.0f : 0.0f));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 compare_ge(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    float fa = fp8_e5m2_to_f32(a.data);
    float fb = fp8_e5m2_to_f32(b.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2((fa >= fb) ? 1.0f : 0.0f));
}

// Generic broadcast comparison kernel template
template<typename T, typename CompFunc>
__device__ void compare_broadcast_kernel_impl(
    const T* a, const T* b, T* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n,
    CompFunc op
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

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
// F32 Comparison Operations
// ============================================================================

__global__ void eq_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void ne_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] != b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void lt_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void le_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] <= b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void gt_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void ge_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] >= b[idx]) ? 1.0f : 0.0f;
    }
}

// ============================================================================
// F64 Comparison Operations
// ============================================================================

__global__ void eq_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] == b[idx]) ? 1.0 : 0.0;
    }
}

__global__ void ne_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] != b[idx]) ? 1.0 : 0.0;
    }
}

__global__ void lt_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? 1.0 : 0.0;
    }
}

__global__ void le_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] <= b[idx]) ? 1.0 : 0.0;
    }
}

__global__ void gt_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? 1.0 : 0.0;
    }
}

__global__ void ge_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] >= b[idx]) ? 1.0 : 0.0;
    }
}

// ============================================================================
// I32 Comparison Operations
// ============================================================================

__global__ void eq_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] == b[idx]) ? 1 : 0;
    }
}

__global__ void ne_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] != b[idx]) ? 1 : 0;
    }
}

__global__ void lt_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? 1 : 0;
    }
}

__global__ void le_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] <= b[idx]) ? 1 : 0;
    }
}

__global__ void gt_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? 1 : 0;
    }
}

__global__ void ge_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] >= b[idx]) ? 1 : 0;
    }
}

// ============================================================================
// I64 Comparison Operations
// ============================================================================

__global__ void eq_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] == b[idx]) ? 1LL : 0LL;
    }
}

__global__ void ne_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] != b[idx]) ? 1LL : 0LL;
    }
}

__global__ void lt_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? 1LL : 0LL;
    }
}

__global__ void le_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] <= b[idx]) ? 1LL : 0LL;
    }
}

__global__ void gt_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? 1LL : 0LL;
    }
}

__global__ void ge_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] >= b[idx]) ? 1LL : 0LL;
    }
}

// ============================================================================
// F16 Comparison Operations
// ============================================================================

__global__ void eq_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __heq(a[idx], b[idx]) ? __float2half(1.0f) : __float2half(0.0f);
    }
}

__global__ void ne_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hne(a[idx], b[idx]) ? __float2half(1.0f) : __float2half(0.0f);
    }
}

__global__ void lt_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hlt(a[idx], b[idx]) ? __float2half(1.0f) : __float2half(0.0f);
    }
}

__global__ void le_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hle(a[idx], b[idx]) ? __float2half(1.0f) : __float2half(0.0f);
    }
}

__global__ void gt_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hgt(a[idx], b[idx]) ? __float2half(1.0f) : __float2half(0.0f);
    }
}

__global__ void ge_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hge(a[idx], b[idx]) ? __float2half(1.0f) : __float2half(0.0f);
    }
}

// ============================================================================
// BF16 Comparison Operations
// ============================================================================

__global__ void eq_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __heq(a[idx], b[idx]) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #else
        out[idx] = (__bfloat162float(a[idx]) == __bfloat162float(b[idx])) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #endif
    }
}

__global__ void ne_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hne(a[idx], b[idx]) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #else
        out[idx] = (__bfloat162float(a[idx]) != __bfloat162float(b[idx])) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #endif
    }
}

__global__ void lt_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hlt(a[idx], b[idx]) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #else
        out[idx] = (__bfloat162float(a[idx]) < __bfloat162float(b[idx])) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #endif
    }
}

__global__ void le_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hle(a[idx], b[idx]) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #else
        out[idx] = (__bfloat162float(a[idx]) <= __bfloat162float(b[idx])) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #endif
    }
}

__global__ void gt_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hgt(a[idx], b[idx]) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #else
        out[idx] = (__bfloat162float(a[idx]) > __bfloat162float(b[idx])) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #endif
    }
}

__global__ void ge_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hge(a[idx], b[idx]) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #else
        out[idx] = (__bfloat162float(a[idx]) >= __bfloat162float(b[idx])) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #endif
    }
}

// ============================================================================
// Broadcasting Comparison Operations (F32)
// ============================================================================

__global__ void eq_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_eq<float>);
}

__global__ void ne_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_ne<float>);
}

__global__ void lt_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_lt<float>);
}

__global__ void le_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_le<float>);
}

__global__ void gt_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_gt<float>);
}

__global__ void ge_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<float>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_ge<float>);
}

// ============================================================================
// Broadcasting Comparison Operations (F64)
// ============================================================================

__global__ void eq_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_eq<double>);
}

__global__ void ne_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_ne<double>);
}

__global__ void lt_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_lt<double>);
}

__global__ void le_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_le<double>);
}

__global__ void gt_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_gt<double>);
}

__global__ void ge_broadcast_f64(
    const double* a, const double* b, double* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<double>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_ge<double>);
}

// ============================================================================
// Broadcasting Comparison Operations (F16)
// ============================================================================

__global__ void eq_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_eq<__half>);
}

__global__ void ne_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_ne<__half>);
}

__global__ void lt_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_lt<__half>);
}

__global__ void le_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_le<__half>);
}

__global__ void gt_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_gt<__half>);
}

__global__ void ge_broadcast_f16(
    const __half* a, const __half* b, __half* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<__half>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_ge<__half>);
}

// ============================================================================
// Broadcasting Comparison Operations (BF16)
// ============================================================================

__global__ void eq_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_eq<__nv_bfloat16>);
}

__global__ void ne_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_ne<__nv_bfloat16>);
}

__global__ void lt_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_lt<__nv_bfloat16>);
}

__global__ void le_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_le<__nv_bfloat16>);
}

__global__ void gt_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_gt<__nv_bfloat16>);
}

__global__ void ge_broadcast_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<__nv_bfloat16>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_ge<__nv_bfloat16>);
}

// ============================================================================
// Broadcasting Comparison Operations (I32)
// ============================================================================

__global__ void eq_broadcast_i32(
    const int* a, const int* b, int* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<int>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_eq<int>);
}

__global__ void ne_broadcast_i32(
    const int* a, const int* b, int* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<int>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_ne<int>);
}

__global__ void lt_broadcast_i32(
    const int* a, const int* b, int* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<int>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_lt<int>);
}

__global__ void le_broadcast_i32(
    const int* a, const int* b, int* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<int>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_le<int>);
}

__global__ void gt_broadcast_i32(
    const int* a, const int* b, int* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<int>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_gt<int>);
}

__global__ void ge_broadcast_i32(
    const int* a, const int* b, int* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<int>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_ge<int>);
}

// ============================================================================
// Broadcasting Comparison Operations (I64)
// ============================================================================

__global__ void eq_broadcast_i64(
    const long long* a, const long long* b, long long* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<long long>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_eq<long long>);
}

__global__ void ne_broadcast_i64(
    const long long* a, const long long* b, long long* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<long long>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_ne<long long>);
}

__global__ void lt_broadcast_i64(
    const long long* a, const long long* b, long long* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<long long>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_lt<long long>);
}

__global__ void le_broadcast_i64(
    const long long* a, const long long* b, long long* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<long long>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_le<long long>);
}

__global__ void gt_broadcast_i64(
    const long long* a, const long long* b, long long* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<long long>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_gt<long long>);
}

__global__ void ge_broadcast_i64(
    const long long* a, const long long* b, long long* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<long long>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_ge<long long>);
}

// ============================================================================
// Broadcasting Comparison Operations (FP8E4M3)
// ============================================================================

__global__ void eq_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_eq<numr_fp8_e4m3>);
}

__global__ void ne_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_ne<numr_fp8_e4m3>);
}

__global__ void lt_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_lt<numr_fp8_e4m3>);
}

__global__ void le_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_le<numr_fp8_e4m3>);
}

__global__ void gt_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_gt<numr_fp8_e4m3>);
}

__global__ void ge_broadcast_fp8_e4m3(
    const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<numr_fp8_e4m3>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_ge<numr_fp8_e4m3>);
}

// ============================================================================
// Broadcasting Comparison Operations (FP8E5M2)
// ============================================================================

__global__ void eq_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_eq<numr_fp8_e5m2>);
}

__global__ void ne_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_ne<numr_fp8_e5m2>);
}

__global__ void lt_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_lt<numr_fp8_e5m2>);
}

__global__ void le_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_le<numr_fp8_e5m2>);
}

__global__ void gt_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_gt<numr_fp8_e5m2>);
}

__global__ void ge_broadcast_fp8_e5m2(
    const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    compare_broadcast_kernel_impl<numr_fp8_e5m2>(a, b, out, a_strides, b_strides, shape, ndim, n, compare_ge<numr_fp8_e5m2>);
}

} // extern "C"
