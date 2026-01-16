// Scalar CUDA kernels (tensor-scalar operations)
// Supports: add_scalar, sub_scalar, mul_scalar, div_scalar, pow_scalar
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2, i32, i64
//
// FP8 operations compute in FP32 and convert back to FP8 for storage.
// Hopper (SM 8.9+) uses native PTX intrinsics for FP8 conversion.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

extern "C" {

// ============================================================================
// F32 Scalar Operations
// ============================================================================

__global__ void add_scalar_f32(const float* a, float scalar, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

__global__ void sub_scalar_f32(const float* a, float scalar, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - scalar;
    }
}

__global__ void mul_scalar_f32(const float* a, float scalar, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void div_scalar_f32(const float* a, float scalar, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / scalar;
    }
}

__global__ void pow_scalar_f32(const float* a, float scalar, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = powf(a[idx], scalar);
    }
}

// ============================================================================
// F64 Scalar Operations
// ============================================================================

__global__ void add_scalar_f64(const double* a, double scalar, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

__global__ void sub_scalar_f64(const double* a, double scalar, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - scalar;
    }
}

__global__ void mul_scalar_f64(const double* a, double scalar, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void div_scalar_f64(const double* a, double scalar, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / scalar;
    }
}

__global__ void pow_scalar_f64(const double* a, double scalar, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = pow(a[idx], scalar);
    }
}

// ============================================================================
// F16 Scalar Operations (half precision)
// Note: Scalar is passed as float and converted to half for compatibility
// ============================================================================

__global__ void add_scalar_f16(const __half* a, float scalar, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half s = __float2half(scalar);
        out[idx] = __hadd(a[idx], s);
    }
}

__global__ void sub_scalar_f16(const __half* a, float scalar, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half s = __float2half(scalar);
        out[idx] = __hsub(a[idx], s);
    }
}

__global__ void mul_scalar_f16(const __half* a, float scalar, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half s = __float2half(scalar);
        out[idx] = __hmul(a[idx], s);
    }
}

__global__ void div_scalar_f16(const __half* a, float scalar, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half s = __float2half(scalar);
        out[idx] = __hdiv(a[idx], s);
    }
}

__global__ void pow_scalar_f16(const __half* a, float scalar, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use FP32 for pow computation (more accurate)
        float af = __half2float(a[idx]);
        out[idx] = __float2half(powf(af, scalar));
    }
}

// ============================================================================
// BF16 Scalar Operations (bfloat16)
// Note: Scalar is passed as float and converted to bfloat16
// ============================================================================

__global__ void add_scalar_bf16(const __nv_bfloat16* a, float scalar, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        __nv_bfloat16 s = __float2bfloat16(scalar);
        out[idx] = __hadd(a[idx], s);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) + scalar);
        #endif
    }
}

__global__ void sub_scalar_bf16(const __nv_bfloat16* a, float scalar, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        __nv_bfloat16 s = __float2bfloat16(scalar);
        out[idx] = __hsub(a[idx], s);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) - scalar);
        #endif
    }
}

__global__ void mul_scalar_bf16(const __nv_bfloat16* a, float scalar, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        __nv_bfloat16 s = __float2bfloat16(scalar);
        out[idx] = __hmul(a[idx], s);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) * scalar);
        #endif
    }
}

__global__ void div_scalar_bf16(const __nv_bfloat16* a, float scalar, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        __nv_bfloat16 s = __float2bfloat16(scalar);
        out[idx] = __hdiv(a[idx], s);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) / scalar);
        #endif
    }
}

__global__ void pow_scalar_bf16(const __nv_bfloat16* a, float scalar, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use FP32 for pow computation (more accurate)
        float af = __bfloat162float(a[idx]);
        out[idx] = __float2bfloat16(powf(af, scalar));
    }
}

// ============================================================================
// I32 Scalar Operations
// ============================================================================

__global__ void add_scalar_i32(const int* a, int scalar, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

__global__ void sub_scalar_i32(const int* a, int scalar, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - scalar;
    }
}

__global__ void mul_scalar_i32(const int* a, int scalar, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void div_scalar_i32(const int* a, int scalar, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / scalar;
    }
}

// ============================================================================
// I64 Scalar Operations
// ============================================================================

__global__ void add_scalar_i64(const long long* a, long long scalar, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

__global__ void sub_scalar_i64(const long long* a, long long scalar, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - scalar;
    }
}

__global__ void mul_scalar_i64(const long long* a, long long scalar, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

__global__ void div_scalar_i64(const long long* a, long long scalar, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / scalar;
    }
}

// ============================================================================
// FP8 E4M3 Scalar Operations
// Scalar is passed as float, operations compute in FP32
// Hopper (SM 8.9+) uses native PTX intrinsics for conversion
// ============================================================================

__global__ void add_scalar_fp8_e4m3(const numr_fp8_e4m3* a, float scalar, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(af + scalar));
    }
}

__global__ void sub_scalar_fp8_e4m3(const numr_fp8_e4m3* a, float scalar, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(af - scalar));
    }
}

__global__ void mul_scalar_fp8_e4m3(const numr_fp8_e4m3* a, float scalar, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(af * scalar));
    }
}

__global__ void div_scalar_fp8_e4m3(const numr_fp8_e4m3* a, float scalar, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(af / scalar));
    }
}

__global__ void pow_scalar_fp8_e4m3(const numr_fp8_e4m3* a, float scalar, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(powf(af, scalar)));
    }
}

// ============================================================================
// FP8 E5M2 Scalar Operations
// Same pattern as E4M3 but with E5M2 conversion functions
// ============================================================================

__global__ void add_scalar_fp8_e5m2(const numr_fp8_e5m2* a, float scalar, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(af + scalar));
    }
}

__global__ void sub_scalar_fp8_e5m2(const numr_fp8_e5m2* a, float scalar, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(af - scalar));
    }
}

__global__ void mul_scalar_fp8_e5m2(const numr_fp8_e5m2* a, float scalar, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(af * scalar));
    }
}

__global__ void div_scalar_fp8_e5m2(const numr_fp8_e5m2* a, float scalar, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(af / scalar));
    }
}

__global__ void pow_scalar_fp8_e5m2(const numr_fp8_e5m2* a, float scalar, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float af = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(powf(af, scalar)));
    }
}

} // extern "C"
