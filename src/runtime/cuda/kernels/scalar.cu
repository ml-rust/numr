// Scalar CUDA kernels (tensor-scalar operations)
// Supports: add_scalar, sub_scalar, mul_scalar, div_scalar, pow_scalar
// Types: f32, f64, f16, bf16, i32, i64

#include <cuda_fp16.h>
#include <cuda_bf16.h>

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

} // extern "C"
