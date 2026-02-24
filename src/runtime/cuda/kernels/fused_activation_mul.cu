// Fused activation-mul CUDA kernels
// Forward: output = activation(a) * b
// Supports: silu_mul, gelu_mul, relu_mul, sigmoid_mul
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Helper device functions (shared across dtypes)
// ============================================================================

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float gelu_f(float x) {
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    return x * cdf;
}

__device__ __forceinline__ float relu_f(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ double silu_d(double x) {
    return x / (1.0 + exp(-x));
}

__device__ __forceinline__ double gelu_d(double x) {
    double cdf = 0.5 * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
    return x * cdf;
}

__device__ __forceinline__ double relu_d(double x) {
    return fmax(0.0, x);
}

__device__ __forceinline__ double sigmoid_d(double x) {
    return 1.0 / (1.0 + exp(-x));
}

extern "C" {

// ============================================================================
// F32 Fused Activation-Mul Forward
// ============================================================================

__global__ void silu_mul_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = silu_f(a[idx]) * b[idx];
    }
}

__global__ void gelu_mul_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = gelu_f(a[idx]) * b[idx];
    }
}

__global__ void relu_mul_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = relu_f(a[idx]) * b[idx];
    }
}

__global__ void sigmoid_mul_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sigmoid_f(a[idx]) * b[idx];
    }
}

// ============================================================================
// F64 Fused Activation-Mul Forward
// ============================================================================

__global__ void silu_mul_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = silu_d(a[idx]) * b[idx];
    }
}

__global__ void gelu_mul_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = gelu_d(a[idx]) * b[idx];
    }
}

__global__ void relu_mul_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = relu_d(a[idx]) * b[idx];
    }
}

__global__ void sigmoid_mul_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sigmoid_d(a[idx]) * b[idx];
    }
}

// ============================================================================
// F16 Fused Activation-Mul Forward (compute in F32)
// ============================================================================

__global__ void silu_mul_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = __half2float(a[idx]);
        float bx = __half2float(b[idx]);
        out[idx] = __float2half(silu_f(ax) * bx);
    }
}

__global__ void gelu_mul_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = __half2float(a[idx]);
        float bx = __half2float(b[idx]);
        out[idx] = __float2half(gelu_f(ax) * bx);
    }
}

__global__ void relu_mul_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = __half2float(a[idx]);
        float bx = __half2float(b[idx]);
        out[idx] = __float2half(relu_f(ax) * bx);
    }
}

__global__ void sigmoid_mul_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = __half2float(a[idx]);
        float bx = __half2float(b[idx]);
        out[idx] = __float2half(sigmoid_f(ax) * bx);
    }
}

// ============================================================================
// BF16 Fused Activation-Mul Forward (compute in F32)
// ============================================================================

__global__ void silu_mul_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = __bfloat162float(a[idx]);
        float bx = __bfloat162float(b[idx]);
        out[idx] = __float2bfloat16(silu_f(ax) * bx);
    }
}

__global__ void gelu_mul_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = __bfloat162float(a[idx]);
        float bx = __bfloat162float(b[idx]);
        out[idx] = __float2bfloat16(gelu_f(ax) * bx);
    }
}

__global__ void relu_mul_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = __bfloat162float(a[idx]);
        float bx = __bfloat162float(b[idx]);
        out[idx] = __float2bfloat16(relu_f(ax) * bx);
    }
}

__global__ void sigmoid_mul_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = __bfloat162float(a[idx]);
        float bx = __bfloat162float(b[idx]);
        out[idx] = __float2bfloat16(sigmoid_f(ax) * bx);
    }
}

// ============================================================================
// FP8 E4M3 Fused Activation-Mul Forward (compute in F32)
// ============================================================================

__global__ void silu_mul_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = fp8_e4m3_to_f32(a[idx].data);
        float bx = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(silu_f(ax) * bx));
    }
}

__global__ void gelu_mul_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = fp8_e4m3_to_f32(a[idx].data);
        float bx = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(gelu_f(ax) * bx));
    }
}

__global__ void relu_mul_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = fp8_e4m3_to_f32(a[idx].data);
        float bx = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(relu_f(ax) * bx));
    }
}

__global__ void sigmoid_mul_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = fp8_e4m3_to_f32(a[idx].data);
        float bx = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(sigmoid_f(ax) * bx));
    }
}

// ============================================================================
// FP8 E5M2 Fused Activation-Mul Forward (compute in F32)
// ============================================================================

__global__ void silu_mul_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = fp8_e5m2_to_f32(a[idx].data);
        float bx = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(silu_f(ax) * bx));
    }
}

__global__ void gelu_mul_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = fp8_e5m2_to_f32(a[idx].data);
        float bx = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(gelu_f(ax) * bx));
    }
}

__global__ void relu_mul_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = fp8_e5m2_to_f32(a[idx].data);
        float bx = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(relu_f(ax) * bx));
    }
}

__global__ void sigmoid_mul_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float ax = fp8_e5m2_to_f32(a[idx].data);
        float bx = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(sigmoid_f(ax) * bx));
    }
}

} // extern "C"
