// Fused elementwise CUDA kernels
// fused_mul_add: out = a * b + c (FMA)
// fused_add_mul: out = (a + b) * c
// fused_mul_add_scalar: out = a * scale + bias
// Types: f32, f64, f16, bf16

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

extern "C" {

// ============================================================================
// fused_mul_add: out = a * b + c
// ============================================================================

__global__ void fused_mul_add_f32(const float* a, const float* b, const float* c, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaf(a[idx], b[idx], c[idx]);
    }
}

__global__ void fused_mul_add_f64(const double* a, const double* b, const double* c, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fma(a[idx], b[idx], c[idx]);
    }
}

__global__ void fused_mul_add_f16(const __half* a, const __half* b, const __half* c, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        float vc = __half2float(c[idx]);
        out[idx] = __float2half(fmaf(va, vb, vc));
    }
}

__global__ void fused_mul_add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* c, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        float vc = __bfloat162float(c[idx]);
        out[idx] = __float2bfloat16(fmaf(va, vb, vc));
    }
}

// ============================================================================
// fused_add_mul: out = (a + b) * c
// ============================================================================

__global__ void fused_add_mul_f32(const float* a, const float* b, const float* c, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] + b[idx]) * c[idx];
    }
}

__global__ void fused_add_mul_f64(const double* a, const double* b, const double* c, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] + b[idx]) * c[idx];
    }
}

__global__ void fused_add_mul_f16(const __half* a, const __half* b, const __half* c, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        float vc = __half2float(c[idx]);
        out[idx] = __float2half((va + vb) * vc);
    }
}

__global__ void fused_add_mul_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* c, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        float vc = __bfloat162float(c[idx]);
        out[idx] = __float2bfloat16((va + vb) * vc);
    }
}

// ============================================================================
// fused_mul_add_scalar: out = a * scale + bias
// ============================================================================

__global__ void fused_mul_add_scalar_f32(const float* a, float* out, unsigned int n, float scale, float bias) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaf(a[idx], scale, bias);
    }
}

__global__ void fused_mul_add_scalar_f64(const double* a, double* out, unsigned int n, double scale, double bias) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fma(a[idx], scale, bias);
    }
}

__global__ void fused_mul_add_scalar_f16(const __half* a, __half* out, unsigned int n, float scale, float bias) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __half2float(a[idx]);
        out[idx] = __float2half(fmaf(va, scale, bias));
    }
}

__global__ void fused_mul_add_scalar_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n, float scale, float bias) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __bfloat162float(a[idx]);
        out[idx] = __float2bfloat16(fmaf(va, scale, bias));
    }
}

} // extern "C"
