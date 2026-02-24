// Element-wise activation CUDA kernels
// Supports: relu, sigmoid, silu, gelu, leaky_relu, elu
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2
//
// Softmax kernels are in softmax.cu

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

extern "C" {

// ============================================================================
// F32 Activation Operations
// ============================================================================

__global__ void relu_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaxf(0.0f, a[idx]);
    }
}

__global__ void sigmoid_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / (1.0f + expf(-a[idx]));
    }
}

__global__ void silu_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        out[idx] = x / (1.0f + expf(-x));
    }
}

__global__ void gelu_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        out[idx] = x * cdf;
    }
}

// ============================================================================
// F64 Activation Operations
// ============================================================================

__global__ void relu_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmax(0.0, a[idx]);
    }
}

__global__ void sigmoid_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0 / (1.0 + exp(-a[idx]));
    }
}

__global__ void silu_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = a[idx];
        out[idx] = x / (1.0 + exp(-x));
    }
}

__global__ void gelu_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = a[idx];
        double cdf = 0.5 * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
        out[idx] = x * cdf;
    }
}

// ============================================================================
// F16 Activation Operations (FP32 internally for accuracy)
// ============================================================================

__global__ void relu_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half zero = __float2half(0.0f);
        out[idx] = __hgt(a[idx], zero) ? a[idx] : zero;
    }
}

__global__ void sigmoid_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(a[idx]);
        out[idx] = __float2half(1.0f / (1.0f + expf(-x)));
    }
}

__global__ void silu_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(a[idx]);
        out[idx] = __float2half(x / (1.0f + expf(-x)));
    }
}

__global__ void gelu_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(a[idx]);
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        out[idx] = __float2half(x * cdf);
    }
}

// ============================================================================
// BF16 Activation Operations (FP32 internally for accuracy)
// ============================================================================

__global__ void relu_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(a[idx]);
        out[idx] = __float2bfloat16(fmaxf(0.0f, x));
    }
}

__global__ void sigmoid_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(a[idx]);
        out[idx] = __float2bfloat16(1.0f / (1.0f + expf(-x)));
    }
}

__global__ void silu_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(a[idx]);
        out[idx] = __float2bfloat16(x / (1.0f + expf(-x)));
    }
}

__global__ void gelu_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(a[idx]);
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        out[idx] = __float2bfloat16(x * cdf);
    }
}

// ============================================================================
// FP8 E4M3 Activation Operations (FP32 internally)
// ============================================================================

__global__ void relu_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fmaxf(0.0f, x)));
    }
}

__global__ void sigmoid_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(1.0f / (1.0f + expf(-x))));
    }
}

__global__ void silu_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(x / (1.0f + expf(-x))));
    }
}

__global__ void gelu_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e4m3_to_f32(a[idx].data);
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(x * cdf));
    }
}

// ============================================================================
// FP8 E5M2 Activation Operations (FP32 internally)
// ============================================================================

__global__ void relu_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fmaxf(0.0f, x)));
    }
}

__global__ void sigmoid_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(1.0f / (1.0f + expf(-x))));
    }
}

__global__ void silu_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(x / (1.0f + expf(-x))));
    }
}

__global__ void gelu_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e5m2_to_f32(a[idx].data);
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(x * cdf));
    }
}

// ============================================================================
// Leaky ReLU: max(negative_slope * x, x)
// ============================================================================

__global__ void leaky_relu_f32(const float* a, float* out, unsigned int n, float negative_slope) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        out[idx] = x > 0.0f ? x : negative_slope * x;
    }
}

__global__ void leaky_relu_f64(const double* a, double* out, unsigned int n, float negative_slope) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = a[idx];
        double ns = (double)negative_slope;
        out[idx] = x > 0.0 ? x : ns * x;
    }
}

__global__ void leaky_relu_f16(const __half* a, __half* out, unsigned int n, float negative_slope) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(a[idx]);
        out[idx] = __float2half(x > 0.0f ? x : negative_slope * x);
    }
}

__global__ void leaky_relu_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n, float negative_slope) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(a[idx]);
        out[idx] = __float2bfloat16(x > 0.0f ? x : negative_slope * x);
    }
}

__global__ void leaky_relu_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n, float negative_slope) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(x > 0.0f ? x : negative_slope * x));
    }
}

__global__ void leaky_relu_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n, float negative_slope) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(x > 0.0f ? x : negative_slope * x));
    }
}

// ============================================================================
// ELU: x if x > 0, else alpha * (exp(x) - 1)
// ============================================================================

__global__ void elu_f32(const float* a, float* out, unsigned int n, float alpha) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        out[idx] = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
    }
}

__global__ void elu_f64(const double* a, double* out, unsigned int n, float alpha) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = a[idx];
        double a_d = (double)alpha;
        out[idx] = x > 0.0 ? x : a_d * (exp(x) - 1.0);
    }
}

__global__ void elu_f16(const __half* a, __half* out, unsigned int n, float alpha) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(a[idx]);
        out[idx] = __float2half(x > 0.0f ? x : alpha * (expf(x) - 1.0f));
    }
}

__global__ void elu_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n, float alpha) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(a[idx]);
        out[idx] = __float2bfloat16(x > 0.0f ? x : alpha * (expf(x) - 1.0f));
    }
}

__global__ void elu_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n, float alpha) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(x > 0.0f ? x : alpha * (expf(x) - 1.0f)));
    }
}

__global__ void elu_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n, float alpha) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(x > 0.0f ? x : alpha * (expf(x) - 1.0f)));
    }
}

} // extern "C"
