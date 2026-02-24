// Fused activation-mul backward CUDA kernels
// Given forward: output = activation(a) * b
// Backward: d_a = grad * b * activation'(a), d_b = grad * activation(a)
// Fused: computes activation(a), activation'(a), d_a, d_b in single pass
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

extern "C" {

// ============================================================================
// F32 Fused Activation-Mul Backward
// ============================================================================

// SiLU backward: silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
__global__ void silu_mul_bwd_f32(
    const float* grad, const float* a, const float* b,
    float* d_a, float* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        float g = grad[idx];
        float bv = b[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        float silu_val = x * sig;
        float silu_deriv = sig * (1.0f + x * (1.0f - sig));
        d_b[idx] = g * silu_val;
        d_a[idx] = g * bv * silu_deriv;
    }
}

// GELU backward: uses tanh approximation derivative
__global__ void gelu_mul_bwd_f32(
    const float* grad, const float* a, const float* b,
    float* d_a, float* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        float g = grad[idx];
        float bv = b[idx];
        float c = 0.7978845608f;
        float k = 0.044715f;
        float inner = c * (x + k * x * x * x);
        float t = tanhf(inner);
        float gelu_val = 0.5f * x * (1.0f + t);
        // gelu'(x) = 0.5 * (1 + t) + 0.5 * x * (1 - t*t) * c * (1 + 3*k*x*x)
        float gelu_deriv = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * c * (1.0f + 3.0f * k * x * x);
        d_b[idx] = g * gelu_val;
        d_a[idx] = g * bv * gelu_deriv;
    }
}

// ReLU backward: relu'(x) = 1 if x > 0 else 0
__global__ void relu_mul_bwd_f32(
    const float* grad, const float* a, const float* b,
    float* d_a, float* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        float g = grad[idx];
        float bv = b[idx];
        float relu_val = fmaxf(0.0f, x);
        float relu_deriv = x > 0.0f ? 1.0f : 0.0f;
        d_b[idx] = g * relu_val;
        d_a[idx] = g * bv * relu_deriv;
    }
}

// Sigmoid backward: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
__global__ void sigmoid_mul_bwd_f32(
    const float* grad, const float* a, const float* b,
    float* d_a, float* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        float g = grad[idx];
        float bv = b[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        float sig_deriv = sig * (1.0f - sig);
        d_b[idx] = g * sig;
        d_a[idx] = g * bv * sig_deriv;
    }
}

// ============================================================================
// F64 Fused Activation-Mul Backward
// ============================================================================

__global__ void silu_mul_bwd_f64(
    const double* grad, const double* a, const double* b,
    double* d_a, double* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = a[idx];
        double g = grad[idx];
        double bv = b[idx];
        double sig = 1.0 / (1.0 + exp(-x));
        double silu_val = x * sig;
        double silu_deriv = sig * (1.0 + x * (1.0 - sig));
        d_b[idx] = g * silu_val;
        d_a[idx] = g * bv * silu_deriv;
    }
}

__global__ void gelu_mul_bwd_f64(
    const double* grad, const double* a, const double* b,
    double* d_a, double* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = a[idx];
        double g = grad[idx];
        double bv = b[idx];
        double c = 0.7978845608028654;
        double k = 0.044715;
        double inner = c * (x + k * x * x * x);
        double t = tanh(inner);
        double gelu_val = 0.5 * x * (1.0 + t);
        double gelu_deriv = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3.0 * k * x * x);
        d_b[idx] = g * gelu_val;
        d_a[idx] = g * bv * gelu_deriv;
    }
}

__global__ void relu_mul_bwd_f64(
    const double* grad, const double* a, const double* b,
    double* d_a, double* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = a[idx];
        double g = grad[idx];
        double bv = b[idx];
        double relu_val = fmax(0.0, x);
        double relu_deriv = x > 0.0 ? 1.0 : 0.0;
        d_b[idx] = g * relu_val;
        d_a[idx] = g * bv * relu_deriv;
    }
}

__global__ void sigmoid_mul_bwd_f64(
    const double* grad, const double* a, const double* b,
    double* d_a, double* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = a[idx];
        double g = grad[idx];
        double bv = b[idx];
        double sig = 1.0 / (1.0 + exp(-x));
        double sig_deriv = sig * (1.0 - sig);
        d_b[idx] = g * sig;
        d_a[idx] = g * bv * sig_deriv;
    }
}

// ============================================================================
// F16 Fused Activation-Mul Backward (compute in F32)
// ============================================================================

__global__ void silu_mul_bwd_f16(
    const __half* grad, const __half* a, const __half* b,
    __half* d_a, __half* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(a[idx]);
        float g = __half2float(grad[idx]);
        float bv = __half2float(b[idx]);
        float sig = 1.0f / (1.0f + expf(-x));
        float silu_val = x * sig;
        float silu_deriv = sig * (1.0f + x * (1.0f - sig));
        d_b[idx] = __float2half(g * silu_val);
        d_a[idx] = __float2half(g * bv * silu_deriv);
    }
}

__global__ void gelu_mul_bwd_f16(
    const __half* grad, const __half* a, const __half* b,
    __half* d_a, __half* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(a[idx]);
        float g = __half2float(grad[idx]);
        float bv = __half2float(b[idx]);
        float c = 0.7978845608f;
        float k = 0.044715f;
        float inner = c * (x + k * x * x * x);
        float t = tanhf(inner);
        float gelu_val = 0.5f * x * (1.0f + t);
        float gelu_deriv = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * c * (1.0f + 3.0f * k * x * x);
        d_b[idx] = __float2half(g * gelu_val);
        d_a[idx] = __float2half(g * bv * gelu_deriv);
    }
}

__global__ void relu_mul_bwd_f16(
    const __half* grad, const __half* a, const __half* b,
    __half* d_a, __half* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(a[idx]);
        float g = __half2float(grad[idx]);
        float bv = __half2float(b[idx]);
        float relu_val = fmaxf(0.0f, x);
        float relu_deriv = x > 0.0f ? 1.0f : 0.0f;
        d_b[idx] = __float2half(g * relu_val);
        d_a[idx] = __float2half(g * bv * relu_deriv);
    }
}

__global__ void sigmoid_mul_bwd_f16(
    const __half* grad, const __half* a, const __half* b,
    __half* d_a, __half* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(a[idx]);
        float g = __half2float(grad[idx]);
        float bv = __half2float(b[idx]);
        float sig = 1.0f / (1.0f + expf(-x));
        float sig_deriv = sig * (1.0f - sig);
        d_b[idx] = __float2half(g * sig);
        d_a[idx] = __float2half(g * bv * sig_deriv);
    }
}

// ============================================================================
// BF16 Fused Activation-Mul Backward (compute in F32)
// ============================================================================

__global__ void silu_mul_bwd_bf16(
    const __nv_bfloat16* grad, const __nv_bfloat16* a, const __nv_bfloat16* b,
    __nv_bfloat16* d_a, __nv_bfloat16* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(a[idx]);
        float g = __bfloat162float(grad[idx]);
        float bv = __bfloat162float(b[idx]);
        float sig = 1.0f / (1.0f + expf(-x));
        float silu_val = x * sig;
        float silu_deriv = sig * (1.0f + x * (1.0f - sig));
        d_b[idx] = __float2bfloat16(g * silu_val);
        d_a[idx] = __float2bfloat16(g * bv * silu_deriv);
    }
}

__global__ void gelu_mul_bwd_bf16(
    const __nv_bfloat16* grad, const __nv_bfloat16* a, const __nv_bfloat16* b,
    __nv_bfloat16* d_a, __nv_bfloat16* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(a[idx]);
        float g = __bfloat162float(grad[idx]);
        float bv = __bfloat162float(b[idx]);
        float c = 0.7978845608f;
        float k = 0.044715f;
        float inner = c * (x + k * x * x * x);
        float t = tanhf(inner);
        float gelu_val = 0.5f * x * (1.0f + t);
        float gelu_deriv = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * c * (1.0f + 3.0f * k * x * x);
        d_b[idx] = __float2bfloat16(g * gelu_val);
        d_a[idx] = __float2bfloat16(g * bv * gelu_deriv);
    }
}

__global__ void relu_mul_bwd_bf16(
    const __nv_bfloat16* grad, const __nv_bfloat16* a, const __nv_bfloat16* b,
    __nv_bfloat16* d_a, __nv_bfloat16* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(a[idx]);
        float g = __bfloat162float(grad[idx]);
        float bv = __bfloat162float(b[idx]);
        float relu_val = fmaxf(0.0f, x);
        float relu_deriv = x > 0.0f ? 1.0f : 0.0f;
        d_b[idx] = __float2bfloat16(g * relu_val);
        d_a[idx] = __float2bfloat16(g * bv * relu_deriv);
    }
}

__global__ void sigmoid_mul_bwd_bf16(
    const __nv_bfloat16* grad, const __nv_bfloat16* a, const __nv_bfloat16* b,
    __nv_bfloat16* d_a, __nv_bfloat16* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(a[idx]);
        float g = __bfloat162float(grad[idx]);
        float bv = __bfloat162float(b[idx]);
        float sig = 1.0f / (1.0f + expf(-x));
        float sig_deriv = sig * (1.0f - sig);
        d_b[idx] = __float2bfloat16(g * sig);
        d_a[idx] = __float2bfloat16(g * bv * sig_deriv);
    }
}

// ============================================================================
// FP8 E4M3 Fused Activation-Mul Backward (compute in F32)
// ============================================================================

__global__ void silu_mul_bwd_fp8_e4m3(
    const numr_fp8_e4m3* grad, const numr_fp8_e4m3* a, const numr_fp8_e4m3* b,
    numr_fp8_e4m3* d_a, numr_fp8_e4m3* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e4m3_to_f32(a[idx].data);
        float g = fp8_e4m3_to_f32(grad[idx].data);
        float bv = fp8_e4m3_to_f32(b[idx].data);
        float sig = 1.0f / (1.0f + expf(-x));
        float silu_val = x * sig;
        float silu_deriv = sig * (1.0f + x * (1.0f - sig));
        d_b[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * silu_val));
        d_a[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * bv * silu_deriv));
    }
}

__global__ void gelu_mul_bwd_fp8_e4m3(
    const numr_fp8_e4m3* grad, const numr_fp8_e4m3* a, const numr_fp8_e4m3* b,
    numr_fp8_e4m3* d_a, numr_fp8_e4m3* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e4m3_to_f32(a[idx].data);
        float g = fp8_e4m3_to_f32(grad[idx].data);
        float bv = fp8_e4m3_to_f32(b[idx].data);
        float c = 0.7978845608f;
        float k = 0.044715f;
        float inner = c * (x + k * x * x * x);
        float t = tanhf(inner);
        float gelu_val = 0.5f * x * (1.0f + t);
        float gelu_deriv = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * c * (1.0f + 3.0f * k * x * x);
        d_b[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * gelu_val));
        d_a[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * bv * gelu_deriv));
    }
}

__global__ void relu_mul_bwd_fp8_e4m3(
    const numr_fp8_e4m3* grad, const numr_fp8_e4m3* a, const numr_fp8_e4m3* b,
    numr_fp8_e4m3* d_a, numr_fp8_e4m3* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e4m3_to_f32(a[idx].data);
        float g = fp8_e4m3_to_f32(grad[idx].data);
        float bv = fp8_e4m3_to_f32(b[idx].data);
        float relu_val = fmaxf(0.0f, x);
        float relu_deriv = x > 0.0f ? 1.0f : 0.0f;
        d_b[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * relu_val));
        d_a[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * bv * relu_deriv));
    }
}

__global__ void sigmoid_mul_bwd_fp8_e4m3(
    const numr_fp8_e4m3* grad, const numr_fp8_e4m3* a, const numr_fp8_e4m3* b,
    numr_fp8_e4m3* d_a, numr_fp8_e4m3* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e4m3_to_f32(a[idx].data);
        float g = fp8_e4m3_to_f32(grad[idx].data);
        float bv = fp8_e4m3_to_f32(b[idx].data);
        float sig = 1.0f / (1.0f + expf(-x));
        float sig_deriv = sig * (1.0f - sig);
        d_b[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * sig));
        d_a[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * bv * sig_deriv));
    }
}

// ============================================================================
// FP8 E5M2 Fused Activation-Mul Backward (compute in F32)
// ============================================================================

__global__ void silu_mul_bwd_fp8_e5m2(
    const numr_fp8_e5m2* grad, const numr_fp8_e5m2* a, const numr_fp8_e5m2* b,
    numr_fp8_e5m2* d_a, numr_fp8_e5m2* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e5m2_to_f32(a[idx].data);
        float g = fp8_e5m2_to_f32(grad[idx].data);
        float bv = fp8_e5m2_to_f32(b[idx].data);
        float sig = 1.0f / (1.0f + expf(-x));
        float silu_val = x * sig;
        float silu_deriv = sig * (1.0f + x * (1.0f - sig));
        d_b[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * silu_val));
        d_a[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * bv * silu_deriv));
    }
}

__global__ void gelu_mul_bwd_fp8_e5m2(
    const numr_fp8_e5m2* grad, const numr_fp8_e5m2* a, const numr_fp8_e5m2* b,
    numr_fp8_e5m2* d_a, numr_fp8_e5m2* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e5m2_to_f32(a[idx].data);
        float g = fp8_e5m2_to_f32(grad[idx].data);
        float bv = fp8_e5m2_to_f32(b[idx].data);
        float c = 0.7978845608f;
        float k = 0.044715f;
        float inner = c * (x + k * x * x * x);
        float t = tanhf(inner);
        float gelu_val = 0.5f * x * (1.0f + t);
        float gelu_deriv = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * c * (1.0f + 3.0f * k * x * x);
        d_b[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * gelu_val));
        d_a[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * bv * gelu_deriv));
    }
}

__global__ void relu_mul_bwd_fp8_e5m2(
    const numr_fp8_e5m2* grad, const numr_fp8_e5m2* a, const numr_fp8_e5m2* b,
    numr_fp8_e5m2* d_a, numr_fp8_e5m2* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e5m2_to_f32(a[idx].data);
        float g = fp8_e5m2_to_f32(grad[idx].data);
        float bv = fp8_e5m2_to_f32(b[idx].data);
        float relu_val = fmaxf(0.0f, x);
        float relu_deriv = x > 0.0f ? 1.0f : 0.0f;
        d_b[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * relu_val));
        d_a[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * bv * relu_deriv));
    }
}

__global__ void sigmoid_mul_bwd_fp8_e5m2(
    const numr_fp8_e5m2* grad, const numr_fp8_e5m2* a, const numr_fp8_e5m2* b,
    numr_fp8_e5m2* d_a, numr_fp8_e5m2* d_b, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = fp8_e5m2_to_f32(a[idx].data);
        float g = fp8_e5m2_to_f32(grad[idx].data);
        float bv = fp8_e5m2_to_f32(b[idx].data);
        float sig = 1.0f / (1.0f + expf(-x));
        float sig_deriv = sig * (1.0f - sig);
        d_b[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * sig));
        d_a[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * bv * sig_deriv));
    }
}

} // extern "C"
