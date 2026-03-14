// Fused activation-mul backward CUDA kernels
// Given forward: output = activation(a) * b
// Backward: d_a = grad * b * activation'(a), d_b = grad * activation(a)
// Fused: computes activation(a), activation'(a), d_a, d_b in single pass
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"
#include "activation_deriv.cuh"

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
        d_b[idx] = g * silu_fwd_f32(x);
        d_a[idx] = g * bv * silu_deriv_f32(x);
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
        d_b[idx] = g * gelu_fwd_f32(x);
        d_a[idx] = g * bv * gelu_deriv_f32(x);
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
        d_b[idx] = g * relu_fwd_f32(x);
        d_a[idx] = g * bv * relu_deriv_f32(x);
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
        d_b[idx] = g * sigmoid_fwd_f32(x);
        d_a[idx] = g * bv * sigmoid_deriv_f32(x);
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
        d_b[idx] = g * silu_fwd_f64(x);
        d_a[idx] = g * bv * silu_deriv_f64(x);
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
        d_b[idx] = g * gelu_fwd_f64(x);
        d_a[idx] = g * bv * gelu_deriv_f64(x);
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
        d_b[idx] = g * relu_fwd_f64(x);
        d_a[idx] = g * bv * relu_deriv_f64(x);
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
        d_b[idx] = g * sigmoid_fwd_f64(x);
        d_a[idx] = g * bv * sigmoid_deriv_f64(x);
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
        d_b[idx] = __float2half(g * silu_fwd_f32(x));
        d_a[idx] = __float2half(g * bv * silu_deriv_f32(x));
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
        d_b[idx] = __float2half(g * gelu_fwd_f32(x));
        d_a[idx] = __float2half(g * bv * gelu_deriv_f32(x));
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
        d_b[idx] = __float2half(g * relu_fwd_f32(x));
        d_a[idx] = __float2half(g * bv * relu_deriv_f32(x));
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
        d_b[idx] = __float2half(g * sigmoid_fwd_f32(x));
        d_a[idx] = __float2half(g * bv * sigmoid_deriv_f32(x));
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
        d_b[idx] = __float2bfloat16(g * silu_fwd_f32(x));
        d_a[idx] = __float2bfloat16(g * bv * silu_deriv_f32(x));
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
        d_b[idx] = __float2bfloat16(g * gelu_fwd_f32(x));
        d_a[idx] = __float2bfloat16(g * bv * gelu_deriv_f32(x));
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
        d_b[idx] = __float2bfloat16(g * relu_fwd_f32(x));
        d_a[idx] = __float2bfloat16(g * bv * relu_deriv_f32(x));
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
        d_b[idx] = __float2bfloat16(g * sigmoid_fwd_f32(x));
        d_a[idx] = __float2bfloat16(g * bv * sigmoid_deriv_f32(x));
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
        d_b[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * silu_fwd_f32(x)));
        d_a[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * bv * silu_deriv_f32(x)));
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
        d_b[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * gelu_fwd_f32(x)));
        d_a[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * bv * gelu_deriv_f32(x)));
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
        d_b[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * relu_fwd_f32(x)));
        d_a[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * bv * relu_deriv_f32(x)));
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
        d_b[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * sigmoid_fwd_f32(x)));
        d_a[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(g * bv * sigmoid_deriv_f32(x)));
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
        d_b[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * silu_fwd_f32(x)));
        d_a[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * bv * silu_deriv_f32(x)));
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
        d_b[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * gelu_fwd_f32(x)));
        d_a[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * bv * gelu_deriv_f32(x)));
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
        d_b[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * relu_fwd_f32(x)));
        d_a[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * bv * relu_deriv_f32(x)));
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
        d_b[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * sigmoid_fwd_f32(x)));
        d_a[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(g * bv * sigmoid_deriv_f32(x)));
    }
}

} // extern "C"
