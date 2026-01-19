// Activation CUDA kernels
// Supports: relu, sigmoid, softmax, silu, gelu
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2

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

// SiLU (Swish): x * sigmoid(x) = x / (1 + exp(-x))
__global__ void silu_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        out[idx] = x / (1.0f + expf(-x));
    }
}

// GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Using the tanh approximation for better performance
__global__ void gelu_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        // sqrt(2/pi) ≈ 0.7978845608
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        out[idx] = x * cdf;
    }
}

// Softmax over the last dimension
// outer_size = product of all dims except last
// dim_size = size of last dimension
__global__ void softmax_f32(
    const float* input, float* output,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_exp = shared + blockDim.x;

    const float* row_in = input + outer_idx * dim_size;
    float* row_out = output + outer_idx * dim_size;

    // Phase 1: Find max value for numerical stability
    float thread_max = -INFINITY;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_in[i]);
    }
    max_val[threadIdx.x] = thread_max;
    __syncthreads();

    // Reduce max across threads
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = max_val[0];
    __syncthreads();

    // Phase 2: Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = expf(row_in[i] - row_max);
        row_out[i] = val;  // Temporarily store exp values
        thread_sum += val;
    }
    sum_exp[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduce sum across threads
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_exp[threadIdx.x] += sum_exp[threadIdx.x + s];
        }
        __syncthreads();
    }
    float row_sum = sum_exp[0];
    __syncthreads();

    // Phase 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
}

// Softmax over non-last dimension
// For shape [A, B, C] with softmax over dim=1:
// outer_size = A, dim_size = B, inner_size = C
__global__ void softmax_dim_f32(
    const float* input, float* output,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    // Base offset for this (outer, inner) position
    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    // Find max
    float max_val = -INFINITY;
    for (unsigned int i = 0; i < dim_size; i++) {
        max_val = fmaxf(max_val, input[base + i * stride]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (unsigned int i = 0; i < dim_size; i++) {
        float val = expf(input[base + i * stride] - max_val);
        output[base + i * stride] = val;
        sum += val;
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (unsigned int i = 0; i < dim_size; i++) {
        output[base + i * stride] *= inv_sum;
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

__global__ void softmax_f64(
    const double* input, double* output,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ double shared_f64[];
    double* max_val = shared_f64;
    double* sum_exp = shared_f64 + blockDim.x;

    const double* row_in = input + outer_idx * dim_size;
    double* row_out = output + outer_idx * dim_size;

    // Phase 1: Find max
    double thread_max = -INFINITY;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_max = fmax(thread_max, row_in[i]);
    }
    max_val[threadIdx.x] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            max_val[threadIdx.x] = fmax(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        }
        __syncthreads();
    }
    double row_max = max_val[0];
    __syncthreads();

    // Phase 2: Compute exp and sum
    double thread_sum = 0.0;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        double val = exp(row_in[i] - row_max);
        row_out[i] = val;
        thread_sum += val;
    }
    sum_exp[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_exp[threadIdx.x] += sum_exp[threadIdx.x + s];
        }
        __syncthreads();
    }
    double row_sum = sum_exp[0];
    __syncthreads();

    // Phase 3: Normalize
    double inv_sum = 1.0 / row_sum;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
}

__global__ void softmax_dim_f64(
    const double* input, double* output,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    double max_val = -INFINITY;
    for (unsigned int i = 0; i < dim_size; i++) {
        max_val = fmax(max_val, input[base + i * stride]);
    }

    double sum = 0.0;
    for (unsigned int i = 0; i < dim_size; i++) {
        double val = exp(input[base + i * stride] - max_val);
        output[base + i * stride] = val;
        sum += val;
    }

    double inv_sum = 1.0 / sum;
    for (unsigned int i = 0; i < dim_size; i++) {
        output[base + i * stride] *= inv_sum;
    }
}

// ============================================================================
// F16 Activation Operations
// Note: Uses FP32 internally for accuracy where needed
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
        // Use FP32 for accuracy
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

// F16 Softmax: Uses FP32 accumulation internally for numerical stability
__global__ void softmax_f16(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_exp = shared + blockDim.x;

    const __half* row_in = input + outer_idx * dim_size;
    __half* row_out = output + outer_idx * dim_size;

    // Phase 1: Find max value (using FP32)
    float thread_max = -INFINITY;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, __half2float(row_in[i]));
    }
    max_val[threadIdx.x] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = max_val[0];
    __syncthreads();

    // Phase 2: Compute exp(x - max) and sum (FP32 accumulation)
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = expf(__half2float(row_in[i]) - row_max);
        row_out[i] = __float2half(val);
        thread_sum += val;
    }
    sum_exp[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_exp[threadIdx.x] += sum_exp[threadIdx.x + s];
        }
        __syncthreads();
    }
    float row_sum = sum_exp[0];
    __syncthreads();

    // Phase 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        row_out[i] = __float2half(__half2float(row_out[i]) * inv_sum);
    }
}

__global__ void softmax_dim_f16(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    // FP32 accumulation for stability
    float max_val = -INFINITY;
    for (unsigned int i = 0; i < dim_size; i++) {
        max_val = fmaxf(max_val, __half2float(input[base + i * stride]));
    }

    float sum = 0.0f;
    for (unsigned int i = 0; i < dim_size; i++) {
        float val = expf(__half2float(input[base + i * stride]) - max_val);
        output[base + i * stride] = __float2half(val);
        sum += val;
    }

    float inv_sum = 1.0f / sum;
    for (unsigned int i = 0; i < dim_size; i++) {
        output[base + i * stride] = __float2half(__half2float(output[base + i * stride]) * inv_sum);
    }
}

// ============================================================================
// BF16 Activation Operations
// Note: Uses FP32 internally for accuracy where needed
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

// BF16 Softmax: Uses FP32 accumulation internally for numerical stability
__global__ void softmax_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_exp = shared + blockDim.x;

    const __nv_bfloat16* row_in = input + outer_idx * dim_size;
    __nv_bfloat16* row_out = output + outer_idx * dim_size;

    // Phase 1: Find max value (using FP32)
    float thread_max = -INFINITY;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, __bfloat162float(row_in[i]));
    }
    max_val[threadIdx.x] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = max_val[0];
    __syncthreads();

    // Phase 2: Compute exp(x - max) and sum (FP32 accumulation)
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = expf(__bfloat162float(row_in[i]) - row_max);
        row_out[i] = __float2bfloat16(val);
        thread_sum += val;
    }
    sum_exp[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_exp[threadIdx.x] += sum_exp[threadIdx.x + s];
        }
        __syncthreads();
    }
    float row_sum = sum_exp[0];
    __syncthreads();

    // Phase 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        row_out[i] = __float2bfloat16(__bfloat162float(row_out[i]) * inv_sum);
    }
}

__global__ void softmax_dim_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    // FP32 accumulation for stability
    float max_val = -INFINITY;
    for (unsigned int i = 0; i < dim_size; i++) {
        max_val = fmaxf(max_val, __bfloat162float(input[base + i * stride]));
    }

    float sum = 0.0f;
    for (unsigned int i = 0; i < dim_size; i++) {
        float val = expf(__bfloat162float(input[base + i * stride]) - max_val);
        output[base + i * stride] = __float2bfloat16(val);
        sum += val;
    }

    float inv_sum = 1.0f / sum;
    for (unsigned int i = 0; i < dim_size; i++) {
        output[base + i * stride] = __float2bfloat16(__bfloat162float(output[base + i * stride]) * inv_sum);
    }
}

// ============================================================================
// FP8 E4M3 Activation Operations
// All computation done in F32, stored back as FP8
// Uses Hopper PTX intrinsics on SM 8.9+, software emulation on SM 8.0+
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

// FP8 E4M3 Softmax: Uses FP32 accumulation internally for numerical stability
__global__ void softmax_fp8_e4m3(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_exp = shared + blockDim.x;

    const numr_fp8_e4m3* row_in = input + outer_idx * dim_size;
    numr_fp8_e4m3* row_out = output + outer_idx * dim_size;

    // Phase 1: Find max value (using FP32)
    float thread_max = -INFINITY;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, fp8_e4m3_to_f32(row_in[i].data));
    }
    max_val[threadIdx.x] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = max_val[0];
    __syncthreads();

    // Phase 2: Compute exp(x - max) and sum (FP32 accumulation)
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = expf(fp8_e4m3_to_f32(row_in[i].data) - row_max);
        row_out[i] = numr_fp8_e4m3(f32_to_fp8_e4m3(val));
        thread_sum += val;
    }
    sum_exp[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_exp[threadIdx.x] += sum_exp[threadIdx.x + s];
        }
        __syncthreads();
    }
    float row_sum = sum_exp[0];
    __syncthreads();

    // Phase 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        row_out[i] = numr_fp8_e4m3(f32_to_fp8_e4m3(fp8_e4m3_to_f32(row_out[i].data) * inv_sum));
    }
}

__global__ void softmax_dim_fp8_e4m3(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    // FP32 accumulation for stability
    float max_val = -INFINITY;
    for (unsigned int i = 0; i < dim_size; i++) {
        max_val = fmaxf(max_val, fp8_e4m3_to_f32(input[base + i * stride].data));
    }

    float sum = 0.0f;
    for (unsigned int i = 0; i < dim_size; i++) {
        float val = expf(fp8_e4m3_to_f32(input[base + i * stride].data) - max_val);
        output[base + i * stride] = numr_fp8_e4m3(f32_to_fp8_e4m3(val));
        sum += val;
    }

    float inv_sum = 1.0f / sum;
    for (unsigned int i = 0; i < dim_size; i++) {
        output[base + i * stride] = numr_fp8_e4m3(f32_to_fp8_e4m3(
            fp8_e4m3_to_f32(output[base + i * stride].data) * inv_sum));
    }
}

// ============================================================================
// FP8 E5M2 Activation Operations
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

// FP8 E5M2 Softmax
__global__ void softmax_fp8_e5m2(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_exp = shared + blockDim.x;

    const numr_fp8_e5m2* row_in = input + outer_idx * dim_size;
    numr_fp8_e5m2* row_out = output + outer_idx * dim_size;

    // Phase 1: Find max value (using FP32)
    float thread_max = -INFINITY;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, fp8_e5m2_to_f32(row_in[i].data));
    }
    max_val[threadIdx.x] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = max_val[0];
    __syncthreads();

    // Phase 2: Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = expf(fp8_e5m2_to_f32(row_in[i].data) - row_max);
        row_out[i] = numr_fp8_e5m2(f32_to_fp8_e5m2(val));
        thread_sum += val;
    }
    sum_exp[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_exp[threadIdx.x] += sum_exp[threadIdx.x + s];
        }
        __syncthreads();
    }
    float row_sum = sum_exp[0];
    __syncthreads();

    // Phase 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        row_out[i] = numr_fp8_e5m2(f32_to_fp8_e5m2(fp8_e5m2_to_f32(row_out[i].data) * inv_sum));
    }
}

__global__ void softmax_dim_fp8_e5m2(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    // FP32 accumulation for stability
    float max_val = -INFINITY;
    for (unsigned int i = 0; i < dim_size; i++) {
        max_val = fmaxf(max_val, fp8_e5m2_to_f32(input[base + i * stride].data));
    }

    float sum = 0.0f;
    for (unsigned int i = 0; i < dim_size; i++) {
        float val = expf(fp8_e5m2_to_f32(input[base + i * stride].data) - max_val);
        output[base + i * stride] = numr_fp8_e5m2(f32_to_fp8_e5m2(val));
        sum += val;
    }

    float inv_sum = 1.0f / sum;
    for (unsigned int i = 0; i < dim_size; i++) {
        output[base + i * stride] = numr_fp8_e5m2(f32_to_fp8_e5m2(
            fp8_e5m2_to_f32(output[base + i * stride].data) * inv_sum));
    }
}

// ============================================================================
// Leaky ReLU Activation Operations
// leaky_relu(x) = max(negative_slope * x, x)
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
// ELU (Exponential Linear Unit) Activation Operations
// elu(x) = x if x > 0, else alpha * (exp(x) - 1)
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
