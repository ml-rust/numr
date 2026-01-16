// Normalization CUDA kernels
// Supports: rms_norm, layer_norm
// Types: f32, f64, f16, bf16
// Note: All half-precision variants use FP32 accumulation for numerical stability

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" {

// ============================================================================
// F32 Normalization Operations
// ============================================================================

// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
// Each block handles one row (hidden_size elements)
__global__ void rms_norm_f32(
    const float* input, const float* weight, float* output,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];

    const float* row_in = input + row * hidden_size;
    float* row_out = output + row * hidden_size;

    // Phase 1: Compute sum of squares
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = row_in[i];
        thread_sum += val * val;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Compute rsqrt(mean + eps)
    float rms_inv = rsqrtf(shared[0] / hidden_size + eps);
    __syncthreads();

    // Phase 2: Normalize and apply weight
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_out[i] = row_in[i] * rms_inv * weight[i];
    }
}

// LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
__global__ void layer_norm_f32(
    const float* input, const float* weight, const float* bias, float* output,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;

    const float* row_in = input + row * hidden_size;
    float* row_out = output + row * hidden_size;

    // Phase 1: Compute mean
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += row_in[i];
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            mean_shared[threadIdx.x] += mean_shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean = mean_shared[0] / hidden_size;
    __syncthreads();

    // Phase 2: Compute variance
    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = row_in[i] - mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            var_shared[threadIdx.x] += var_shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    float inv_std = rsqrtf(var_shared[0] / hidden_size + eps);
    __syncthreads();

    // Phase 3: Normalize and apply affine transform
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (row_in[i] - mean) * inv_std;
        row_out[i] = normalized * weight[i] + bias[i];
    }
}

// ============================================================================
// F64 Normalization Operations
// ============================================================================

__global__ void rms_norm_f64(
    const double* input, const double* weight, double* output,
    unsigned int batch_size, unsigned int hidden_size, double eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ double shared_f64[];

    const double* row_in = input + row * hidden_size;
    double* row_out = output + row * hidden_size;

    double thread_sum = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double val = row_in[i];
        thread_sum += val * val;
    }
    shared_f64[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_f64[threadIdx.x] += shared_f64[threadIdx.x + s];
        }
        __syncthreads();
    }

    double rms_inv = rsqrt(shared_f64[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_out[i] = row_in[i] * rms_inv * weight[i];
    }
}

__global__ void layer_norm_f64(
    const double* input, const double* weight, const double* bias, double* output,
    unsigned int batch_size, unsigned int hidden_size, double eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ double shared_f64[];
    double* mean_shared = shared_f64;
    double* var_shared = shared_f64 + blockDim.x;

    const double* row_in = input + row * hidden_size;
    double* row_out = output + row * hidden_size;

    double thread_sum = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += row_in[i];
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            mean_shared[threadIdx.x] += mean_shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    double mean = mean_shared[0] / hidden_size;
    __syncthreads();

    double thread_var = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double diff = row_in[i] - mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            var_shared[threadIdx.x] += var_shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    double inv_std = rsqrt(var_shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double normalized = (row_in[i] - mean) * inv_std;
        row_out[i] = normalized * weight[i] + bias[i];
    }
}

// ============================================================================
// F16 Normalization Operations
// Note: Uses FP32 accumulation for numerical stability
// ============================================================================

__global__ void rms_norm_f16(
    const __half* input, const __half* weight, __half* output,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];

    const __half* row_in = input + row * hidden_size;
    __half* row_out = output + row * hidden_size;

    // Accumulate in FP32 for precision
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_in[i]);
        thread_sum += val * val;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms_inv = rsqrtf(shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float result = __half2float(row_in[i]) * rms_inv * __half2float(weight[i]);
        row_out[i] = __float2half(result);
    }
}

__global__ void layer_norm_f16(
    const __half* input, const __half* weight, const __half* bias, __half* output,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;

    const __half* row_in = input + row * hidden_size;
    __half* row_out = output + row * hidden_size;

    // FP32 accumulation for mean
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += __half2float(row_in[i]);
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            mean_shared[threadIdx.x] += mean_shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean = mean_shared[0] / hidden_size;
    __syncthreads();

    // FP32 accumulation for variance
    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = __half2float(row_in[i]) - mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            var_shared[threadIdx.x] += var_shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    float inv_std = rsqrtf(var_shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (__half2float(row_in[i]) - mean) * inv_std;
        float result = normalized * __half2float(weight[i]) + __half2float(bias[i]);
        row_out[i] = __float2half(result);
    }
}

// ============================================================================
// BF16 Normalization Operations
// Note: Uses FP32 accumulation for numerical stability
// ============================================================================

__global__ void rms_norm_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];

    const __nv_bfloat16* row_in = input + row * hidden_size;
    __nv_bfloat16* row_out = output + row * hidden_size;

    // Accumulate in FP32 for precision
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(row_in[i]);
        thread_sum += val * val;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms_inv = rsqrtf(shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float result = __bfloat162float(row_in[i]) * rms_inv * __bfloat162float(weight[i]);
        row_out[i] = __float2bfloat16(result);
    }
}

__global__ void layer_norm_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, const __nv_bfloat16* bias, __nv_bfloat16* output,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;

    const __nv_bfloat16* row_in = input + row * hidden_size;
    __nv_bfloat16* row_out = output + row * hidden_size;

    // FP32 accumulation for mean
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += __bfloat162float(row_in[i]);
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            mean_shared[threadIdx.x] += mean_shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean = mean_shared[0] / hidden_size;
    __syncthreads();

    // FP32 accumulation for variance
    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = __bfloat162float(row_in[i]) - mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            var_shared[threadIdx.x] += var_shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    float inv_std = rsqrtf(var_shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (__bfloat162float(row_in[i]) - mean) * inv_std;
        float result = normalized * __bfloat162float(weight[i]) + __bfloat162float(bias[i]);
        row_out[i] = __float2bfloat16(result);
    }
}

} // extern "C"
