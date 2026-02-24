// Fused Add + Normalization CUDA kernels
// Supports: fused_add_rms_norm, fused_add_layer_norm (forward + backward)
// Types: f32, f64, f16, bf16
// Note: All half-precision variants use FP32 accumulation for numerical stability

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" {

// ============================================================================
// Helper: atomicAdd for half-precision types via atomicCAS
// ============================================================================

__device__ void atomicAddHalf(__half* address, float val) {
    unsigned short int* address_as_us = (unsigned short int*)address;
    unsigned short int old = *address_as_us, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_us, assumed,
            __half_as_ushort(__float2half(__half2float(__ushort_as_half(assumed)) + val)));
    } while (assumed != old);
}

__device__ void atomicAddBf16(__nv_bfloat16* address, float val) {
    // Use atomicCAS with bit manipulation for BF16
    unsigned short int* address_as_us = (unsigned short int*)address;
    unsigned short int old = *address_as_us, assumed;
    do {
        assumed = old;
        // Extract as uint16, convert to bfloat16, then float, add, convert back
        __nv_bfloat16 old_val;
        unsigned short int* old_val_ptr = (unsigned short int*)&old_val;
        *old_val_ptr = assumed;
        float new_float = __bfloat162float(old_val) + val;
        __nv_bfloat16 new_val = __float2bfloat16(new_float);
        unsigned short int* new_val_ptr = (unsigned short int*)&new_val;
        old = atomicCAS(address_as_us, assumed, *new_val_ptr);
    } while (assumed != old);
}

// ============================================================================
// F32 Fused Add + RMSNorm Forward
// ============================================================================

__global__ void fused_add_rms_norm_f32(
    const float* input, const float* residual, const float* weight,
    float* output, float* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];

    const float* row_in = input + row * hidden_size;
    const float* row_res = residual + row * hidden_size;
    float* row_pn = pre_norm + row * hidden_size;
    float* row_out = output + row * hidden_size;

    // Phase 1: Add residual + compute sum of squares
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = row_in[i] + row_res[i];
        row_pn[i] = pn;
        thread_sum += pn * pn;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduce
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(shared[0] / hidden_size + eps);
    __syncthreads();

    // Phase 2: Normalize and apply weight
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_out[i] = row_pn[i] * rms_inv * weight[i];
    }
}

// ============================================================================
// F64 Fused Add + RMSNorm Forward
// ============================================================================

__global__ void fused_add_rms_norm_f64(
    const double* input, const double* residual, const double* weight,
    double* output, double* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, double eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ double shared_f64[];

    const double* row_in = input + row * hidden_size;
    const double* row_res = residual + row * hidden_size;
    double* row_pn = pre_norm + row * hidden_size;
    double* row_out = output + row * hidden_size;

    double thread_sum = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double pn = row_in[i] + row_res[i];
        row_pn[i] = pn;
        thread_sum += pn * pn;
    }
    shared_f64[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared_f64[threadIdx.x] += shared_f64[threadIdx.x + s];
        __syncthreads();
    }

    double rms_inv = rsqrt(shared_f64[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_out[i] = row_pn[i] * rms_inv * weight[i];
    }
}

// ============================================================================
// F16 Fused Add + RMSNorm Forward (FP32 accumulation)
// ============================================================================

__global__ void fused_add_rms_norm_f16(
    const __half* input, const __half* residual, const __half* weight,
    __half* output, __half* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];

    const __half* row_in = input + row * hidden_size;
    const __half* row_res = residual + row * hidden_size;
    __half* row_pn = pre_norm + row * hidden_size;
    __half* row_out = output + row * hidden_size;

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = __half2float(row_in[i]) + __half2float(row_res[i]);
        row_pn[i] = __float2half(pn);
        thread_sum += pn * pn;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = __half2float(row_pn[i]);
        float result = pn * rms_inv * __half2float(weight[i]);
        row_out[i] = __float2half(result);
    }
}

// ============================================================================
// BF16 Fused Add + RMSNorm Forward (FP32 accumulation)
// ============================================================================

__global__ void fused_add_rms_norm_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* residual, const __nv_bfloat16* weight,
    __nv_bfloat16* output, __nv_bfloat16* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];

    const __nv_bfloat16* row_in = input + row * hidden_size;
    const __nv_bfloat16* row_res = residual + row * hidden_size;
    __nv_bfloat16* row_pn = pre_norm + row * hidden_size;
    __nv_bfloat16* row_out = output + row * hidden_size;

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = __bfloat162float(row_in[i]) + __bfloat162float(row_res[i]);
        row_pn[i] = __float2bfloat16(pn);
        thread_sum += pn * pn;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }

    float rms_inv = rsqrtf(shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = __bfloat162float(row_pn[i]);
        float result = pn * rms_inv * __bfloat162float(weight[i]);
        row_out[i] = __float2bfloat16(result);
    }
}

// ============================================================================
// F32 Fused Add + RMSNorm Backward
// ============================================================================

__global__ void fused_add_rms_norm_bwd_f32(
    const float* grad, const float* pre_norm, const float* weight,
    float* d_input_residual, float* d_weight,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* sum_sq_shared = shared;
    float* dot_shared = shared + blockDim.x;

    // Phase 1: Compute sum_sq and dot = sum(grad * weight * pre_norm)
    float thread_sq = 0.0f, thread_dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = pre_norm[row * hidden_size + i];
        float g = grad[row * hidden_size + i];
        float w = weight[i];
        thread_sq += pn * pn;
        thread_dot += g * w * pn;
    }
    sum_sq_shared[threadIdx.x] = thread_sq;
    dot_shared[threadIdx.x] = thread_dot;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_sq_shared[threadIdx.x] += sum_sq_shared[threadIdx.x + s];
            dot_shared[threadIdx.x] += dot_shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float mean_sq = sum_sq_shared[0] / hidden_size;
    float inv_rms = rsqrtf(mean_sq + eps);
    float dot = dot_shared[0];
    float coeff = dot * inv_rms / (hidden_size * (mean_sq + eps));
    __syncthreads();

    // Phase 2: Compute d_input_residual and atomicAdd d_weight
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = grad[row * hidden_size + i];
        float w = weight[i];
        float pn = pre_norm[row * hidden_size + i];
        d_input_residual[row * hidden_size + i] = (g * w - pn * coeff) * inv_rms;
        atomicAdd(&d_weight[i], g * pn * inv_rms);
    }
}

// ============================================================================
// F64 Fused Add + RMSNorm Backward
// ============================================================================

__global__ void fused_add_rms_norm_bwd_f64(
    const double* grad, const double* pre_norm, const double* weight,
    double* d_input_residual, double* d_weight,
    unsigned int batch_size, unsigned int hidden_size, double eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ double shared_f64[];
    double* sum_sq_shared = shared_f64;
    double* dot_shared = shared_f64 + blockDim.x;

    double thread_sq = 0.0, thread_dot = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double pn = pre_norm[row * hidden_size + i];
        double g = grad[row * hidden_size + i];
        double w = weight[i];
        thread_sq += pn * pn;
        thread_dot += g * w * pn;
    }
    sum_sq_shared[threadIdx.x] = thread_sq;
    dot_shared[threadIdx.x] = thread_dot;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_sq_shared[threadIdx.x] += sum_sq_shared[threadIdx.x + s];
            dot_shared[threadIdx.x] += dot_shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    double mean_sq = sum_sq_shared[0] / hidden_size;
    double inv_rms = rsqrt(mean_sq + eps);
    double dot = dot_shared[0];
    double coeff = dot * inv_rms / (hidden_size * (mean_sq + eps));
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double g = grad[row * hidden_size + i];
        double w = weight[i];
        double pn = pre_norm[row * hidden_size + i];
        d_input_residual[row * hidden_size + i] = (g * w - pn * coeff) * inv_rms;
        atomicAdd(&d_weight[i], g * pn * inv_rms);
    }
}

// ============================================================================
// F16 Fused Add + RMSNorm Backward (FP32 accumulation)
// ============================================================================

__global__ void fused_add_rms_norm_bwd_f16(
    const __half* grad, const __half* pre_norm, const __half* weight,
    __half* d_input_residual, __half* d_weight,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* sum_sq_shared = shared;
    float* dot_shared = shared + blockDim.x;

    float thread_sq = 0.0f, thread_dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = __half2float(pre_norm[row * hidden_size + i]);
        float g = __half2float(grad[row * hidden_size + i]);
        float w = __half2float(weight[i]);
        thread_sq += pn * pn;
        thread_dot += g * w * pn;
    }
    sum_sq_shared[threadIdx.x] = thread_sq;
    dot_shared[threadIdx.x] = thread_dot;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_sq_shared[threadIdx.x] += sum_sq_shared[threadIdx.x + s];
            dot_shared[threadIdx.x] += dot_shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float mean_sq = sum_sq_shared[0] / hidden_size;
    float inv_rms = rsqrtf(mean_sq + eps);
    float dot = dot_shared[0];
    float coeff = dot * inv_rms / (hidden_size * (mean_sq + eps));
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = __half2float(grad[row * hidden_size + i]);
        float w = __half2float(weight[i]);
        float pn = __half2float(pre_norm[row * hidden_size + i]);
        float dir = (g * w - pn * coeff) * inv_rms;
        d_input_residual[row * hidden_size + i] = __float2half(dir);
        atomicAddHalf(&d_weight[i], g * pn * inv_rms);
    }
}

// ============================================================================
// BF16 Fused Add + RMSNorm Backward (FP32 accumulation)
// ============================================================================

__global__ void fused_add_rms_norm_bwd_bf16(
    const __nv_bfloat16* grad, const __nv_bfloat16* pre_norm, const __nv_bfloat16* weight,
    __nv_bfloat16* d_input_residual, __nv_bfloat16* d_weight,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* sum_sq_shared = shared;
    float* dot_shared = shared + blockDim.x;

    float thread_sq = 0.0f, thread_dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = __bfloat162float(pre_norm[row * hidden_size + i]);
        float g = __bfloat162float(grad[row * hidden_size + i]);
        float w = __bfloat162float(weight[i]);
        thread_sq += pn * pn;
        thread_dot += g * w * pn;
    }
    sum_sq_shared[threadIdx.x] = thread_sq;
    dot_shared[threadIdx.x] = thread_dot;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_sq_shared[threadIdx.x] += sum_sq_shared[threadIdx.x + s];
            dot_shared[threadIdx.x] += dot_shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float mean_sq = sum_sq_shared[0] / hidden_size;
    float inv_rms = rsqrtf(mean_sq + eps);
    float dot = dot_shared[0];
    float coeff = dot * inv_rms / (hidden_size * (mean_sq + eps));
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = __bfloat162float(grad[row * hidden_size + i]);
        float w = __bfloat162float(weight[i]);
        float pn = __bfloat162float(pre_norm[row * hidden_size + i]);
        float dir = (g * w - pn * coeff) * inv_rms;
        d_input_residual[row * hidden_size + i] = __float2bfloat16(dir);
        atomicAddBf16(&d_weight[i], g * pn * inv_rms);
    }
}

// ============================================================================
// F32 Fused Add + LayerNorm Forward
// ============================================================================

__global__ void fused_add_layer_norm_f32(
    const float* input, const float* residual, const float* weight, const float* bias,
    float* output, float* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;

    const float* row_in = input + row * hidden_size;
    const float* row_res = residual + row * hidden_size;
    float* row_pn = pre_norm + row * hidden_size;
    float* row_out = output + row * hidden_size;

    // Phase 1: Add residual + compute mean
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = row_in[i] + row_res[i];
        row_pn[i] = pn;
        thread_sum += pn;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) mean_shared[threadIdx.x] += mean_shared[threadIdx.x + s];
        __syncthreads();
    }
    float mean = mean_shared[0] / hidden_size;
    __syncthreads();

    // Phase 2: Compute variance
    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = row_pn[i] - mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) var_shared[threadIdx.x] += var_shared[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(var_shared[0] / hidden_size + eps);
    __syncthreads();

    // Phase 3: Normalize and apply affine
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (row_pn[i] - mean) * inv_std;
        row_out[i] = normalized * weight[i] + bias[i];
    }
}

// ============================================================================
// F64 Fused Add + LayerNorm Forward
// ============================================================================

__global__ void fused_add_layer_norm_f64(
    const double* input, const double* residual, const double* weight, const double* bias,
    double* output, double* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, double eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ double shared_f64[];
    double* mean_shared = shared_f64;
    double* var_shared = shared_f64 + blockDim.x;

    const double* row_in = input + row * hidden_size;
    const double* row_res = residual + row * hidden_size;
    double* row_pn = pre_norm + row * hidden_size;
    double* row_out = output + row * hidden_size;

    double thread_sum = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double pn = row_in[i] + row_res[i];
        row_pn[i] = pn;
        thread_sum += pn;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) mean_shared[threadIdx.x] += mean_shared[threadIdx.x + s];
        __syncthreads();
    }
    double mean = mean_shared[0] / hidden_size;
    __syncthreads();

    double thread_var = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double diff = row_pn[i] - mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) var_shared[threadIdx.x] += var_shared[threadIdx.x + s];
        __syncthreads();
    }
    double inv_std = rsqrt(var_shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double normalized = (row_pn[i] - mean) * inv_std;
        row_out[i] = normalized * weight[i] + bias[i];
    }
}

// ============================================================================
// F16 Fused Add + LayerNorm Forward (FP32 accumulation)
// ============================================================================

__global__ void fused_add_layer_norm_f16(
    const __half* input, const __half* residual, const __half* weight, const __half* bias,
    __half* output, __half* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;

    const __half* row_in = input + row * hidden_size;
    const __half* row_res = residual + row * hidden_size;
    __half* row_pn = pre_norm + row * hidden_size;
    __half* row_out = output + row * hidden_size;

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = __half2float(row_in[i]) + __half2float(row_res[i]);
        row_pn[i] = __float2half(pn);
        thread_sum += pn;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) mean_shared[threadIdx.x] += mean_shared[threadIdx.x + s];
        __syncthreads();
    }
    float mean = mean_shared[0] / hidden_size;
    __syncthreads();

    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = __half2float(row_pn[i]) - mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) var_shared[threadIdx.x] += var_shared[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(var_shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (__half2float(row_pn[i]) - mean) * inv_std;
        float result = normalized * __half2float(weight[i]) + __half2float(bias[i]);
        row_out[i] = __float2half(result);
    }
}

// ============================================================================
// BF16 Fused Add + LayerNorm Forward (FP32 accumulation)
// ============================================================================

__global__ void fused_add_layer_norm_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* residual, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    __nv_bfloat16* output, __nv_bfloat16* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;

    const __nv_bfloat16* row_in = input + row * hidden_size;
    const __nv_bfloat16* row_res = residual + row * hidden_size;
    __nv_bfloat16* row_pn = pre_norm + row * hidden_size;
    __nv_bfloat16* row_out = output + row * hidden_size;

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = __bfloat162float(row_in[i]) + __bfloat162float(row_res[i]);
        row_pn[i] = __float2bfloat16(pn);
        thread_sum += pn;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) mean_shared[threadIdx.x] += mean_shared[threadIdx.x + s];
        __syncthreads();
    }
    float mean = mean_shared[0] / hidden_size;
    __syncthreads();

    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = __bfloat162float(row_pn[i]) - mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) var_shared[threadIdx.x] += var_shared[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(var_shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (__bfloat162float(row_pn[i]) - mean) * inv_std;
        float result = normalized * __bfloat162float(weight[i]) + __bfloat162float(bias[i]);
        row_out[i] = __float2bfloat16(result);
    }
}

// ============================================================================
// F32 Fused Add + LayerNorm Backward
// ============================================================================

__global__ void fused_add_layer_norm_bwd_f32(
    const float* grad, const float* pre_norm, const float* weight,
    float* d_input_residual, float* d_weight, float* d_bias,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;
    float* gs_shared = shared + 2 * blockDim.x;
    float* gsn_shared = shared + 3 * blockDim.x;

    // Phase 1: Compute mean
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += pre_norm[row * hidden_size + i];
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) mean_shared[threadIdx.x] += mean_shared[threadIdx.x + s];
        __syncthreads();
    }
    float mean = mean_shared[0] / hidden_size;
    __syncthreads();

    // Phase 2: Compute variance
    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = pre_norm[row * hidden_size + i] - mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) var_shared[threadIdx.x] += var_shared[threadIdx.x + s];
        __syncthreads();
    }
    float var = var_shared[0] / hidden_size;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    // Phase 3: mean_gs and mean_gsn
    float thread_gs = 0.0f, thread_gsn = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = grad[row * hidden_size + i];
        float w = weight[i];
        float normalized = (pre_norm[row * hidden_size + i] - mean) * inv_std;
        thread_gs += g * w;
        thread_gsn += g * w * normalized;
    }
    gs_shared[threadIdx.x] = thread_gs;
    gsn_shared[threadIdx.x] = thread_gsn;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            gs_shared[threadIdx.x] += gs_shared[threadIdx.x + s];
            gsn_shared[threadIdx.x] += gsn_shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean_gs = gs_shared[0] / hidden_size;
    float mean_gsn = gsn_shared[0] / hidden_size;
    __syncthreads();

    // Phase 4: Compute gradients
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = grad[row * hidden_size + i];
        float w = weight[i];
        float normalized = (pre_norm[row * hidden_size + i] - mean) * inv_std;
        float d_ir = inv_std * (g * w - mean_gs - normalized * mean_gsn);
        d_input_residual[row * hidden_size + i] = d_ir;
        atomicAdd(&d_weight[i], g * normalized);
        atomicAdd(&d_bias[i], g);
    }
}

// ============================================================================
// F64 Fused Add + LayerNorm Backward
// ============================================================================

__global__ void fused_add_layer_norm_bwd_f64(
    const double* grad, const double* pre_norm, const double* weight,
    double* d_input_residual, double* d_weight, double* d_bias,
    unsigned int batch_size, unsigned int hidden_size, double eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ double shared_f64[];
    double* mean_shared = shared_f64;
    double* var_shared = shared_f64 + blockDim.x;
    double* gs_shared = shared_f64 + 2 * blockDim.x;
    double* gsn_shared = shared_f64 + 3 * blockDim.x;

    double thread_sum = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += pre_norm[row * hidden_size + i];
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) mean_shared[threadIdx.x] += mean_shared[threadIdx.x + s];
        __syncthreads();
    }
    double mean = mean_shared[0] / hidden_size;
    __syncthreads();

    double thread_var = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double diff = pre_norm[row * hidden_size + i] - mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) var_shared[threadIdx.x] += var_shared[threadIdx.x + s];
        __syncthreads();
    }
    double var = var_shared[0] / hidden_size;
    double inv_std = rsqrt(var + eps);
    __syncthreads();

    double thread_gs = 0.0, thread_gsn = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double g = grad[row * hidden_size + i];
        double w = weight[i];
        double normalized = (pre_norm[row * hidden_size + i] - mean) * inv_std;
        thread_gs += g * w;
        thread_gsn += g * w * normalized;
    }
    gs_shared[threadIdx.x] = thread_gs;
    gsn_shared[threadIdx.x] = thread_gsn;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            gs_shared[threadIdx.x] += gs_shared[threadIdx.x + s];
            gsn_shared[threadIdx.x] += gsn_shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    double mean_gs = gs_shared[0] / hidden_size;
    double mean_gsn = gsn_shared[0] / hidden_size;
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double g = grad[row * hidden_size + i];
        double w = weight[i];
        double normalized = (pre_norm[row * hidden_size + i] - mean) * inv_std;
        double d_ir = inv_std * (g * w - mean_gs - normalized * mean_gsn);
        d_input_residual[row * hidden_size + i] = d_ir;
        atomicAdd(&d_weight[i], g * normalized);
        atomicAdd(&d_bias[i], g);
    }
}

// ============================================================================
// F16 Fused Add + LayerNorm Backward (FP32 accumulation)
// ============================================================================

__global__ void fused_add_layer_norm_bwd_f16(
    const __half* grad, const __half* pre_norm, const __half* weight,
    __half* d_input_residual, __half* d_weight, __half* d_bias,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;
    float* gs_shared = shared + 2 * blockDim.x;
    float* gsn_shared = shared + 3 * blockDim.x;

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += __half2float(pre_norm[row * hidden_size + i]);
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) mean_shared[threadIdx.x] += mean_shared[threadIdx.x + s];
        __syncthreads();
    }
    float mean = mean_shared[0] / hidden_size;
    __syncthreads();

    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = __half2float(pre_norm[row * hidden_size + i]) - mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) var_shared[threadIdx.x] += var_shared[threadIdx.x + s];
        __syncthreads();
    }
    float var = var_shared[0] / hidden_size;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    float thread_gs = 0.0f, thread_gsn = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = __half2float(grad[row * hidden_size + i]);
        float w = __half2float(weight[i]);
        float normalized = (__half2float(pre_norm[row * hidden_size + i]) - mean) * inv_std;
        thread_gs += g * w;
        thread_gsn += g * w * normalized;
    }
    gs_shared[threadIdx.x] = thread_gs;
    gsn_shared[threadIdx.x] = thread_gsn;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            gs_shared[threadIdx.x] += gs_shared[threadIdx.x + s];
            gsn_shared[threadIdx.x] += gsn_shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean_gs = gs_shared[0] / hidden_size;
    float mean_gsn = gsn_shared[0] / hidden_size;
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = __half2float(grad[row * hidden_size + i]);
        float w = __half2float(weight[i]);
        float normalized = (__half2float(pre_norm[row * hidden_size + i]) - mean) * inv_std;
        float d_ir = inv_std * (g * w - mean_gs - normalized * mean_gsn);
        d_input_residual[row * hidden_size + i] = __float2half(d_ir);
        atomicAddHalf(&d_weight[i], g * normalized);
        atomicAddHalf(&d_bias[i], g);
    }
}

// ============================================================================
// BF16 Fused Add + LayerNorm Backward (FP32 accumulation)
// ============================================================================

__global__ void fused_add_layer_norm_bwd_bf16(
    const __nv_bfloat16* grad, const __nv_bfloat16* pre_norm, const __nv_bfloat16* weight,
    __nv_bfloat16* d_input_residual, __nv_bfloat16* d_weight, __nv_bfloat16* d_bias,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;
    float* gs_shared = shared + 2 * blockDim.x;
    float* gsn_shared = shared + 3 * blockDim.x;

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += __bfloat162float(pre_norm[row * hidden_size + i]);
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) mean_shared[threadIdx.x] += mean_shared[threadIdx.x + s];
        __syncthreads();
    }
    float mean = mean_shared[0] / hidden_size;
    __syncthreads();

    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = __bfloat162float(pre_norm[row * hidden_size + i]) - mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) var_shared[threadIdx.x] += var_shared[threadIdx.x + s];
        __syncthreads();
    }
    float var = var_shared[0] / hidden_size;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    float thread_gs = 0.0f, thread_gsn = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = __bfloat162float(grad[row * hidden_size + i]);
        float w = __bfloat162float(weight[i]);
        float normalized = (__bfloat162float(pre_norm[row * hidden_size + i]) - mean) * inv_std;
        thread_gs += g * w;
        thread_gsn += g * w * normalized;
    }
    gs_shared[threadIdx.x] = thread_gs;
    gsn_shared[threadIdx.x] = thread_gsn;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            gs_shared[threadIdx.x] += gs_shared[threadIdx.x + s];
            gsn_shared[threadIdx.x] += gsn_shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    float mean_gs = gs_shared[0] / hidden_size;
    float mean_gsn = gsn_shared[0] / hidden_size;
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = __bfloat162float(grad[row * hidden_size + i]);
        float w = __bfloat162float(weight[i]);
        float normalized = (__bfloat162float(pre_norm[row * hidden_size + i]) - mean) * inv_std;
        float d_ir = inv_std * (g * w - mean_gs - normalized * mean_gsn);
        d_input_residual[row * hidden_size + i] = __float2bfloat16(d_ir);
        atomicAddBf16(&d_weight[i], g * normalized);
        atomicAddBf16(&d_bias[i], g);
    }
}

} // extern "C"
