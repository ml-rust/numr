// Reduction CUDA kernels
// Supports: sum, max, min, mean
// Types: f32, f64, f16, bf16
//
// Uses two-phase reduction:
// 1. Block-level reduction with shared memory
// 2. Final reduction across blocks
//
// F16/BF16 ACCUMULATION OPTIONS:
// - Default kernels (e.g., reduce_sum_f16): Native half accumulation (faster, less memory)
// - FP32 accumulation variants (e.g., reduce_sum_f16_fp32acc): More precise for large tensors
// Choose based on your precision vs. performance requirements.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" {

// ============================================================================
// Warp-level primitives
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max_f32(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_min_f32(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ double warp_reduce_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ double warp_reduce_max_f64(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ double warp_reduce_min_f64(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmin(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// ============================================================================
// F32 Reduction Kernels
// ============================================================================

// Sum reduction - produces partial sums per block
__global__ void reduce_sum_f32(const float* input, float* output, unsigned int n) {
    __shared__ float shared[32]; // One element per warp

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    // Load and accumulate multiple elements per thread (grid-stride loop)
    float sum = 0.0f;
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // Warp-level reduction
    sum = warp_reduce_sum_f32(sum);

    // First thread of each warp writes to shared memory
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum_f32(sum);
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

// Max reduction
__global__ void reduce_max_f32(const float* input, float* output, unsigned int n) {
    __shared__ float shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    float max_val = -INFINITY;
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        max_val = fmaxf(max_val, input[i]);
    }

    max_val = warp_reduce_max_f32(max_val);

    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : -INFINITY;
        max_val = warp_reduce_max_f32(max_val);
        if (tid == 0) {
            output[blockIdx.x] = max_val;
        }
    }
}

// Min reduction
__global__ void reduce_min_f32(const float* input, float* output, unsigned int n) {
    __shared__ float shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    float min_val = INFINITY;
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        min_val = fminf(min_val, input[i]);
    }

    min_val = warp_reduce_min_f32(min_val);

    if (lane == 0) {
        shared[warp_id] = min_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        min_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : INFINITY;
        min_val = warp_reduce_min_f32(min_val);
        if (tid == 0) {
            output[blockIdx.x] = min_val;
        }
    }
}

// ============================================================================
// F64 Reduction Kernels
// ============================================================================

__global__ void reduce_sum_f64(const double* input, double* output, unsigned int n) {
    __shared__ double shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    double sum = 0.0;
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    sum = warp_reduce_sum_f64(sum);

    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[lane] : 0.0;
        sum = warp_reduce_sum_f64(sum);
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

__global__ void reduce_max_f64(const double* input, double* output, unsigned int n) {
    __shared__ double shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    double max_val = -INFINITY;
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        max_val = fmax(max_val, input[i]);
    }

    max_val = warp_reduce_max_f64(max_val);

    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : -INFINITY;
        max_val = warp_reduce_max_f64(max_val);
        if (tid == 0) {
            output[blockIdx.x] = max_val;
        }
    }
}

__global__ void reduce_min_f64(const double* input, double* output, unsigned int n) {
    __shared__ double shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    double min_val = INFINITY;
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        min_val = fmin(min_val, input[i]);
    }

    min_val = warp_reduce_min_f64(min_val);

    if (lane == 0) {
        shared[warp_id] = min_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        min_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : INFINITY;
        min_val = warp_reduce_min_f64(min_val);
        if (tid == 0) {
            output[blockIdx.x] = min_val;
        }
    }
}

// ============================================================================
// Dimension-wise Reduction Kernels
// ============================================================================

// Reduce along a specific dimension (row-wise reduction over last dim)
// For shape [..., M, N] reducing over dim -1: produces [..., M]
__global__ void reduce_sum_dim_f32(
    const float* input, float* output,
    unsigned int outer_size,    // Product of all dims before reduce dim
    unsigned int reduce_size,   // Size of dimension being reduced
    unsigned int inner_size     // Product of all dims after reduce dim (1 for last dim)
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ float shared[256];
    unsigned int tid = threadIdx.x;

    float sum = 0.0f;
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        sum += input[idx];
    }

    shared[tid] = sum;
    __syncthreads();

    // Block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared[0];
    }
}

__global__ void reduce_max_dim_f32(
    const float* input, float* output,
    unsigned int outer_size,
    unsigned int reduce_size,
    unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ float shared[256];
    unsigned int tid = threadIdx.x;

    float max_val = -INFINITY;
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        max_val = fmaxf(max_val, input[idx]);
    }

    shared[tid] = max_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared[0];
    }
}

__global__ void reduce_min_dim_f32(
    const float* input, float* output,
    unsigned int outer_size,
    unsigned int reduce_size,
    unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ float shared[256];
    unsigned int tid = threadIdx.x;

    float min_val = INFINITY;
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        min_val = fminf(min_val, input[idx]);
    }

    shared[tid] = min_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fminf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared[0];
    }
}

// F64 dimension-wise reductions
__global__ void reduce_sum_dim_f64(
    const double* input, double* output,
    unsigned int outer_size,
    unsigned int reduce_size,
    unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ double shared[256];
    unsigned int tid = threadIdx.x;

    double sum = 0.0;
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        sum += input[idx];
    }

    shared[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared[0];
    }
}

__global__ void reduce_max_dim_f64(
    const double* input, double* output,
    unsigned int outer_size,
    unsigned int reduce_size,
    unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ double shared[256];
    unsigned int tid = threadIdx.x;

    double max_val = -INFINITY;
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        max_val = fmax(max_val, input[idx]);
    }

    shared[tid] = max_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmax(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared[0];
    }
}

__global__ void reduce_min_dim_f64(
    const double* input, double* output,
    unsigned int outer_size,
    unsigned int reduce_size,
    unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ double shared[256];
    unsigned int tid = threadIdx.x;

    double min_val = INFINITY;
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        min_val = fmin(min_val, input[idx]);
    }

    shared[tid] = min_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmin(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared[0];
    }
}

// ============================================================================
// F16 Reduction Kernels (half precision) - Native Accumulation
// Faster, uses less memory. Use _fp32acc variants for better precision.
// ============================================================================

// Warp-level reduction for half precision
__device__ __forceinline__ __half warp_reduce_sum_f16(__half val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = __hadd(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ __half warp_reduce_max_f16(__half val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        __half other = __shfl_down_sync(0xffffffff, val, offset);
        val = __hgt(val, other) ? val : other;
    }
    return val;
}

__device__ __forceinline__ __half warp_reduce_min_f16(__half val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        __half other = __shfl_down_sync(0xffffffff, val, offset);
        val = __hlt(val, other) ? val : other;
    }
    return val;
}

// Native F16 sum reduction
__global__ void reduce_sum_f16(const __half* input, __half* output, unsigned int n) {
    __shared__ __half shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    __half sum = __float2half(0.0f);
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum = __hadd(sum, input[i]);
    }

    sum = warp_reduce_sum_f16(sum);

    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[lane] : __float2half(0.0f);
        sum = warp_reduce_sum_f16(sum);
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

// Native F16 max reduction
__global__ void reduce_max_f16(const __half* input, __half* output, unsigned int n) {
    __shared__ __half shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    __half max_val = __float2half(-INFINITY);
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        __half val = input[i];
        max_val = __hgt(val, max_val) ? val : max_val;
    }

    max_val = warp_reduce_max_f16(max_val);

    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : __float2half(-INFINITY);
        max_val = warp_reduce_max_f16(max_val);
        if (tid == 0) {
            output[blockIdx.x] = max_val;
        }
    }
}

// Native F16 min reduction
__global__ void reduce_min_f16(const __half* input, __half* output, unsigned int n) {
    __shared__ __half shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    __half min_val = __float2half(INFINITY);
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        __half val = input[i];
        min_val = __hlt(val, min_val) ? val : min_val;
    }

    min_val = warp_reduce_min_f16(min_val);

    if (lane == 0) {
        shared[warp_id] = min_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        min_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : __float2half(INFINITY);
        min_val = warp_reduce_min_f16(min_val);
        if (tid == 0) {
            output[blockIdx.x] = min_val;
        }
    }
}

// Native F16 dimension-wise reductions
__global__ void reduce_sum_dim_f16(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ __half shared[256];
    unsigned int tid = threadIdx.x;

    __half sum = __float2half(0.0f);
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        sum = __hadd(sum, input[idx]);
    }

    shared[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = __hadd(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared[0];
    }
}

__global__ void reduce_max_dim_f16(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ __half shared[256];
    unsigned int tid = threadIdx.x;

    __half max_val = __float2half(-INFINITY);
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        __half val = input[idx];
        max_val = __hgt(val, max_val) ? val : max_val;
    }

    shared[tid] = max_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            __half a = shared[tid];
            __half b = shared[tid + s];
            shared[tid] = __hgt(a, b) ? a : b;
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared[0];
    }
}

__global__ void reduce_min_dim_f16(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ __half shared[256];
    unsigned int tid = threadIdx.x;

    __half min_val = __float2half(INFINITY);
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        __half val = input[idx];
        min_val = __hlt(val, min_val) ? val : min_val;
    }

    shared[tid] = min_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            __half a = shared[tid];
            __half b = shared[tid + s];
            shared[tid] = __hlt(a, b) ? a : b;
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared[0];
    }
}

// ============================================================================
// F16 Reduction Kernels - FP32 Accumulation (more precise)
// Use these for large tensors where precision matters
// ============================================================================

__global__ void reduce_sum_f16_fp32acc(const __half* input, __half* output, unsigned int n) {
    __shared__ float shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    float sum = 0.0f;
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += __half2float(input[i]);
    }

    sum = warp_reduce_sum_f32(sum);

    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum_f32(sum);
        if (tid == 0) {
            output[blockIdx.x] = __float2half(sum);
        }
    }
}

__global__ void reduce_max_f16_fp32acc(const __half* input, __half* output, unsigned int n) {
    __shared__ float shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    float max_val = -INFINITY;
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        max_val = fmaxf(max_val, __half2float(input[i]));
    }

    max_val = warp_reduce_max_f32(max_val);

    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : -INFINITY;
        max_val = warp_reduce_max_f32(max_val);
        if (tid == 0) {
            output[blockIdx.x] = __float2half(max_val);
        }
    }
}

__global__ void reduce_min_f16_fp32acc(const __half* input, __half* output, unsigned int n) {
    __shared__ float shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    float min_val = INFINITY;
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        min_val = fminf(min_val, __half2float(input[i]));
    }

    min_val = warp_reduce_min_f32(min_val);

    if (lane == 0) {
        shared[warp_id] = min_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        min_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : INFINITY;
        min_val = warp_reduce_min_f32(min_val);
        if (tid == 0) {
            output[blockIdx.x] = __float2half(min_val);
        }
    }
}

__global__ void reduce_sum_dim_f16_fp32acc(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ float shared[256];
    unsigned int tid = threadIdx.x;

    float sum = 0.0f;
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        sum += __half2float(input[idx]);
    }

    shared[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = __float2half(shared[0]);
    }
}

__global__ void reduce_max_dim_f16_fp32acc(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ float shared[256];
    unsigned int tid = threadIdx.x;

    float max_val = -INFINITY;
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        max_val = fmaxf(max_val, __half2float(input[idx]));
    }

    shared[tid] = max_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = __float2half(shared[0]);
    }
}

__global__ void reduce_min_dim_f16_fp32acc(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ float shared[256];
    unsigned int tid = threadIdx.x;

    float min_val = INFINITY;
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        min_val = fminf(min_val, __half2float(input[idx]));
    }

    shared[tid] = min_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fminf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = __float2half(shared[0]);
    }
}

// ============================================================================
// BF16 Reduction Kernels (bfloat16) - Native Accumulation
// Faster, uses less memory. Use _fp32acc variants for better precision.
// Note: BF16 native ops require SM >= 8.0, fallback uses FP32 on older GPUs
// ============================================================================

// Warp-level reduction for bfloat16
__device__ __forceinline__ __nv_bfloat16 warp_reduce_sum_bf16(__nv_bfloat16 val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        __nv_bfloat16 other = __shfl_down_sync(0xffffffff, val, offset);
        #if __CUDA_ARCH__ >= 800
        val = __hadd(val, other);
        #else
        val = __float2bfloat16(__bfloat162float(val) + __bfloat162float(other));
        #endif
    }
    return val;
}

__device__ __forceinline__ __nv_bfloat16 warp_reduce_max_bf16(__nv_bfloat16 val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        __nv_bfloat16 other = __shfl_down_sync(0xffffffff, val, offset);
        #if __CUDA_ARCH__ >= 800
        val = __hgt(val, other) ? val : other;
        #else
        val = (__bfloat162float(val) > __bfloat162float(other)) ? val : other;
        #endif
    }
    return val;
}

__device__ __forceinline__ __nv_bfloat16 warp_reduce_min_bf16(__nv_bfloat16 val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        __nv_bfloat16 other = __shfl_down_sync(0xffffffff, val, offset);
        #if __CUDA_ARCH__ >= 800
        val = __hlt(val, other) ? val : other;
        #else
        val = (__bfloat162float(val) < __bfloat162float(other)) ? val : other;
        #endif
    }
    return val;
}

// Native BF16 sum reduction
__global__ void reduce_sum_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int n) {
    __shared__ __nv_bfloat16 shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    __nv_bfloat16 sum = __float2bfloat16(0.0f);
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        #if __CUDA_ARCH__ >= 800
        sum = __hadd(sum, input[i]);
        #else
        sum = __float2bfloat16(__bfloat162float(sum) + __bfloat162float(input[i]));
        #endif
    }

    sum = warp_reduce_sum_bf16(sum);

    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[lane] : __float2bfloat16(0.0f);
        sum = warp_reduce_sum_bf16(sum);
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

// Native BF16 max reduction
__global__ void reduce_max_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int n) {
    __shared__ __nv_bfloat16 shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    __nv_bfloat16 max_val = __float2bfloat16(-INFINITY);
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        __nv_bfloat16 val = input[i];
        #if __CUDA_ARCH__ >= 800
        max_val = __hgt(val, max_val) ? val : max_val;
        #else
        max_val = (__bfloat162float(val) > __bfloat162float(max_val)) ? val : max_val;
        #endif
    }

    max_val = warp_reduce_max_bf16(max_val);

    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : __float2bfloat16(-INFINITY);
        max_val = warp_reduce_max_bf16(max_val);
        if (tid == 0) {
            output[blockIdx.x] = max_val;
        }
    }
}

// Native BF16 min reduction
__global__ void reduce_min_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int n) {
    __shared__ __nv_bfloat16 shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    __nv_bfloat16 min_val = __float2bfloat16(INFINITY);
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        __nv_bfloat16 val = input[i];
        #if __CUDA_ARCH__ >= 800
        min_val = __hlt(val, min_val) ? val : min_val;
        #else
        min_val = (__bfloat162float(val) < __bfloat162float(min_val)) ? val : min_val;
        #endif
    }

    min_val = warp_reduce_min_bf16(min_val);

    if (lane == 0) {
        shared[warp_id] = min_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        min_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : __float2bfloat16(INFINITY);
        min_val = warp_reduce_min_bf16(min_val);
        if (tid == 0) {
            output[blockIdx.x] = min_val;
        }
    }
}

// Native BF16 dimension-wise reductions
__global__ void reduce_sum_dim_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ __nv_bfloat16 shared[256];
    unsigned int tid = threadIdx.x;

    __nv_bfloat16 sum = __float2bfloat16(0.0f);
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        #if __CUDA_ARCH__ >= 800
        sum = __hadd(sum, input[idx]);
        #else
        sum = __float2bfloat16(__bfloat162float(sum) + __bfloat162float(input[idx]));
        #endif
    }

    shared[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #if __CUDA_ARCH__ >= 800
            shared[tid] = __hadd(shared[tid], shared[tid + s]);
            #else
            shared[tid] = __float2bfloat16(__bfloat162float(shared[tid]) + __bfloat162float(shared[tid + s]));
            #endif
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared[0];
    }
}

__global__ void reduce_max_dim_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ __nv_bfloat16 shared[256];
    unsigned int tid = threadIdx.x;

    __nv_bfloat16 max_val = __float2bfloat16(-INFINITY);
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        __nv_bfloat16 val = input[idx];
        #if __CUDA_ARCH__ >= 800
        max_val = __hgt(val, max_val) ? val : max_val;
        #else
        max_val = (__bfloat162float(val) > __bfloat162float(max_val)) ? val : max_val;
        #endif
    }

    shared[tid] = max_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            __nv_bfloat16 a = shared[tid];
            __nv_bfloat16 b = shared[tid + s];
            #if __CUDA_ARCH__ >= 800
            shared[tid] = __hgt(a, b) ? a : b;
            #else
            shared[tid] = (__bfloat162float(a) > __bfloat162float(b)) ? a : b;
            #endif
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared[0];
    }
}

__global__ void reduce_min_dim_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ __nv_bfloat16 shared[256];
    unsigned int tid = threadIdx.x;

    __nv_bfloat16 min_val = __float2bfloat16(INFINITY);
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        __nv_bfloat16 val = input[idx];
        #if __CUDA_ARCH__ >= 800
        min_val = __hlt(val, min_val) ? val : min_val;
        #else
        min_val = (__bfloat162float(val) < __bfloat162float(min_val)) ? val : min_val;
        #endif
    }

    shared[tid] = min_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            __nv_bfloat16 a = shared[tid];
            __nv_bfloat16 b = shared[tid + s];
            #if __CUDA_ARCH__ >= 800
            shared[tid] = __hlt(a, b) ? a : b;
            #else
            shared[tid] = (__bfloat162float(a) < __bfloat162float(b)) ? a : b;
            #endif
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared[0];
    }
}

// ============================================================================
// BF16 Reduction Kernels - FP32 Accumulation (more precise)
// Use these for large tensors where precision matters
// ============================================================================

__global__ void reduce_sum_bf16_fp32acc(const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int n) {
    __shared__ float shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    float sum = 0.0f;
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += __bfloat162float(input[i]);
    }

    sum = warp_reduce_sum_f32(sum);

    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum_f32(sum);
        if (tid == 0) {
            output[blockIdx.x] = __float2bfloat16(sum);
        }
    }
}

__global__ void reduce_max_bf16_fp32acc(const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int n) {
    __shared__ float shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    float max_val = -INFINITY;
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        max_val = fmaxf(max_val, __bfloat162float(input[i]));
    }

    max_val = warp_reduce_max_f32(max_val);

    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : -INFINITY;
        max_val = warp_reduce_max_f32(max_val);
        if (tid == 0) {
            output[blockIdx.x] = __float2bfloat16(max_val);
        }
    }
}

__global__ void reduce_min_bf16_fp32acc(const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int n) {
    __shared__ float shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    float min_val = INFINITY;
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        min_val = fminf(min_val, __bfloat162float(input[i]));
    }

    min_val = warp_reduce_min_f32(min_val);

    if (lane == 0) {
        shared[warp_id] = min_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        min_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : INFINITY;
        min_val = warp_reduce_min_f32(min_val);
        if (tid == 0) {
            output[blockIdx.x] = __float2bfloat16(min_val);
        }
    }
}

__global__ void reduce_sum_dim_bf16_fp32acc(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ float shared[256];
    unsigned int tid = threadIdx.x;

    float sum = 0.0f;
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        sum += __bfloat162float(input[idx]);
    }

    shared[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = __float2bfloat16(shared[0]);
    }
}

__global__ void reduce_max_dim_bf16_fp32acc(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ float shared[256];
    unsigned int tid = threadIdx.x;

    float max_val = -INFINITY;
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        max_val = fmaxf(max_val, __bfloat162float(input[idx]));
    }

    shared[tid] = max_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = __float2bfloat16(shared[0]);
    }
}

__global__ void reduce_min_dim_bf16_fp32acc(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ float shared[256];
    unsigned int tid = threadIdx.x;

    float min_val = INFINITY;
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        min_val = fminf(min_val, __bfloat162float(input[idx]));
    }

    shared[tid] = min_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fminf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = __float2bfloat16(shared[0]);
    }
}

// ============================================================================
// Argmax Kernels - Return indices of maximum values (output: int64_t)
// ============================================================================

// F32 argmax along dimension
__global__ void argmax_dim_f32(
    const float* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ float shared_val[256];
    __shared__ long long shared_idx[256];
    unsigned int tid = threadIdx.x;

    float max_val = -INFINITY;
    long long max_idx = 0;

    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        float val = input[idx];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    shared_val[tid] = max_val;
    shared_idx[tid] = max_idx;
    __syncthreads();

    // Block reduction - find global max and its index
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_val[tid + s] > shared_val[tid]) {
                shared_val[tid] = shared_val[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared_idx[0];
    }
}

// F64 argmax along dimension
__global__ void argmax_dim_f64(
    const double* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ double shared_val[256];
    __shared__ long long shared_idx[256];
    unsigned int tid = threadIdx.x;

    double max_val = -INFINITY;
    long long max_idx = 0;

    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        double val = input[idx];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    shared_val[tid] = max_val;
    shared_idx[tid] = max_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_val[tid + s] > shared_val[tid]) {
                shared_val[tid] = shared_val[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared_idx[0];
    }
}

// F16 argmax along dimension (uses FP32 for comparison)
__global__ void argmax_dim_f16(
    const __half* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ float shared_val[256];
    __shared__ long long shared_idx[256];
    unsigned int tid = threadIdx.x;

    float max_val = -INFINITY;
    long long max_idx = 0;

    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        float val = __half2float(input[idx]);
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    shared_val[tid] = max_val;
    shared_idx[tid] = max_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_val[tid + s] > shared_val[tid]) {
                shared_val[tid] = shared_val[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared_idx[0];
    }
}

// BF16 argmax along dimension (uses FP32 for comparison)
__global__ void argmax_dim_bf16(
    const __nv_bfloat16* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ float shared_val[256];
    __shared__ long long shared_idx[256];
    unsigned int tid = threadIdx.x;

    float max_val = -INFINITY;
    long long max_idx = 0;

    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        float val = __bfloat162float(input[idx]);
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    shared_val[tid] = max_val;
    shared_idx[tid] = max_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_val[tid + s] > shared_val[tid]) {
                shared_val[tid] = shared_val[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared_idx[0];
    }
}

// ============================================================================
// Argmin Kernels - Return indices of minimum values (output: int64_t)
// ============================================================================

// F32 argmin along dimension
__global__ void argmin_dim_f32(
    const float* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ float shared_val[256];
    __shared__ long long shared_idx[256];
    unsigned int tid = threadIdx.x;

    float min_val = INFINITY;
    long long min_idx = 0;

    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        float val = input[idx];
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }

    shared_val[tid] = min_val;
    shared_idx[tid] = min_idx;
    __syncthreads();

    // Block reduction - find global min and its index
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_val[tid + s] < shared_val[tid]) {
                shared_val[tid] = shared_val[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared_idx[0];
    }
}

// F64 argmin along dimension
__global__ void argmin_dim_f64(
    const double* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ double shared_val[256];
    __shared__ long long shared_idx[256];
    unsigned int tid = threadIdx.x;

    double min_val = INFINITY;
    long long min_idx = 0;

    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        double val = input[idx];
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }

    shared_val[tid] = min_val;
    shared_idx[tid] = min_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_val[tid + s] < shared_val[tid]) {
                shared_val[tid] = shared_val[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared_idx[0];
    }
}

// F16 argmin along dimension (uses FP32 for comparison)
__global__ void argmin_dim_f16(
    const __half* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ float shared_val[256];
    __shared__ long long shared_idx[256];
    unsigned int tid = threadIdx.x;

    float min_val = INFINITY;
    long long min_idx = 0;

    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        float val = __half2float(input[idx]);
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }

    shared_val[tid] = min_val;
    shared_idx[tid] = min_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_val[tid + s] < shared_val[tid]) {
                shared_val[tid] = shared_val[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared_idx[0];
    }
}

// BF16 argmin along dimension (uses FP32 for comparison)
__global__ void argmin_dim_bf16(
    const __nv_bfloat16* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ float shared_val[256];
    __shared__ long long shared_idx[256];
    unsigned int tid = threadIdx.x;

    float min_val = INFINITY;
    long long min_idx = 0;

    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        float val = __bfloat162float(input[idx]);
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }

    shared_val[tid] = min_val;
    shared_idx[tid] = min_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_val[tid + s] < shared_val[tid]) {
                shared_val[tid] = shared_val[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx * inner_size + inner_idx] = shared_idx[0];
    }
}

} // extern "C"
