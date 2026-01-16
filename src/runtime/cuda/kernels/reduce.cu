// Reduction CUDA kernels
// Supports: sum, max, min, mean
// Types: f32, f64
//
// Uses two-phase reduction:
// 1. Block-level reduction with shared memory
// 2. Final reduction across blocks

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

} // extern "C"
