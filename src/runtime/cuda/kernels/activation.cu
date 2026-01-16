// Activation CUDA kernels
// Supports: relu, sigmoid, softmax
// Types: f32, f64

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

} // extern "C"
