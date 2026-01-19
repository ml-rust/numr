// Reduction CUDA kernels - Templated implementation using AccumTraits
// Supports: sum, max, min, mean
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2
//
// Uses AccumTraits<T, Acc> for DRY reduction code across all type/accumulation combinations.
// Similar to CPU's Accumulator trait pattern.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Templated Reduction Device Functions
// These are called by the extern "C" __global__ wrapper kernels
// ============================================================================

// Global sum reduction - produces partial sums per block
template<typename T, typename Acc>
__device__ void reduce_sum_impl(const T* input, T* output, unsigned int n) {
    using Traits = AccumTraits<T, Acc>;
    __shared__ Acc shared[32]; // One element per warp

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    Acc sum = Traits::zero();
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum = Traits::add(sum, Traits::load(input, i));
    }

    sum = Traits::warp_sum(sum);

    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[lane] : Traits::zero();
        sum = Traits::warp_sum(sum);
        if (tid == 0) {
            Traits::store(output, blockIdx.x, sum);
        }
    }
}

// Global max reduction
template<typename T, typename Acc>
__device__ void reduce_max_impl(const T* input, T* output, unsigned int n) {
    using Traits = AccumTraits<T, Acc>;
    __shared__ Acc shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    Acc max_val = Traits::neg_inf();
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        max_val = Traits::max(max_val, Traits::load(input, i));
    }

    max_val = Traits::warp_max(max_val);

    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : Traits::neg_inf();
        max_val = Traits::warp_max(max_val);
        if (tid == 0) {
            Traits::store(output, blockIdx.x, max_val);
        }
    }
}

// Global min reduction
template<typename T, typename Acc>
__device__ void reduce_min_impl(const T* input, T* output, unsigned int n) {
    using Traits = AccumTraits<T, Acc>;
    __shared__ Acc shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    Acc min_val = Traits::pos_inf();
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        min_val = Traits::min(min_val, Traits::load(input, i));
    }

    min_val = Traits::warp_min(min_val);

    if (lane == 0) {
        shared[warp_id] = min_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        min_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : Traits::pos_inf();
        min_val = Traits::warp_min(min_val);
        if (tid == 0) {
            Traits::store(output, blockIdx.x, min_val);
        }
    }
}

// Global prod reduction - produces partial products per block
template<typename T, typename Acc>
__device__ void reduce_prod_impl(const T* input, T* output, unsigned int n) {
    using Traits = AccumTraits<T, Acc>;
    __shared__ Acc shared[32]; // One element per warp

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    Acc prod = Traits::one();
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        prod = Traits::mul(prod, Traits::load(input, i));
    }

    prod = Traits::warp_prod(prod);

    if (lane == 0) {
        shared[warp_id] = prod;
    }
    __syncthreads();

    if (warp_id == 0) {
        prod = (tid < (blockDim.x + 31) / 32) ? shared[lane] : Traits::one();
        prod = Traits::warp_prod(prod);
        if (tid == 0) {
            Traits::store(output, blockIdx.x, prod);
        }
    }
}

// Dimension-wise sum reduction
template<typename T, typename Acc>
__device__ void reduce_sum_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, Acc>;
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ Acc shared[256];
    unsigned int tid = threadIdx.x;

    Acc sum = Traits::zero();
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        sum = Traits::add(sum, Traits::load(input, idx));
    }

    shared[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = Traits::add(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        Traits::store(output, outer_idx * inner_size + inner_idx, shared[0]);
    }
}

// Dimension-wise max reduction
template<typename T, typename Acc>
__device__ void reduce_max_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, Acc>;
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ Acc shared[256];
    unsigned int tid = threadIdx.x;

    Acc max_val = Traits::neg_inf();
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        max_val = Traits::max(max_val, Traits::load(input, idx));
    }

    shared[tid] = max_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = Traits::max(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        Traits::store(output, outer_idx * inner_size + inner_idx, shared[0]);
    }
}

// Dimension-wise min reduction
template<typename T, typename Acc>
__device__ void reduce_min_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, Acc>;
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ Acc shared[256];
    unsigned int tid = threadIdx.x;

    Acc min_val = Traits::pos_inf();
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        min_val = Traits::min(min_val, Traits::load(input, idx));
    }

    shared[tid] = min_val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = Traits::min(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        Traits::store(output, outer_idx * inner_size + inner_idx, shared[0]);
    }
}

// Argmax along dimension - uses float for comparison (or double for f64)
template<typename T, typename CompareType>
__device__ void argmax_dim_impl(
    const T* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, CompareType>;
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ CompareType shared_val[256];
    __shared__ long long shared_idx[256];
    unsigned int tid = threadIdx.x;

    CompareType max_val = Traits::neg_inf();
    long long max_idx = 0;

    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        CompareType val = Traits::load(input, idx);
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

// Argmin along dimension
template<typename T, typename CompareType>
__device__ void argmin_dim_impl(
    const T* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, CompareType>;
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ CompareType shared_val[256];
    __shared__ long long shared_idx[256];
    unsigned int tid = threadIdx.x;

    CompareType min_val = Traits::pos_inf();
    long long min_idx = 0;

    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        CompareType val = Traits::load(input, idx);
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

// Dimension-wise prod reduction
template<typename T, typename Acc>
__device__ void reduce_prod_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, Acc>;
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ Acc shared[256];
    unsigned int tid = threadIdx.x;

    Acc prod = Traits::one();
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        prod = Traits::mul(prod, Traits::load(input, idx));
    }

    shared[tid] = prod;
    __syncthreads();

    // Tree reduction with multiplication
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = Traits::mul(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        Traits::store(output, outer_idx * inner_size + inner_idx, shared[0]);
    }
}

// Dimension-wise any reduction (logical OR - true if any element is non-zero)
// Works on any dtype by converting to bool (non-zero = true)
template<typename T, typename CompareType>
__device__ void reduce_any_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, CompareType>;
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ int shared[256];  // Use int for boolean reduction
    unsigned int tid = threadIdx.x;

    int any_true = 0;  // false
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        CompareType val = Traits::load(input, idx);
        // Non-zero means true
        if (val != Traits::zero()) {
            any_true = 1;
        }
    }

    shared[tid] = any_true;
    __syncthreads();

    // Tree reduction with OR
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = shared[tid] | shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Store 1 if any true, 0 otherwise (in output dtype)
        Traits::store(output, outer_idx * inner_size + inner_idx,
                      shared[0] ? Traits::one() : Traits::zero());
    }
}

// Dimension-wise all reduction (logical AND - true if all elements are non-zero)
template<typename T, typename CompareType>
__device__ void reduce_all_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, CompareType>;
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    __shared__ int shared[256];  // Use int for boolean reduction
    unsigned int tid = threadIdx.x;

    int all_true = 1;  // true
    for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
        unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        CompareType val = Traits::load(input, idx);
        // Zero means false
        if (val == Traits::zero()) {
            all_true = 0;
        }
    }

    shared[tid] = all_true;
    __syncthreads();

    // Tree reduction with AND
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = shared[tid] & shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Store 1 if all true, 0 otherwise (in output dtype)
        Traits::store(output, outer_idx * inner_size + inner_idx,
                      shared[0] ? Traits::one() : Traits::zero());
    }
}

// ============================================================================
// extern "C" wrapper kernels for Rust FFI
// Each kernel calls the corresponding templated __device__ function
// ============================================================================

extern "C" {

// ----------------------------------------------------------------------------
// F32 kernels (native accumulation)
// ----------------------------------------------------------------------------

__global__ void reduce_sum_f32(const float* input, float* output, unsigned int n) {
    reduce_sum_impl<float, float>(input, output, n);
}
__global__ void reduce_max_f32(const float* input, float* output, unsigned int n) {
    reduce_max_impl<float, float>(input, output, n);
}
__global__ void reduce_min_f32(const float* input, float* output, unsigned int n) {
    reduce_min_impl<float, float>(input, output, n);
}
__global__ void reduce_prod_f32(const float* input, float* output, unsigned int n) {
    reduce_prod_impl<float, float>(input, output, n);
}

__global__ void reduce_sum_dim_f32(
    const float* input, float* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_sum_dim_impl<float, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_max_dim_f32(
    const float* input, float* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_max_dim_impl<float, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_min_dim_f32(
    const float* input, float* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_min_dim_impl<float, float>(input, output, outer_size, reduce_size, inner_size);
}

// F32 with F64 accumulation
__global__ void reduce_sum_dim_f32_fp64acc(
    const float* input, float* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_sum_dim_impl<float, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_max_dim_f32_fp64acc(
    const float* input, float* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_max_dim_impl<float, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_min_dim_f32_fp64acc(
    const float* input, float* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_min_dim_impl<float, double>(input, output, outer_size, reduce_size, inner_size);
}

__global__ void argmax_dim_f32(
    const float* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    argmax_dim_impl<float, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void argmin_dim_f32(
    const float* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    argmin_dim_impl<float, float>(input, output, outer_size, reduce_size, inner_size);
}

__global__ void reduce_prod_dim_f32(
    const float* input, float* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_prod_dim_impl<float, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_any_dim_f32(
    const float* input, float* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_any_dim_impl<float, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_all_dim_f32(
    const float* input, float* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_all_dim_impl<float, float>(input, output, outer_size, reduce_size, inner_size);
}

// F32 with F64 accumulation for prod
__global__ void reduce_prod_dim_f32_fp64acc(
    const float* input, float* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_prod_dim_impl<float, double>(input, output, outer_size, reduce_size, inner_size);
}

// ----------------------------------------------------------------------------
// F64 kernels (native accumulation only)
// ----------------------------------------------------------------------------

__global__ void reduce_sum_f64(const double* input, double* output, unsigned int n) {
    reduce_sum_impl<double, double>(input, output, n);
}
__global__ void reduce_max_f64(const double* input, double* output, unsigned int n) {
    reduce_max_impl<double, double>(input, output, n);
}
__global__ void reduce_min_f64(const double* input, double* output, unsigned int n) {
    reduce_min_impl<double, double>(input, output, n);
}
__global__ void reduce_prod_f64(const double* input, double* output, unsigned int n) {
    reduce_prod_impl<double, double>(input, output, n);
}

__global__ void reduce_sum_dim_f64(
    const double* input, double* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_sum_dim_impl<double, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_max_dim_f64(
    const double* input, double* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_max_dim_impl<double, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_min_dim_f64(
    const double* input, double* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_min_dim_impl<double, double>(input, output, outer_size, reduce_size, inner_size);
}

__global__ void argmax_dim_f64(
    const double* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    argmax_dim_impl<double, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void argmin_dim_f64(
    const double* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    argmin_dim_impl<double, double>(input, output, outer_size, reduce_size, inner_size);
}

__global__ void reduce_prod_dim_f64(
    const double* input, double* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_prod_dim_impl<double, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_any_dim_f64(
    const double* input, double* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_any_dim_impl<double, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_all_dim_f64(
    const double* input, double* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_all_dim_impl<double, double>(input, output, outer_size, reduce_size, inner_size);
}

// ----------------------------------------------------------------------------
// F16 kernels - native and fp32acc/fp64acc variants
// ----------------------------------------------------------------------------

__global__ void reduce_sum_f16(const __half* input, __half* output, unsigned int n) {
    reduce_sum_impl<__half, __half>(input, output, n);
}
__global__ void reduce_max_f16(const __half* input, __half* output, unsigned int n) {
    reduce_max_impl<__half, __half>(input, output, n);
}
__global__ void reduce_min_f16(const __half* input, __half* output, unsigned int n) {
    reduce_min_impl<__half, __half>(input, output, n);
}
__global__ void reduce_prod_f16(const __half* input, __half* output, unsigned int n) {
    reduce_prod_impl<__half, __half>(input, output, n);
}

__global__ void reduce_sum_dim_f16(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_sum_dim_impl<__half, __half>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_max_dim_f16(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_max_dim_impl<__half, __half>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_min_dim_f16(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_min_dim_impl<__half, __half>(input, output, outer_size, reduce_size, inner_size);
}

// F16 with FP32 accumulation
__global__ void reduce_sum_f16_fp32acc(const __half* input, __half* output, unsigned int n) {
    reduce_sum_impl<__half, float>(input, output, n);
}
__global__ void reduce_max_f16_fp32acc(const __half* input, __half* output, unsigned int n) {
    reduce_max_impl<__half, float>(input, output, n);
}
__global__ void reduce_min_f16_fp32acc(const __half* input, __half* output, unsigned int n) {
    reduce_min_impl<__half, float>(input, output, n);
}
__global__ void reduce_prod_f16_fp32acc(const __half* input, __half* output, unsigned int n) {
    reduce_prod_impl<__half, float>(input, output, n);
}

__global__ void reduce_sum_dim_f16_fp32acc(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_sum_dim_impl<__half, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_max_dim_f16_fp32acc(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_max_dim_impl<__half, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_min_dim_f16_fp32acc(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_min_dim_impl<__half, float>(input, output, outer_size, reduce_size, inner_size);
}

// F16 with FP64 accumulation
__global__ void reduce_sum_dim_f16_fp64acc(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_sum_dim_impl<__half, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_max_dim_f16_fp64acc(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_max_dim_impl<__half, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_min_dim_f16_fp64acc(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_min_dim_impl<__half, double>(input, output, outer_size, reduce_size, inner_size);
}

__global__ void argmax_dim_f16(
    const __half* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    argmax_dim_impl<__half, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void argmin_dim_f16(
    const __half* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    argmin_dim_impl<__half, float>(input, output, outer_size, reduce_size, inner_size);
}

__global__ void reduce_prod_dim_f16(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_prod_dim_impl<__half, __half>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_prod_dim_f16_fp32acc(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_prod_dim_impl<__half, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_prod_dim_f16_fp64acc(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_prod_dim_impl<__half, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_any_dim_f16(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_any_dim_impl<__half, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_all_dim_f16(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_all_dim_impl<__half, float>(input, output, outer_size, reduce_size, inner_size);
}

// ----------------------------------------------------------------------------
// BF16 kernels - native and fp32acc/fp64acc variants
// ----------------------------------------------------------------------------

__global__ void reduce_sum_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int n) {
    reduce_sum_impl<__nv_bfloat16, __nv_bfloat16>(input, output, n);
}
__global__ void reduce_max_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int n) {
    reduce_max_impl<__nv_bfloat16, __nv_bfloat16>(input, output, n);
}
__global__ void reduce_min_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int n) {
    reduce_min_impl<__nv_bfloat16, __nv_bfloat16>(input, output, n);
}
__global__ void reduce_prod_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int n) {
    reduce_prod_impl<__nv_bfloat16, __nv_bfloat16>(input, output, n);
}

__global__ void reduce_sum_dim_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_sum_dim_impl<__nv_bfloat16, __nv_bfloat16>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_max_dim_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_max_dim_impl<__nv_bfloat16, __nv_bfloat16>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_min_dim_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_min_dim_impl<__nv_bfloat16, __nv_bfloat16>(input, output, outer_size, reduce_size, inner_size);
}

// BF16 with FP32 accumulation
__global__ void reduce_sum_bf16_fp32acc(const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int n) {
    reduce_sum_impl<__nv_bfloat16, float>(input, output, n);
}
__global__ void reduce_max_bf16_fp32acc(const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int n) {
    reduce_max_impl<__nv_bfloat16, float>(input, output, n);
}
__global__ void reduce_min_bf16_fp32acc(const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int n) {
    reduce_min_impl<__nv_bfloat16, float>(input, output, n);
}
__global__ void reduce_prod_bf16_fp32acc(const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int n) {
    reduce_prod_impl<__nv_bfloat16, float>(input, output, n);
}

__global__ void reduce_sum_dim_bf16_fp32acc(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_sum_dim_impl<__nv_bfloat16, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_max_dim_bf16_fp32acc(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_max_dim_impl<__nv_bfloat16, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_min_dim_bf16_fp32acc(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_min_dim_impl<__nv_bfloat16, float>(input, output, outer_size, reduce_size, inner_size);
}

// BF16 with FP64 accumulation
__global__ void reduce_sum_dim_bf16_fp64acc(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_sum_dim_impl<__nv_bfloat16, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_max_dim_bf16_fp64acc(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_max_dim_impl<__nv_bfloat16, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_min_dim_bf16_fp64acc(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_min_dim_impl<__nv_bfloat16, double>(input, output, outer_size, reduce_size, inner_size);
}

__global__ void argmax_dim_bf16(
    const __nv_bfloat16* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    argmax_dim_impl<__nv_bfloat16, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void argmin_dim_bf16(
    const __nv_bfloat16* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    argmin_dim_impl<__nv_bfloat16, float>(input, output, outer_size, reduce_size, inner_size);
}

// BF16 prod/any/all - native accumulation
__global__ void reduce_prod_dim_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_prod_dim_impl<__nv_bfloat16, __nv_bfloat16>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_prod_dim_bf16_fp32acc(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_prod_dim_impl<__nv_bfloat16, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_prod_dim_bf16_fp64acc(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_prod_dim_impl<__nv_bfloat16, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_any_dim_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_any_dim_impl<__nv_bfloat16, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_all_dim_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_all_dim_impl<__nv_bfloat16, float>(input, output, outer_size, reduce_size, inner_size);
}

// ----------------------------------------------------------------------------
// FP8 E4M3 kernels - FP32 accumulation by default, BF16 and FP64 variants
// ----------------------------------------------------------------------------

__global__ void reduce_sum_fp8_e4m3(const numr_fp8_e4m3* input, numr_fp8_e4m3* output, unsigned int n) {
    reduce_sum_impl<numr_fp8_e4m3, float>(input, output, n);
}
__global__ void reduce_max_fp8_e4m3(const numr_fp8_e4m3* input, numr_fp8_e4m3* output, unsigned int n) {
    reduce_max_impl<numr_fp8_e4m3, float>(input, output, n);
}
__global__ void reduce_min_fp8_e4m3(const numr_fp8_e4m3* input, numr_fp8_e4m3* output, unsigned int n) {
    reduce_min_impl<numr_fp8_e4m3, float>(input, output, n);
}
__global__ void reduce_prod_fp8_e4m3(const numr_fp8_e4m3* input, numr_fp8_e4m3* output, unsigned int n) {
    reduce_prod_impl<numr_fp8_e4m3, float>(input, output, n);
}

__global__ void reduce_sum_dim_fp8_e4m3(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_sum_dim_impl<numr_fp8_e4m3, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_max_dim_fp8_e4m3(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_max_dim_impl<numr_fp8_e4m3, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_min_dim_fp8_e4m3(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_min_dim_impl<numr_fp8_e4m3, float>(input, output, outer_size, reduce_size, inner_size);
}

// FP8 E4M3 with BF16 accumulation
__global__ void reduce_sum_fp8_e4m3_bf16acc(const numr_fp8_e4m3* input, numr_fp8_e4m3* output, unsigned int n) {
    reduce_sum_impl<numr_fp8_e4m3, __nv_bfloat16>(input, output, n);
}
__global__ void reduce_max_fp8_e4m3_bf16acc(const numr_fp8_e4m3* input, numr_fp8_e4m3* output, unsigned int n) {
    reduce_max_impl<numr_fp8_e4m3, __nv_bfloat16>(input, output, n);
}
__global__ void reduce_min_fp8_e4m3_bf16acc(const numr_fp8_e4m3* input, numr_fp8_e4m3* output, unsigned int n) {
    reduce_min_impl<numr_fp8_e4m3, __nv_bfloat16>(input, output, n);
}
__global__ void reduce_prod_fp8_e4m3_bf16acc(const numr_fp8_e4m3* input, numr_fp8_e4m3* output, unsigned int n) {
    reduce_prod_impl<numr_fp8_e4m3, __nv_bfloat16>(input, output, n);
}

__global__ void reduce_sum_dim_fp8_e4m3_bf16acc(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_sum_dim_impl<numr_fp8_e4m3, __nv_bfloat16>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_max_dim_fp8_e4m3_bf16acc(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_max_dim_impl<numr_fp8_e4m3, __nv_bfloat16>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_min_dim_fp8_e4m3_bf16acc(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_min_dim_impl<numr_fp8_e4m3, __nv_bfloat16>(input, output, outer_size, reduce_size, inner_size);
}

// FP8 E4M3 with FP64 accumulation
__global__ void reduce_sum_dim_fp8_e4m3_fp64acc(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_sum_dim_impl<numr_fp8_e4m3, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_max_dim_fp8_e4m3_fp64acc(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_max_dim_impl<numr_fp8_e4m3, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_min_dim_fp8_e4m3_fp64acc(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_min_dim_impl<numr_fp8_e4m3, double>(input, output, outer_size, reduce_size, inner_size);
}

__global__ void argmax_dim_fp8_e4m3(
    const numr_fp8_e4m3* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    argmax_dim_impl<numr_fp8_e4m3, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void argmin_dim_fp8_e4m3(
    const numr_fp8_e4m3* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    argmin_dim_impl<numr_fp8_e4m3, float>(input, output, outer_size, reduce_size, inner_size);
}

// FP8 E4M3 prod/any/all - FP32 accumulation by default
__global__ void reduce_prod_dim_fp8_e4m3(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_prod_dim_impl<numr_fp8_e4m3, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_prod_dim_fp8_e4m3_bf16acc(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_prod_dim_impl<numr_fp8_e4m3, __nv_bfloat16>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_prod_dim_fp8_e4m3_fp64acc(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_prod_dim_impl<numr_fp8_e4m3, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_any_dim_fp8_e4m3(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_any_dim_impl<numr_fp8_e4m3, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_all_dim_fp8_e4m3(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_all_dim_impl<numr_fp8_e4m3, float>(input, output, outer_size, reduce_size, inner_size);
}

// ----------------------------------------------------------------------------
// FP8 E5M2 kernels - FP32 accumulation by default, BF16 and FP64 variants
// ----------------------------------------------------------------------------

__global__ void reduce_sum_fp8_e5m2(const numr_fp8_e5m2* input, numr_fp8_e5m2* output, unsigned int n) {
    reduce_sum_impl<numr_fp8_e5m2, float>(input, output, n);
}
__global__ void reduce_max_fp8_e5m2(const numr_fp8_e5m2* input, numr_fp8_e5m2* output, unsigned int n) {
    reduce_max_impl<numr_fp8_e5m2, float>(input, output, n);
}
__global__ void reduce_min_fp8_e5m2(const numr_fp8_e5m2* input, numr_fp8_e5m2* output, unsigned int n) {
    reduce_min_impl<numr_fp8_e5m2, float>(input, output, n);
}
__global__ void reduce_prod_fp8_e5m2(const numr_fp8_e5m2* input, numr_fp8_e5m2* output, unsigned int n) {
    reduce_prod_impl<numr_fp8_e5m2, float>(input, output, n);
}

__global__ void reduce_sum_dim_fp8_e5m2(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_sum_dim_impl<numr_fp8_e5m2, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_max_dim_fp8_e5m2(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_max_dim_impl<numr_fp8_e5m2, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_min_dim_fp8_e5m2(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_min_dim_impl<numr_fp8_e5m2, float>(input, output, outer_size, reduce_size, inner_size);
}

// FP8 E5M2 with BF16 accumulation
__global__ void reduce_sum_fp8_e5m2_bf16acc(const numr_fp8_e5m2* input, numr_fp8_e5m2* output, unsigned int n) {
    reduce_sum_impl<numr_fp8_e5m2, __nv_bfloat16>(input, output, n);
}
__global__ void reduce_max_fp8_e5m2_bf16acc(const numr_fp8_e5m2* input, numr_fp8_e5m2* output, unsigned int n) {
    reduce_max_impl<numr_fp8_e5m2, __nv_bfloat16>(input, output, n);
}
__global__ void reduce_min_fp8_e5m2_bf16acc(const numr_fp8_e5m2* input, numr_fp8_e5m2* output, unsigned int n) {
    reduce_min_impl<numr_fp8_e5m2, __nv_bfloat16>(input, output, n);
}
__global__ void reduce_prod_fp8_e5m2_bf16acc(const numr_fp8_e5m2* input, numr_fp8_e5m2* output, unsigned int n) {
    reduce_prod_impl<numr_fp8_e5m2, __nv_bfloat16>(input, output, n);
}

__global__ void reduce_sum_dim_fp8_e5m2_bf16acc(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_sum_dim_impl<numr_fp8_e5m2, __nv_bfloat16>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_max_dim_fp8_e5m2_bf16acc(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_max_dim_impl<numr_fp8_e5m2, __nv_bfloat16>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_min_dim_fp8_e5m2_bf16acc(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_min_dim_impl<numr_fp8_e5m2, __nv_bfloat16>(input, output, outer_size, reduce_size, inner_size);
}

// FP8 E5M2 with FP64 accumulation
__global__ void reduce_sum_dim_fp8_e5m2_fp64acc(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_sum_dim_impl<numr_fp8_e5m2, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_max_dim_fp8_e5m2_fp64acc(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_max_dim_impl<numr_fp8_e5m2, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_min_dim_fp8_e5m2_fp64acc(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_min_dim_impl<numr_fp8_e5m2, double>(input, output, outer_size, reduce_size, inner_size);
}

__global__ void argmax_dim_fp8_e5m2(
    const numr_fp8_e5m2* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    argmax_dim_impl<numr_fp8_e5m2, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void argmin_dim_fp8_e5m2(
    const numr_fp8_e5m2* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    argmin_dim_impl<numr_fp8_e5m2, float>(input, output, outer_size, reduce_size, inner_size);
}

// FP8 E5M2 prod/any/all - FP32 accumulation by default
__global__ void reduce_prod_dim_fp8_e5m2(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_prod_dim_impl<numr_fp8_e5m2, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_prod_dim_fp8_e5m2_bf16acc(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_prod_dim_impl<numr_fp8_e5m2, __nv_bfloat16>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_prod_dim_fp8_e5m2_fp64acc(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_prod_dim_impl<numr_fp8_e5m2, double>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_any_dim_fp8_e5m2(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_any_dim_impl<numr_fp8_e5m2, float>(input, output, outer_size, reduce_size, inner_size);
}
__global__ void reduce_all_dim_fp8_e5m2(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    reduce_all_dim_impl<numr_fp8_e5m2, float>(input, output, outer_size, reduce_size, inner_size);
}

} // extern "C"
