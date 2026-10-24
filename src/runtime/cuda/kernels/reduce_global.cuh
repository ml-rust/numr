// Global (whole-tensor) reduction device functions for `reduce.cu`.
// Each function produces one partial result per block.

#pragma once

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
