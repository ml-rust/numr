// Argmax, argmin, any and all device functions for `reduce.cu`.

#pragma once

#include "dtype_traits.cuh"
#include "arg_reduce.cuh"

// Argmax or argmin along a dimension. Compares in float, or double for f64.
// Ties and NaN follow `arg_take` in `arg_reduce.cuh`.
template<typename T, typename CompareType, bool IsMax>
__device__ void arg_dim_impl(
    const T* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, CompareType>;
    __shared__ CompareType shared_val[256];
    __shared__ long long shared_idx[256];
    unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
            CompareType best = IsMax ? Traits::neg_inf() : Traits::pos_inf();
            long long best_idx = ARG_NO_INDEX;

            for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
                unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
                CompareType val = Traits::load(input, idx);
                if (arg_take<IsMax>(val, (long long)i, best, best_idx)) {
                    best = val;
                    best_idx = i;
                }
            }

            shared_val[tid] = best;
            shared_idx[tid] = best_idx;
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s && arg_take<IsMax>(shared_val[tid + s], shared_idx[tid + s],
                                               shared_val[tid], shared_idx[tid])) {
                    shared_val[tid] = shared_val[tid + s];
                    shared_idx[tid] = shared_idx[tid + s];
                }
                __syncthreads();
            }

            if (tid == 0) {
                long long found = shared_idx[0];
                output[outer_idx * inner_size + inner_idx] = found == ARG_NO_INDEX ? 0 : found;
            }
            __syncthreads();
        }
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
    __shared__ int shared[256];  // Use int for boolean reduction
    unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
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
            __syncthreads();
        }
    }
}

// Dimension-wise all reduction (logical AND - true if all elements are non-zero)
template<typename T, typename CompareType>
__device__ void reduce_all_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, CompareType>;
    __shared__ int shared[256];  // Use int for boolean reduction
    unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
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
            __syncthreads();
        }
    }
}
