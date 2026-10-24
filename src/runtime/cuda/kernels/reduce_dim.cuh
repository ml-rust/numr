// Dimension-wise reduction device functions for `reduce.cu`:
// sum, max, min and prod, plus the serial variants for a short reduced axis.

#pragma once

#include "dtype_traits.cuh"

// ============================================================================
// Dimension-wise reductions: grid-stride over BOTH output axes
//
// The tensor is viewed as [outer_size, reduce_size, inner_size] and reduced
// along the middle axis. One thread block owns one (outer, inner) output
// element at a time.
//
// CUDA caps grid dimensions at [2147483647, 65535, 65535], so mapping outer to
// blockIdx.x and inner to blockIdx.y one-to-one makes any launch with
// inner_size > 65535 fail with CUDA_ERROR_INVALID_VALUE (hit by e.g. summing
// [1, 32, 4096, 64] over dim 1, where inner_size = 262144). Both axes therefore
// grid-stride: the launcher clamps each grid axis to its architectural maximum
// and these loops cover the remainder.
//
// The stride loops change only WHICH block computes an output element, never
// how that element is accumulated: each output is still reduced by a single
// block, sequentially over `i += blockDim.x` then through the same shared-memory
// tree. Results are bit-identical to the pre-fix kernels.
//
// The trip counts depend only on blockIdx/gridDim/sizes, so they are uniform
// across every thread of a block and the __syncthreads() calls stay legal.
// ============================================================================

// Dimension-wise sum reduction
template<typename T, typename Acc>
__device__ void reduce_sum_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, Acc>;
    __shared__ Acc shared[256];
    unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
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
            __syncthreads();
        }
    }
}

// ============================================================================
// Serial dim reductions for a short reduced axis
// ============================================================================
//
// The block-per-output kernels above give every (outer, inner) pair a whole
// block: 256 threads, a shared-memory tree and its barriers. When the reduced
// axis is only a handful of elements almost every thread idles and the grid
// runs to millions of near-empty blocks — a weight of shape [2048, 1024, 16]
// summed over its last axis is two million blocks for sixteen adds each.
//
// Below `reduce_size <= 32` (see the launcher) one THREAD owns one output and
// walks the axis in order. Threads are numbered inner-fastest, so for an
// inner reduction they read consecutive short runs and for an outer one they
// read consecutive addresses; either way the warp coalesces.
//
// Accumulation is sequential over the axis, which is the CPU backend's order
// as well; the tree above rounds differently, and both sit inside the parity
// tolerance.

template<typename T, typename Acc>
struct SerialSum {
    static __device__ __forceinline__ Acc identity() { return AccumTraits<T, Acc>::zero(); }
    static __device__ __forceinline__ Acc combine(Acc a, Acc b) { return AccumTraits<T, Acc>::add(a, b); }
};
template<typename T, typename Acc>
struct SerialMax {
    static __device__ __forceinline__ Acc identity() { return AccumTraits<T, Acc>::neg_inf(); }
    static __device__ __forceinline__ Acc combine(Acc a, Acc b) { return AccumTraits<T, Acc>::max(a, b); }
};
template<typename T, typename Acc>
struct SerialMin {
    static __device__ __forceinline__ Acc identity() { return AccumTraits<T, Acc>::pos_inf(); }
    static __device__ __forceinline__ Acc combine(Acc a, Acc b) { return AccumTraits<T, Acc>::min(a, b); }
};
template<typename T, typename Acc>
struct SerialProd {
    static __device__ __forceinline__ Acc identity() { return AccumTraits<T, Acc>::one(); }
    static __device__ __forceinline__ Acc combine(Acc a, Acc b) { return AccumTraits<T, Acc>::mul(a, b); }
};

template<typename T, typename Acc, class Op>
__device__ void reduce_dim_serial_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, Acc>;
    const size_t total = (size_t)outer_size * inner_size;
    const size_t step = (size_t)gridDim.x * blockDim.x;
    for (size_t o = (size_t)blockIdx.x * blockDim.x + threadIdx.x; o < total; o += step) {
        const size_t outer_idx = o / inner_size;
        const size_t inner_idx = o - outer_idx * inner_size;
        const T* base = input + outer_idx * reduce_size * inner_size + inner_idx;
        Acc acc = Op::identity();
        for (unsigned int i = 0; i < reduce_size; i++) {
            acc = Op::combine(acc, Traits::load(base, (int)(i * inner_size)));
        }
        Traits::store(output + o, 0, acc);
    }
}

// Dimension-wise max reduction
template<typename T, typename Acc>
__device__ void reduce_max_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, Acc>;
    __shared__ Acc shared[256];
    unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
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
            __syncthreads();
        }
    }
}

// Dimension-wise min reduction
template<typename T, typename Acc>
__device__ void reduce_min_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, Acc>;
    __shared__ Acc shared[256];
    unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
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
            __syncthreads();
        }
    }
}

// Dimension-wise prod reduction
template<typename T, typename Acc>
__device__ void reduce_prod_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, Acc>;
    __shared__ Acc shared[256];
    unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
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
            __syncthreads();
        }
    }
}
