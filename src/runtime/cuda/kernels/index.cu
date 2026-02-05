// Indexing CUDA kernels - gather, scatter, index_select, masked_select, masked_fill
// Supports: f32, f64, f16, bf16, i32, i64
//
// Each operation is defined per-dtype (no templates for extern "C" compatibility)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Helper Macros for Multi-Dtype Kernel Generation
// ============================================================================

// Macro for gather kernel
#define DEFINE_GATHER_KERNEL(suffix, dtype) \
__global__ void gather_##suffix( \
    const dtype* __restrict__ input, \
    const long long* __restrict__ indices, \
    dtype* __restrict__ output, \
    unsigned int ndim, \
    unsigned int dim, \
    const unsigned int* __restrict__ input_shape, \
    const unsigned int* __restrict__ input_strides, \
    const unsigned int* __restrict__ output_shape, \
    const unsigned int* __restrict__ output_strides, \
    unsigned int total_elements \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= total_elements) return; \
    \
    unsigned int remaining = idx; \
    unsigned int src_offset = 0; \
    \
    for (unsigned int d = 0; d < ndim; d++) { \
        unsigned int coord = remaining / output_strides[d]; \
        remaining %= output_strides[d]; \
        \
        if (d == dim) { \
            long long index_val = indices[idx]; \
            if (index_val < 0 || (unsigned int)index_val >= input_shape[d]) { \
                output[idx] = (dtype)0; \
                return; \
            } \
            src_offset += (unsigned int)index_val * input_strides[d]; \
        } else { \
            src_offset += coord * input_strides[d]; \
        } \
    } \
    \
    output[idx] = input[src_offset]; \
}

// Macro for scatter kernel
#define DEFINE_SCATTER_KERNEL(suffix, dtype) \
__global__ void scatter_##suffix( \
    const dtype* __restrict__ input, \
    const long long* __restrict__ indices, \
    const dtype* __restrict__ src, \
    dtype* __restrict__ output, \
    unsigned int ndim, \
    unsigned int dim, \
    const unsigned int* __restrict__ output_shape, \
    const unsigned int* __restrict__ output_strides, \
    const unsigned int* __restrict__ src_shape, \
    const unsigned int* __restrict__ src_strides, \
    unsigned int src_total \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= src_total) return; \
    \
    unsigned int remaining = idx; \
    unsigned int dst_offset = 0; \
    \
    for (unsigned int d = 0; d < ndim; d++) { \
        unsigned int coord = remaining / src_strides[d]; \
        remaining %= src_strides[d]; \
        \
        if (d == dim) { \
            long long index_val = indices[idx]; \
            if (index_val < 0 || (unsigned int)index_val >= output_shape[d]) { \
                return; \
            } \
            dst_offset += (unsigned int)index_val * output_strides[d]; \
        } else { \
            dst_offset += coord * output_strides[d]; \
        } \
    } \
    \
    output[dst_offset] = src[idx]; \
}

// Macro for copy kernel (for scatter initialization)
#define DEFINE_COPY_KERNEL(suffix, dtype) \
__global__ void copy_##suffix( \
    const dtype* __restrict__ src, \
    dtype* __restrict__ dst, \
    unsigned int n \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < n) { \
        dst[idx] = src[idx]; \
    } \
}

// Macro for index_select kernel
#define DEFINE_INDEX_SELECT_KERNEL(suffix, dtype) \
__global__ void index_select_##suffix( \
    const dtype* __restrict__ input, \
    const long long* __restrict__ indices, \
    dtype* __restrict__ output, \
    unsigned int outer_size, \
    unsigned int dim_size, \
    unsigned int inner_size, \
    unsigned int index_len \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = outer_size * index_len * inner_size; \
    if (idx >= total) return; \
    \
    unsigned int inner = idx % inner_size; \
    unsigned int sel_idx = (idx / inner_size) % index_len; \
    unsigned int outer = idx / (index_len * inner_size); \
    \
    long long index_val = indices[sel_idx]; \
    if (index_val < 0 || (unsigned int)index_val >= dim_size) { \
        output[idx] = (dtype)0; \
        return; \
    } \
    \
    unsigned int src_offset = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner; \
    output[idx] = input[src_offset]; \
}

// Macro for masked_select kernel
#define DEFINE_MASKED_SELECT_KERNEL(suffix, dtype) \
__global__ void masked_select_##suffix( \
    const dtype* __restrict__ input, \
    const unsigned char* __restrict__ mask, \
    dtype* __restrict__ output, \
    const unsigned int* __restrict__ prefix_sum, \
    unsigned int n \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return; \
    \
    if (mask[idx] != 0) { \
        unsigned int out_idx = prefix_sum[idx]; \
        output[out_idx] = input[idx]; \
    } \
}

// Macro for masked_fill kernel (float types)
#define DEFINE_MASKED_FILL_KERNEL(suffix, dtype) \
__global__ void masked_fill_##suffix( \
    const dtype* __restrict__ input, \
    const unsigned char* __restrict__ mask, \
    dtype* __restrict__ output, \
    dtype fill_value, \
    unsigned int n \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return; \
    \
    output[idx] = (mask[idx] != 0) ? fill_value : input[idx]; \
}

// Macro for index_put kernel
// Inverse of index_select: puts values from src at positions specified by indices.
// Output is pre-initialized with a copy of input tensor.
// For each position (outer, sel_idx, inner):
//   out[outer * dim_size * inner_size + indices[sel_idx] * inner_size + inner] = src[outer * index_len * inner_size + sel_idx * inner_size + inner]
#define DEFINE_INDEX_PUT_KERNEL(suffix, dtype) \
__global__ void index_put_##suffix( \
    const long long* __restrict__ indices, \
    const dtype* __restrict__ src, \
    dtype* __restrict__ output, \
    unsigned int outer_size, \
    unsigned int dim_size, \
    unsigned int inner_size, \
    unsigned int index_len \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = outer_size * index_len * inner_size; \
    if (idx >= total) return; \
    \
    unsigned int inner = idx % inner_size; \
    unsigned int sel_idx = (idx / inner_size) % index_len; \
    unsigned int outer = idx / (index_len * inner_size); \
    \
    long long index_val = indices[sel_idx]; \
    if (index_val < 0 || (unsigned int)index_val >= dim_size) { \
        return; /* Out of bounds - skip */ \
    } \
    \
    unsigned int dst_offset = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner; \
    output[dst_offset] = src[idx]; \
}

// Macro for embedding_lookup kernel
// Industry-standard embedding table lookup operation for neural networks.
// Input: embeddings [vocab_size, embedding_dim], indices [num_indices]
// Output: output [num_indices, embedding_dim]
// Each thread handles one index, copying the entire embedding row to output.
#define DEFINE_EMBEDDING_LOOKUP_KERNEL(suffix, dtype) \
__global__ void embedding_lookup_##suffix( \
    const dtype* __restrict__ embeddings, \
    const long long* __restrict__ indices, \
    dtype* __restrict__ output, \
    unsigned int num_indices, \
    unsigned int vocab_size, \
    unsigned int embedding_dim \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= num_indices) return; \
    \
    long long index_val = indices[idx]; \
    dtype* out_row = output + (idx * embedding_dim); \
    \
    /* Check bounds */ \
    if (index_val < 0 || (unsigned int)index_val >= vocab_size) { \
        /* Out of bounds - fill with zeros */ \
        for (unsigned int i = 0; i < embedding_dim; i++) { \
            out_row[i] = (dtype)0; \
        } \
        return; \
    } \
    \
    const dtype* emb_row = embeddings + ((unsigned int)index_val * embedding_dim); \
    \
    /* Copy the entire embedding row to output (coalesced writes) */ \
    for (unsigned int i = 0; i < embedding_dim; i++) { \
        out_row[i] = emb_row[i]; \
    } \
}

extern "C" {

// ============================================================================
// Masked Count Kernel (dtype-independent)
// ============================================================================

__global__ void masked_count_kernel(
    const unsigned char* __restrict__ mask,
    unsigned int* __restrict__ count,
    unsigned int n
) {
    __shared__ unsigned int shared_count;
    if (threadIdx.x == 0) {
        shared_count = 0;
    }
    __syncthreads();

    unsigned int local_count = 0;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (mask[i] != 0) {
            local_count++;
        }
    }

    atomicAdd(&shared_count, local_count);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(count, shared_count);
    }
}

// ============================================================================
// Masked Prefix Sum Kernel (dtype-independent)
// ============================================================================

__global__ void masked_prefix_sum_kernel(
    const unsigned char* __restrict__ mask,
    unsigned int* __restrict__ prefix_sum,
    unsigned int n
) {
    // Simple sequential prefix sum - for small tensors or as fallback
    // For large tensors, a parallel scan algorithm would be used
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned int sum = 0;
        for (unsigned int i = 0; i < n; i++) {
            prefix_sum[i] = sum;
            if (mask[i] != 0) {
                sum++;
            }
        }
    }
}

// ============================================================================
// Broadcast Masked Operations
// These kernels support broadcasting the mask to the input/output shape.
// Uses stride-based indexing where stride=0 means broadcast dimension.
// ============================================================================

// Device function to compute mask index with broadcasting
__device__ __forceinline__ unsigned int compute_broadcast_index(
    unsigned int linear_idx,
    const unsigned int* __restrict__ mask_strides,
    const unsigned int* __restrict__ out_shape,
    unsigned int ndim
) {
    unsigned int mask_offset = 0;
    unsigned int remaining = linear_idx;

    for (int d = ndim - 1; d >= 0; d--) {
        unsigned int coord = remaining % out_shape[d];
        remaining /= out_shape[d];
        mask_offset += coord * mask_strides[d];
    }
    return mask_offset;
}

// Broadcast masked count - counts true elements in mask broadcast to output shape
__global__ void masked_count_broadcast_kernel(
    const unsigned char* __restrict__ mask,
    unsigned int* __restrict__ count,
    const unsigned int* __restrict__ mask_strides,
    const unsigned int* __restrict__ out_shape,
    unsigned int ndim,
    unsigned int n
) {
    __shared__ unsigned int shared_count;
    if (threadIdx.x == 0) {
        shared_count = 0;
    }
    __syncthreads();

    unsigned int local_count = 0;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        unsigned int mask_idx = compute_broadcast_index(i, mask_strides, out_shape, ndim);
        if (mask[mask_idx] != 0) {
            local_count++;
        }
    }

    atomicAdd(&shared_count, local_count);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(count, shared_count);
    }
}

// Broadcast masked prefix sum - computes prefix sum with broadcast mask
__global__ void masked_prefix_sum_broadcast_kernel(
    const unsigned char* __restrict__ mask,
    unsigned int* __restrict__ prefix_sum,
    const unsigned int* __restrict__ mask_strides,
    const unsigned int* __restrict__ out_shape,
    unsigned int ndim,
    unsigned int n
) {
    // Sequential prefix sum (for small tensors or as fallback)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned int sum = 0;
        for (unsigned int i = 0; i < n; i++) {
            prefix_sum[i] = sum;
            unsigned int mask_idx = compute_broadcast_index(i, mask_strides, out_shape, ndim);
            if (mask[mask_idx] != 0) {
                sum++;
            }
        }
    }
}

// Macro for broadcast masked_select kernel
#define DEFINE_MASKED_SELECT_BROADCAST_KERNEL(suffix, dtype) \
__global__ void masked_select_broadcast_##suffix( \
    const dtype* __restrict__ input, \
    const unsigned char* __restrict__ mask, \
    dtype* __restrict__ output, \
    const unsigned int* __restrict__ prefix_sum, \
    const unsigned int* __restrict__ mask_strides, \
    const unsigned int* __restrict__ out_shape, \
    unsigned int ndim, \
    unsigned int n \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return; \
    \
    unsigned int mask_idx = compute_broadcast_index(idx, mask_strides, out_shape, ndim); \
    if (mask[mask_idx] != 0) { \
        unsigned int out_idx = prefix_sum[idx]; \
        output[out_idx] = input[idx]; \
    } \
}

// Macro for broadcast masked_fill kernel
#define DEFINE_MASKED_FILL_BROADCAST_KERNEL(suffix, dtype) \
__global__ void masked_fill_broadcast_##suffix( \
    const dtype* __restrict__ input, \
    const unsigned char* __restrict__ mask, \
    dtype* __restrict__ output, \
    dtype fill_value, \
    const unsigned int* __restrict__ mask_strides, \
    const unsigned int* __restrict__ out_shape, \
    unsigned int ndim, \
    unsigned int n \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return; \
    \
    unsigned int mask_idx = compute_broadcast_index(idx, mask_strides, out_shape, ndim); \
    output[idx] = (mask[mask_idx] != 0) ? fill_value : input[idx]; \
}

// Instantiate broadcast masked operations for all types
DEFINE_MASKED_SELECT_BROADCAST_KERNEL(f32, float)
DEFINE_MASKED_SELECT_BROADCAST_KERNEL(f64, double)
DEFINE_MASKED_SELECT_BROADCAST_KERNEL(f16, __half)
DEFINE_MASKED_SELECT_BROADCAST_KERNEL(bf16, __nv_bfloat16)
DEFINE_MASKED_SELECT_BROADCAST_KERNEL(i32, int)
DEFINE_MASKED_SELECT_BROADCAST_KERNEL(i64, long long)

DEFINE_MASKED_FILL_BROADCAST_KERNEL(f32, float)
DEFINE_MASKED_FILL_BROADCAST_KERNEL(f64, double)
DEFINE_MASKED_FILL_BROADCAST_KERNEL(f16, __half)
DEFINE_MASKED_FILL_BROADCAST_KERNEL(bf16, __nv_bfloat16)
DEFINE_MASKED_FILL_BROADCAST_KERNEL(i32, int)
DEFINE_MASKED_FILL_BROADCAST_KERNEL(i64, long long)

// ============================================================================
// Index Bounds Validation Kernel (dtype-independent)
// ============================================================================

// Validates that all indices are within bounds [0, dim_size).
// Atomically counts the number of out-of-bounds indices.
// Returns count in error_count[0]. If count > 0, some indices are invalid.
__global__ void validate_indices_kernel(
    const long long* __restrict__ indices,
    unsigned int* __restrict__ error_count,
    unsigned int index_len,
    unsigned int dim_size
) {
    __shared__ unsigned int shared_count;
    if (threadIdx.x == 0) {
        shared_count = 0;
    }
    __syncthreads();

    unsigned int local_count = 0;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < index_len; i += blockDim.x * gridDim.x) {
        long long idx = indices[i];
        if (idx < 0 || idx >= (long long)dim_size) {
            local_count++;
        }
    }

    if (local_count > 0) {
        atomicAdd(&shared_count, local_count);
    }
    __syncthreads();

    if (threadIdx.x == 0 && shared_count > 0) {
        atomicAdd(error_count, shared_count);
    }
}

// ============================================================================
// F32 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(f32, float)
DEFINE_SCATTER_KERNEL(f32, float)
DEFINE_COPY_KERNEL(f32, float)
DEFINE_INDEX_SELECT_KERNEL(f32, float)
DEFINE_INDEX_PUT_KERNEL(f32, float)
DEFINE_MASKED_SELECT_KERNEL(f32, float)
DEFINE_MASKED_FILL_KERNEL(f32, float)
DEFINE_EMBEDDING_LOOKUP_KERNEL(f32, float)

// ============================================================================
// F64 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(f64, double)
DEFINE_SCATTER_KERNEL(f64, double)
DEFINE_COPY_KERNEL(f64, double)
DEFINE_INDEX_SELECT_KERNEL(f64, double)
DEFINE_INDEX_PUT_KERNEL(f64, double)
DEFINE_MASKED_SELECT_KERNEL(f64, double)
DEFINE_MASKED_FILL_KERNEL(f64, double)
DEFINE_EMBEDDING_LOOKUP_KERNEL(f64, double)

// ============================================================================
// F16 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(f16, __half)
DEFINE_SCATTER_KERNEL(f16, __half)
DEFINE_COPY_KERNEL(f16, __half)
DEFINE_INDEX_SELECT_KERNEL(f16, __half)
DEFINE_INDEX_PUT_KERNEL(f16, __half)
DEFINE_MASKED_SELECT_KERNEL(f16, __half)
DEFINE_MASKED_FILL_KERNEL(f16, __half)
DEFINE_EMBEDDING_LOOKUP_KERNEL(f16, __half)

// ============================================================================
// BF16 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(bf16, __nv_bfloat16)
DEFINE_SCATTER_KERNEL(bf16, __nv_bfloat16)
DEFINE_COPY_KERNEL(bf16, __nv_bfloat16)
DEFINE_INDEX_SELECT_KERNEL(bf16, __nv_bfloat16)
DEFINE_INDEX_PUT_KERNEL(bf16, __nv_bfloat16)
DEFINE_MASKED_SELECT_KERNEL(bf16, __nv_bfloat16)
DEFINE_MASKED_FILL_KERNEL(bf16, __nv_bfloat16)
DEFINE_EMBEDDING_LOOKUP_KERNEL(bf16, __nv_bfloat16)

// ============================================================================
// I32 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(i32, int)
DEFINE_SCATTER_KERNEL(i32, int)
DEFINE_COPY_KERNEL(i32, int)
DEFINE_INDEX_SELECT_KERNEL(i32, int)
DEFINE_INDEX_PUT_KERNEL(i32, int)
DEFINE_MASKED_SELECT_KERNEL(i32, int)
DEFINE_MASKED_FILL_KERNEL(i32, int)
DEFINE_EMBEDDING_LOOKUP_KERNEL(i32, int)

// ============================================================================
// I64 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(i64, long long)
DEFINE_SCATTER_KERNEL(i64, long long)
DEFINE_COPY_KERNEL(i64, long long)
DEFINE_INDEX_SELECT_KERNEL(i64, long long)
DEFINE_INDEX_PUT_KERNEL(i64, long long)
DEFINE_MASKED_SELECT_KERNEL(i64, long long)
DEFINE_MASKED_FILL_KERNEL(i64, long long)
DEFINE_EMBEDDING_LOOKUP_KERNEL(i64, long long)

// ============================================================================
// Gather ND - N-dimensional gather operation
// Gathers slices from input at positions specified by indices tensor.
// indices: (num_slices, index_depth) where index_depth <= ndim
// output: (num_slices, remaining_dims...)
// ============================================================================

#define DEFINE_GATHER_ND_KERNEL(suffix, dtype) \
__global__ void gather_nd_##suffix( \
    const dtype* __restrict__ input, \
    const long long* __restrict__ indices, \
    dtype* __restrict__ output, \
    const unsigned int* __restrict__ input_shape, \
    const unsigned int* __restrict__ input_strides, \
    unsigned int num_slices, \
    unsigned int slice_size, \
    unsigned int index_depth, \
    unsigned int ndim \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = num_slices * slice_size; \
    if (idx >= total) return; \
    \
    unsigned int slice_idx = idx / slice_size; \
    unsigned int within_slice = idx % slice_size; \
    \
    /* Compute offset into input from indices */ \
    unsigned int src_offset = 0; \
    bool out_of_bounds = false; \
    for (unsigned int d = 0; d < index_depth; d++) { \
        long long index_val = indices[slice_idx * index_depth + d]; \
        if (index_val < 0 || (unsigned int)index_val >= input_shape[d]) { \
            out_of_bounds = true; \
            break; \
        } \
        src_offset += (unsigned int)index_val * input_strides[d]; \
    } \
    \
    if (out_of_bounds) { \
        output[idx] = (dtype)0; \
        return; \
    } \
    \
    /* Add offset for remaining dimensions */ \
    src_offset += within_slice; \
    output[idx] = input[src_offset]; \
}

// Instantiate gather_nd for all dtypes
DEFINE_GATHER_ND_KERNEL(f32, float)
DEFINE_GATHER_ND_KERNEL(f64, double)
DEFINE_GATHER_ND_KERNEL(f16, __half)
DEFINE_GATHER_ND_KERNEL(bf16, __nv_bfloat16)
DEFINE_GATHER_ND_KERNEL(i32, int)
DEFINE_GATHER_ND_KERNEL(i64, long long)

// ============================================================================
// Bincount - Count occurrences of each value in an integer tensor
// input: 1D tensor of non-negative integers
// weights: Optional 1D tensor of weights (same length as input)
// output: 1D tensor of counts/sums with length = minlength
// Uses atomicAdd for thread-safe accumulation
// ============================================================================

// Bincount without weights (counting)
__global__ void bincount_i32(
    const int* __restrict__ input,
    float* __restrict__ output,
    unsigned int n,
    unsigned int minlength
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int val = input[idx];
    if (val >= 0 && (unsigned int)val < minlength) {
        atomicAdd(&output[val], 1.0f);
    }
}

__global__ void bincount_i64(
    const long long* __restrict__ input,
    float* __restrict__ output,
    unsigned int n,
    unsigned int minlength
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    long long val = input[idx];
    if (val >= 0 && (unsigned long long)val < minlength) {
        atomicAdd(&output[val], 1.0f);
    }
}

// Bincount with f32 weights
__global__ void bincount_weighted_f32(
    const int* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ output,
    unsigned int n,
    unsigned int minlength
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int val = input[idx];
    if (val >= 0 && (unsigned int)val < minlength) {
        atomicAdd(&output[val], weights[idx]);
    }
}

// Bincount with f64 weights
__global__ void bincount_weighted_f64(
    const int* __restrict__ input,
    const double* __restrict__ weights,
    double* __restrict__ output,
    unsigned int n,
    unsigned int minlength
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int val = input[idx];
    if (val >= 0 && (unsigned int)val < minlength) {
        atomicAdd(&output[val], weights[idx]);
    }
}

// Bincount with i64 input and f32 weights
__global__ void bincount_i64_weighted_f32(
    const long long* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ output,
    unsigned int n,
    unsigned int minlength
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    long long val = input[idx];
    if (val >= 0 && (unsigned long long)val < minlength) {
        atomicAdd(&output[val], weights[idx]);
    }
}

// ============================================================================
// Scatter Reduce - Scatter with reduction operations
// Scatters values from src to dst at positions specified by index with a
// reduction operation (sum, prod, max, min, mean).
// Uses atomic operations for thread-safe reduction.
// Reduce ops: 0=sum, 1=prod, 2=max, 3=min, 4=mean
// ============================================================================

// Atomic max for float using CAS
__device__ __forceinline__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        float new_val = fmaxf(old_val, val);
        old = atomicCAS(address_as_int, assumed, __float_as_int(new_val));
    } while (assumed != old);
    return __int_as_float(old);
}

// Atomic min for float using CAS
__device__ __forceinline__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        float new_val = fminf(old_val, val);
        old = atomicCAS(address_as_int, assumed, __float_as_int(new_val));
    } while (assumed != old);
    return __int_as_float(old);
}

// Atomic max for double using CAS
__device__ __forceinline__ double atomicMaxDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        double new_val = fmax(old_val, val);
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Atomic min for double using CAS
__device__ __forceinline__ double atomicMinDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        double new_val = fmin(old_val, val);
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Scatter reduce for F32 - sum operation
__global__ void scatter_reduce_sum_f32(
    const float* __restrict__ src,
    const long long* __restrict__ indices,
    float* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    long long index_val = indices[src_d];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;
    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    atomicAdd(&dst[dst_idx], src[src_idx]);
}

// Scatter reduce for F32 - max operation
__global__ void scatter_reduce_max_f32(
    const float* __restrict__ src,
    const long long* __restrict__ indices,
    float* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    long long index_val = indices[src_d];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;
    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    atomicMaxFloat(&dst[dst_idx], src[src_idx]);
}

// Scatter reduce for F32 - min operation
__global__ void scatter_reduce_min_f32(
    const float* __restrict__ src,
    const long long* __restrict__ indices,
    float* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    long long index_val = indices[src_d];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;
    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    atomicMinFloat(&dst[dst_idx], src[src_idx]);
}

// Scatter reduce for F64 - sum operation
__global__ void scatter_reduce_sum_f64(
    const double* __restrict__ src,
    const long long* __restrict__ indices,
    double* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    long long index_val = indices[src_d];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;
    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    atomicAdd(&dst[dst_idx], src[src_idx]);
}

// Scatter reduce for F64 - max operation
__global__ void scatter_reduce_max_f64(
    const double* __restrict__ src,
    const long long* __restrict__ indices,
    double* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    long long index_val = indices[src_d];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;
    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    atomicMaxDouble(&dst[dst_idx], src[src_idx]);
}

// Scatter reduce for F64 - min operation
__global__ void scatter_reduce_min_f64(
    const double* __restrict__ src,
    const long long* __restrict__ indices,
    double* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    long long index_val = indices[src_d];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;
    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    atomicMinDouble(&dst[dst_idx], src[src_idx]);
}

// Scatter reduce for I32 - sum operation
__global__ void scatter_reduce_sum_i32(
    const int* __restrict__ src,
    const long long* __restrict__ indices,
    int* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    long long index_val = indices[src_d];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;
    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    atomicAdd(&dst[dst_idx], src[src_idx]);
}

// Scatter reduce for I32 - max operation
__global__ void scatter_reduce_max_i32(
    const int* __restrict__ src,
    const long long* __restrict__ indices,
    int* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    long long index_val = indices[src_d];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;
    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    atomicMax(&dst[dst_idx], src[src_idx]);
}

// Scatter reduce for I32 - min operation
__global__ void scatter_reduce_min_i32(
    const int* __restrict__ src,
    const long long* __restrict__ indices,
    int* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    long long index_val = indices[src_d];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;
    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    atomicMin(&dst[dst_idx], src[src_idx]);
}

// ============================================================================
// Gather 2D - Gathers from 2D matrix at (row, col) positions
// input: 2D tensor [nrows, ncols]
// rows: 1D tensor of row indices [num_indices]
// cols: 1D tensor of column indices [num_indices]
// output: 1D tensor [num_indices]
// ============================================================================

#define DEFINE_GATHER_2D_KERNEL(suffix, dtype) \
__global__ void gather_2d_##suffix( \
    const dtype* __restrict__ input, \
    const long long* __restrict__ rows, \
    const long long* __restrict__ cols, \
    dtype* __restrict__ output, \
    unsigned int nrows, \
    unsigned int ncols, \
    unsigned int num_indices \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= num_indices) return; \
    \
    long long r = rows[idx]; \
    long long c = cols[idx]; \
    \
    /* Bounds checking */ \
    if (r < 0 || (unsigned int)r >= nrows || c < 0 || (unsigned int)c >= ncols) { \
        output[idx] = (dtype)0; \
        return; \
    } \
    \
    /* Row-major indexing: input[r, c] = input[r * ncols + c] */ \
    unsigned int input_idx = (unsigned int)r * ncols + (unsigned int)c; \
    output[idx] = input[input_idx]; \
}

// Instantiate gather_2d for all dtypes
DEFINE_GATHER_2D_KERNEL(f32, float)
DEFINE_GATHER_2D_KERNEL(f64, double)
DEFINE_GATHER_2D_KERNEL(f16, __half)
DEFINE_GATHER_2D_KERNEL(bf16, __nv_bfloat16)
DEFINE_GATHER_2D_KERNEL(i32, int)
DEFINE_GATHER_2D_KERNEL(i64, long long)

} // extern "C"
