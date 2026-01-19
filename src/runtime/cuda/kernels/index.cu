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
// F32 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(f32, float)
DEFINE_SCATTER_KERNEL(f32, float)
DEFINE_COPY_KERNEL(f32, float)
DEFINE_INDEX_SELECT_KERNEL(f32, float)
DEFINE_MASKED_SELECT_KERNEL(f32, float)
DEFINE_MASKED_FILL_KERNEL(f32, float)

// ============================================================================
// F64 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(f64, double)
DEFINE_SCATTER_KERNEL(f64, double)
DEFINE_COPY_KERNEL(f64, double)
DEFINE_INDEX_SELECT_KERNEL(f64, double)
DEFINE_MASKED_SELECT_KERNEL(f64, double)
DEFINE_MASKED_FILL_KERNEL(f64, double)

// ============================================================================
// F16 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(f16, __half)
DEFINE_SCATTER_KERNEL(f16, __half)
DEFINE_COPY_KERNEL(f16, __half)
DEFINE_INDEX_SELECT_KERNEL(f16, __half)
DEFINE_MASKED_SELECT_KERNEL(f16, __half)
DEFINE_MASKED_FILL_KERNEL(f16, __half)

// ============================================================================
// BF16 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(bf16, __nv_bfloat16)
DEFINE_SCATTER_KERNEL(bf16, __nv_bfloat16)
DEFINE_COPY_KERNEL(bf16, __nv_bfloat16)
DEFINE_INDEX_SELECT_KERNEL(bf16, __nv_bfloat16)
DEFINE_MASKED_SELECT_KERNEL(bf16, __nv_bfloat16)
DEFINE_MASKED_FILL_KERNEL(bf16, __nv_bfloat16)

// ============================================================================
// I32 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(i32, int)
DEFINE_SCATTER_KERNEL(i32, int)
DEFINE_COPY_KERNEL(i32, int)
DEFINE_INDEX_SELECT_KERNEL(i32, int)
DEFINE_MASKED_SELECT_KERNEL(i32, int)
DEFINE_MASKED_FILL_KERNEL(i32, int)

// ============================================================================
// I64 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(i64, long long)
DEFINE_SCATTER_KERNEL(i64, long long)
DEFINE_COPY_KERNEL(i64, long long)
DEFINE_INDEX_SELECT_KERNEL(i64, long long)
DEFINE_MASKED_SELECT_KERNEL(i64, long long)
DEFINE_MASKED_FILL_KERNEL(i64, long long)

} // extern "C"
