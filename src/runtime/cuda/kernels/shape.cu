// Shape operation CUDA kernels - cat, stack
// Supports: f32, f64, f16, bf16, i32, i64, u32, u64
//
// cat: Concatenates tensors along a specified dimension
// stack: Uses unsqueeze + cat (implemented in Rust layer)
// split/chunk: Zero-copy views using narrow (no kernel needed)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Cat Kernel
// ============================================================================
//
// Algorithm:
// For each element in the output tensor:
//   1. Decompose linear index into (outer, cat_idx, inner) coordinates
//   2. Find which source tensor this cat_idx belongs to using offsets array
//   3. Compute local cat index within that source tensor
//   4. Read from source tensor and write to output
//
// Memory layout:
//   - outer_size = product of dims before cat dimension
//   - cat_dim_total = sum of all tensor sizes along cat dimension
//   - inner_size = product of dims after cat dimension
//
// Note: This is a multi-pass approach - one kernel launch per input tensor
// for simplicity and to avoid complex offset calculations in a single kernel.

#define DEFINE_CAT_KERNEL(suffix, dtype) \
__global__ void cat_copy_##suffix( \
    const dtype* __restrict__ src, \
    dtype* __restrict__ dst, \
    unsigned int outer_size, \
    unsigned int src_cat_size, \
    unsigned int dst_cat_size, \
    unsigned int cat_offset, \
    unsigned int inner_size, \
    unsigned int total_elements \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= total_elements) return; \
    \
    /* Decompose idx into (outer, cat_i, inner) for source tensor */ \
    unsigned int inner = idx % inner_size; \
    unsigned int remaining = idx / inner_size; \
    unsigned int cat_i = remaining % src_cat_size; \
    unsigned int outer = remaining / src_cat_size; \
    \
    /* Compute destination index */ \
    unsigned int dst_idx = outer * dst_cat_size * inner_size \
                         + (cat_offset + cat_i) * inner_size \
                         + inner; \
    \
    dst[dst_idx] = src[idx]; \
}

extern "C" {

// Kernel instantiations for all supported dtypes
DEFINE_CAT_KERNEL(f32, float)
DEFINE_CAT_KERNEL(f64, double)
DEFINE_CAT_KERNEL(f16, __half)
DEFINE_CAT_KERNEL(bf16, __nv_bfloat16)
DEFINE_CAT_KERNEL(i32, int)
DEFINE_CAT_KERNEL(i64, long long)
DEFINE_CAT_KERNEL(u32, unsigned int)
DEFINE_CAT_KERNEL(u64, unsigned long long)
DEFINE_CAT_KERNEL(i16, short)
DEFINE_CAT_KERNEL(i8, signed char)
DEFINE_CAT_KERNEL(u16, unsigned short)
DEFINE_CAT_KERNEL(u8, unsigned char)

} // extern "C"
