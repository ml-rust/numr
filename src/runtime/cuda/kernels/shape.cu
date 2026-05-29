// Shape operation CUDA kernels - cat, stack
// Supports: f32, f64, f16, bf16, i32, i64, u32, u64, c64, c128
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
DEFINE_CAT_KERNEL(c64, numr_complex64)
DEFINE_CAT_KERNEL(c128, numr_complex128)
DEFINE_CAT_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_CAT_KERNEL(fp8_e5m2, numr_fp8_e5m2)

} // extern "C"

// ============================================================================
// Repeat Kernel
// ============================================================================
//
// Algorithm:
// For each element in the output tensor (which is larger by factor of repeats):
//   1. Decompose linear index into multi-dimensional coordinates
//   2. Map each coordinate back to source tensor using modulo
//   3. Read from source tensor and write to output
//
// Example: repeat([2,3], [2,3]) -> [4,9]
//   out[i,j] = src[i%2, j%3]
//
// Shape/stride arrays are passed as 8 individual scalar args (MAX_DIMS=8)
// rather than device pointers, making this safe for CUDA graph capture/replay.
// Unused dimension slots are zero-padded by the Rust launcher.

// Maximum number of dimensions supported (must match shape.rs MAX_DIMS)
#ifndef SHAPE_MAX_DIMS
#define SHAPE_MAX_DIMS 8
#endif

#define DEFINE_REPEAT_KERNEL(suffix, dtype) \
__global__ void repeat_##suffix( \
    const dtype* __restrict__ src, \
    dtype* __restrict__ dst, \
    unsigned int src_shape0, unsigned int src_shape1, unsigned int src_shape2, unsigned int src_shape3, \
    unsigned int src_shape4, unsigned int src_shape5, unsigned int src_shape6, unsigned int src_shape7, \
    unsigned int out_shape0, unsigned int out_shape1, unsigned int out_shape2, unsigned int out_shape3, \
    unsigned int out_shape4, unsigned int out_shape5, unsigned int out_shape6, unsigned int out_shape7, \
    unsigned int out_strides0, unsigned int out_strides1, unsigned int out_strides2, unsigned int out_strides3, \
    unsigned int out_strides4, unsigned int out_strides5, unsigned int out_strides6, unsigned int out_strides7, \
    unsigned int ndim, \
    unsigned int total_elements \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= total_elements) return; \
    \
    /* Load scalar args into shared arrays for indexed access */ \
    __shared__ unsigned int s_src_shape[SHAPE_MAX_DIMS]; \
    __shared__ unsigned int s_out_shape[SHAPE_MAX_DIMS]; \
    if (threadIdx.x == 0) { \
        s_src_shape[0] = src_shape0; s_src_shape[1] = src_shape1; \
        s_src_shape[2] = src_shape2; s_src_shape[3] = src_shape3; \
        s_src_shape[4] = src_shape4; s_src_shape[5] = src_shape5; \
        s_src_shape[6] = src_shape6; s_src_shape[7] = src_shape7; \
        s_out_shape[0] = out_shape0; s_out_shape[1] = out_shape1; \
        s_out_shape[2] = out_shape2; s_out_shape[3] = out_shape3; \
        s_out_shape[4] = out_shape4; s_out_shape[5] = out_shape5; \
        s_out_shape[6] = out_shape6; s_out_shape[7] = out_shape7; \
    } \
    __syncthreads(); \
    \
    /* Decompose idx into multi-dimensional output coordinates */ \
    unsigned int remaining = idx; \
    unsigned int src_idx = 0; \
    unsigned int src_stride = 1; \
    \
    /* Process dimensions from last to first */ \
    for (int d = (int)ndim - 1; d >= 0; d--) { \
        unsigned int coord = remaining % s_out_shape[d]; \
        remaining /= s_out_shape[d]; \
        \
        /* Map to source coordinate using modulo */ \
        unsigned int src_coord = coord % s_src_shape[d]; \
        src_idx += src_coord * src_stride; \
        src_stride *= s_src_shape[d]; \
    } \
    \
    dst[idx] = src[src_idx]; \
}

extern "C" {

DEFINE_REPEAT_KERNEL(f32, float)
DEFINE_REPEAT_KERNEL(f64, double)
DEFINE_REPEAT_KERNEL(f16, __half)
DEFINE_REPEAT_KERNEL(bf16, __nv_bfloat16)
DEFINE_REPEAT_KERNEL(i32, int)
DEFINE_REPEAT_KERNEL(i64, long long)
DEFINE_REPEAT_KERNEL(u32, unsigned int)
DEFINE_REPEAT_KERNEL(u64, unsigned long long)
DEFINE_REPEAT_KERNEL(i16, short)
DEFINE_REPEAT_KERNEL(i8, signed char)
DEFINE_REPEAT_KERNEL(u16, unsigned short)
DEFINE_REPEAT_KERNEL(u8, unsigned char)
DEFINE_REPEAT_KERNEL(c64, numr_complex64)
DEFINE_REPEAT_KERNEL(c128, numr_complex128)
DEFINE_REPEAT_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_REPEAT_KERNEL(fp8_e5m2, numr_fp8_e5m2)

} // extern "C"

// ============================================================================
// Pad Kernel
// ============================================================================
//
// Algorithm:
// For each element in the output tensor:
//   1. Decompose linear index into multi-dimensional coordinates
//   2. Check if coordinate falls within padded region or original tensor
//   3. If in original region, read from source; otherwise use fill value
//
// Memory layout (for dimension d):
//   - pad_before[d]: padding before original data
//   - src_shape[d]: original tensor size
//   - pad_after[d]: padding after original data
//   - out_shape[d] = pad_before[d] + src_shape[d] + pad_after[d]
//
// Shape/pad arrays are passed as 8 individual scalar args (MAX_DIMS=8)
// rather than device pointers, making this safe for CUDA graph capture/replay.
// Unused dimension slots are zero-padded by the Rust launcher.

#define DEFINE_PAD_KERNEL(suffix, dtype) \
__global__ void pad_##suffix( \
    const dtype* __restrict__ src, \
    dtype* __restrict__ dst, \
    dtype fill_value, \
    unsigned int src_shape0, unsigned int src_shape1, unsigned int src_shape2, unsigned int src_shape3, \
    unsigned int src_shape4, unsigned int src_shape5, unsigned int src_shape6, unsigned int src_shape7, \
    unsigned int out_shape0, unsigned int out_shape1, unsigned int out_shape2, unsigned int out_shape3, \
    unsigned int out_shape4, unsigned int out_shape5, unsigned int out_shape6, unsigned int out_shape7, \
    unsigned int pad_before0, unsigned int pad_before1, unsigned int pad_before2, unsigned int pad_before3, \
    unsigned int pad_before4, unsigned int pad_before5, unsigned int pad_before6, unsigned int pad_before7, \
    unsigned int ndim, \
    unsigned int total_elements \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= total_elements) return; \
    \
    /* Load scalar args into shared arrays for indexed access */ \
    __shared__ unsigned int s_src_shape[SHAPE_MAX_DIMS]; \
    __shared__ unsigned int s_out_shape[SHAPE_MAX_DIMS]; \
    __shared__ unsigned int s_pad_before[SHAPE_MAX_DIMS]; \
    if (threadIdx.x == 0) { \
        s_src_shape[0] = src_shape0; s_src_shape[1] = src_shape1; \
        s_src_shape[2] = src_shape2; s_src_shape[3] = src_shape3; \
        s_src_shape[4] = src_shape4; s_src_shape[5] = src_shape5; \
        s_src_shape[6] = src_shape6; s_src_shape[7] = src_shape7; \
        s_out_shape[0] = out_shape0; s_out_shape[1] = out_shape1; \
        s_out_shape[2] = out_shape2; s_out_shape[3] = out_shape3; \
        s_out_shape[4] = out_shape4; s_out_shape[5] = out_shape5; \
        s_out_shape[6] = out_shape6; s_out_shape[7] = out_shape7; \
        s_pad_before[0] = pad_before0; s_pad_before[1] = pad_before1; \
        s_pad_before[2] = pad_before2; s_pad_before[3] = pad_before3; \
        s_pad_before[4] = pad_before4; s_pad_before[5] = pad_before5; \
        s_pad_before[6] = pad_before6; s_pad_before[7] = pad_before7; \
    } \
    __syncthreads(); \
    \
    /* Decompose idx into multi-dimensional output coordinates */ \
    unsigned int remaining = idx; \
    unsigned int coords[SHAPE_MAX_DIMS]; \
    bool in_bounds = true; \
    \
    /* Process dimensions from last to first */ \
    for (int d = (int)ndim - 1; d >= 0; d--) { \
        coords[d] = remaining % s_out_shape[d]; \
        remaining /= s_out_shape[d]; \
        \
        /* Check if this coordinate is in the original tensor region */ \
        if (coords[d] < s_pad_before[d] || coords[d] >= s_pad_before[d] + s_src_shape[d]) { \
            in_bounds = false; \
        } \
    } \
    \
    if (in_bounds) { \
        /* Compute source index */ \
        unsigned int src_idx = 0; \
        unsigned int src_stride = 1; \
        for (int d = (int)ndim - 1; d >= 0; d--) { \
            unsigned int src_coord = coords[d] - s_pad_before[d]; \
            src_idx += src_coord * src_stride; \
            src_stride *= s_src_shape[d]; \
        } \
        dst[idx] = src[src_idx]; \
    } else { \
        dst[idx] = fill_value; \
    } \
}

extern "C" {

DEFINE_PAD_KERNEL(f32, float)
DEFINE_PAD_KERNEL(f64, double)
DEFINE_PAD_KERNEL(f16, __half)
DEFINE_PAD_KERNEL(bf16, __nv_bfloat16)
DEFINE_PAD_KERNEL(i32, int)
DEFINE_PAD_KERNEL(i64, long long)
DEFINE_PAD_KERNEL(u32, unsigned int)
DEFINE_PAD_KERNEL(u64, unsigned long long)
DEFINE_PAD_KERNEL(i16, short)
DEFINE_PAD_KERNEL(i8, signed char)
DEFINE_PAD_KERNEL(u16, unsigned short)
DEFINE_PAD_KERNEL(u8, unsigned char)
DEFINE_PAD_KERNEL(c64, numr_complex64)
DEFINE_PAD_KERNEL(c128, numr_complex128)
DEFINE_PAD_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_PAD_KERNEL(fp8_e5m2, numr_fp8_e5m2)

} // extern "C"

// ============================================================================
// Roll Kernel
// ============================================================================
//
// Algorithm:
// For each element in the output tensor:
//   1. Decompose linear index into (outer, dim_coord, inner)
//   2. Compute source coordinate by rolling: src_coord = (dim_coord - shift + dim_size) % dim_size
//   3. Read from source at computed index
//
// Example: roll([1,2,3,4,5], shift=2, dim=0) -> [4,5,1,2,3]
//   Elements are shifted right by 2, wrapping around

#define DEFINE_ROLL_KERNEL(suffix, dtype) \
__global__ void roll_##suffix( \
    const dtype* __restrict__ src, \
    dtype* __restrict__ dst, \
    unsigned int outer_size, \
    unsigned int dim_size, \
    unsigned int inner_size, \
    unsigned int shift, \
    unsigned int total_elements \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= total_elements) return; \
    \
    /* Decompose idx into (outer, dim_coord, inner) */ \
    unsigned int inner = idx % inner_size; \
    unsigned int remaining = idx / inner_size; \
    unsigned int dim_coord = remaining % dim_size; \
    unsigned int outer = remaining / dim_size; \
    \
    /* Compute source coordinate with roll (shift goes right, so source is shift positions left) */ \
    unsigned int src_dim_coord = (dim_coord + dim_size - shift) % dim_size; \
    \
    /* Compute source linear index */ \
    unsigned int src_idx = outer * dim_size * inner_size \
                         + src_dim_coord * inner_size \
                         + inner; \
    \
    dst[idx] = src[src_idx]; \
}

extern "C" {

DEFINE_ROLL_KERNEL(f32, float)
DEFINE_ROLL_KERNEL(f64, double)
DEFINE_ROLL_KERNEL(f16, __half)
DEFINE_ROLL_KERNEL(bf16, __nv_bfloat16)
DEFINE_ROLL_KERNEL(i32, int)
DEFINE_ROLL_KERNEL(i64, long long)
DEFINE_ROLL_KERNEL(u32, unsigned int)
DEFINE_ROLL_KERNEL(u64, unsigned long long)
DEFINE_ROLL_KERNEL(i16, short)
DEFINE_ROLL_KERNEL(i8, signed char)
DEFINE_ROLL_KERNEL(u16, unsigned short)
DEFINE_ROLL_KERNEL(u8, unsigned char)
DEFINE_ROLL_KERNEL(c64, numr_complex64)
DEFINE_ROLL_KERNEL(c128, numr_complex128)
DEFINE_ROLL_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_ROLL_KERNEL(fp8_e5m2, numr_fp8_e5m2)

} // extern "C"
