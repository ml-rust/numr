// Reflect-mode pad kernel: every output element gathers from a reflected
// source coordinate instead of a fill value.
//
// Algorithm, per output element:
//   1. Decompose the linear index into per-dimension output coordinates.
//   2. For each dimension, map the output coordinate back to a source
//      coordinate by mirroring across the boundary, excluding the edge
//      element (PyTorch `F.pad(mode="reflect")` semantics).
//   3. Gather from the source tensor at the resulting coordinate.
//
// The Rust launcher only reaches this kernel after `validate_reflect_pad`
// guarantees every pad size is strictly less than its dimension's size, so a
// single mirror bounce always lands back inside `0..src_shape[d]` — no loop
// over multiple reflection periods is needed.
//
// Shape/pad arrays are passed as 8 individual scalar args (MAX_DIMS=8)
// rather than device pointers, making this safe for CUDA graph capture/replay.
// Unused dimension slots are zero-padded by the Rust launcher.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

#ifndef SHAPE_MAX_DIMS
#define SHAPE_MAX_DIMS 8
#endif

__device__ __forceinline__ int reflect_coord(int out_coord, int before, int size) {
    int rel = out_coord - before;
    if (rel < 0) {
        return -rel;
    }
    if (rel < size) {
        return rel;
    }
    int over = rel - size;
    return size - 2 - over;
}

#define DEFINE_PAD_REFLECT_KERNEL(suffix, dtype) \
__global__ void pad_reflect_##suffix( \
    const dtype* __restrict__ src, \
    dtype* __restrict__ dst, \
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
    unsigned int remaining = idx; \
    unsigned int coords[SHAPE_MAX_DIMS]; \
    for (int d = (int)ndim - 1; d >= 0; d--) { \
        coords[d] = remaining % s_out_shape[d]; \
        remaining /= s_out_shape[d]; \
    } \
    \
    unsigned int src_idx = 0; \
    unsigned int src_stride = 1; \
    for (int d = (int)ndim - 1; d >= 0; d--) { \
        int src_coord = reflect_coord((int)coords[d], (int)s_pad_before[d], (int)s_src_shape[d]); \
        src_idx += (unsigned int)src_coord * src_stride; \
        src_stride *= s_src_shape[d]; \
    } \
    dst[idx] = src[src_idx]; \
}

extern "C" {

DEFINE_PAD_REFLECT_KERNEL(f32, float)
DEFINE_PAD_REFLECT_KERNEL(f64, double)
DEFINE_PAD_REFLECT_KERNEL(f16, __half)
DEFINE_PAD_REFLECT_KERNEL(bf16, __nv_bfloat16)
DEFINE_PAD_REFLECT_KERNEL(i32, int)
DEFINE_PAD_REFLECT_KERNEL(i64, long long)
DEFINE_PAD_REFLECT_KERNEL(u32, unsigned int)
DEFINE_PAD_REFLECT_KERNEL(u64, unsigned long long)
DEFINE_PAD_REFLECT_KERNEL(i16, short)
DEFINE_PAD_REFLECT_KERNEL(i8, signed char)
DEFINE_PAD_REFLECT_KERNEL(u16, unsigned short)
DEFINE_PAD_REFLECT_KERNEL(u8, unsigned char)
DEFINE_PAD_REFLECT_KERNEL(c64, numr_complex64)
DEFINE_PAD_REFLECT_KERNEL(c128, numr_complex128)
DEFINE_PAD_REFLECT_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_PAD_REFLECT_KERNEL(fp8_e5m2, numr_fp8_e5m2)

} // extern "C"
