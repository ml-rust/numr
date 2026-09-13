// Row-wise pad kernels: padding confined to the last two dimensions of a
// contiguous tensor. The generic 8-dim kernel in shape.cu covers every other
// pad layout; the Rust launcher picks between them.
//
// The tensor is viewed as `batch` matrices of `src_rows x src_cols`. The
// output is `batch` matrices of `out_rows x out_cols` where
//   out_rows = pad_before_row + src_rows + pad_after_row
//   out_cols = pad_before_col + src_cols + pad_after_col
// Leading dimensions are untouched, so their product folds into `batch`.
//
// Grid: x spans the output columns, y the output rows, z the batch. Threads
// grid-stride on y and z so a row or batch count above the 65535 grid limit
// still launches. Consecutive threads write consecutive columns, so every
// warp-level store (and the matching interior load) coalesces into full
// cache lines. No index decode chain, no shared-memory shape arrays.
//
// Two variants per dtype:
//   pad_rows_<dtype>      one element per thread; the reference semantics
//   pad_rows_vec_<dtype>  one 16-byte chunk per thread
// The vector kernel needs every row start on both sides 16-byte aligned:
// base pointers aligned AND the source, output, and pad_before_col byte
// widths all multiples of 16. Under those guards a chunk never straddles
// the interior boundary, so each chunk is either a single 128-bit copy or a
// single 128-bit fill store. The launcher checks the guards and otherwise
// runs the scalar kernel. Both kernels write bit-identical output.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

#define PAD_ROWS_VEC_BYTES 16u

// Each thread owns PAD_ROWS_ITEMS columns spaced blockDim.x apart, so a warp
// still touches one contiguous run per item while every thread keeps several
// independent loads in flight. The launcher sizes grid.x by
// blockDim.x * PAD_ROWS_ITEMS.
#define PAD_ROWS_ITEMS 4u

#define DEFINE_PAD_ROWS_KERNEL(suffix, dtype) \
__global__ void pad_rows_##suffix( \
    const dtype* __restrict__ src, \
    dtype* __restrict__ dst, \
    dtype fill_value, \
    unsigned int batch, \
    unsigned int src_rows, \
    unsigned int src_cols, \
    unsigned int out_rows, \
    unsigned int out_cols, \
    unsigned int pad_before_row, \
    unsigned int pad_before_col \
) { \
    unsigned int col0 = blockIdx.x * blockDim.x * PAD_ROWS_ITEMS + threadIdx.x; \
    if (col0 >= out_cols) return; \
    for (unsigned int b = blockIdx.z; b < batch; b += gridDim.z) { \
        for (unsigned int r = blockIdx.y; r < out_rows; r += gridDim.y) { \
            unsigned int src_row = r - pad_before_row; \
            bool row_in = (r >= pad_before_row) && (src_row < src_rows); \
            dtype* dst_row = dst + ((size_t)b * out_rows + r) * out_cols; \
            /* Clamped so the pointer stays inside the buffer even when */ \
            /* row_in is false; it is then never dereferenced. */ \
            unsigned int src_row_c = row_in ? src_row : 0u; \
            const dtype* src_row_p = src + ((size_t)b * src_rows + src_row_c) * src_cols; \
            _Pragma("unroll") \
            for (unsigned int k = 0; k < PAD_ROWS_ITEMS; k++) { \
                unsigned int col = col0 + k * blockDim.x; \
                if (col >= out_cols) break; \
                unsigned int src_col = col - pad_before_col; \
                bool col_in = (col >= pad_before_col) && (src_col < src_cols); \
                dst_row[col] = (row_in && col_in) ? src_row_p[src_col] : fill_value; \
            } \
        } \
    } \
}

// Column counts here are in 16-byte chunks: `out_vecs = out_cols / VEC`,
// `src_vecs = src_cols / VEC`, `pad_before_vec = pad_before_col / VEC`, where
// VEC = 16 / sizeof(dtype). The launcher performs that division after
// checking divisibility.
#define DEFINE_PAD_ROWS_VEC_KERNEL(suffix, dtype) \
__global__ void pad_rows_vec_##suffix( \
    const uint4* __restrict__ src, \
    uint4* __restrict__ dst, \
    dtype fill_value, \
    unsigned int batch, \
    unsigned int src_rows, \
    unsigned int src_vecs, \
    unsigned int out_rows, \
    unsigned int out_vecs, \
    unsigned int pad_before_row, \
    unsigned int pad_before_vec \
) { \
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x; \
    if (v >= out_vecs) return; \
    constexpr unsigned int VEC = PAD_ROWS_VEC_BYTES / sizeof(dtype); \
    __align__(16) dtype fill_lane[VEC]; \
    _Pragma("unroll") \
    for (unsigned int i = 0; i < VEC; i++) fill_lane[i] = fill_value; \
    uint4 fill_chunk = *reinterpret_cast<const uint4*>(fill_lane); \
    unsigned int src_v = v - pad_before_vec; \
    bool col_in = (v >= pad_before_vec) && (src_v < src_vecs); \
    for (unsigned int b = blockIdx.z; b < batch; b += gridDim.z) { \
        for (unsigned int r = blockIdx.y; r < out_rows; r += gridDim.y) { \
            unsigned int src_row = r - pad_before_row; \
            bool row_in = (r >= pad_before_row) && (src_row < src_rows); \
            size_t dst_idx = ((size_t)b * out_rows + r) * out_vecs + v; \
            if (col_in && row_in) { \
                size_t src_idx = ((size_t)b * src_rows + src_row) * src_vecs + src_v; \
                dst[dst_idx] = src[src_idx]; \
            } else { \
                dst[dst_idx] = fill_chunk; \
            } \
        } \
    } \
}

extern "C" {

DEFINE_PAD_ROWS_KERNEL(f32, float)
DEFINE_PAD_ROWS_KERNEL(f64, double)
DEFINE_PAD_ROWS_KERNEL(f16, __half)
DEFINE_PAD_ROWS_KERNEL(bf16, __nv_bfloat16)
DEFINE_PAD_ROWS_KERNEL(i32, int)
DEFINE_PAD_ROWS_KERNEL(i64, long long)
DEFINE_PAD_ROWS_KERNEL(u32, unsigned int)
DEFINE_PAD_ROWS_KERNEL(u64, unsigned long long)
DEFINE_PAD_ROWS_KERNEL(i16, short)
DEFINE_PAD_ROWS_KERNEL(i8, signed char)
DEFINE_PAD_ROWS_KERNEL(u16, unsigned short)
DEFINE_PAD_ROWS_KERNEL(u8, unsigned char)
DEFINE_PAD_ROWS_KERNEL(c64, numr_complex64)
DEFINE_PAD_ROWS_KERNEL(c128, numr_complex128)
DEFINE_PAD_ROWS_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_PAD_ROWS_KERNEL(fp8_e5m2, numr_fp8_e5m2)

DEFINE_PAD_ROWS_VEC_KERNEL(f32, float)
DEFINE_PAD_ROWS_VEC_KERNEL(f64, double)
DEFINE_PAD_ROWS_VEC_KERNEL(f16, __half)
DEFINE_PAD_ROWS_VEC_KERNEL(bf16, __nv_bfloat16)
DEFINE_PAD_ROWS_VEC_KERNEL(i32, int)
DEFINE_PAD_ROWS_VEC_KERNEL(i64, long long)
DEFINE_PAD_ROWS_VEC_KERNEL(u32, unsigned int)
DEFINE_PAD_ROWS_VEC_KERNEL(u64, unsigned long long)
DEFINE_PAD_ROWS_VEC_KERNEL(i16, short)
DEFINE_PAD_ROWS_VEC_KERNEL(i8, signed char)
DEFINE_PAD_ROWS_VEC_KERNEL(u16, unsigned short)
DEFINE_PAD_ROWS_VEC_KERNEL(u8, unsigned char)
DEFINE_PAD_ROWS_VEC_KERNEL(c64, numr_complex64)
DEFINE_PAD_ROWS_VEC_KERNEL(c128, numr_complex128)
DEFINE_PAD_ROWS_VEC_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_PAD_ROWS_VEC_KERNEL(fp8_e5m2, numr_fp8_e5m2)

} // extern "C"
