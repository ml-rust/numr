// col2im_transpose1d CUDA kernels - fold conv_transpose1d's GEMM output into
// the output signal.
// Supports: f32, f64, f16, bf16
//
// Input:  col  (N, L, C_out*K)   the GEMM product  x^T @ W  per batch
//         bias (C_out)           optional
// Output: out  (N, C_out, L_out)
//
// GEMM FIRST, THEN GATHER. `col_transpose1d.cu` gathers the input into a
// column buffer of `C_in*K x L_out` and contracts afterwards; that buffer
// grows with the OUTPUT length, `stride` times the input. This kernel is the
// other order: the GEMM runs on the input length, producing every
// (tap, channel) product per input sample, and the fold below sums the taps
// that land on each output position:
//
//   out[n, oc, ox] = bias[oc] + sum over k of col[n, l, oc*K + k]
//     where  num = ox + pad_left - k * dilation
//            num >= 0                 (checked BEFORE the division)
//            num % stride == 0        (otherwise this tap lands between
//                                      input samples and contributes nothing)
//            l = num / stride  <  L
//
// GATHER, NOT SCATTER. The textbook col2im scatter-adds each product into its
// output slot and needs atomics on overlapping writes. Read backwards, each
// output position owns its taps, so one thread writes each output once and the
// tap order is fixed - k ascending - which keeps the result deterministic.
//
// ACCUMULATION ORDER. `conv_transpose.rs` on the CPU sums `for tap { for ic }`.
// The GEMM sums the input channels of one tap, and this kernel sums the taps,
// so the nesting is the same. The channel sum inside the GEMM is tiled, so the
// backends agree to tolerance, not bit for bit - as conv1d's im2col path does.
//
// THREAD MAP. threadIdx.x walks consecutive output positions, so the writes
// coalesce and the reads fall on short contiguous tap runs. blockIdx.y walks
// output channels and blockIdx.z the batch, each with a grid-stride loop so
// neither axis can exceed the 65535 grid limit.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

template<typename T, typename Acc>
__device__ __forceinline__ void col2im_transpose1d_impl(
    const T* __restrict__ col,
    const T* __restrict__ bias,
    T* __restrict__ out,
    unsigned int batch,
    unsigned int length,
    unsigned int c_out,
    unsigned int kernel_size,
    unsigned int output_length,
    unsigned int stride,
    unsigned int pad_left,
    unsigned int dilation
) {
    const unsigned int ox = blockIdx.x * blockDim.x + threadIdx.x;
    if (ox >= output_length) return;

    const size_t row_stride = (size_t)c_out * kernel_size;
    const int num_base = (int)ox + (int)pad_left;

    for (unsigned int oc = blockIdx.y; oc < c_out; oc += gridDim.y) {
        const Acc b = (bias != nullptr) ? AccumTraits<T, Acc>::load(bias, (int)oc)
                                        : AccumTraits<T, Acc>::zero();
        for (unsigned int n = blockIdx.z; n < batch; n += gridDim.z) {
            Acc acc = AccumTraits<T, Acc>::zero();
            for (unsigned int k = 0; k < kernel_size; k++) {
                const int num = num_base - (int)(k * dilation);
                // Sign first: a negative numerator has no input sample at all.
                if (num < 0) continue;
                const unsigned int unum = (unsigned int)num;
                if (unum % stride != 0) continue;
                const unsigned int l = unum / stride;
                if (l >= length) continue;
                const T* p = col + ((size_t)n * length + l) * row_stride
                                 + (size_t)oc * kernel_size + k;
                acc = AccumTraits<T, Acc>::add(acc, AccumTraits<T, Acc>::load(p, 0));
            }
            T* o = out + ((size_t)n * c_out + oc) * output_length + ox;
            AccumTraits<T, Acc>::store(o, 0, AccumTraits<T, Acc>::add(acc, b));
        }
    }
}

#define DEFINE_COL2IM_TRANSPOSE1D_KERNEL(suffix, dtype, acc) \
__global__ void col2im_transpose1d_##suffix( \
    const dtype* __restrict__ col, \
    const dtype* __restrict__ bias, \
    dtype* __restrict__ out, \
    unsigned int batch, \
    unsigned int length, \
    unsigned int c_out, \
    unsigned int kernel_size, \
    unsigned int output_length, \
    unsigned int stride, \
    unsigned int pad_left, \
    unsigned int dilation \
) { \
    col2im_transpose1d_impl<dtype, acc>(col, bias, out, batch, length, c_out, \
                                        kernel_size, output_length, stride, \
                                        pad_left, dilation); \
}

// Instantiations must stay inside `extern "C"` so the launcher can look the
// kernels up by their unmangled `col2im_transpose1d_<dtype>` names. Half
// widths accumulate in F32, as the direct kernel does.
extern "C" {

DEFINE_COL2IM_TRANSPOSE1D_KERNEL(f32, float, float)
DEFINE_COL2IM_TRANSPOSE1D_KERNEL(f64, double, double)
DEFINE_COL2IM_TRANSPOSE1D_KERNEL(f16, __half, float)
DEFINE_COL2IM_TRANSPOSE1D_KERNEL(bf16, __nv_bfloat16, float)

} // extern "C"
