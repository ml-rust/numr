// Convolution CUDA kernels - conv1d, conv2d, depthwise_conv2d
// Supports: f32, f64, f16, bf16
//
// Direct convolution approach - each thread computes one output element.
// Input layout: NCHW (batch, channels, height, width)
// Weight layout: (C_out, C_in/groups, K_h, K_w)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Conv1d Kernel Template
// Input: (N, C_in, L)
// Weight: (C_out, C_in/groups, K)
// Output: (N, C_out, L_out)
// ============================================================================

#define DEFINE_CONV1D_KERNEL(suffix, dtype) \
__global__ void conv1d_##suffix( \
    const dtype* __restrict__ input, \
    const dtype* __restrict__ weight, \
    const dtype* __restrict__ bias, \
    dtype* __restrict__ output, \
    unsigned int batch, \
    unsigned int c_in, \
    unsigned int length, \
    unsigned int c_out, \
    unsigned int kernel_size, \
    unsigned int output_length, \
    unsigned int stride, \
    unsigned int padding, \
    unsigned int dilation, \
    unsigned int groups, \
    unsigned int has_bias \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = batch * c_out * output_length; \
    if (idx >= total) return; \
    \
    unsigned int ox = idx % output_length; \
    unsigned int oc = (idx / output_length) % c_out; \
    unsigned int b = idx / (c_out * output_length); \
    \
    unsigned int c_in_per_group = c_in / groups; \
    unsigned int c_out_per_group = c_out / groups; \
    unsigned int g = oc / c_out_per_group; \
    unsigned int oc_local = oc % c_out_per_group; \
    unsigned int c_in_start = g * c_in_per_group; \
    \
    dtype sum = (dtype)0; \
    \
    for (unsigned int ic = 0; ic < c_in_per_group; ic++) { \
        unsigned int c_in_idx = c_in_start + ic; \
        \
        for (unsigned int kx = 0; kx < kernel_size; kx++) { \
            int ix = (int)(ox * stride + kx * dilation) - (int)padding; \
            \
            if (ix >= 0 && ix < (int)length) { \
                unsigned int input_idx = b * c_in * length + c_in_idx * length + (unsigned int)ix; \
                unsigned int weight_idx = oc * c_in_per_group * kernel_size + ic * kernel_size + kx; \
                sum = sum + input[input_idx] * weight[weight_idx]; \
            } \
        } \
    } \
    \
    if (has_bias != 0u && bias != nullptr) { \
        sum = sum + bias[oc]; \
    } \
    \
    output[idx] = sum; \
}

// ============================================================================
// Conv2d Kernel Template
// Input: (N, C_in, H, W)
// Weight: (C_out, C_in/groups, K_h, K_w)
// Output: (N, C_out, H_out, W_out)
// ============================================================================

#define DEFINE_CONV2D_KERNEL(suffix, dtype) \
__global__ void conv2d_##suffix( \
    const dtype* __restrict__ input, \
    const dtype* __restrict__ weight, \
    const dtype* __restrict__ bias, \
    dtype* __restrict__ output, \
    unsigned int batch, \
    unsigned int c_in, \
    unsigned int height, \
    unsigned int width, \
    unsigned int c_out, \
    unsigned int kernel_h, \
    unsigned int kernel_w, \
    unsigned int output_h, \
    unsigned int output_w, \
    unsigned int stride_h, \
    unsigned int stride_w, \
    unsigned int pad_h, \
    unsigned int pad_w, \
    unsigned int dilation_h, \
    unsigned int dilation_w, \
    unsigned int groups, \
    unsigned int has_bias \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = batch * c_out * output_h * output_w; \
    if (idx >= total) return; \
    \
    unsigned int ox = idx % output_w; \
    unsigned int oy = (idx / output_w) % output_h; \
    unsigned int oc = (idx / (output_w * output_h)) % c_out; \
    unsigned int b = idx / (c_out * output_h * output_w); \
    \
    unsigned int c_in_per_group = c_in / groups; \
    unsigned int c_out_per_group = c_out / groups; \
    unsigned int g = oc / c_out_per_group; \
    unsigned int c_in_start = g * c_in_per_group; \
    \
    dtype sum = (dtype)0; \
    \
    for (unsigned int ic = 0; ic < c_in_per_group; ic++) { \
        unsigned int c_in_idx = c_in_start + ic; \
        \
        for (unsigned int ky = 0; ky < kernel_h; ky++) { \
            for (unsigned int kx = 0; kx < kernel_w; kx++) { \
                int iy = (int)(oy * stride_h + ky * dilation_h) - (int)pad_h; \
                int ix = (int)(ox * stride_w + kx * dilation_w) - (int)pad_w; \
                \
                if (iy >= 0 && iy < (int)height && ix >= 0 && ix < (int)width) { \
                    unsigned int input_idx = b * c_in * height * width \
                        + c_in_idx * height * width \
                        + (unsigned int)iy * width \
                        + (unsigned int)ix; \
                    unsigned int weight_idx = oc * c_in_per_group * kernel_h * kernel_w \
                        + ic * kernel_h * kernel_w \
                        + ky * kernel_w \
                        + kx; \
                    sum = sum + input[input_idx] * weight[weight_idx]; \
                } \
            } \
        } \
    } \
    \
    if (has_bias != 0u && bias != nullptr) { \
        sum = sum + bias[oc]; \
    } \
    \
    output[idx] = sum; \
}

// ============================================================================
// Depthwise Conv2d Kernel Template
// Input: (N, C, H, W)
// Weight: (C, 1, K_h, K_w)
// Output: (N, C, H_out, W_out)
// Each channel has its own independent filter
// ============================================================================

#define DEFINE_DEPTHWISE_CONV2D_KERNEL(suffix, dtype) \
__global__ void depthwise_conv2d_##suffix( \
    const dtype* __restrict__ input, \
    const dtype* __restrict__ weight, \
    const dtype* __restrict__ bias, \
    dtype* __restrict__ output, \
    unsigned int batch, \
    unsigned int channels, \
    unsigned int height, \
    unsigned int width, \
    unsigned int kernel_h, \
    unsigned int kernel_w, \
    unsigned int output_h, \
    unsigned int output_w, \
    unsigned int stride_h, \
    unsigned int stride_w, \
    unsigned int pad_h, \
    unsigned int pad_w, \
    unsigned int dilation_h, \
    unsigned int dilation_w, \
    unsigned int has_bias \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = batch * channels * output_h * output_w; \
    if (idx >= total) return; \
    \
    unsigned int ox = idx % output_w; \
    unsigned int oy = (idx / output_w) % output_h; \
    unsigned int c = (idx / (output_w * output_h)) % channels; \
    unsigned int b = idx / (channels * output_h * output_w); \
    \
    dtype sum = (dtype)0; \
    \
    for (unsigned int ky = 0; ky < kernel_h; ky++) { \
        for (unsigned int kx = 0; kx < kernel_w; kx++) { \
            int iy = (int)(oy * stride_h + ky * dilation_h) - (int)pad_h; \
            int ix = (int)(ox * stride_w + kx * dilation_w) - (int)pad_w; \
            \
            if (iy >= 0 && iy < (int)height && ix >= 0 && ix < (int)width) { \
                unsigned int input_idx = b * channels * height * width \
                    + c * height * width \
                    + (unsigned int)iy * width \
                    + (unsigned int)ix; \
                unsigned int weight_idx = c * kernel_h * kernel_w + ky * kernel_w + kx; \
                sum = sum + input[input_idx] * weight[weight_idx]; \
            } \
        } \
    } \
    \
    if (has_bias != 0u && bias != nullptr) { \
        sum = sum + bias[c]; \
    } \
    \
    output[idx] = sum; \
}

// ============================================================================
// Instantiate kernels for all supported dtypes
// ============================================================================

extern "C" {

// F32 kernels
DEFINE_CONV1D_KERNEL(f32, float)
DEFINE_CONV2D_KERNEL(f32, float)
DEFINE_DEPTHWISE_CONV2D_KERNEL(f32, float)

// F64 kernels
DEFINE_CONV1D_KERNEL(f64, double)
DEFINE_CONV2D_KERNEL(f64, double)
DEFINE_DEPTHWISE_CONV2D_KERNEL(f64, double)

// F16 kernels (half precision)
DEFINE_CONV1D_KERNEL(f16, __half)
DEFINE_CONV2D_KERNEL(f16, __half)
DEFINE_DEPTHWISE_CONV2D_KERNEL(f16, __half)

// BF16 kernels (bfloat16)
DEFINE_CONV1D_KERNEL(bf16, __nv_bfloat16)
DEFINE_CONV2D_KERNEL(bf16, __nv_bfloat16)
DEFINE_DEPTHWISE_CONV2D_KERNEL(bf16, __nv_bfloat16)

} // extern "C"
