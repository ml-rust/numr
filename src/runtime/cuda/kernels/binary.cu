// Binary element-wise CUDA kernels
// Supports: add, sub, mul, div, pow, max, min
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2, i32, i64

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

extern "C" {

// ============================================================================
// F32 Binary Operations
// ============================================================================

__global__ void add_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void div_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void pow_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = powf(a[idx], b[idx]);
    }
}

__global__ void max_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaxf(a[idx], b[idx]);
    }
}

__global__ void min_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fminf(a[idx], b[idx]);
    }
}

// ============================================================================
// F16 Binary Operations (half precision)
// Uses half2 vectorization for 2x throughput where possible
// ============================================================================

__global__ void add_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hadd(a[idx], b[idx]);
    }
}

__global__ void sub_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hsub(a[idx], b[idx]);
    }
}

__global__ void mul_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hmul(a[idx], b[idx]);
    }
}

__global__ void div_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hdiv(a[idx], b[idx]);
    }
}

__global__ void pow_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use FP32 for pow computation (more accurate)
        float af = __half2float(a[idx]);
        float bf = __half2float(b[idx]);
        out[idx] = __float2half(powf(af, bf));
    }
}

__global__ void max_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hgt(a[idx], b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void min_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hlt(a[idx], b[idx]) ? a[idx] : b[idx];
    }
}

// ============================================================================
// BF16 Binary Operations (bfloat16)
// Note: BF16 arithmetic requires SM 8.0+ (Ampere)
// ============================================================================

__global__ void add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hadd(a[idx], b[idx]);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
        #endif
    }
}

__global__ void sub_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hsub(a[idx], b[idx]);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) - __bfloat162float(b[idx]));
        #endif
    }
}

__global__ void mul_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hmul(a[idx], b[idx]);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) * __bfloat162float(b[idx]));
        #endif
    }
}

__global__ void div_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hdiv(a[idx], b[idx]);
        #else
        out[idx] = __float2bfloat16(__bfloat162float(a[idx]) / __bfloat162float(b[idx]));
        #endif
    }
}

__global__ void pow_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use FP32 for pow computation (more accurate)
        float af = __bfloat162float(a[idx]);
        float bf = __bfloat162float(b[idx]);
        out[idx] = __float2bfloat16(powf(af, bf));
    }
}

__global__ void max_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hgt(a[idx], b[idx]) ? a[idx] : b[idx];
        #else
        out[idx] = (__bfloat162float(a[idx]) > __bfloat162float(b[idx])) ? a[idx] : b[idx];
        #endif
    }
}

__global__ void min_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hlt(a[idx], b[idx]) ? a[idx] : b[idx];
        #else
        out[idx] = (__bfloat162float(a[idx]) < __bfloat162float(b[idx])) ? a[idx] : b[idx];
        #endif
    }
}

// ============================================================================
// F64 Binary Operations
// ============================================================================

__global__ void add_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void div_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void pow_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = pow(a[idx], b[idx]);
    }
}

__global__ void max_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmax(a[idx], b[idx]);
    }
}

__global__ void min_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmin(a[idx], b[idx]);
    }
}

// ============================================================================
// I32 Binary Operations
// ============================================================================

__global__ void add_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void div_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void max_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void min_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? a[idx] : b[idx];
    }
}

// ============================================================================
// I64 Binary Operations
// ============================================================================

__global__ void add_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void sub_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void mul_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void div_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void max_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
    }
}

__global__ void min_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? a[idx] : b[idx];
    }
}

// ============================================================================
// Broadcasting Binary Operations (strided access)
// ============================================================================

__global__ void add_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Convert linear index to multi-dimensional indices
    unsigned int remaining = idx;
    unsigned int a_offset = 0, b_offset = 0;

    for (int d = ndim - 1; d >= 0; d--) {
        unsigned int coord = remaining % shape[d];
        remaining /= shape[d];
        a_offset += coord * a_strides[d];
        b_offset += coord * b_strides[d];
    }

    out[idx] = a[a_offset] + b[b_offset];
}

__global__ void sub_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int remaining = idx;
    unsigned int a_offset = 0, b_offset = 0;

    for (int d = ndim - 1; d >= 0; d--) {
        unsigned int coord = remaining % shape[d];
        remaining /= shape[d];
        a_offset += coord * a_strides[d];
        b_offset += coord * b_strides[d];
    }

    out[idx] = a[a_offset] - b[b_offset];
}

__global__ void mul_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int remaining = idx;
    unsigned int a_offset = 0, b_offset = 0;

    for (int d = ndim - 1; d >= 0; d--) {
        unsigned int coord = remaining % shape[d];
        remaining /= shape[d];
        a_offset += coord * a_strides[d];
        b_offset += coord * b_strides[d];
    }

    out[idx] = a[a_offset] * b[b_offset];
}

__global__ void div_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides, const unsigned int* out_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int remaining = idx;
    unsigned int a_offset = 0, b_offset = 0;

    for (int d = ndim - 1; d >= 0; d--) {
        unsigned int coord = remaining % shape[d];
        remaining /= shape[d];
        a_offset += coord * a_strides[d];
        b_offset += coord * b_strides[d];
    }

    out[idx] = a[a_offset] / b[b_offset];
}

// ============================================================================
// FP8 E4M3 Binary Operations
// All computation done in F32, stored back as FP8
// Uses Hopper PTX intrinsics on SM 8.9+, software emulation on SM 8.0+
// ============================================================================

__global__ void add_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fa + fb));
    }
}

__global__ void sub_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fa - fb));
    }
}

__global__ void mul_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fa * fb));
    }
}

__global__ void div_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fa / fb));
    }
}

__global__ void pow_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(powf(fa, fb)));
    }
}

__global__ void max_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fmaxf(fa, fb)));
    }
}

__global__ void min_fp8_e4m3(const numr_fp8_e4m3* a, const numr_fp8_e4m3* b, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        float fb = fp8_e4m3_to_f32(b[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fminf(fa, fb)));
    }
}

// ============================================================================
// FP8 E5M2 Binary Operations
// ============================================================================

__global__ void add_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fa + fb));
    }
}

__global__ void sub_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fa - fb));
    }
}

__global__ void mul_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fa * fb));
    }
}

__global__ void div_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fa / fb));
    }
}

__global__ void pow_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(powf(fa, fb)));
    }
}

__global__ void max_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fmaxf(fa, fb)));
    }
}

__global__ void min_fp8_e5m2(const numr_fp8_e5m2* a, const numr_fp8_e5m2* b, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        float fb = fp8_e5m2_to_f32(b[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fminf(fa, fb)));
    }
}

} // extern "C"
