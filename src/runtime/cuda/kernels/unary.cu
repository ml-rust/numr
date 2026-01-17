// Unary element-wise CUDA kernels
// Supports: neg, abs, sqrt, exp, log, sin, cos, tan, tanh, recip, square, floor, ceil, round, clamp, sign, isnan, isinf, logical_not
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2, i32, i64, u8 (bool)
// Clamp operation: clamp(x, min_scalar, max_scalar) = min(max(x, lo), hi)
// Sign operation: returns -1, 0, or +1
// IsNaN/IsInf: returns u8 (bool) - 1 if true, 0 if false
// Logical NOT: boolean negation (u8 input/output)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Clamp Helper Device Function (Templated - must be outside extern "C")
// ============================================================================

template<typename T>
__device__ __forceinline__ T clamp_impl(T val, T min_val, T max_val) {
    return (val < min_val) ? min_val : (val > max_val ? max_val : val);
}

// Specialization for float
template<>
__device__ __forceinline__ float clamp_impl(float val, float min_val, float max_val) {
    return fminf(fmaxf(val, min_val), max_val);
}

// Specialization for double
template<>
__device__ __forceinline__ double clamp_impl(double val, double min_val, double max_val) {
    return fmin(fmax(val, min_val), max_val);
}

// Specialization for half
template<>
__device__ __forceinline__ __half clamp_impl(__half val, __half min_val, __half max_val) {
    return __hmin(__hmax(val, min_val), max_val);
}

// Specialization for bfloat16
template<>
__device__ __forceinline__ __nv_bfloat16 clamp_impl(__nv_bfloat16 val, __nv_bfloat16 min_val, __nv_bfloat16 max_val) {
    #if __CUDA_ARCH__ >= 800
    return __hmin(__hmax(val, min_val), max_val);
    #else
    float vf = __bfloat162float(val);
    float minf = __bfloat162float(min_val);
    float maxf = __bfloat162float(max_val);
    return __float2bfloat16(fminf(fmaxf(vf, minf), maxf));
    #endif
}

// Specialization for FP8E4M3 (clamp in F32)
template<>
__device__ __forceinline__ numr_fp8_e4m3 clamp_impl(numr_fp8_e4m3 val, numr_fp8_e4m3 min_val, numr_fp8_e4m3 max_val) {
    float vf = fp8_e4m3_to_f32(val.data);
    float minf = fp8_e4m3_to_f32(min_val.data);
    float maxf = fp8_e4m3_to_f32(max_val.data);
    return numr_fp8_e4m3(f32_to_fp8_e4m3(fminf(fmaxf(vf, minf), maxf)));
}

// Specialization for FP8E5M2 (clamp in F32)
template<>
__device__ __forceinline__ numr_fp8_e5m2 clamp_impl(numr_fp8_e5m2 val, numr_fp8_e5m2 min_val, numr_fp8_e5m2 max_val) {
    float vf = fp8_e5m2_to_f32(val.data);
    float minf = fp8_e5m2_to_f32(min_val.data);
    float maxf = fp8_e5m2_to_f32(max_val.data);
    return numr_fp8_e5m2(f32_to_fp8_e5m2(fminf(fmaxf(vf, minf), maxf)));
}

extern "C" {

// ============================================================================
// F32 Unary Operations
// ============================================================================

__global__ void neg_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -a[idx];
    }
}

__global__ void abs_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fabsf(a[idx]);
    }
}

__global__ void sqrt_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sqrtf(a[idx]);
    }
}

__global__ void exp_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = expf(a[idx]);
    }
}

__global__ void log_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = logf(a[idx]);
    }
}

__global__ void sin_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sinf(a[idx]);
    }
}

__global__ void cos_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = cosf(a[idx]);
    }
}

__global__ void tan_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanf(a[idx]);
    }
}

__global__ void tanh_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanhf(a[idx]);
    }
}

__global__ void recip_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / a[idx];
    }
}

__global__ void square_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = a[idx];
        out[idx] = val * val;
    }
}

__global__ void floor_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = floorf(a[idx]);
    }
}

__global__ void ceil_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = ceilf(a[idx]);
    }
}

__global__ void round_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = roundf(a[idx]);
    }
}

// ============================================================================
// F64 Unary Operations
// ============================================================================

__global__ void neg_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -a[idx];
    }
}

__global__ void abs_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fabs(a[idx]);
    }
}

__global__ void sqrt_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sqrt(a[idx]);
    }
}

__global__ void exp_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = exp(a[idx]);
    }
}

__global__ void log_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = log(a[idx]);
    }
}

__global__ void sin_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sin(a[idx]);
    }
}

__global__ void cos_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = cos(a[idx]);
    }
}

__global__ void tan_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tan(a[idx]);
    }
}

__global__ void tanh_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanh(a[idx]);
    }
}

__global__ void recip_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0 / a[idx];
    }
}

__global__ void square_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = a[idx];
        out[idx] = val * val;
    }
}

__global__ void floor_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = floor(a[idx]);
    }
}

__global__ void ceil_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = ceil(a[idx]);
    }
}

__global__ void round_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = round(a[idx]);
    }
}

// ============================================================================
// F16 Unary Operations (half precision)
// Transcendental functions use FP32 internally for accuracy
// ============================================================================

__global__ void neg_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hneg(a[idx]);
    }
}

__global__ void abs_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __habs(a[idx]);
    }
}

__global__ void sqrt_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hsqrt(a[idx]);
    }
}

__global__ void exp_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hexp(a[idx]);
    }
}

__global__ void log_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hlog(a[idx]);
    }
}

__global__ void sin_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hsin(a[idx]);
    }
}

__global__ void cos_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hcos(a[idx]);
    }
}

__global__ void tan_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // No native htan, use FP32
        out[idx] = __float2half(tanf(__half2float(a[idx])));
    }
}

__global__ void tanh_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Use FP32 for accuracy
        out[idx] = __float2half(tanhf(__half2float(a[idx])));
    }
}

__global__ void recip_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hrcp(a[idx]);
    }
}

__global__ void square_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half val = a[idx];
        out[idx] = __hmul(val, val);
    }
}

__global__ void floor_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hfloor(a[idx]);
    }
}

__global__ void ceil_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hceil(a[idx]);
    }
}

__global__ void round_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = hrint(a[idx]);
    }
}

// ============================================================================
// BF16 Unary Operations (bfloat16)
// All transcendental functions use FP32 internally (no native BF16 math)
// ============================================================================

__global__ void neg_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hneg(a[idx]);
        #else
        out[idx] = __float2bfloat16(-__bfloat162float(a[idx]));
        #endif
    }
}

__global__ void abs_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __habs(a[idx]);
        #else
        out[idx] = __float2bfloat16(fabsf(__bfloat162float(a[idx])));
        #endif
    }
}

__global__ void sqrt_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(sqrtf(__bfloat162float(a[idx])));
    }
}

__global__ void exp_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(expf(__bfloat162float(a[idx])));
    }
}

__global__ void log_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(logf(__bfloat162float(a[idx])));
    }
}

__global__ void sin_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(sinf(__bfloat162float(a[idx])));
    }
}

__global__ void cos_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(cosf(__bfloat162float(a[idx])));
    }
}

__global__ void tan_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(tanf(__bfloat162float(a[idx])));
    }
}

__global__ void tanh_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(tanhf(__bfloat162float(a[idx])));
    }
}

__global__ void recip_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(1.0f / __bfloat162float(a[idx]));
    }
}

__global__ void square_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __bfloat162float(a[idx]);
        out[idx] = __float2bfloat16(val * val);
    }
}

__global__ void floor_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(floorf(__bfloat162float(a[idx])));
    }
}

__global__ void ceil_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(ceilf(__bfloat162float(a[idx])));
    }
}

__global__ void round_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(roundf(__bfloat162float(a[idx])));
    }
}

// ============================================================================
// I32 Unary Operations
// ============================================================================

__global__ void neg_i32(const int* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -a[idx];
    }
}

__global__ void abs_i32(const int* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = abs(a[idx]);
    }
}

__global__ void square_i32(const int* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = a[idx];
        out[idx] = val * val;
    }
}

// ============================================================================
// I64 Unary Operations
// ============================================================================

__global__ void neg_i64(const long long* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -a[idx];
    }
}

__global__ void abs_i64(const long long* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = llabs(a[idx]);
    }
}

__global__ void square_i64(const long long* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        long long val = a[idx];
        out[idx] = val * val;
    }
}

// ============================================================================
// FP8 E4M3 Unary Operations
// All computation done in F32, stored back as FP8
// Uses Hopper PTX intrinsics on SM 8.9+, software emulation on SM 8.0+
// ============================================================================

__global__ void neg_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(-fa));
    }
}

__global__ void abs_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fabsf(fa)));
    }
}

__global__ void sqrt_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(sqrtf(fa)));
    }
}

__global__ void exp_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(expf(fa)));
    }
}

__global__ void log_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(logf(fa)));
    }
}

__global__ void sin_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(sinf(fa)));
    }
}

__global__ void cos_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(cosf(fa)));
    }
}

__global__ void tan_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(tanf(fa)));
    }
}

__global__ void tanh_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(tanhf(fa)));
    }
}

__global__ void recip_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(1.0f / fa));
    }
}

__global__ void square_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fa * fa));
    }
}

__global__ void floor_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(floorf(fa)));
    }
}

__global__ void ceil_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(ceilf(fa)));
    }
}

__global__ void round_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(roundf(fa)));
    }
}

// ============================================================================
// FP8 E5M2 Unary Operations
// ============================================================================

__global__ void neg_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(-fa));
    }
}

__global__ void abs_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fabsf(fa)));
    }
}

__global__ void sqrt_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(sqrtf(fa)));
    }
}

__global__ void exp_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(expf(fa)));
    }
}

__global__ void log_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(logf(fa)));
    }
}

__global__ void sin_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(sinf(fa)));
    }
}

__global__ void cos_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(cosf(fa)));
    }
}

__global__ void tan_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(tanf(fa)));
    }
}

__global__ void tanh_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(tanhf(fa)));
    }
}

__global__ void recip_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(1.0f / fa));
    }
}

__global__ void square_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fa * fa));
    }
}

__global__ void floor_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(floorf(fa)));
    }
}

__global__ void ceil_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(ceilf(fa)));
    }
}

__global__ void round_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(roundf(fa)));
    }
}

// ============================================================================
// Clamp Operations (tensor, min_val, max_val) - Kernel Wrappers
// ============================================================================

__global__ void clamp_f32(const float* a, float min_val, float max_val, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_impl<float>(a[idx], min_val, max_val);
    }
}

__global__ void clamp_f64(const double* a, double min_val, double max_val, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_impl<double>(a[idx], min_val, max_val);
    }
}

__global__ void clamp_f16(const __half* a, __half min_val, __half max_val, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_impl<__half>(a[idx], min_val, max_val);
    }
}

__global__ void clamp_bf16(const __nv_bfloat16* a, __nv_bfloat16 min_val, __nv_bfloat16 max_val, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_impl<__nv_bfloat16>(a[idx], min_val, max_val);
    }
}

__global__ void clamp_i32(const int* a, int min_val, int max_val, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_impl<int>(a[idx], min_val, max_val);
    }
}

__global__ void clamp_i64(const long long* a, long long min_val, long long max_val, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_impl<long long>(a[idx], min_val, max_val);
    }
}

__global__ void clamp_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3 min_val, numr_fp8_e4m3 max_val, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_impl<numr_fp8_e4m3>(a[idx], min_val, max_val);
    }
}

__global__ void clamp_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2 min_val, numr_fp8_e5m2 max_val, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = clamp_impl<numr_fp8_e5m2>(a[idx], min_val, max_val);
    }
}

// ============================================================================
// Sign Operations - Returns -1, 0, or +1
// ============================================================================

__global__ void sign_f32(const float* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = a[idx];
        out[idx] = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
    }
}

__global__ void sign_f64(const double* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = a[idx];
        out[idx] = (val > 0.0) ? 1.0 : ((val < 0.0) ? -1.0 : 0.0);
    }
}

__global__ void sign_f16(const __half* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(a[idx]);
        out[idx] = __float2half((val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f));
    }
}

__global__ void sign_bf16(const __nv_bfloat16* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __bfloat162float(a[idx]);
        out[idx] = __float2bfloat16((val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f));
    }
}

__global__ void sign_i32(const int* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = a[idx];
        out[idx] = (val > 0) ? 1 : ((val < 0) ? -1 : 0);
    }
}

__global__ void sign_i64(const long long* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        long long val = a[idx];
        out[idx] = (val > 0) ? 1 : ((val < 0) ? -1 : 0);
    }
}

__global__ void sign_fp8_e4m3(const numr_fp8_e4m3* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = fp8_e4m3_to_f32(a[idx].data);
        float sign = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(sign));
    }
}

__global__ void sign_fp8_e5m2(const numr_fp8_e5m2* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = fp8_e5m2_to_f32(a[idx].data);
        float sign = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(sign));
    }
}

// ============================================================================
// IsNaN Operations - Returns 1 if NaN, 0 otherwise (output is u8)
// ============================================================================

__global__ void isnan_f32(const float* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = isnan(a[idx]) ? 1 : 0;
    }
}

__global__ void isnan_f64(const double* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = isnan(a[idx]) ? 1 : 0;
    }
}

__global__ void isnan_f16(const __half* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hisnan(a[idx]) ? 1 : 0;
    }
}

__global__ void isnan_bf16(const __nv_bfloat16* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = isnan(__bfloat162float(a[idx])) ? 1 : 0;
    }
}

__global__ void isnan_fp8_e4m3(const numr_fp8_e4m3* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = isnan(val) ? 1 : 0;
    }
}

__global__ void isnan_fp8_e5m2(const numr_fp8_e5m2* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = isnan(val) ? 1 : 0;
    }
}

// ============================================================================
// IsInf Operations - Returns 1 if Inf, 0 otherwise (output is u8)
// ============================================================================

__global__ void isinf_f32(const float* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = isinf(a[idx]) ? 1 : 0;
    }
}

__global__ void isinf_f64(const double* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = isinf(a[idx]) ? 1 : 0;
    }
}

__global__ void isinf_f16(const __half* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hisinf(a[idx]) ? 1 : 0;
    }
}

__global__ void isinf_bf16(const __nv_bfloat16* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = isinf(__bfloat162float(a[idx])) ? 1 : 0;
    }
}

__global__ void isinf_fp8_e4m3(const numr_fp8_e4m3* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = isinf(val) ? 1 : 0;
    }
}

__global__ void isinf_fp8_e5m2(const numr_fp8_e5m2* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = isinf(val) ? 1 : 0;
    }
}

// ============================================================================
// Logical NOT - Boolean negation (input and output are u8)
// ============================================================================

__global__ void logical_not_u8(const unsigned char* a, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] ? 0 : 1;
    }
}

} // extern "C"
