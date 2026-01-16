// Unary element-wise CUDA kernels
// Supports: neg, abs, sqrt, exp, log, sin, cos, tan, tanh, recip, square, floor, ceil, round
// Types: f32, f64, f16, bf16

#include <cuda_fp16.h>
#include <cuda_bf16.h>

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

} // extern "C"
