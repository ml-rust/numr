// Type casting CUDA kernels
// Supports: cast between f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2, i32, i64
//
// Kernel naming: cast_{src_dtype}_{dst_dtype}
// Example: cast_f32_f16 converts from f32 to f16

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

extern "C" {

// ============================================================================
// F32 -> Other Types
// ============================================================================

__global__ void cast_f32_f64(const float* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (double)a[idx];
    }
}

__global__ void cast_f32_f16(const float* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(a[idx]);
    }
}

__global__ void cast_f32_bf16(const float* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(a[idx]);
    }
}

__global__ void cast_f32_fp8_e4m3(const float* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(a[idx]));
    }
}

__global__ void cast_f32_fp8_e5m2(const float* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(a[idx]));
    }
}

__global__ void cast_f32_i32(const float* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (int)a[idx];
    }
}

__global__ void cast_f32_i64(const float* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (long long)a[idx];
    }
}

// ============================================================================
// F64 -> Other Types
// ============================================================================

__global__ void cast_f64_f32(const double* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (float)a[idx];
    }
}

__global__ void cast_f64_f16(const double* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half((float)a[idx]);
    }
}

__global__ void cast_f64_bf16(const double* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16((float)a[idx]);
    }
}

__global__ void cast_f64_fp8_e4m3(const double* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3((float)a[idx]));
    }
}

__global__ void cast_f64_fp8_e5m2(const double* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2((float)a[idx]));
    }
}

__global__ void cast_f64_i32(const double* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (int)a[idx];
    }
}

__global__ void cast_f64_i64(const double* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (long long)a[idx];
    }
}

// ============================================================================
// F16 -> Other Types
// ============================================================================

__global__ void cast_f16_f32(const __half* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(a[idx]);
    }
}

__global__ void cast_f16_f64(const __half* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (double)__half2float(a[idx]);
    }
}

__global__ void cast_f16_bf16(const __half* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(__half2float(a[idx]));
    }
}

__global__ void cast_f16_fp8_e4m3(const __half* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(__half2float(a[idx])));
    }
}

__global__ void cast_f16_fp8_e5m2(const __half* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(__half2float(a[idx])));
    }
}

__global__ void cast_f16_i32(const __half* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (int)__half2float(a[idx]);
    }
}

__global__ void cast_f16_i64(const __half* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (long long)__half2float(a[idx]);
    }
}

// ============================================================================
// BF16 -> Other Types
// ============================================================================

__global__ void cast_bf16_f32(const __nv_bfloat16* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __bfloat162float(a[idx]);
    }
}

__global__ void cast_bf16_f64(const __nv_bfloat16* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (double)__bfloat162float(a[idx]);
    }
}

__global__ void cast_bf16_f16(const __nv_bfloat16* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(__bfloat162float(a[idx]));
    }
}

__global__ void cast_bf16_fp8_e4m3(const __nv_bfloat16* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(__bfloat162float(a[idx])));
    }
}

__global__ void cast_bf16_fp8_e5m2(const __nv_bfloat16* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(__bfloat162float(a[idx])));
    }
}

__global__ void cast_bf16_i32(const __nv_bfloat16* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (int)__bfloat162float(a[idx]);
    }
}

__global__ void cast_bf16_i64(const __nv_bfloat16* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (long long)__bfloat162float(a[idx]);
    }
}

// ============================================================================
// FP8 E4M3 -> Other Types
// ============================================================================

__global__ void cast_fp8_e4m3_f32(const numr_fp8_e4m3* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fp8_e4m3_to_f32(a[idx].data);
    }
}

__global__ void cast_fp8_e4m3_f64(const numr_fp8_e4m3* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (double)fp8_e4m3_to_f32(a[idx].data);
    }
}

__global__ void cast_fp8_e4m3_f16(const numr_fp8_e4m3* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(fp8_e4m3_to_f32(a[idx].data));
    }
}

__global__ void cast_fp8_e4m3_bf16(const numr_fp8_e4m3* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(fp8_e4m3_to_f32(a[idx].data));
    }
}

__global__ void cast_fp8_e4m3_fp8_e5m2(const numr_fp8_e4m3* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float f = fp8_e4m3_to_f32(a[idx].data);
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(f));
    }
}

__global__ void cast_fp8_e4m3_i32(const numr_fp8_e4m3* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (int)fp8_e4m3_to_f32(a[idx].data);
    }
}

__global__ void cast_fp8_e4m3_i64(const numr_fp8_e4m3* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (long long)fp8_e4m3_to_f32(a[idx].data);
    }
}

// ============================================================================
// FP8 E5M2 -> Other Types
// ============================================================================

__global__ void cast_fp8_e5m2_f32(const numr_fp8_e5m2* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fp8_e5m2_to_f32(a[idx].data);
    }
}

__global__ void cast_fp8_e5m2_f64(const numr_fp8_e5m2* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (double)fp8_e5m2_to_f32(a[idx].data);
    }
}

__global__ void cast_fp8_e5m2_f16(const numr_fp8_e5m2* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(fp8_e5m2_to_f32(a[idx].data));
    }
}

__global__ void cast_fp8_e5m2_bf16(const numr_fp8_e5m2* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(fp8_e5m2_to_f32(a[idx].data));
    }
}

__global__ void cast_fp8_e5m2_fp8_e4m3(const numr_fp8_e5m2* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float f = fp8_e5m2_to_f32(a[idx].data);
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(f));
    }
}

__global__ void cast_fp8_e5m2_i32(const numr_fp8_e5m2* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (int)fp8_e5m2_to_f32(a[idx].data);
    }
}

__global__ void cast_fp8_e5m2_i64(const numr_fp8_e5m2* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (long long)fp8_e5m2_to_f32(a[idx].data);
    }
}

// ============================================================================
// I32 -> Other Types
// ============================================================================

__global__ void cast_i32_f32(const int* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (float)a[idx];
    }
}

__global__ void cast_i32_f64(const int* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (double)a[idx];
    }
}

__global__ void cast_i32_f16(const int* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half((float)a[idx]);
    }
}

__global__ void cast_i32_bf16(const int* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16((float)a[idx]);
    }
}

__global__ void cast_i32_fp8_e4m3(const int* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3((float)a[idx]));
    }
}

__global__ void cast_i32_fp8_e5m2(const int* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2((float)a[idx]));
    }
}

__global__ void cast_i32_i64(const int* a, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (long long)a[idx];
    }
}

// ============================================================================
// I64 -> Other Types
// ============================================================================

__global__ void cast_i64_f32(const long long* a, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (float)a[idx];
    }
}

__global__ void cast_i64_f64(const long long* a, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (double)a[idx];
    }
}

__global__ void cast_i64_f16(const long long* a, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half((float)a[idx]);
    }
}

__global__ void cast_i64_bf16(const long long* a, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16((float)a[idx]);
    }
}

__global__ void cast_i64_fp8_e4m3(const long long* a, numr_fp8_e4m3* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3((float)a[idx]));
    }
}

__global__ void cast_i64_fp8_e5m2(const long long* a, numr_fp8_e5m2* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2((float)a[idx]));
    }
}

__global__ void cast_i64_i32(const long long* a, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (int)a[idx];
    }
}

} // extern "C"
