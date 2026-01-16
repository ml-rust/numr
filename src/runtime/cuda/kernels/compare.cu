// Comparison CUDA kernels
// Supports: eq, ne, lt, le, gt, ge
// Types: f32, f64, f16, bf16, i32, i64
// Output: same type as input (1 for true, 0 for false)

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" {

// ============================================================================
// F32 Comparison Operations
// ============================================================================

__global__ void eq_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void ne_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] != b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void lt_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void le_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] <= b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void gt_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void ge_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] >= b[idx]) ? 1.0f : 0.0f;
    }
}

// ============================================================================
// F64 Comparison Operations
// ============================================================================

__global__ void eq_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] == b[idx]) ? 1.0 : 0.0;
    }
}

__global__ void ne_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] != b[idx]) ? 1.0 : 0.0;
    }
}

__global__ void lt_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? 1.0 : 0.0;
    }
}

__global__ void le_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] <= b[idx]) ? 1.0 : 0.0;
    }
}

__global__ void gt_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? 1.0 : 0.0;
    }
}

__global__ void ge_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] >= b[idx]) ? 1.0 : 0.0;
    }
}

// ============================================================================
// I32 Comparison Operations
// ============================================================================

__global__ void eq_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] == b[idx]) ? 1 : 0;
    }
}

__global__ void ne_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] != b[idx]) ? 1 : 0;
    }
}

__global__ void lt_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? 1 : 0;
    }
}

__global__ void le_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] <= b[idx]) ? 1 : 0;
    }
}

__global__ void gt_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? 1 : 0;
    }
}

__global__ void ge_i32(const int* a, const int* b, int* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] >= b[idx]) ? 1 : 0;
    }
}

// ============================================================================
// I64 Comparison Operations
// ============================================================================

__global__ void eq_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] == b[idx]) ? 1LL : 0LL;
    }
}

__global__ void ne_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] != b[idx]) ? 1LL : 0LL;
    }
}

__global__ void lt_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? 1LL : 0LL;
    }
}

__global__ void le_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] <= b[idx]) ? 1LL : 0LL;
    }
}

__global__ void gt_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? 1LL : 0LL;
    }
}

__global__ void ge_i64(const long long* a, const long long* b, long long* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] >= b[idx]) ? 1LL : 0LL;
    }
}

// ============================================================================
// Broadcasting Comparison Operations (F32)
// ============================================================================

__global__ void eq_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
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

    out[idx] = (a[a_offset] == b[b_offset]) ? 1.0f : 0.0f;
}

// ============================================================================
// F16 Comparison Operations
// ============================================================================

__global__ void eq_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __heq(a[idx], b[idx]) ? __float2half(1.0f) : __float2half(0.0f);
    }
}

__global__ void ne_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hne(a[idx], b[idx]) ? __float2half(1.0f) : __float2half(0.0f);
    }
}

__global__ void lt_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hlt(a[idx], b[idx]) ? __float2half(1.0f) : __float2half(0.0f);
    }
}

__global__ void le_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hle(a[idx], b[idx]) ? __float2half(1.0f) : __float2half(0.0f);
    }
}

__global__ void gt_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hgt(a[idx], b[idx]) ? __float2half(1.0f) : __float2half(0.0f);
    }
}

__global__ void ge_f16(const __half* a, const __half* b, __half* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hge(a[idx], b[idx]) ? __float2half(1.0f) : __float2half(0.0f);
    }
}

// ============================================================================
// BF16 Comparison Operations
// ============================================================================

__global__ void eq_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __heq(a[idx], b[idx]) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #else
        out[idx] = (__bfloat162float(a[idx]) == __bfloat162float(b[idx])) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #endif
    }
}

__global__ void ne_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hne(a[idx], b[idx]) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #else
        out[idx] = (__bfloat162float(a[idx]) != __bfloat162float(b[idx])) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #endif
    }
}

__global__ void lt_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hlt(a[idx], b[idx]) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #else
        out[idx] = (__bfloat162float(a[idx]) < __bfloat162float(b[idx])) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #endif
    }
}

__global__ void le_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hle(a[idx], b[idx]) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #else
        out[idx] = (__bfloat162float(a[idx]) <= __bfloat162float(b[idx])) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #endif
    }
}

__global__ void gt_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hgt(a[idx], b[idx]) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #else
        out[idx] = (__bfloat162float(a[idx]) > __bfloat162float(b[idx])) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #endif
    }
}

__global__ void ge_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        #if __CUDA_ARCH__ >= 800
        out[idx] = __hge(a[idx], b[idx]) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #else
        out[idx] = (__bfloat162float(a[idx]) >= __bfloat162float(b[idx])) ? __float2bfloat16(1.0f) : __float2bfloat16(0.0f);
        #endif
    }
}

// ============================================================================
// Broadcasting Comparison Operations (F32)
// ============================================================================

__global__ void lt_broadcast_f32(
    const float* a, const float* b, float* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
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

    out[idx] = (a[a_offset] < b[b_offset]) ? 1.0f : 0.0f;
}

} // extern "C"
