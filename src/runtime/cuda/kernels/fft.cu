// FFT CUDA kernels using Stockham autosort algorithm
// Supports: Complex64 (float2), Complex128 (double2)
// Uses shared memory for small FFTs (N <= 1024) and global memory for larger FFTs

#include <cuda_runtime.h>
#include <math.h>
#include "dtype_traits.cuh"  // Provides M_PI and M_PI_F constants

// ============================================================================
// Complex Number Helpers
// ============================================================================

__device__ __forceinline__ float2 cmul_f32(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ float2 cadd_f32(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 csub_f32(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ float2 cscale_f32(float2 a, float s) {
    return make_float2(a.x * s, a.y * s);
}

__device__ __forceinline__ double2 cmul_f64(double2 a, double2 b) {
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ double2 cadd_f64(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ double2 csub_f64(double2 a, double2 b) {
    return make_double2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ double2 cscale_f64(double2 a, double s) {
    return make_double2(a.x * s, a.y * s);
}

// ============================================================================
// Twiddle Factor Computation
// ============================================================================

__device__ __forceinline__ float2 twiddle_f32(int k, int n, int inverse) {
    float angle = (inverse ? 1.0f : -1.0f) * 2.0f * (float)M_PI * (float)k / (float)n;
    float c, s;
    sincosf(angle, &s, &c);
    return make_float2(c, s);
}

__device__ __forceinline__ double2 twiddle_f64(int k, int n, int inverse) {
    double angle = (inverse ? 1.0 : -1.0) * 2.0 * M_PI * (double)k / (double)n;
    double c, s;
    sincos(angle, &s, &c);
    return make_double2(c, s);
}

extern "C" {

// ============================================================================
// Batched Stockham FFT - Complex64 (float2)
// Each block processes one FFT of size n
// ============================================================================

__global__ void stockham_fft_batched_c64(
    const float2* __restrict__ input,
    float2* __restrict__ output,
    unsigned int n,
    unsigned int log_n,
    int inverse,
    float scale,
    unsigned int batch_size
) {
    extern __shared__ float2 smem[];
    float2* src = smem;
    float2* dst = smem + n;

    unsigned int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float2* batch_in = input + batch_idx * n;
    float2* batch_out = output + batch_idx * n;

    // Load input to shared memory
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        src[i] = batch_in[i];
    }
    __syncthreads();

    // Stockham FFT stages
    for (unsigned int stage = 0; stage < log_n; stage++) {
        unsigned int m = 1u << (stage + 1);
        unsigned int half_m = 1u << stage;
        unsigned int groups = n / m;

        for (unsigned int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
            unsigned int g = idx / half_m;
            unsigned int b = idx % half_m;

            unsigned int even_idx = g * half_m + b;
            unsigned int odd_idx = even_idx + n / 2;

            float2 tw = twiddle_f32(b * groups, n, inverse);

            float2 even = src[even_idx];
            float2 odd = cmul_f32(src[odd_idx], tw);

            unsigned int out_lo = g * m + b;
            unsigned int out_hi = out_lo + half_m;

            dst[out_lo] = cadd_f32(even, odd);
            dst[out_hi] = csub_f32(even, odd);
        }
        __syncthreads();

        // Swap buffers
        float2* tmp = src;
        src = dst;
        dst = tmp;
    }

    // Write output with scaling
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        batch_out[i] = cscale_f32(src[i], scale);
    }
}

// ============================================================================
// Batched Stockham FFT - Complex128 (double2)
// ============================================================================

__global__ void stockham_fft_batched_c128(
    const double2* __restrict__ input,
    double2* __restrict__ output,
    unsigned int n,
    unsigned int log_n,
    int inverse,
    double scale,
    unsigned int batch_size
) {
    extern __shared__ double2 smem_d[];
    double2* src = smem_d;
    double2* dst = smem_d + n;

    unsigned int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const double2* batch_in = input + batch_idx * n;
    double2* batch_out = output + batch_idx * n;

    // Load input to shared memory
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        src[i] = batch_in[i];
    }
    __syncthreads();

    // Stockham FFT stages
    for (unsigned int stage = 0; stage < log_n; stage++) {
        unsigned int m = 1u << (stage + 1);
        unsigned int half_m = 1u << stage;
        unsigned int groups = n / m;

        for (unsigned int idx = threadIdx.x; idx < n / 2; idx += blockDim.x) {
            unsigned int g = idx / half_m;
            unsigned int b = idx % half_m;

            unsigned int even_idx = g * half_m + b;
            unsigned int odd_idx = even_idx + n / 2;

            double2 tw = twiddle_f64(b * groups, n, inverse);

            double2 even = src[even_idx];
            double2 odd = cmul_f64(src[odd_idx], tw);

            unsigned int out_lo = g * m + b;
            unsigned int out_hi = out_lo + half_m;

            dst[out_lo] = cadd_f64(even, odd);
            dst[out_hi] = csub_f64(even, odd);
        }
        __syncthreads();

        // Swap buffers
        double2* tmp = src;
        src = dst;
        dst = tmp;
    }

    // Write output with scaling
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        batch_out[i] = cscale_f64(src[i], scale);
    }
}

// ============================================================================
// Large FFT Kernels (N > 1024) - Global Memory
// Single stage of Stockham FFT for large transforms
// ============================================================================

__global__ void stockham_fft_stage_c64(
    const float2* __restrict__ src,
    float2* __restrict__ dst,
    unsigned int n,
    unsigned int stage,
    int inverse,
    unsigned int batch_size
) {
    unsigned int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n / 2) return;

    const float2* batch_src = src + batch_idx * n;
    float2* batch_dst = dst + batch_idx * n;

    unsigned int m = 1u << (stage + 1);
    unsigned int half_m = 1u << stage;
    unsigned int groups = n / m;

    unsigned int g = idx / half_m;
    unsigned int b = idx % half_m;

    unsigned int even_idx = g * half_m + b;
    unsigned int odd_idx = even_idx + n / 2;

    float2 tw = twiddle_f32(b * groups, n, inverse);

    float2 even = batch_src[even_idx];
    float2 odd = cmul_f32(batch_src[odd_idx], tw);

    unsigned int out_lo = g * m + b;
    unsigned int out_hi = out_lo + half_m;

    batch_dst[out_lo] = cadd_f32(even, odd);
    batch_dst[out_hi] = csub_f32(even, odd);
}

__global__ void stockham_fft_stage_c128(
    const double2* __restrict__ src,
    double2* __restrict__ dst,
    unsigned int n,
    unsigned int stage,
    int inverse,
    unsigned int batch_size
) {
    unsigned int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n / 2) return;

    const double2* batch_src = src + batch_idx * n;
    double2* batch_dst = dst + batch_idx * n;

    unsigned int m = 1u << (stage + 1);
    unsigned int half_m = 1u << stage;
    unsigned int groups = n / m;

    unsigned int g = idx / half_m;
    unsigned int b = idx % half_m;

    unsigned int even_idx = g * half_m + b;
    unsigned int odd_idx = even_idx + n / 2;

    double2 tw = twiddle_f64(b * groups, n, inverse);

    double2 even = batch_src[even_idx];
    double2 odd = cmul_f64(batch_src[odd_idx], tw);

    unsigned int out_lo = g * m + b;
    unsigned int out_hi = out_lo + half_m;

    batch_dst[out_lo] = cadd_f64(even, odd);
    batch_dst[out_hi] = csub_f64(even, odd);
}

// ============================================================================
// Scale Kernel - Apply normalization factor
// ============================================================================

__global__ void scale_complex_c64(
    float2* __restrict__ data,
    float scale,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = cscale_f32(data[idx], scale);
    }
}

__global__ void scale_complex_c128(
    double2* __restrict__ data,
    double scale,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = cscale_f64(data[idx], scale);
    }
}

// ============================================================================
// Copy Kernel - Simple element-wise copy
// ============================================================================

__global__ void copy_complex_c64(
    const float2* __restrict__ src,
    float2* __restrict__ dst,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

__global__ void copy_complex_c128(
    const double2* __restrict__ src,
    double2* __restrict__ dst,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// ============================================================================
// Real FFT Kernels - Real to Complex and Complex to Real
// ============================================================================

// rfft: Convert N real values to N/2+1 complex values
__global__ void rfft_pack_c64(
    const float* __restrict__ real_input,
    float2* __restrict__ complex_input,
    unsigned int n,
    unsigned int batch_size
) {
    unsigned int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float* batch_real = real_input + batch_idx * n;
    float2* batch_complex = complex_input + batch_idx * n;

    batch_complex[idx] = make_float2(batch_real[idx], 0.0f);
}

__global__ void rfft_pack_c128(
    const double* __restrict__ real_input,
    double2* __restrict__ complex_input,
    unsigned int n,
    unsigned int batch_size
) {
    unsigned int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const double* batch_real = real_input + batch_idx * n;
    double2* batch_complex = complex_input + batch_idx * n;

    batch_complex[idx] = make_double2(batch_real[idx], 0.0);
}

// irfft: Extract real values from complex FFT output
__global__ void irfft_unpack_c64(
    const float2* __restrict__ complex_output,
    float* __restrict__ real_output,
    unsigned int n,
    float scale,
    unsigned int batch_size
) {
    unsigned int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float2* batch_complex = complex_output + batch_idx * n;
    float* batch_real = real_output + batch_idx * n;

    batch_real[idx] = batch_complex[idx].x * scale;
}

__global__ void irfft_unpack_c128(
    const double2* __restrict__ complex_output,
    double* __restrict__ real_output,
    unsigned int n,
    double scale,
    unsigned int batch_size
) {
    unsigned int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const double2* batch_complex = complex_output + batch_idx * n;
    double* batch_real = real_output + batch_idx * n;

    batch_real[idx] = batch_complex[idx].x * scale;
}

// Hermitian extension for irfft: N/2+1 complex -> N complex
__global__ void hermitian_extend_c64(
    const float2* __restrict__ half_spectrum,
    float2* __restrict__ full_spectrum,
    unsigned int half_n,
    unsigned int full_n,
    unsigned int batch_size
) {
    unsigned int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= full_n) return;

    const float2* batch_half = half_spectrum + batch_idx * half_n;
    float2* batch_full = full_spectrum + batch_idx * full_n;

    if (idx < half_n) {
        batch_full[idx] = batch_half[idx];
    } else {
        // Conjugate symmetry: X[N-k] = conj(X[k])
        unsigned int mirror = full_n - idx;
        float2 val = batch_half[mirror];
        batch_full[idx] = make_float2(val.x, -val.y);
    }
}

__global__ void hermitian_extend_c128(
    const double2* __restrict__ half_spectrum,
    double2* __restrict__ full_spectrum,
    unsigned int half_n,
    unsigned int full_n,
    unsigned int batch_size
) {
    unsigned int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= full_n) return;

    const double2* batch_half = half_spectrum + batch_idx * half_n;
    double2* batch_full = full_spectrum + batch_idx * full_n;

    if (idx < half_n) {
        batch_full[idx] = batch_half[idx];
    } else {
        // Conjugate symmetry: X[N-k] = conj(X[k])
        unsigned int mirror = full_n - idx;
        double2 val = batch_half[mirror];
        batch_full[idx] = make_double2(val.x, -val.y);
    }
}

// rfft truncation: keep only first N/2+1 elements
__global__ void rfft_truncate_c64(
    const float2* __restrict__ full_spectrum,
    float2* __restrict__ half_spectrum,
    unsigned int full_n,
    unsigned int half_n,
    unsigned int batch_size
) {
    unsigned int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half_n) return;

    const float2* batch_full = full_spectrum + batch_idx * full_n;
    float2* batch_half = half_spectrum + batch_idx * half_n;

    batch_half[idx] = batch_full[idx];
}

__global__ void rfft_truncate_c128(
    const double2* __restrict__ full_spectrum,
    double2* __restrict__ half_spectrum,
    unsigned int full_n,
    unsigned int half_n,
    unsigned int batch_size
) {
    unsigned int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half_n) return;

    const double2* batch_full = full_spectrum + batch_idx * full_n;
    double2* batch_half = half_spectrum + batch_idx * half_n;

    batch_half[idx] = batch_full[idx];
}

// ============================================================================
// FFT Shift Kernels
// ============================================================================

__global__ void fftshift_c64(
    const float2* __restrict__ input,
    float2* __restrict__ output,
    unsigned int n,
    unsigned int batch_size
) {
    unsigned int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float2* batch_in = input + batch_idx * n;
    float2* batch_out = output + batch_idx * n;

    unsigned int half = n / 2;
    unsigned int out_idx = (idx + half) % n;
    batch_out[out_idx] = batch_in[idx];
}

__global__ void fftshift_c128(
    const double2* __restrict__ input,
    double2* __restrict__ output,
    unsigned int n,
    unsigned int batch_size
) {
    unsigned int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const double2* batch_in = input + batch_idx * n;
    double2* batch_out = output + batch_idx * n;

    unsigned int half = n / 2;
    unsigned int out_idx = (idx + half) % n;
    batch_out[out_idx] = batch_in[idx];
}

__global__ void ifftshift_c64(
    const float2* __restrict__ input,
    float2* __restrict__ output,
    unsigned int n,
    unsigned int batch_size
) {
    unsigned int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float2* batch_in = input + batch_idx * n;
    float2* batch_out = output + batch_idx * n;

    unsigned int half = (n + 1) / 2;  // Ceiling division for odd n
    unsigned int out_idx = (idx + half) % n;
    batch_out[out_idx] = batch_in[idx];
}

__global__ void ifftshift_c128(
    const double2* __restrict__ input,
    double2* __restrict__ output,
    unsigned int n,
    unsigned int batch_size
) {
    unsigned int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const double2* batch_in = input + batch_idx * n;
    double2* batch_out = output + batch_idx * n;

    unsigned int half = (n + 1) / 2;  // Ceiling division for odd n
    unsigned int out_idx = (idx + half) % n;
    batch_out[out_idx] = batch_in[idx];
}

} // extern "C"
