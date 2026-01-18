// Utility CUDA kernels
// Supports: fill (initialize tensor with constant value)
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2, i32, i64, u8 (bool)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

extern "C" {

// ============================================================================
// Fill Operations - Initialize tensor with constant value
// ============================================================================

__global__ void fill_f32(float* out, float value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void fill_f64(double* out, double value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void fill_f16(__half* out, __half value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void fill_bf16(__nv_bfloat16* out, __nv_bfloat16 value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void fill_i32(int* out, int value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void fill_i64(long long* out, long long value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void fill_u8(unsigned char* out, unsigned char value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void fill_fp8_e4m3(numr_fp8_e4m3* out, numr_fp8_e4m3 value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

__global__ void fill_fp8_e5m2(numr_fp8_e5m2* out, numr_fp8_e5m2 value, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = value;
    }
}

} // extern "C"

// ============================================================================
// Random Number Generation - Device Functions (outside extern "C")
// ============================================================================

// xorshift128+ state per thread
struct XorShift128PlusState {
    unsigned long long s0;
    unsigned long long s1;
};

// Initialize state from seed and thread index
__device__ __forceinline__ void xorshift128plus_init(XorShift128PlusState* state, unsigned long long seed, unsigned int idx) {
    // Use splitmix64 to initialize both state values from seed + idx
    unsigned long long z = seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    state->s0 = z ^ (z >> 31);

    z = seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL + 1;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    state->s1 = z ^ (z >> 31);

    // Ensure non-zero state
    if (state->s0 == 0) state->s0 = 1;
    if (state->s1 == 0) state->s1 = 1;
}

// Generate next random 64-bit value
__device__ __forceinline__ unsigned long long xorshift128plus_next(XorShift128PlusState* state) {
    unsigned long long s1 = state->s0;
    unsigned long long s0 = state->s1;
    unsigned long long result = s0 + s1;
    state->s0 = s0;
    s1 ^= s1 << 23;
    state->s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return result;
}

// Convert to uniform [0, 1)
__device__ __forceinline__ double xorshift128plus_uniform(XorShift128PlusState* state) {
    // Use upper 53 bits for double precision
    return (double)(xorshift128plus_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

// Box-Muller transform for normal distribution
__device__ __forceinline__ void box_muller(XorShift128PlusState* state, float* z0, float* z1) {
    double u1 = xorshift128plus_uniform(state);
    double u2 = xorshift128plus_uniform(state);

    // Avoid log(0)
    if (u1 < 1e-12) u1 = 1e-12;

    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;

    *z0 = (float)(r * cos(theta));
    *z1 = (float)(r * sin(theta));
}

__device__ __forceinline__ void box_muller_f64(XorShift128PlusState* state, double* z0, double* z1) {
    double u1 = xorshift128plus_uniform(state);
    double u2 = xorshift128plus_uniform(state);

    if (u1 < 1e-15) u1 = 1e-15;

    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;

    *z0 = r * cos(theta);
    *z1 = r * sin(theta);
}

// ============================================================================
// Uniform Random [0, 1) - Native CUDA kernels
// ============================================================================

extern "C" {

__global__ void rand_f32(float* out, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = (float)xorshift128plus_uniform(&state);
    }
}

__global__ void rand_f64(double* out, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);
        out[idx] = xorshift128plus_uniform(&state);
    }
}

// ============================================================================
// Normal Random (mean=0, std=1) - Native CUDA kernels using Box-Muller
// ============================================================================

__global__ void randn_f32(float* out, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Box-Muller generates pairs, so we handle two elements per thread when possible
    unsigned int pair_idx = idx * 2;

    if (pair_idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);

        float z0, z1;
        box_muller(&state, &z0, &z1);

        out[pair_idx] = z0;
        if (pair_idx + 1 < n) {
            out[pair_idx + 1] = z1;
        }
    }
}

__global__ void randn_f64(double* out, unsigned long long seed, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pair_idx = idx * 2;

    if (pair_idx < n) {
        XorShift128PlusState state;
        xorshift128plus_init(&state, seed, idx);

        double z0, z1;
        box_muller_f64(&state, &z0, &z1);

        out[pair_idx] = z0;
        if (pair_idx + 1 < n) {
            out[pair_idx + 1] = z1;
        }
    }
}

} // extern "C"
