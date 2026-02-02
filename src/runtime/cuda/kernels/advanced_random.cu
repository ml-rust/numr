// CUDA kernels for advanced PRNGs: Philox4x32-10, ThreeFry4x64-20, PCG64, Xoshiro256++

#include <cuda_runtime.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Philox4x32-10 (JAX/TensorFlow default)
// ============================================================================

#define PHILOX_M2X32_0 0xD2511F53u
#define PHILOX_M2X32_1 0xCD9E8D57u
#define PHILOX_W32_0 0x9E3779B9u
#define PHILOX_W32_1 0xBB67AE85u

__device__ __forceinline__ void philox_round(uint32_t* ctr, const uint32_t* key) {
    uint64_t prod0 = ((uint64_t)ctr[0]) * PHILOX_M2X32_0;
    uint64_t prod1 = ((uint64_t)ctr[2]) * PHILOX_M2X32_1;
    uint32_t new_ctr[4] = {
        (uint32_t)(prod1 >> 32) ^ ctr[1] ^ key[0],
        (uint32_t)prod1,
        (uint32_t)(prod0 >> 32) ^ ctr[3] ^ key[1],
        (uint32_t)prod0
    };
    ctr[0] = new_ctr[0]; ctr[1] = new_ctr[1]; ctr[2] = new_ctr[2]; ctr[3] = new_ctr[3];
}

__device__ __forceinline__ void philox4x32_10(uint32_t* ctr, uint32_t* key) {
    for (int i = 0; i < 10; i++) {
        philox_round(ctr, key);
        key[0] += PHILOX_W32_0;
        key[1] += PHILOX_W32_1;
    }
}

template<typename T>
__global__ void philox_uniform_impl(T* out, unsigned int n, uint64_t key, uint64_t counter_base) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 4 >= n) return;

    uint32_t key_split[2] = {(uint32_t)(key & 0xFFFFFFFFULL), (uint32_t)(key >> 32)};
    uint64_t counter = counter_base + idx;
    uint32_t ctr[4] = {(uint32_t)(counter & 0xFFFFFFFFULL), (uint32_t)(counter >> 32), 0, 0};

    philox4x32_10(ctr, key_split);

    for (int j = 0; j < 4 && idx * 4 + j < n; j++) {
        T val = (sizeof(T) == 4) ? (T)((ctr[j] >> 8)) / (T)(1u << 24)
                                 : (T)ctr[j] / (T)0xFFFFFFFFU;
        out[idx * 4 + j] = val;
    }
}

template<typename T>
__global__ void philox_randn_impl(T* out, unsigned int n, uint64_t key, uint64_t counter_base) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 4 >= n) return;

    uint32_t key_split[2] = {(uint32_t)(key & 0xFFFFFFFFULL), (uint32_t)(key >> 32)};
    uint64_t counter = counter_base + idx;
    uint32_t ctr[4] = {(uint32_t)(counter & 0xFFFFFFFFULL), (uint32_t)(counter >> 32), 0, 0};

    philox4x32_10(ctr, key_split);

    for (int j = 0; j < 4 && idx * 4 + j < n; j += 2) {
        T u1 = (T)((ctr[j] >> 8)) / (T)(1u << 24);
        T u2 = (T)((ctr[j + 1] >> 8)) / (T)(1u << 24);
        u1 = fmax(u1, (T)1e-10); u1 = fmin(u1, (T)(1.0 - 1e-10));
        T r = sqrt((T)(-2.0) * log(u1));
        T theta = (T)(2.0 * M_PI) * u2;
        out[idx * 4 + j] = r * cos(theta);
        if (idx * 4 + j + 1 < n) out[idx * 4 + j + 1] = r * sin(theta);
    }
}

// ============================================================================
// ThreeFry4x64-20 (Cryptographic quality)
// ============================================================================

__constant__ uint32_t THREEFRY_ROTATION[8][4] = {
    {14, 16, 52, 57}, {23, 40, 5, 37}, {33, 48, 46, 12}, {17, 34, 22, 32},
    {13, 50, 10, 17}, {25, 29, 39, 43}, {26, 24, 20, 10}, {37, 38, 19, 22}
};
#define THREEFRY_PARITY64 0x1BD11BDAA9FC1A22ULL

__device__ __forceinline__ void threefry_round(uint64_t* x, const uint64_t* ks, int r) {
    if (r % 4 == 0) {
        int d = r / 4;
        x[0] += ks[d % 5]; x[1] += ks[(d + 1) % 5];
        x[2] += ks[(d + 2) % 5]; x[3] += ks[(d + 3) % 5] + d;
    }
    const uint32_t* rot = THREEFRY_ROTATION[r % 8];
    x[0] += x[1]; x[1] = (x[1] << rot[0]) | (x[1] >> (64 - rot[0])); x[1] ^= x[0];
    x[2] += x[3]; x[3] = (x[3] << rot[1]) | (x[3] >> (64 - rot[1])); x[3] ^= x[2];
    uint64_t tmp = x[1]; x[1] = x[3]; x[3] = tmp;
}

__device__ __forceinline__ void threefry4x64_20(uint64_t* ctr, const uint64_t* key) {
    uint64_t ks[5] = {key[0], key[1], 0, 0, key[0] ^ key[1] ^ THREEFRY_PARITY64};
    uint64_t x[4] = {ctr[0], ctr[1], ctr[2], ctr[3]};
    for (int r = 0; r < 20; r++) threefry_round(x, ks, r);
    x[0] += ks[0]; x[1] += ks[1]; x[2] += ks[2]; x[3] += ks[3] + 5;
    ctr[0] = x[0]; ctr[1] = x[1]; ctr[2] = x[2]; ctr[3] = x[3];
}

template<typename T>
__global__ void threefry_uniform_impl(T* out, unsigned int n, uint64_t key, uint64_t counter_base) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 4 >= n) return;

    uint64_t key_arr[2] = {key, 0};
    uint64_t ctr[4] = {counter_base + idx, 0, 0, 0};
    threefry4x64_20(ctr, key_arr);

    for (int j = 0; j < 4 && idx * 4 + j < n; j++) {
        out[idx * 4 + j] = (T)(ctr[j] >> 11) / (T)(1ULL << 53);
    }
}

template<typename T>
__global__ void threefry_randn_impl(T* out, unsigned int n, uint64_t key, uint64_t counter_base) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 4 >= n) return;

    uint64_t key_arr[2] = {key, 0};
    uint64_t ctr[4] = {counter_base + idx, 0, 0, 0};
    threefry4x64_20(ctr, key_arr);

    for (int j = 0; j < 4 && idx * 4 + j < n; j += 2) {
        T u1 = (T)(ctr[j] >> 11) / (T)(1ULL << 53);
        T u2 = (T)(ctr[j + 1] >> 11) / (T)(1ULL << 53);
        u1 = fmax(u1, (T)1e-10); u1 = fmin(u1, (T)(1.0 - 1e-10));
        T r = sqrt((T)(-2.0) * log(u1));
        T theta = (T)(2.0 * M_PI) * u2;
        out[idx * 4 + j] = r * cos(theta);
        if (idx * 4 + j + 1 < n) out[idx * 4 + j + 1] = r * sin(theta);
    }
}

// ============================================================================
// PCG64 (NumPy default)
// ============================================================================

#define PCG64_MULT_LO 0x4385df649fccf645ULL
#define PCG64_MULT_HI 0x2360ed051fc65da4ULL

__device__ __forceinline__ void pcg64_step(uint64_t* lo, uint64_t* hi, uint64_t* output) {
    uint64_t old_lo = *lo, old_hi = *hi;
    uint64_t new_lo = old_lo * PCG64_MULT_LO;
    uint64_t new_hi = __umul64hi(old_lo, PCG64_MULT_LO) + old_hi * PCG64_MULT_LO + old_lo * PCG64_MULT_HI;
    new_lo += 1; if (new_lo == 0) new_hi += 1;
    *lo = new_lo; *hi = new_hi;
    uint64_t xorshifted = ((old_hi ^ old_lo) >> 59);
    uint32_t rot = (uint32_t)(old_hi >> 58);
    *output = (xorshifted >> rot) | (xorshifted << ((-rot) & 63));
}

template<typename T>
__global__ void pcg64_uniform_impl(T* out, unsigned int n, uint64_t seed, uint64_t stream) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t state_lo = stream + idx, state_hi = seed, dummy, output;
    pcg64_step(&state_lo, &state_hi, &dummy);
    pcg64_step(&state_lo, &state_hi, &output);
    out[idx] = (T)(output >> 11) / (T)(1ULL << 53);
}

template<typename T>
__global__ void pcg64_randn_impl(T* out, unsigned int n, uint64_t seed, uint64_t stream) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 >= n) return;

    uint64_t state_lo = stream + idx, state_hi = seed, dummy, out1, out2;
    pcg64_step(&state_lo, &state_hi, &dummy);
    pcg64_step(&state_lo, &state_hi, &out1);
    pcg64_step(&state_lo, &state_hi, &out2);

    T u1 = (T)(out1 >> 11) / (T)(1ULL << 53);
    T u2 = (T)(out2 >> 11) / (T)(1ULL << 53);
    u1 = fmax(u1, (T)1e-10); u1 = fmin(u1, (T)(1.0 - 1e-10));
    T r = sqrt((T)(-2.0) * log(u1));
    T theta = (T)(2.0 * M_PI) * u2;
    out[idx * 2] = r * cos(theta);
    if (idx * 2 + 1 < n) out[idx * 2 + 1] = r * sin(theta);
}

// ============================================================================
// Xoshiro256++ (Rust rand default)
// ============================================================================

__device__ __forceinline__ uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ uint64_t xoshiro256_next(uint64_t* s) {
    uint64_t result = ((s[0] + s[3]) << 23 | (s[0] + s[3]) >> 41) + s[0];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = (s[3] << 45) | (s[3] >> 19);
    return result;
}

template<typename T>
__global__ void xoshiro256_uniform_impl(T* out, unsigned int n, uint64_t seed) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t sm_state = seed + idx;
    uint64_t s[4] = {splitmix64(&sm_state), splitmix64(&sm_state),
                     splitmix64(&sm_state), splitmix64(&sm_state)};
    out[idx] = (T)(xoshiro256_next(s) >> 11) / (T)(1ULL << 53);
}

template<typename T>
__global__ void xoshiro256_randn_impl(T* out, unsigned int n, uint64_t seed) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 >= n) return;

    uint64_t sm_state = seed + idx;
    uint64_t s[4] = {splitmix64(&sm_state), splitmix64(&sm_state),
                     splitmix64(&sm_state), splitmix64(&sm_state)};

    T u1 = (T)(xoshiro256_next(s) >> 11) / (T)(1ULL << 53);
    T u2 = (T)(xoshiro256_next(s) >> 11) / (T)(1ULL << 53);
    u1 = fmax(u1, (T)1e-10); u1 = fmin(u1, (T)(1.0 - 1e-10));
    T r = sqrt((T)(-2.0) * log(u1));
    T theta = (T)(2.0 * M_PI) * u2;
    out[idx * 2] = r * cos(theta);
    if (idx * 2 + 1 < n) out[idx * 2 + 1] = r * sin(theta);
}

// ============================================================================
// Extern "C" wrappers - instantiate templates for F32 and F64
// ============================================================================

extern "C" {
    // Philox
    __global__ void philox_uniform_f32(float* out, unsigned int n, uint64_t key, uint64_t ctr) {
        philox_uniform_impl<float>(out, n, key, ctr);
    }
    __global__ void philox_uniform_f64(double* out, unsigned int n, uint64_t key, uint64_t ctr) {
        philox_uniform_impl<double>(out, n, key, ctr);
    }
    __global__ void philox_randn_f32(float* out, unsigned int n, uint64_t key, uint64_t ctr) {
        philox_randn_impl<float>(out, n, key, ctr);
    }
    __global__ void philox_randn_f64(double* out, unsigned int n, uint64_t key, uint64_t ctr) {
        philox_randn_impl<double>(out, n, key, ctr);
    }

    // ThreeFry
    __global__ void threefry_uniform_f32(float* out, unsigned int n, uint64_t key, uint64_t ctr) {
        threefry_uniform_impl<float>(out, n, key, ctr);
    }
    __global__ void threefry_uniform_f64(double* out, unsigned int n, uint64_t key, uint64_t ctr) {
        threefry_uniform_impl<double>(out, n, key, ctr);
    }
    __global__ void threefry_randn_f32(float* out, unsigned int n, uint64_t key, uint64_t ctr) {
        threefry_randn_impl<float>(out, n, key, ctr);
    }
    __global__ void threefry_randn_f64(double* out, unsigned int n, uint64_t key, uint64_t ctr) {
        threefry_randn_impl<double>(out, n, key, ctr);
    }

    // PCG64
    __global__ void pcg64_uniform_f32(float* out, unsigned int n, uint64_t seed, uint64_t stream) {
        pcg64_uniform_impl<float>(out, n, seed, stream);
    }
    __global__ void pcg64_uniform_f64(double* out, unsigned int n, uint64_t seed, uint64_t stream) {
        pcg64_uniform_impl<double>(out, n, seed, stream);
    }
    __global__ void pcg64_randn_f32(float* out, unsigned int n, uint64_t seed, uint64_t stream) {
        pcg64_randn_impl<float>(out, n, seed, stream);
    }
    __global__ void pcg64_randn_f64(double* out, unsigned int n, uint64_t seed, uint64_t stream) {
        pcg64_randn_impl<double>(out, n, seed, stream);
    }

    // Xoshiro256++
    __global__ void xoshiro256_uniform_f32(float* out, unsigned int n, uint64_t seed) {
        xoshiro256_uniform_impl<float>(out, n, seed);
    }
    __global__ void xoshiro256_uniform_f64(double* out, unsigned int n, uint64_t seed) {
        xoshiro256_uniform_impl<double>(out, n, seed);
    }
    __global__ void xoshiro256_randn_f32(float* out, unsigned int n, uint64_t seed) {
        xoshiro256_randn_impl<float>(out, n, seed);
    }
    __global__ void xoshiro256_randn_f64(double* out, unsigned int n, uint64_t seed) {
        xoshiro256_randn_impl<double>(out, n, seed);
    }
}
