// numr DType Traits - Multi-precision CUDA kernel support
// Provides FP8 types with Hopper PTX intrinsics (SM 8.9+) and software fallback (SM 8.0+)
//
// Supports: F32, F64, F16, BF16, FP8E4M3, FP8E5M2, I32, I64
// Note: Using numr_ prefix to avoid collision with NVIDIA's __nv_fp8 types

#ifndef NUMR_DTYPE_TRAITS_CUH
#define NUMR_DTYPE_TRAITS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ============================================================================
// Mathematical Constants
// ============================================================================
// Following C99/C++11 standards for mathematical constants
// These match std::f32::consts::PI and std::f64::consts::PI exactly

#ifndef M_PI
#define M_PI 3.14159265358979323846  // π to F64 precision (20 digits, ~17 significant)
#endif

#ifndef M_PI_F
#define M_PI_F 3.14159265f  // π to F32 precision (~7 significant digits)
#endif

// ============================================================================
// FP8 Type Definitions
// ============================================================================

// numr_fp8_e4m3: 1 sign + 4 exponent + 3 mantissa bits
// Range: ~[-448, 448], better precision
struct numr_fp8_e4m3 {
    uint8_t data;
    __device__ __forceinline__ numr_fp8_e4m3() : data(0) {}
    __device__ __forceinline__ explicit numr_fp8_e4m3(uint8_t v) : data(v) {}
    __device__ __forceinline__ operator uint8_t() const { return data; }
};

// numr_fp8_e5m2: 1 sign + 5 exponent + 2 mantissa bits
// Range: ~[-57344, 57344], larger dynamic range
struct numr_fp8_e5m2 {
    uint8_t data;
    __device__ __forceinline__ numr_fp8_e5m2() : data(0) {}
    __device__ __forceinline__ explicit numr_fp8_e5m2(uint8_t v) : data(v) {}
    __device__ __forceinline__ operator uint8_t() const { return data; }
};

// ============================================================================
// FP8 Conversion Utilities
// - SM 8.9+ (Hopper): Hardware PTX intrinsics for ~2x speedup
// - SM 8.0+ (Ampere): Software emulation (still functional)
// ============================================================================

#if __CUDA_ARCH__ >= 800  // Ampere and newer support FP8

// ============================================================================
// FP8 E4M3 <-> F32 Conversion
// ============================================================================

__device__ __forceinline__ float fp8_e4m3_to_f32(uint8_t u) {
#if __CUDA_ARCH__ >= 890
    // Hopper+: Native PTX instruction
    // cvt.rn.f32.e4m3x2 unpacks 2xFP8 -> 2xF32 (we use only the first result)
    float result;
    asm volatile (
        "{ .reg .f32 dummy; \n\t"
        "cvt.rn.f32.e4m3x2 {%0, dummy}, %1; \n\t"
        "}"
        : "=f"(result)
        : "r"((uint16_t)u)
    );
    return result;
#else
    // Ampere: Software emulation
    int sign = (u >> 7) & 1;
    int exp_bits = (u >> 3) & 0xF;
    int mant_bits = u & 0x7;

    if (exp_bits == 0 && mant_bits == 0) {
        return sign ? -0.0f : 0.0f;
    }

    float result = 0.0f;
    if (exp_bits != 0) {
        float mant = 1.0f + mant_bits * (1.0f / 8.0f);
        int exp = exp_bits - 7;
        result = ldexpf(mant, exp);
    } else {
        // Subnormal: implicit leading bit is 0
        float mant = mant_bits * (1.0f / 8.0f);
        result = ldexpf(mant, -6);
    }

    return sign ? -result : result;
#endif
}

__device__ __forceinline__ uint8_t f32_to_fp8_e4m3(float x) {
#if __CUDA_ARCH__ >= 890
    // Hopper+: Native PTX instruction with saturation
    uint32_t result;
    asm ("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
         : "=r"(result)
         : "f"(x), "f"(0.0f));
    return (uint8_t)result;
#else
    // Ampere: Software emulation with saturation
    if (x == 0.0f) return 0;
    if (isnan(x)) return 0x7F;  // NaN -> max positive

    int sign = (x < 0.0f) ? 1 : 0;
    float abs_x = fabsf(x);

    // Saturate to E4M3 range
    if (abs_x > 448.0f) abs_x = 448.0f;

    int exp_bits;
    float mant = frexpf(abs_x, &exp_bits);
    exp_bits = exp_bits - 1;
    mant = ldexpf(abs_x, -exp_bits);

    exp_bits = max(-7, min(8, exp_bits));
    int exp_encoded = exp_bits + 7;

    int mant_bits = (int)roundf((mant - 1.0f) * 8.0f);
    mant_bits = max(0, min(7, mant_bits));

    return (sign << 7) | (exp_encoded << 3) | mant_bits;
#endif
}

// ============================================================================
// FP8 E5M2 <-> F32 Conversion
// ============================================================================

__device__ __forceinline__ float fp8_e5m2_to_f32(uint8_t u) {
#if __CUDA_ARCH__ >= 890
    // Hopper+: Native PTX instruction
    float result;
    asm volatile (
        "{ .reg .f32 dummy; \n\t"
        "cvt.rn.f32.e5m2x2 {%0, dummy}, %1; \n\t"
        "}"
        : "=f"(result)
        : "r"((uint16_t)u)
    );
    return result;
#else
    // Ampere: Software emulation
    int sign = (u >> 7) & 1;
    int exp_bits = (u >> 2) & 0x1F;
    int mant_bits = u & 0x3;

    if (exp_bits == 0 && mant_bits == 0) {
        return sign ? -0.0f : 0.0f;
    }

    float result = 0.0f;
    if (exp_bits != 0) {
        float mant = 1.0f + mant_bits * 0.25f;
        int exp = exp_bits - 15;
        result = ldexpf(mant, exp);
    } else {
        // Subnormal
        float mant = mant_bits * 0.25f;
        result = ldexpf(mant, -14);
    }

    return sign ? -result : result;
#endif
}

__device__ __forceinline__ uint8_t f32_to_fp8_e5m2(float x) {
#if __CUDA_ARCH__ >= 890
    // Hopper+: Native PTX instruction with saturation
    uint32_t result;
    asm ("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
         : "=r"(result)
         : "f"(x), "f"(0.0f));
    return (uint8_t)result;
#else
    // Ampere: Software emulation with saturation
    if (x == 0.0f) return 0;
    if (isnan(x)) return 0x7F;  // NaN -> max positive

    int sign = (x < 0.0f) ? 1 : 0;
    float abs_x = fabsf(x);

    // Saturate to E5M2 range
    if (abs_x > 57344.0f) abs_x = 57344.0f;

    int exp_bits;
    float mant = frexpf(abs_x, &exp_bits);
    exp_bits = exp_bits - 1;
    mant = ldexpf(abs_x, -exp_bits);

    exp_bits = max(-15, min(16, exp_bits));
    int exp_encoded = exp_bits + 15;

    int mant_bits = (int)roundf((mant - 1.0f) * 4.0f);
    mant_bits = max(0, min(3, mant_bits));

    return (sign << 7) | (exp_encoded << 2) | mant_bits;
#endif
}

#else  // __CUDA_ARCH__ < 800

// Pre-Ampere fallback (very slow, should rarely be used)
__device__ __forceinline__ float fp8_e4m3_to_f32(uint8_t u) {
    int sign = (u >> 7) & 1;
    int exp_bits = (u >> 3) & 0xF;
    int mant_bits = u & 0x7;
    if (exp_bits == 0) return 0.0f;
    float mant = 1.0f + mant_bits * (1.0f / 8.0f);
    int exp = exp_bits - 7;
    float result = ldexpf(mant, exp);
    return sign ? -result : result;
}

__device__ __forceinline__ uint8_t f32_to_fp8_e4m3(float x) {
    if (x == 0.0f) return 0;
    int sign = (x < 0.0f) ? 1 : 0;
    float abs_x = fabsf(x);
    if (abs_x > 448.0f) abs_x = 448.0f;
    int exp_bits;
    frexpf(abs_x, &exp_bits);
    exp_bits = max(0, min(15, exp_bits - 1 + 7));
    float mant = ldexpf(abs_x, -(exp_bits - 7));
    int mant_bits = max(0, min(7, (int)roundf((mant - 1.0f) * 8.0f)));
    return (sign << 7) | (exp_bits << 3) | mant_bits;
}

__device__ __forceinline__ float fp8_e5m2_to_f32(uint8_t u) {
    int sign = (u >> 7) & 1;
    int exp_bits = (u >> 2) & 0x1F;
    int mant_bits = u & 0x3;
    if (exp_bits == 0) return 0.0f;
    float mant = 1.0f + mant_bits * 0.25f;
    int exp = exp_bits - 15;
    float result = ldexpf(mant, exp);
    return sign ? -result : result;
}

__device__ __forceinline__ uint8_t f32_to_fp8_e5m2(float x) {
    if (x == 0.0f) return 0;
    int sign = (x < 0.0f) ? 1 : 0;
    float abs_x = fabsf(x);
    if (abs_x > 57344.0f) abs_x = 57344.0f;
    int exp_bits;
    frexpf(abs_x, &exp_bits);
    exp_bits = max(0, min(31, exp_bits - 1 + 15));
    float mant = ldexpf(abs_x, -(exp_bits - 15));
    int mant_bits = max(0, min(3, (int)roundf((mant - 1.0f) * 4.0f)));
    return (sign << 7) | (exp_bits << 2) | mant_bits;
}

#endif  // __CUDA_ARCH__

// ============================================================================
// Helper Macros for FP8 Kernel Generation
// ============================================================================

// Load FP8 value and convert to F32 for computation
#define LOAD_FP8_E4M3(ptr, idx) fp8_e4m3_to_f32(((const uint8_t*)(ptr))[idx])
#define LOAD_FP8_E5M2(ptr, idx) fp8_e5m2_to_f32(((const uint8_t*)(ptr))[idx])

// Convert F32 result and store as FP8
#define STORE_FP8_E4M3(ptr, idx, val) ((uint8_t*)(ptr))[idx] = f32_to_fp8_e4m3(val)
#define STORE_FP8_E5M2(ptr, idx, val) ((uint8_t*)(ptr))[idx] = f32_to_fp8_e5m2(val)

// ============================================================================
// DType Constants for FP8
// ============================================================================

// E4M3 range constants
#define FP8_E4M3_MAX 448.0f
#define FP8_E4M3_MIN (-448.0f)
#define FP8_E4M3_EPS 0.125f

// E5M2 range constants
#define FP8_E5M2_MAX 57344.0f
#define FP8_E5M2_MIN (-57344.0f)
#define FP8_E5M2_EPS 0.25f

// ============================================================================
// Accumulator Traits for Templated Reduction Kernels
// ============================================================================
// Similar pattern to CPU's Accumulator trait - provides:
// - load: T -> Acc conversion
// - store: Acc -> T conversion
// - identity values (zero, neg_inf, pos_inf)
// - warp reduction primitives

// Warp reduction primitives for f32
__device__ __forceinline__ float warp_sum_f32(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
__device__ __forceinline__ float warp_max_f32(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}
__device__ __forceinline__ float warp_min_f32(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Warp reduction primitives for f64
__device__ __forceinline__ double warp_sum_f64(double val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
__device__ __forceinline__ double warp_max_f64(double val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}
__device__ __forceinline__ double warp_min_f64(double val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmin(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Warp product reduction primitives
__device__ __forceinline__ float warp_prod_f32(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val *= __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
__device__ __forceinline__ double warp_prod_f64(double val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val *= __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Warp boolean reduction primitives (for any/all)
__device__ __forceinline__ int warp_or(int val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val |= __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
__device__ __forceinline__ int warp_and(int val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val &= __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Primary template (not defined - specializations required)
template<typename T, typename Acc> struct AccumTraits;

// ============================================================================
// F32 -> F32 (native)
// ============================================================================
template<> struct AccumTraits<float, float> {
    static __device__ __forceinline__ float load(const float* p, int i) { return p[i]; }
    static __device__ __forceinline__ void store(float* p, int i, float v) { p[i] = v; }
    static __device__ __forceinline__ float zero() { return 0.0f; }
    static __device__ __forceinline__ float one() { return 1.0f; }
    static __device__ __forceinline__ float neg_inf() { return -INFINITY; }
    static __device__ __forceinline__ float pos_inf() { return INFINITY; }
    static __device__ __forceinline__ float warp_sum(float v) { return warp_sum_f32(v); }
    static __device__ __forceinline__ float warp_max(float v) { return warp_max_f32(v); }
    static __device__ __forceinline__ float warp_min(float v) { return warp_min_f32(v); }
    static __device__ __forceinline__ float warp_prod(float v) { return warp_prod_f32(v); }
    static __device__ __forceinline__ float add(float a, float b) { return a + b; }
    static __device__ __forceinline__ float mul(float a, float b) { return a * b; }
    static __device__ __forceinline__ float max(float a, float b) { return fmaxf(a, b); }
    static __device__ __forceinline__ float min(float a, float b) { return fminf(a, b); }
};

// F32 -> F64 (high precision)
template<> struct AccumTraits<float, double> {
    static __device__ __forceinline__ double load(const float* p, int i) { return (double)p[i]; }
    static __device__ __forceinline__ void store(float* p, int i, double v) { p[i] = (float)v; }
    static __device__ __forceinline__ double zero() { return 0.0; }
    static __device__ __forceinline__ double one() { return 1.0; }
    static __device__ __forceinline__ double neg_inf() { return -INFINITY; }
    static __device__ __forceinline__ double pos_inf() { return INFINITY; }
    static __device__ __forceinline__ double warp_sum(double v) { return warp_sum_f64(v); }
    static __device__ __forceinline__ double warp_max(double v) { return warp_max_f64(v); }
    static __device__ __forceinline__ double warp_min(double v) { return warp_min_f64(v); }
    static __device__ __forceinline__ double warp_prod(double v) { return warp_prod_f64(v); }
    static __device__ __forceinline__ double add(double a, double b) { return a + b; }
    static __device__ __forceinline__ double mul(double a, double b) { return a * b; }
    static __device__ __forceinline__ double max(double a, double b) { return fmax(a, b); }
    static __device__ __forceinline__ double min(double a, double b) { return fmin(a, b); }
};

// ============================================================================
// F64 -> F64 (native)
// ============================================================================
template<> struct AccumTraits<double, double> {
    static __device__ __forceinline__ double load(const double* p, int i) { return p[i]; }
    static __device__ __forceinline__ void store(double* p, int i, double v) { p[i] = v; }
    static __device__ __forceinline__ double zero() { return 0.0; }
    static __device__ __forceinline__ double one() { return 1.0; }
    static __device__ __forceinline__ double neg_inf() { return -INFINITY; }
    static __device__ __forceinline__ double pos_inf() { return INFINITY; }
    static __device__ __forceinline__ double warp_sum(double v) { return warp_sum_f64(v); }
    static __device__ __forceinline__ double warp_max(double v) { return warp_max_f64(v); }
    static __device__ __forceinline__ double warp_min(double v) { return warp_min_f64(v); }
    static __device__ __forceinline__ double warp_prod(double v) { return warp_prod_f64(v); }
    static __device__ __forceinline__ double add(double a, double b) { return a + b; }
    static __device__ __forceinline__ double mul(double a, double b) { return a * b; }
    static __device__ __forceinline__ double max(double a, double b) { return fmax(a, b); }
    static __device__ __forceinline__ double min(double a, double b) { return fmin(a, b); }
};

// ============================================================================
// F16 -> F16 (native), F16 -> F32, F16 -> F64
// ============================================================================
template<> struct AccumTraits<__half, __half> {
    static __device__ __forceinline__ __half load(const __half* p, int i) { return p[i]; }
    static __device__ __forceinline__ void store(__half* p, int i, __half v) { p[i] = v; }
    static __device__ __forceinline__ __half zero() { return __float2half(0.0f); }
    static __device__ __forceinline__ __half one() { return __float2half(1.0f); }
    static __device__ __forceinline__ __half neg_inf() { return __float2half(-INFINITY); }
    static __device__ __forceinline__ __half pos_inf() { return __float2half(INFINITY); }
    static __device__ __forceinline__ __half warp_sum(__half v) { return __float2half(warp_sum_f32(__half2float(v))); }
    static __device__ __forceinline__ __half warp_max(__half v) { return __float2half(warp_max_f32(__half2float(v))); }
    static __device__ __forceinline__ __half warp_min(__half v) { return __float2half(warp_min_f32(__half2float(v))); }
    static __device__ __forceinline__ __half warp_prod(__half v) { return __float2half(warp_prod_f32(__half2float(v))); }
    static __device__ __forceinline__ __half add(__half a, __half b) { return __hadd(a, b); }
    static __device__ __forceinline__ __half mul(__half a, __half b) { return __hmul(a, b); }
    static __device__ __forceinline__ __half max(__half a, __half b) { return __hmax(a, b); }
    static __device__ __forceinline__ __half min(__half a, __half b) { return __hmin(a, b); }
};

template<> struct AccumTraits<__half, float> {
    static __device__ __forceinline__ float load(const __half* p, int i) { return __half2float(p[i]); }
    static __device__ __forceinline__ void store(__half* p, int i, float v) { p[i] = __float2half(v); }
    static __device__ __forceinline__ float zero() { return 0.0f; }
    static __device__ __forceinline__ float one() { return 1.0f; }
    static __device__ __forceinline__ float neg_inf() { return -INFINITY; }
    static __device__ __forceinline__ float pos_inf() { return INFINITY; }
    static __device__ __forceinline__ float warp_sum(float v) { return warp_sum_f32(v); }
    static __device__ __forceinline__ float warp_max(float v) { return warp_max_f32(v); }
    static __device__ __forceinline__ float warp_min(float v) { return warp_min_f32(v); }
    static __device__ __forceinline__ float warp_prod(float v) { return warp_prod_f32(v); }
    static __device__ __forceinline__ float add(float a, float b) { return a + b; }
    static __device__ __forceinline__ float mul(float a, float b) { return a * b; }
    static __device__ __forceinline__ float max(float a, float b) { return fmaxf(a, b); }
    static __device__ __forceinline__ float min(float a, float b) { return fminf(a, b); }
};

template<> struct AccumTraits<__half, double> {
    static __device__ __forceinline__ double load(const __half* p, int i) { return (double)__half2float(p[i]); }
    static __device__ __forceinline__ void store(__half* p, int i, double v) { p[i] = __float2half((float)v); }
    static __device__ __forceinline__ double zero() { return 0.0; }
    static __device__ __forceinline__ double one() { return 1.0; }
    static __device__ __forceinline__ double neg_inf() { return -INFINITY; }
    static __device__ __forceinline__ double pos_inf() { return INFINITY; }
    static __device__ __forceinline__ double warp_sum(double v) { return warp_sum_f64(v); }
    static __device__ __forceinline__ double warp_max(double v) { return warp_max_f64(v); }
    static __device__ __forceinline__ double warp_min(double v) { return warp_min_f64(v); }
    static __device__ __forceinline__ double warp_prod(double v) { return warp_prod_f64(v); }
    static __device__ __forceinline__ double add(double a, double b) { return a + b; }
    static __device__ __forceinline__ double mul(double a, double b) { return a * b; }
    static __device__ __forceinline__ double max(double a, double b) { return fmax(a, b); }
    static __device__ __forceinline__ double min(double a, double b) { return fmin(a, b); }
};

// ============================================================================
// BF16 -> BF16 (native), BF16 -> F32, BF16 -> F64
// ============================================================================
template<> struct AccumTraits<__nv_bfloat16, __nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 load(const __nv_bfloat16* p, int i) { return p[i]; }
    static __device__ __forceinline__ void store(__nv_bfloat16* p, int i, __nv_bfloat16 v) { p[i] = v; }
    static __device__ __forceinline__ __nv_bfloat16 zero() { return __float2bfloat16(0.0f); }
    static __device__ __forceinline__ __nv_bfloat16 one() { return __float2bfloat16(1.0f); }
    static __device__ __forceinline__ __nv_bfloat16 neg_inf() { return __float2bfloat16(-INFINITY); }
    static __device__ __forceinline__ __nv_bfloat16 pos_inf() { return __float2bfloat16(INFINITY); }
    static __device__ __forceinline__ __nv_bfloat16 warp_sum(__nv_bfloat16 v) { return __float2bfloat16(warp_sum_f32(__bfloat162float(v))); }
    static __device__ __forceinline__ __nv_bfloat16 warp_max(__nv_bfloat16 v) { return __float2bfloat16(warp_max_f32(__bfloat162float(v))); }
    static __device__ __forceinline__ __nv_bfloat16 warp_min(__nv_bfloat16 v) { return __float2bfloat16(warp_min_f32(__bfloat162float(v))); }
    static __device__ __forceinline__ __nv_bfloat16 warp_prod(__nv_bfloat16 v) { return __float2bfloat16(warp_prod_f32(__bfloat162float(v))); }
    static __device__ __forceinline__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); }
    static __device__ __forceinline__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }
    static __device__ __forceinline__ __nv_bfloat16 max(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmax(a, b); }
    static __device__ __forceinline__ __nv_bfloat16 min(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmin(a, b); }
};

template<> struct AccumTraits<__nv_bfloat16, float> {
    static __device__ __forceinline__ float load(const __nv_bfloat16* p, int i) { return __bfloat162float(p[i]); }
    static __device__ __forceinline__ void store(__nv_bfloat16* p, int i, float v) { p[i] = __float2bfloat16(v); }
    static __device__ __forceinline__ float zero() { return 0.0f; }
    static __device__ __forceinline__ float one() { return 1.0f; }
    static __device__ __forceinline__ float neg_inf() { return -INFINITY; }
    static __device__ __forceinline__ float pos_inf() { return INFINITY; }
    static __device__ __forceinline__ float warp_sum(float v) { return warp_sum_f32(v); }
    static __device__ __forceinline__ float warp_max(float v) { return warp_max_f32(v); }
    static __device__ __forceinline__ float warp_min(float v) { return warp_min_f32(v); }
    static __device__ __forceinline__ float warp_prod(float v) { return warp_prod_f32(v); }
    static __device__ __forceinline__ float add(float a, float b) { return a + b; }
    static __device__ __forceinline__ float mul(float a, float b) { return a * b; }
    static __device__ __forceinline__ float max(float a, float b) { return fmaxf(a, b); }
    static __device__ __forceinline__ float min(float a, float b) { return fminf(a, b); }
};

template<> struct AccumTraits<__nv_bfloat16, double> {
    static __device__ __forceinline__ double load(const __nv_bfloat16* p, int i) { return (double)__bfloat162float(p[i]); }
    static __device__ __forceinline__ void store(__nv_bfloat16* p, int i, double v) { p[i] = __float2bfloat16((float)v); }
    static __device__ __forceinline__ double zero() { return 0.0; }
    static __device__ __forceinline__ double one() { return 1.0; }
    static __device__ __forceinline__ double neg_inf() { return -INFINITY; }
    static __device__ __forceinline__ double pos_inf() { return INFINITY; }
    static __device__ __forceinline__ double warp_sum(double v) { return warp_sum_f64(v); }
    static __device__ __forceinline__ double warp_max(double v) { return warp_max_f64(v); }
    static __device__ __forceinline__ double warp_min(double v) { return warp_min_f64(v); }
    static __device__ __forceinline__ double warp_prod(double v) { return warp_prod_f64(v); }
    static __device__ __forceinline__ double add(double a, double b) { return a + b; }
    static __device__ __forceinline__ double mul(double a, double b) { return a * b; }
    static __device__ __forceinline__ double max(double a, double b) { return fmax(a, b); }
    static __device__ __forceinline__ double min(double a, double b) { return fmin(a, b); }
};

// ============================================================================
// FP8 E4M3 -> F32 (default), FP8 E4M3 -> BF16, FP8 E4M3 -> F64
// ============================================================================
template<> struct AccumTraits<numr_fp8_e4m3, float> {
    static __device__ __forceinline__ float load(const numr_fp8_e4m3* p, int i) { return fp8_e4m3_to_f32(p[i].data); }
    static __device__ __forceinline__ void store(numr_fp8_e4m3* p, int i, float v) { p[i].data = f32_to_fp8_e4m3(v); }
    static __device__ __forceinline__ float zero() { return 0.0f; }
    static __device__ __forceinline__ float one() { return 1.0f; }
    static __device__ __forceinline__ float neg_inf() { return -FP8_E4M3_MAX; }
    static __device__ __forceinline__ float pos_inf() { return FP8_E4M3_MAX; }
    static __device__ __forceinline__ float warp_sum(float v) { return warp_sum_f32(v); }
    static __device__ __forceinline__ float warp_max(float v) { return warp_max_f32(v); }
    static __device__ __forceinline__ float warp_min(float v) { return warp_min_f32(v); }
    static __device__ __forceinline__ float warp_prod(float v) { return warp_prod_f32(v); }
    static __device__ __forceinline__ float add(float a, float b) { return a + b; }
    static __device__ __forceinline__ float mul(float a, float b) { return a * b; }
    static __device__ __forceinline__ float max(float a, float b) { return fmaxf(a, b); }
    static __device__ __forceinline__ float min(float a, float b) { return fminf(a, b); }
};

template<> struct AccumTraits<numr_fp8_e4m3, __nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 load(const numr_fp8_e4m3* p, int i) { return __float2bfloat16(fp8_e4m3_to_f32(p[i].data)); }
    static __device__ __forceinline__ void store(numr_fp8_e4m3* p, int i, __nv_bfloat16 v) { p[i].data = f32_to_fp8_e4m3(__bfloat162float(v)); }
    static __device__ __forceinline__ __nv_bfloat16 zero() { return __float2bfloat16(0.0f); }
    static __device__ __forceinline__ __nv_bfloat16 one() { return __float2bfloat16(1.0f); }
    static __device__ __forceinline__ __nv_bfloat16 neg_inf() { return __float2bfloat16(-FP8_E4M3_MAX); }
    static __device__ __forceinline__ __nv_bfloat16 pos_inf() { return __float2bfloat16(FP8_E4M3_MAX); }
    static __device__ __forceinline__ __nv_bfloat16 warp_sum(__nv_bfloat16 v) { return __float2bfloat16(warp_sum_f32(__bfloat162float(v))); }
    static __device__ __forceinline__ __nv_bfloat16 warp_max(__nv_bfloat16 v) { return __float2bfloat16(warp_max_f32(__bfloat162float(v))); }
    static __device__ __forceinline__ __nv_bfloat16 warp_min(__nv_bfloat16 v) { return __float2bfloat16(warp_min_f32(__bfloat162float(v))); }
    static __device__ __forceinline__ __nv_bfloat16 warp_prod(__nv_bfloat16 v) { return __float2bfloat16(warp_prod_f32(__bfloat162float(v))); }
    static __device__ __forceinline__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); }
    static __device__ __forceinline__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }
    static __device__ __forceinline__ __nv_bfloat16 max(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmax(a, b); }
    static __device__ __forceinline__ __nv_bfloat16 min(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmin(a, b); }
};

template<> struct AccumTraits<numr_fp8_e4m3, double> {
    static __device__ __forceinline__ double load(const numr_fp8_e4m3* p, int i) { return (double)fp8_e4m3_to_f32(p[i].data); }
    static __device__ __forceinline__ void store(numr_fp8_e4m3* p, int i, double v) { p[i].data = f32_to_fp8_e4m3((float)v); }
    static __device__ __forceinline__ double zero() { return 0.0; }
    static __device__ __forceinline__ double one() { return 1.0; }
    static __device__ __forceinline__ double neg_inf() { return -FP8_E4M3_MAX; }
    static __device__ __forceinline__ double pos_inf() { return FP8_E4M3_MAX; }
    static __device__ __forceinline__ double warp_sum(double v) { return warp_sum_f64(v); }
    static __device__ __forceinline__ double warp_max(double v) { return warp_max_f64(v); }
    static __device__ __forceinline__ double warp_min(double v) { return warp_min_f64(v); }
    static __device__ __forceinline__ double warp_prod(double v) { return warp_prod_f64(v); }
    static __device__ __forceinline__ double add(double a, double b) { return a + b; }
    static __device__ __forceinline__ double mul(double a, double b) { return a * b; }
    static __device__ __forceinline__ double max(double a, double b) { return fmax(a, b); }
    static __device__ __forceinline__ double min(double a, double b) { return fmin(a, b); }
};

// ============================================================================
// FP8 E5M2 -> F32 (default), FP8 E5M2 -> BF16, FP8 E5M2 -> F64
// ============================================================================
template<> struct AccumTraits<numr_fp8_e5m2, float> {
    static __device__ __forceinline__ float load(const numr_fp8_e5m2* p, int i) { return fp8_e5m2_to_f32(p[i].data); }
    static __device__ __forceinline__ void store(numr_fp8_e5m2* p, int i, float v) { p[i].data = f32_to_fp8_e5m2(v); }
    static __device__ __forceinline__ float zero() { return 0.0f; }
    static __device__ __forceinline__ float one() { return 1.0f; }
    static __device__ __forceinline__ float neg_inf() { return -FP8_E5M2_MAX; }
    static __device__ __forceinline__ float pos_inf() { return FP8_E5M2_MAX; }
    static __device__ __forceinline__ float warp_sum(float v) { return warp_sum_f32(v); }
    static __device__ __forceinline__ float warp_max(float v) { return warp_max_f32(v); }
    static __device__ __forceinline__ float warp_min(float v) { return warp_min_f32(v); }
    static __device__ __forceinline__ float warp_prod(float v) { return warp_prod_f32(v); }
    static __device__ __forceinline__ float add(float a, float b) { return a + b; }
    static __device__ __forceinline__ float mul(float a, float b) { return a * b; }
    static __device__ __forceinline__ float max(float a, float b) { return fmaxf(a, b); }
    static __device__ __forceinline__ float min(float a, float b) { return fminf(a, b); }
};

template<> struct AccumTraits<numr_fp8_e5m2, __nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 load(const numr_fp8_e5m2* p, int i) { return __float2bfloat16(fp8_e5m2_to_f32(p[i].data)); }
    static __device__ __forceinline__ void store(numr_fp8_e5m2* p, int i, __nv_bfloat16 v) { p[i].data = f32_to_fp8_e5m2(__bfloat162float(v)); }
    static __device__ __forceinline__ __nv_bfloat16 zero() { return __float2bfloat16(0.0f); }
    static __device__ __forceinline__ __nv_bfloat16 one() { return __float2bfloat16(1.0f); }
    static __device__ __forceinline__ __nv_bfloat16 neg_inf() { return __float2bfloat16(-FP8_E5M2_MAX); }
    static __device__ __forceinline__ __nv_bfloat16 pos_inf() { return __float2bfloat16(FP8_E5M2_MAX); }
    static __device__ __forceinline__ __nv_bfloat16 warp_sum(__nv_bfloat16 v) { return __float2bfloat16(warp_sum_f32(__bfloat162float(v))); }
    static __device__ __forceinline__ __nv_bfloat16 warp_max(__nv_bfloat16 v) { return __float2bfloat16(warp_max_f32(__bfloat162float(v))); }
    static __device__ __forceinline__ __nv_bfloat16 warp_min(__nv_bfloat16 v) { return __float2bfloat16(warp_min_f32(__bfloat162float(v))); }
    static __device__ __forceinline__ __nv_bfloat16 warp_prod(__nv_bfloat16 v) { return __float2bfloat16(warp_prod_f32(__bfloat162float(v))); }
    static __device__ __forceinline__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); }
    static __device__ __forceinline__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul(a, b); }
    static __device__ __forceinline__ __nv_bfloat16 max(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmax(a, b); }
    static __device__ __forceinline__ __nv_bfloat16 min(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmin(a, b); }
};

template<> struct AccumTraits<numr_fp8_e5m2, double> {
    static __device__ __forceinline__ double load(const numr_fp8_e5m2* p, int i) { return (double)fp8_e5m2_to_f32(p[i].data); }
    static __device__ __forceinline__ void store(numr_fp8_e5m2* p, int i, double v) { p[i].data = f32_to_fp8_e5m2((float)v); }
    static __device__ __forceinline__ double zero() { return 0.0; }
    static __device__ __forceinline__ double one() { return 1.0; }
    static __device__ __forceinline__ double neg_inf() { return -FP8_E5M2_MAX; }
    static __device__ __forceinline__ double pos_inf() { return FP8_E5M2_MAX; }
    static __device__ __forceinline__ double warp_sum(double v) { return warp_sum_f64(v); }
    static __device__ __forceinline__ double warp_max(double v) { return warp_max_f64(v); }
    static __device__ __forceinline__ double warp_min(double v) { return warp_min_f64(v); }
    static __device__ __forceinline__ double warp_prod(double v) { return warp_prod_f64(v); }
    static __device__ __forceinline__ double add(double a, double b) { return a + b; }
    static __device__ __forceinline__ double mul(double a, double b) { return a * b; }
    static __device__ __forceinline__ double max(double a, double b) { return fmax(a, b); }
    static __device__ __forceinline__ double min(double a, double b) { return fmin(a, b); }
};

#endif // NUMR_DTYPE_TRAITS_CUH
