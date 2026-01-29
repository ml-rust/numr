// Special mathematical functions for CUDA
//
// CUDA provides built-in functions for many special functions:
// - erff/erf, erfcf/erfc, erfinvf/erfinv
// - tgammaf/tgamma (gamma function)
// - lgammaf/lgamma (log-gamma function)
//
// For functions not built into CUDA (digamma, betainc, gammainc),
// we implement the same algorithms as the CPU version.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>

// NaN constants (fallback if not defined)
#ifndef CUDART_NAN_F
#define CUDART_NAN_F __int_as_float(0x7fffffff)
#endif
#ifndef CUDART_NAN
#define CUDART_NAN __longlong_as_double(0x7ff8000000000000ULL)
#endif

// ============================================================================
// Numerical Constants
// ============================================================================

// Maximum iterations for continued fraction / series expansions
#define SPECIAL_MAX_ITER 200

// Convergence thresholds
#define EPSILON_F32 1e-7f
#define EPSILON_F64 1e-15

// Small value to prevent division by zero in continued fractions
#define TINY_F32 1e-30f
#define TINY_F64 1e-30

// Mathematical constants
#define EULER_MASCHERONI_F32 0.5772156649015329f
#define EULER_MASCHERONI_F64 0.5772156649015328606065120900824024310422
#define PI_F32 3.14159265359f
#define PI_F64 3.14159265358979323846

// ============================================================================
// Device Helper Functions
// ============================================================================

// Digamma function using asymptotic expansion
__device__ __forceinline__ float digamma_f32(float x) {
    if (x <= 0.0f && x == floorf(x)) {
        return CUDART_NAN_F;  // poles at non-positive integers
    }

    // Reflection formula for x < 0
    if (x < 0.0f) {
        return digamma_f32(1.0f - x) - PI_F32 / tanf(PI_F32 * x);
    }

    float result = 0.0f;

    // Use recurrence to shift x to larger value where asymptotic expansion works
    while (x < 6.0f) {
        result -= 1.0f / x;
        x += 1.0f;
    }

    // Asymptotic expansion for large x
    float x2 = 1.0f / (x * x);
    result += logf(x) - 0.5f / x
            - x2 * (1.0f/12.0f - x2 * (1.0f/120.0f - x2 * (1.0f/252.0f)));

    return result;
}

__device__ __forceinline__ double digamma_f64(double x) {
    if (x <= 0.0 && x == floor(x)) {
        return CUDART_NAN;  // poles at non-positive integers
    }

    // Reflection formula for x < 0
    if (x < 0.0) {
        return digamma_f64(1.0 - x) - PI_F64 / tan(PI_F64 * x);
    }

    double result = 0.0;

    // Use recurrence to shift x to larger value
    while (x < 6.0) {
        result -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic expansion
    double x2 = 1.0 / (x * x);
    result += log(x) - 0.5 / x
            - x2 * (1.0/12.0 - x2 * (1.0/120.0 - x2 * (1.0/252.0 - x2 * (1.0/240.0))));

    return result;
}

// Lower incomplete gamma using series expansion
__device__ __forceinline__ float gammainc_series_f32(float a, float x) {
    if (x == 0.0f) return 0.0f;

    float term = 1.0f / a;
    float sum = term;

    for (int n = 1; n < SPECIAL_MAX_ITER; n++) {
        term *= x / (a + n);
        sum += term;
        if (fabsf(term) < fabsf(sum) * EPSILON_F32) break;
    }

    return expf(-x + a * logf(x) - lgammaf(a)) * sum;
}

__device__ __forceinline__ double gammainc_series_f64(double a, double x) {
    if (x == 0.0) return 0.0;

    double term = 1.0 / a;
    double sum = term;

    for (int n = 1; n < SPECIAL_MAX_ITER; n++) {
        term *= x / (a + n);
        sum += term;
        if (fabs(term) < fabs(sum) * EPSILON_F64) break;
    }

    return exp(-x + a * log(x) - lgamma(a)) * sum;
}

// Upper incomplete gamma using continued fraction
__device__ __forceinline__ float gammaincc_cf_f32(float a, float x) {
    float f = 1e30f;
    float c = 1e30f;
    float d = 0.0f;

    for (int n = 1; n < SPECIAL_MAX_ITER; n++) {
        float an = (n % 2 == 1) ? ((n + 1) / 2) : (a - n / 2);
        float bn = x + n - a;

        d = bn + an * d;
        if (fabsf(d) < TINY_F32) d = TINY_F32;
        c = bn + an / c;
        if (fabsf(c) < TINY_F32) c = TINY_F32;

        d = 1.0f / d;
        float delta = c * d;
        f *= delta;

        if (fabsf(delta - 1.0f) < EPSILON_F32) break;
    }

    return expf(-x + a * logf(x) - lgammaf(a)) / f;
}

__device__ __forceinline__ double gammaincc_cf_f64(double a, double x) {
    double f = 1e30;
    double c = 1e30;
    double d = 0.0;

    for (int n = 1; n < SPECIAL_MAX_ITER; n++) {
        double an = (n % 2 == 1) ? (double)((n + 1) / 2) : (a - n / 2);
        double bn = x + n - a;

        d = bn + an * d;
        if (fabs(d) < TINY_F64) d = TINY_F64;
        c = bn + an / c;
        if (fabs(c) < TINY_F64) c = TINY_F64;

        d = 1.0 / d;
        double delta = c * d;
        f *= delta;

        if (fabs(delta - 1.0) < EPSILON_F64) break;
    }

    return exp(-x + a * log(x) - lgamma(a)) / f;
}

// Regularized incomplete beta function using continued fraction
__device__ __forceinline__ float betainc_cf_f32(float a, float b, float x) {
    if (x == 0.0f) return 0.0f;
    if (x == 1.0f) return 1.0f;

    // Use symmetry for better convergence
    if (x > (a + 1.0f) / (a + b + 2.0f)) {
        return 1.0f - betainc_cf_f32(b, a, 1.0f - x);
    }

    float qab = a + b;
    float qap = a + 1.0f;
    float qam = a - 1.0f;

    float c = 1.0f;
    float d = 1.0f - qab * x / qap;
    if (fabsf(d) < TINY_F32) d = TINY_F32;
    d = 1.0f / d;
    float h = d;

    for (int m = 1; m < SPECIAL_MAX_ITER; m++) {
        int m2 = 2 * m;

        float aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0f + aa * d;
        if (fabsf(d) < TINY_F32) d = TINY_F32;
        c = 1.0f + aa / c;
        if (fabsf(c) < TINY_F32) c = TINY_F32;
        d = 1.0f / d;
        h *= d * c;

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0f + aa * d;
        if (fabsf(d) < TINY_F32) d = TINY_F32;
        c = 1.0f + aa / c;
        if (fabsf(c) < TINY_F32) c = TINY_F32;
        d = 1.0f / d;
        float delta = d * c;
        h *= delta;

        if (fabsf(delta - 1.0f) < EPSILON_F32) break;
    }

    float lnbeta = lgammaf(a) + lgammaf(b) - lgammaf(a + b);
    return expf(a * logf(x) + b * logf(1.0f - x) - lnbeta) * h / a;
}

__device__ __forceinline__ double betainc_cf_f64(double a, double b, double x) {
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;

    // Use symmetry for better convergence
    if (x > (a + 1.0) / (a + b + 2.0)) {
        return 1.0 - betainc_cf_f64(b, a, 1.0 - x);
    }

    double qab = a + b;
    double qap = a + 1.0;
    double qam = a - 1.0;

    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (fabs(d) < TINY_F64) d = TINY_F64;
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m < SPECIAL_MAX_ITER; m++) {
        int m2 = 2 * m;

        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (fabs(d) < TINY_F64) d = TINY_F64;
        c = 1.0 + aa / c;
        if (fabs(c) < TINY_F64) c = TINY_F64;
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (fabs(d) < TINY_F64) d = TINY_F64;
        c = 1.0 + aa / c;
        if (fabs(c) < TINY_F64) c = TINY_F64;
        d = 1.0 / d;
        double delta = d * c;
        h *= delta;

        if (fabs(delta - 1.0) < EPSILON_F64) break;
    }

    double lnbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
    return exp(a * log(x) + b * log(1.0 - x) - lnbeta) * h / a;
}

// ============================================================================
// Kernel Definitions
// ============================================================================

extern "C" {

// Error function
__global__ void erf_f32(const float* x, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = erff(x[idx]);
    }
}

__global__ void erf_f64(const double* x, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = erf(x[idx]);
    }
}

// Complementary error function
__global__ void erfc_f32(const float* x, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = erfcf(x[idx]);
    }
}

__global__ void erfc_f64(const double* x, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = erfc(x[idx]);
    }
}

// Inverse error function
__global__ void erfinv_f32(const float* x, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = erfinvf(x[idx]);
    }
}

__global__ void erfinv_f64(const double* x, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = erfinv(x[idx]);
    }
}

// Gamma function
__global__ void gamma_f32(const float* x, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tgammaf(x[idx]);
    }
}

__global__ void gamma_f64(const double* x, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tgamma(x[idx]);
    }
}

// Log-gamma function
__global__ void lgamma_f32(const float* x, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = lgammaf(x[idx]);
    }
}

__global__ void lgamma_f64(const double* x, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = lgamma(x[idx]);
    }
}

// Digamma function
__global__ void digamma_f32(const float* x, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = digamma_f32(x[idx]);
    }
}

__global__ void digamma_f64(const double* x, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = digamma_f64(x[idx]);
    }
}

// Beta function B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)
__global__ void beta_f32(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = expf(lgammaf(a[idx]) + lgammaf(b[idx]) - lgammaf(a[idx] + b[idx]));
    }
}

__global__ void beta_f64(const double* a, const double* b, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = exp(lgamma(a[idx]) + lgamma(b[idx]) - lgamma(a[idx] + b[idx]));
    }
}

// Regularized incomplete beta function
__global__ void betainc_f32(const float* a, const float* b, const float* x, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = betainc_cf_f32(a[idx], b[idx], x[idx]);
    }
}

__global__ void betainc_f64(const double* a, const double* b, const double* x, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = betainc_cf_f64(a[idx], b[idx], x[idx]);
    }
}

// Lower regularized incomplete gamma P(a,x)
__global__ void gammainc_f32(const float* a, const float* x, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float aa = a[idx];
        float xx = x[idx];

        if (xx < 0.0f || aa <= 0.0f) {
            out[idx] = CUDART_NAN_F;
        } else if (xx == 0.0f) {
            out[idx] = 0.0f;
        } else if (xx < aa + 1.0f) {
            out[idx] = gammainc_series_f32(aa, xx);
        } else {
            out[idx] = 1.0f - gammaincc_cf_f32(aa, xx);
        }
    }
}

__global__ void gammainc_f64(const double* a, const double* x, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double aa = a[idx];
        double xx = x[idx];

        if (xx < 0.0 || aa <= 0.0) {
            out[idx] = CUDART_NAN;
        } else if (xx == 0.0) {
            out[idx] = 0.0;
        } else if (xx < aa + 1.0) {
            out[idx] = gammainc_series_f64(aa, xx);
        } else {
            out[idx] = 1.0 - gammaincc_cf_f64(aa, xx);
        }
    }
}

// Upper regularized incomplete gamma Q(a,x) = 1 - P(a,x)
__global__ void gammaincc_f32(const float* a, const float* x, float* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float aa = a[idx];
        float xx = x[idx];

        if (xx < 0.0f || aa <= 0.0f) {
            out[idx] = CUDART_NAN_F;
        } else if (xx == 0.0f) {
            out[idx] = 1.0f;
        } else if (xx < aa + 1.0f) {
            out[idx] = 1.0f - gammainc_series_f32(aa, xx);
        } else {
            out[idx] = gammaincc_cf_f32(aa, xx);
        }
    }
}

__global__ void gammaincc_f64(const double* a, const double* x, double* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double aa = a[idx];
        double xx = x[idx];

        if (xx < 0.0 || aa <= 0.0) {
            out[idx] = CUDART_NAN;
        } else if (xx == 0.0) {
            out[idx] = 1.0;
        } else if (xx < aa + 1.0) {
            out[idx] = 1.0 - gammainc_series_f64(aa, xx);
        } else {
            out[idx] = gammaincc_cf_f64(aa, xx);
        }
    }
}

} // extern "C"
