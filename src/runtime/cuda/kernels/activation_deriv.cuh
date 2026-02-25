// Shared activation derivative and forward helpers for CUDA backward kernels.
//
// Used by: gemm_epilogue_bwd.cu, fused_activation_mul_bwd.cu
//
// Activation type encoding (for switch-based dispatch):
//   0 = None (identity), 1 = ReLU, 2 = GELU, 3 = SiLU, 4 = Sigmoid, 5 = Tanh
//
// GELU tanh-approximation clamping ranges:
//   f32: ±15.0 — tanhf(15) saturates to ±1.0f in float32 precision, and
//                expf(30) < FLT_MAX so no overflow in tanh's internal exp(2x).
//   f64: ±20.0 — tanh(20) saturates to ±1.0 in float64 precision, and
//                exp(40) < DBL_MAX so no overflow. Tighter than ±15 would
//                lose valid precision for f64.

#pragma once

// ============================================================================
// Per-activation derivative helpers (scalar, __forceinline__)
// ============================================================================

__device__ __forceinline__ float relu_deriv_f32(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

__device__ __forceinline__ float sigmoid_fwd_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid_deriv_f32(float x) {
    float sig = sigmoid_fwd_f32(x);
    return sig * (1.0f - sig);
}

__device__ __forceinline__ float tanh_deriv_f32(float x) {
    float t = tanhf(x);
    return 1.0f - t * t;
}

__device__ __forceinline__ float silu_deriv_f32(float x) {
    float sig = sigmoid_fwd_f32(x);
    return sig + x * sig * (1.0f - sig);
}

__device__ __forceinline__ float gelu_deriv_f32(float x) {
    const float c = 0.7978845608f;  // sqrt(2/pi)
    const float k = 0.044715f;
    float inner = c * (x + k * x * x * x);
    // Clamp to prevent exp overflow in tanh (see header comment for range rationale)
    inner = fminf(fmaxf(inner, -15.0f), 15.0f);
    float t = tanhf(inner);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * c * (1.0f + 3.0f * k * x * x);
}

// Switch-based dispatcher for f32
__device__ __forceinline__ float activation_deriv_f32(float x, unsigned int act_type) {
    switch (act_type) {
        case 0: return 1.0f;
        case 1: return relu_deriv_f32(x);
        case 2: return gelu_deriv_f32(x);
        case 3: return silu_deriv_f32(x);
        case 4: return sigmoid_deriv_f32(x);
        case 5: return tanh_deriv_f32(x);
        default: return 1.0f;
    }
}

// ============================================================================
// F64 variants
// ============================================================================

__device__ __forceinline__ double relu_deriv_f64(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

__device__ __forceinline__ double sigmoid_fwd_f64(double x) {
    return 1.0 / (1.0 + exp(-x));
}

__device__ __forceinline__ double sigmoid_deriv_f64(double x) {
    double sig = sigmoid_fwd_f64(x);
    return sig * (1.0 - sig);
}

__device__ __forceinline__ double tanh_deriv_f64(double x) {
    double t = tanh(x);
    return 1.0 - t * t;
}

__device__ __forceinline__ double silu_deriv_f64(double x) {
    double sig = sigmoid_fwd_f64(x);
    return sig + x * sig * (1.0 - sig);
}

__device__ __forceinline__ double gelu_deriv_f64(double x) {
    const double c = 0.7978845608028654;  // sqrt(2/pi)
    const double k = 0.044715;
    double inner = c * (x + k * x * x * x);
    // Clamp to prevent exp overflow in tanh (see header comment for range rationale)
    inner = fmin(fmax(inner, -20.0), 20.0);
    double t = tanh(inner);
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3.0 * k * x * x);
}

// Switch-based dispatcher for f64
__device__ __forceinline__ double activation_deriv_f64(double x, unsigned int act_type) {
    switch (act_type) {
        case 0: return 1.0;
        case 1: return relu_deriv_f64(x);
        case 2: return gelu_deriv_f64(x);
        case 3: return silu_deriv_f64(x);
        case 4: return sigmoid_deriv_f64(x);
        case 5: return tanh_deriv_f64(x);
        default: return 1.0;
    }
}

// ============================================================================
// Forward value helpers (used by fused activation-mul backward)
// ============================================================================

__device__ __forceinline__ float relu_fwd_f32(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float silu_fwd_f32(float x) {
    return x * sigmoid_fwd_f32(x);
}

__device__ __forceinline__ float gelu_fwd_f32(float x) {
    const float c = 0.7978845608f;
    const float k = 0.044715f;
    float inner = c * (x + k * x * x * x);
    inner = fminf(fmaxf(inner, -15.0f), 15.0f);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ double relu_fwd_f64(double x) {
    return fmax(0.0, x);
}

__device__ __forceinline__ double sigmoid_fwd_f64_val(double x) {
    return sigmoid_fwd_f64(x);
}

__device__ __forceinline__ double silu_fwd_f64(double x) {
    return x * sigmoid_fwd_f64(x);
}

__device__ __forceinline__ double gelu_fwd_f64(double x) {
    const double c = 0.7978845608028654;
    const double k = 0.044715;
    double inner = c * (x + k * x * x * x);
    inner = fmin(fmax(inner, -20.0), 20.0);
    return 0.5 * x * (1.0 + tanh(inner));
}
