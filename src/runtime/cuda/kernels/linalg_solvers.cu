// Triangular solvers CUDA kernels
// Includes: forward_sub, backward_sub, apply_lu_permutation, det_from_lu
//
// Uses C++ templates to eliminate f32/f64 duplication.

#include "dtype_traits.cuh"

// ============================================================================
// Forward substitution: Solve Lx = b where L is lower triangular
// ============================================================================

template<typename T>
__device__ void forward_sub_impl(const T* __restrict__ l,
                                 const T* __restrict__ b,
                                 T* __restrict__ x,
                                 unsigned int n, int unit_diag) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (unsigned int i = 0; i < n; i++) {
        T sum = b[i];
        for (unsigned int j = 0; j < i; j++) {
            sum -= l[i * n + j] * x[j];
        }
        if (unit_diag) {
            x[i] = sum;
        } else {
            x[i] = sum / l[i * n + i];
        }
    }
}

// ============================================================================
// Backward substitution: Solve Ux = b where U is upper triangular
// ============================================================================

template<typename T>
__device__ void backward_sub_impl(const T* __restrict__ u,
                                  const T* __restrict__ b,
                                  T* __restrict__ x,
                                  unsigned int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int i = (int)n - 1; i >= 0; i--) {
        T sum = b[i];
        for (unsigned int j = i + 1; j < n; j++) {
            sum -= u[i * n + j] * x[j];
        }
        x[i] = sum / u[i * n + i];
    }
}

// ============================================================================
// Determinant from LU
// ============================================================================

template<typename T>
__device__ void det_from_lu_impl(const T* __restrict__ lu,
                                 T* __restrict__ det,
                                 unsigned int n, int num_swaps) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    T result = dtype_traits<T>::one();
    for (unsigned int i = 0; i < n; i++) {
        result *= lu[i * n + i];
    }
    *det = (num_swaps % 2 == 0) ? result : -result;
}

// ============================================================================
// Apply LU Permutation
// ============================================================================

template<typename T>
__device__ void apply_lu_permutation_impl(const T* __restrict__ in,
                                          T* __restrict__ out,
                                          const long long* __restrict__ pivots,
                                          unsigned int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (unsigned int i = 0; i < n; i++) {
        out[i] = in[i];
    }

    for (unsigned int i = 0; i < n; i++) {
        unsigned int pivot = (unsigned int)pivots[i];
        if (pivot != i) {
            T tmp = out[i];
            out[i] = out[pivot];
            out[pivot] = tmp;
        }
    }
}

// ============================================================================
// Extern "C" wrappers for PTX export
// ============================================================================

extern "C" {

__global__ void forward_sub_f32(const float* l, const float* b, float* x,
                                unsigned int n, int unit_diag) {
    forward_sub_impl<float>(l, b, x, n, unit_diag);
}

__global__ void forward_sub_f64(const double* l, const double* b, double* x,
                                unsigned int n, int unit_diag) {
    forward_sub_impl<double>(l, b, x, n, unit_diag);
}

__global__ void backward_sub_f32(const float* u, const float* b, float* x,
                                 unsigned int n) {
    backward_sub_impl<float>(u, b, x, n);
}

__global__ void backward_sub_f64(const double* u, const double* b, double* x,
                                 unsigned int n) {
    backward_sub_impl<double>(u, b, x, n);
}

__global__ void det_from_lu_f32(const float* lu, float* det,
                                unsigned int n, int num_swaps) {
    det_from_lu_impl<float>(lu, det, n, num_swaps);
}

__global__ void det_from_lu_f64(const double* lu, double* det,
                                unsigned int n, int num_swaps) {
    det_from_lu_impl<double>(lu, det, n, num_swaps);
}

__global__ void apply_lu_permutation_f32(const float* in, float* out,
                                         const long long* pivots, unsigned int n) {
    apply_lu_permutation_impl<float>(in, out, pivots, n);
}

__global__ void apply_lu_permutation_f64(const double* in, double* out,
                                         const long long* pivots, unsigned int n) {
    apply_lu_permutation_impl<double>(in, out, pivots, n);
}

// ============================================================================
// F16 (__half) Wrappers
// ============================================================================

__global__ void forward_sub_f16(const __half* l, const __half* b, __half* x,
                                unsigned int n, int unit_diag) {
    forward_sub_impl<__half>(l, b, x, n, unit_diag);
}

__global__ void backward_sub_f16(const __half* u, const __half* b, __half* x,
                                 unsigned int n) {
    backward_sub_impl<__half>(u, b, x, n);
}

__global__ void det_from_lu_f16(const __half* lu, __half* det,
                                unsigned int n, int num_swaps) {
    det_from_lu_impl<__half>(lu, det, n, num_swaps);
}

__global__ void apply_lu_permutation_f16(const __half* in, __half* out,
                                         const long long* pivots, unsigned int n) {
    apply_lu_permutation_impl<__half>(in, out, pivots, n);
}

// ============================================================================
// BF16 (__nv_bfloat16) Wrappers
// ============================================================================

__global__ void forward_sub_bf16(const __nv_bfloat16* l, const __nv_bfloat16* b, __nv_bfloat16* x,
                                 unsigned int n, int unit_diag) {
    forward_sub_impl<__nv_bfloat16>(l, b, x, n, unit_diag);
}

__global__ void backward_sub_bf16(const __nv_bfloat16* u, const __nv_bfloat16* b, __nv_bfloat16* x,
                                  unsigned int n) {
    backward_sub_impl<__nv_bfloat16>(u, b, x, n);
}

__global__ void det_from_lu_bf16(const __nv_bfloat16* lu, __nv_bfloat16* det,
                                 unsigned int n, int num_swaps) {
    det_from_lu_impl<__nv_bfloat16>(lu, det, n, num_swaps);
}

__global__ void apply_lu_permutation_bf16(const __nv_bfloat16* in, __nv_bfloat16* out,
                                          const long long* pivots, unsigned int n) {
    apply_lu_permutation_impl<__nv_bfloat16>(in, out, pivots, n);
}

} // extern "C"
