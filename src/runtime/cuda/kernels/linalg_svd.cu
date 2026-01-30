// SVD CUDA kernels using One-Sided Jacobi algorithm
// Single-thread implementation for backend parity with CPU
//
// Uses C++ templates to eliminate f32/f64 duplication.

#include "dtype_traits.cuh"

// ============================================================================
// SVD Decomposition using One-Sided Jacobi algorithm
// ============================================================================

template<typename T>
__device__ void svd_jacobi_impl(T* __restrict__ b,      // [work_m, work_n] becomes U columns
                                T* __restrict__ v,      // [work_n, work_n] accumulates V
                                T* __restrict__ s,      // [work_n] singular values
                                unsigned int work_m, unsigned int work_n,
                                int* __restrict__ converged_flag) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const T eps = dtype_traits<T>::eps();
    const T tol = (T)work_n * eps;
    const int max_sweeps = 30;

    // Initialize V as identity
    for (unsigned int i = 0; i < work_n; i++) {
        for (unsigned int j = 0; j < work_n; j++) {
            v[i * work_n + j] = (i == j) ? dtype_traits<T>::one() : dtype_traits<T>::zero();
        }
    }

    // One-Sided Jacobi iterations
    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        T off_diag_sum = dtype_traits<T>::zero();

        // Process all column pairs (p, q) where p < q
        for (unsigned int p = 0; p < work_n; p++) {
            for (unsigned int q = p + 1; q < work_n; q++) {
                // Compute Gram matrix elements
                T a_pp = dtype_traits<T>::zero();
                T a_qq = dtype_traits<T>::zero();
                T a_pq = dtype_traits<T>::zero();
                for (unsigned int i = 0; i < work_m; i++) {
                    T bp = b[i * work_n + p];
                    T bq = b[i * work_n + q];
                    a_pp += bp * bp;
                    a_qq += bq * bq;
                    a_pq += bp * bq;
                }

                off_diag_sum += a_pq * a_pq;

                // Skip if off-diagonal is essentially zero
                if (dtype_traits<T>::abs(a_pq) < tol * dtype_traits<T>::sqrt(a_pp * a_qq)) {
                    continue;
                }

                // Compute Jacobi rotation parameters
                T tau_num = a_qq - a_pp;
                T tau_den = dtype_traits<T>::two() * a_pq;

                T c, s_val;
                if (dtype_traits<T>::abs(tau_den) < dtype_traits<T>::tiny_eps()) {
                    c = dtype_traits<T>::one();
                    s_val = dtype_traits<T>::zero();
                } else {
                    T tau = tau_num / tau_den;
                    T t;
                    if (tau >= dtype_traits<T>::zero()) {
                        t = dtype_traits<T>::one() / (tau + dtype_traits<T>::sqrt(dtype_traits<T>::one() + tau * tau));
                    } else {
                        t = -dtype_traits<T>::one() / (-tau + dtype_traits<T>::sqrt(dtype_traits<T>::one() + tau * tau));
                    }
                    c = dtype_traits<T>::one() / dtype_traits<T>::sqrt(dtype_traits<T>::one() + t * t);
                    s_val = t * c;
                }

                // Apply rotation to B columns
                for (unsigned int i = 0; i < work_m; i++) {
                    T bp = b[i * work_n + p];
                    T bq = b[i * work_n + q];
                    b[i * work_n + p] = c * bp - s_val * bq;
                    b[i * work_n + q] = s_val * bp + c * bq;
                }

                // Apply rotation to V columns
                for (unsigned int i = 0; i < work_n; i++) {
                    T vp = v[i * work_n + p];
                    T vq = v[i * work_n + q];
                    v[i * work_n + p] = c * vp - s_val * vq;
                    v[i * work_n + q] = s_val * vp + c * vq;
                }
            }
        }

        // Check convergence
        if (dtype_traits<T>::sqrt(off_diag_sum) < tol) {
            *converged_flag = 1;
            break;
        }
    }

    // Extract singular values (column norms of B)
    for (unsigned int j = 0; j < work_n; j++) {
        T norm_sq = dtype_traits<T>::zero();
        for (unsigned int i = 0; i < work_m; i++) {
            T val = b[i * work_n + j];
            norm_sq += val * val;
        }
        s[j] = dtype_traits<T>::sqrt(norm_sq);

        // Normalize B column to get U column
        T norm = s[j];
        if (norm > eps) {
            for (unsigned int i = 0; i < work_m; i++) {
                b[i * work_n + j] /= norm;
            }
        }
    }
}

// ============================================================================
// Extern "C" wrappers for PTX export
// ============================================================================

extern "C" {

__global__ void svd_jacobi_f32(float* b, float* v, float* s,
                               unsigned int work_m, unsigned int work_n,
                               int* converged_flag) {
    svd_jacobi_impl<float>(b, v, s, work_m, work_n, converged_flag);
}

__global__ void svd_jacobi_f64(double* b, double* v, double* s,
                               unsigned int work_m, unsigned int work_n,
                               int* converged_flag) {
    svd_jacobi_impl<double>(b, v, s, work_m, work_n, converged_flag);
}

// ============================================================================
// F16 (__half) Wrappers
// ============================================================================

__global__ void svd_jacobi_f16(__half* b, __half* v, __half* s,
                               unsigned int work_m, unsigned int work_n,
                               int* converged_flag) {
    svd_jacobi_impl<__half>(b, v, s, work_m, work_n, converged_flag);
}

// ============================================================================
// BF16 (__nv_bfloat16) Wrappers
// ============================================================================

__global__ void svd_jacobi_bf16(__nv_bfloat16* b, __nv_bfloat16* v, __nv_bfloat16* s,
                                unsigned int work_m, unsigned int work_n,
                                int* converged_flag) {
    svd_jacobi_impl<__nv_bfloat16>(b, v, s, work_m, work_n, converged_flag);
}

} // extern "C"
