// Schur decomposition CUDA kernels
// For general (non-symmetric) matrices: A = Z @ T @ Z^T
// T is quasi-upper-triangular (real Schur form), Z is orthogonal
//
// Uses C++ templates to eliminate f32/f64 duplication.

#include "dtype_traits.cuh"

// ============================================================================
// Schur Decomposition - Hessenberg reduction + QR iteration
// ============================================================================

template<typename T>
__device__ void schur_decompose_impl(
    T* __restrict__ t,           // [n, n] input matrix, becomes quasi-triangular T
    T* __restrict__ z,           // [n, n] output orthogonal matrix Z
    unsigned int n,
    int* __restrict__ converged_flag
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const T eps = dtype_traits<T>::eps();
    const int max_sweeps = 30 * (int)n;

    // Initialize Z as identity
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            z[i * n + j] = (i == j) ? dtype_traits<T>::one() : dtype_traits<T>::zero();
        }
    }

    // Step 1: Hessenberg reduction using Householder reflections
    for (unsigned int k = 0; k < n - 2; k++) {
        T norm_sq = dtype_traits<T>::zero();
        for (unsigned int i = k + 1; i < n; i++) {
            T val = t[i * n + k];
            norm_sq += val * val;
        }

        if (norm_sq < eps) continue;

        T norm = dtype_traits<T>::sqrt(norm_sq);
        T x0 = t[(k + 1) * n + k];
        T alpha = (x0 >= dtype_traits<T>::zero()) ? -norm : norm;

        T v0 = x0 - alpha;
        T v_norm_sq = v0 * v0;
        for (unsigned int i = k + 2; i < n; i++) {
            T val = t[i * n + k];
            v_norm_sq += val * val;
        }

        if (v_norm_sq < eps) continue;

        T v_norm = dtype_traits<T>::sqrt(v_norm_sq);
        T v[256];
        v[0] = v0 / v_norm;
        for (unsigned int i = k + 2; i < n; i++) {
            v[i - k - 1] = t[i * n + k] / v_norm;
        }
        unsigned int v_len = n - k - 1;

        // Left multiply: T = (I - 2vv^T) @ T
        for (unsigned int j = 0; j < n; j++) {
            T dot = dtype_traits<T>::zero();
            for (unsigned int i = 0; i < v_len; i++) {
                dot += v[i] * t[(k + 1 + i) * n + j];
            }
            for (unsigned int i = 0; i < v_len; i++) {
                t[(k + 1 + i) * n + j] -= dtype_traits<T>::two() * v[i] * dot;
            }
        }

        // Right multiply: T = T @ (I - 2vv^T)
        for (unsigned int i = 0; i < n; i++) {
            T dot = dtype_traits<T>::zero();
            for (unsigned int j = 0; j < v_len; j++) {
                dot += t[i * n + (k + 1 + j)] * v[j];
            }
            for (unsigned int j = 0; j < v_len; j++) {
                t[i * n + (k + 1 + j)] -= dtype_traits<T>::two() * dot * v[j];
            }
        }

        // Accumulate Z: Z = Z @ (I - 2vv^T)
        for (unsigned int i = 0; i < n; i++) {
            T dot = dtype_traits<T>::zero();
            for (unsigned int j = 0; j < v_len; j++) {
                dot += z[i * n + (k + 1 + j)] * v[j];
            }
            for (unsigned int j = 0; j < v_len; j++) {
                z[i * n + (k + 1 + j)] -= dtype_traits<T>::two() * dot * v[j];
            }
        }
    }

    // Step 2: QR iteration with Wilkinson shift
    for (int iter = 0; iter < max_sweeps; iter++) {
        int converged = 1;
        for (unsigned int i = 0; i < n - 1; i++) {
            T h_ii = dtype_traits<T>::abs(t[i * n + i]);
            T h_ip1 = dtype_traits<T>::abs(t[(i + 1) * n + (i + 1)]);
            T threshold = eps * dtype_traits<T>::max(h_ii + h_ip1, dtype_traits<T>::one());
            if (dtype_traits<T>::abs(t[(i + 1) * n + i]) > threshold) {
                converged = 0;
                break;
            }
        }

        if (converged) {
            *converged_flag = 1;
            break;
        }

        T a = t[(n - 2) * n + (n - 2)];
        T b = t[(n - 2) * n + (n - 1)];
        T c = t[(n - 1) * n + (n - 2)];
        T d = t[(n - 1) * n + (n - 1)];

        T trace = a + d;
        T det = a * d - b * c;
        T disc = trace * trace - (T)4 * det;

        T shift;
        if (disc >= dtype_traits<T>::zero()) {
            T sqrt_disc = dtype_traits<T>::sqrt(disc);
            T lambda1 = (trace + sqrt_disc) / dtype_traits<T>::two();
            T lambda2 = (trace - sqrt_disc) / dtype_traits<T>::two();
            shift = (dtype_traits<T>::abs(lambda1 - d) < dtype_traits<T>::abs(lambda2 - d)) ? lambda1 : lambda2;
        } else {
            shift = trace / dtype_traits<T>::two();
        }

        for (unsigned int i = 0; i < n; i++) {
            t[i * n + i] -= shift;
        }

        // QR step using Givens rotations
        for (unsigned int i = 0; i < n - 1; i++) {
            T a_val = t[i * n + i];
            T b_val = t[(i + 1) * n + i];

            if (dtype_traits<T>::abs(b_val) < eps) continue;

            T r = dtype_traits<T>::sqrt(a_val * a_val + b_val * b_val);
            T cs = a_val / r;
            T sn = -b_val / r;

            for (unsigned int j = 0; j < n; j++) {
                T t1 = t[i * n + j];
                T t2 = t[(i + 1) * n + j];
                t[i * n + j] = cs * t1 - sn * t2;
                t[(i + 1) * n + j] = sn * t1 + cs * t2;
            }

            for (unsigned int kk = 0; kk < n; kk++) {
                T t1 = t[kk * n + i];
                T t2 = t[kk * n + (i + 1)];
                t[kk * n + i] = cs * t1 - sn * t2;
                t[kk * n + (i + 1)] = sn * t1 + cs * t2;
            }

            for (unsigned int kk = 0; kk < n; kk++) {
                T z1 = z[kk * n + i];
                T z2 = z[kk * n + (i + 1)];
                z[kk * n + i] = cs * z1 - sn * z2;
                z[kk * n + (i + 1)] = sn * z1 + cs * z2;
            }
        }

        for (unsigned int i = 0; i < n; i++) {
            t[i * n + i] += shift;
        }
    }

    // Clean up small subdiagonals
    for (unsigned int i = 0; i < n - 1; i++) {
        T h_ii = dtype_traits<T>::abs(t[i * n + i]);
        T h_ip1 = dtype_traits<T>::abs(t[(i + 1) * n + (i + 1)]);
        T threshold = eps * dtype_traits<T>::max(h_ii + h_ip1, dtype_traits<T>::one());
        if (dtype_traits<T>::abs(t[(i + 1) * n + i]) <= threshold) {
            t[(i + 1) * n + i] = dtype_traits<T>::zero();
        }
    }

    // Clear strictly lower triangular (except first subdiagonal for 2x2 blocks)
    for (unsigned int i = 2; i < n; i++) {
        for (unsigned int j = 0; j < i - 1; j++) {
            t[i * n + j] = dtype_traits<T>::zero();
        }
    }
}

// ============================================================================
// Extern "C" wrappers for PTX export
// ============================================================================

extern "C" {

__global__ void schur_decompose_f32(float* t, float* z, unsigned int n,
                                    int* converged_flag) {
    schur_decompose_impl<float>(t, z, n, converged_flag);
}

__global__ void schur_decompose_f64(double* t, double* z, unsigned int n,
                                    int* converged_flag) {
    schur_decompose_impl<double>(t, z, n, converged_flag);
}

// ============================================================================
// F16 (__half) Wrappers
// ============================================================================

__global__ void schur_decompose_f16(__half* t, __half* z, unsigned int n,
                                    int* converged_flag) {
    schur_decompose_impl<__half>(t, z, n, converged_flag);
}

// ============================================================================
// BF16 (__nv_bfloat16) Wrappers
// ============================================================================

__global__ void schur_decompose_bf16(__nv_bfloat16* t, __nv_bfloat16* z, unsigned int n,
                                     int* converged_flag) {
    schur_decompose_impl<__nv_bfloat16>(t, z, n, converged_flag);
}

} // extern "C"
