// Eigenvalue decomposition CUDA kernels
// Includes: Symmetric eigenvalue (Jacobi)
//
// Uses C++ templates to eliminate f32/f64 duplication.

#include "dtype_traits.cuh"

// ============================================================================
// Eigendecomposition for Symmetric Matrices using Jacobi Algorithm
// ============================================================================

template<typename T>
__device__ void eig_jacobi_symmetric_impl(
    T* __restrict__ a,           // [n, n] input matrix, becomes diagonal
    T* __restrict__ eigenvectors,// [n, n] output eigenvectors
    T* __restrict__ eigenvalues, // [n] output eigenvalues
    unsigned int n,
    int* __restrict__ converged_flag
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const T eps = dtype_traits<T>::eps();
    const T tol = (T)n * eps;
    const int max_sweeps = 30;

    // Initialize eigenvector matrix as identity
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            eigenvectors[i * n + j] = (i == j) ? dtype_traits<T>::one() : dtype_traits<T>::zero();
        }
    }

    // Symmetrize input (use lower triangle)
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < i; j++) {
            T val = a[i * n + j];
            a[j * n + i] = val;
        }
    }

    // Jacobi iterations
    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        // Find maximum off-diagonal element
        T max_off_diag = dtype_traits<T>::zero();
        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int j = i + 1; j < n; j++) {
                T val = dtype_traits<T>::abs(a[i * n + j]);
                if (val > max_off_diag) {
                    max_off_diag = val;
                }
            }
        }

        // Check convergence
        if (max_off_diag < tol) {
            *converged_flag = 1;
            break;
        }

        // Process all element pairs (p, q) where p < q
        for (unsigned int p = 0; p < n; p++) {
            for (unsigned int q = p + 1; q < n; q++) {
                T a_pq = a[p * n + q];

                if (dtype_traits<T>::abs(a_pq) < tol) continue;

                T a_pp = a[p * n + p];
                T a_qq = a[q * n + q];

                // Compute Jacobi rotation parameters
                T tau_num = a_qq - a_pp;
                T tau_den = dtype_traits<T>::two() * a_pq;

                T c, s;
                if (dtype_traits<T>::abs(tau_den) < dtype_traits<T>::tiny_eps()) {
                    c = dtype_traits<T>::one();
                    s = dtype_traits<T>::zero();
                } else {
                    T tau = tau_num / tau_den;
                    T t;
                    if (tau >= dtype_traits<T>::zero()) {
                        t = dtype_traits<T>::one() / (tau + dtype_traits<T>::sqrt(dtype_traits<T>::one() + tau * tau));
                    } else {
                        t = -dtype_traits<T>::one() / (-tau + dtype_traits<T>::sqrt(dtype_traits<T>::one() + tau * tau));
                    }
                    c = dtype_traits<T>::one() / dtype_traits<T>::sqrt(dtype_traits<T>::one() + t * t);
                    s = t * c;
                }

                // Apply Jacobi rotation: A' = J^T @ A @ J
                for (unsigned int k = 0; k < n; k++) {
                    if (k != p && k != q) {
                        T a_kp = a[k * n + p];
                        T a_kq = a[k * n + q];

                        T new_kp = c * a_kp - s * a_kq;
                        T new_kq = s * a_kp + c * a_kq;

                        a[k * n + p] = new_kp;
                        a[p * n + k] = new_kp;
                        a[k * n + q] = new_kq;
                        a[q * n + k] = new_kq;
                    }
                }

                // Update diagonal elements
                T c2 = c * c;
                T s2 = s * s;
                T cs2 = dtype_traits<T>::two() * c * s;

                T new_pp = c2 * a_pp - cs2 * a_pq + s2 * a_qq;
                T new_qq = s2 * a_pp + cs2 * a_pq + c2 * a_qq;

                a[p * n + p] = new_pp;
                a[q * n + q] = new_qq;
                a[p * n + q] = dtype_traits<T>::zero();
                a[q * n + p] = dtype_traits<T>::zero();

                // Update eigenvector matrix: V = V @ J
                for (unsigned int i = 0; i < n; i++) {
                    T v_ip = eigenvectors[i * n + p];
                    T v_iq = eigenvectors[i * n + q];

                    eigenvectors[i * n + p] = c * v_ip - s * v_iq;
                    eigenvectors[i * n + q] = s * v_ip + c * v_iq;
                }
            }
        }
    }

    // Extract eigenvalues (diagonal of converged matrix)
    for (unsigned int i = 0; i < n; i++) {
        eigenvalues[i] = a[i * n + i];
    }
}

// ============================================================================
// Extern "C" wrappers for PTX export
// ============================================================================

extern "C" {

__global__ void eig_jacobi_symmetric_f32(float* a, float* eigenvectors,
                                         float* eigenvalues, unsigned int n,
                                         int* converged_flag) {
    eig_jacobi_symmetric_impl<float>(a, eigenvectors, eigenvalues, n, converged_flag);
}

__global__ void eig_jacobi_symmetric_f64(double* a, double* eigenvectors,
                                         double* eigenvalues, unsigned int n,
                                         int* converged_flag) {
    eig_jacobi_symmetric_impl<double>(a, eigenvectors, eigenvalues, n, converged_flag);
}

// ============================================================================
// F16 (__half) Wrappers
// ============================================================================

__global__ void eig_jacobi_symmetric_f16(__half* a, __half* eigenvectors,
                                         __half* eigenvalues, unsigned int n,
                                         int* converged_flag) {
    eig_jacobi_symmetric_impl<__half>(a, eigenvectors, eigenvalues, n, converged_flag);
}

// ============================================================================
// BF16 (__nv_bfloat16) Wrappers
// ============================================================================

__global__ void eig_jacobi_symmetric_bf16(__nv_bfloat16* a, __nv_bfloat16* eigenvectors,
                                          __nv_bfloat16* eigenvalues, unsigned int n,
                                          int* converged_flag) {
    eig_jacobi_symmetric_impl<__nv_bfloat16>(a, eigenvectors, eigenvalues, n, converged_flag);
}

} // extern "C"
