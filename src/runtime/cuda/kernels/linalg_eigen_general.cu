// General eigenvalue decomposition CUDA kernels
// Uses Schur decomposition + back-substitution for eigenvectors
// Returns real and imaginary parts of eigenvalues and eigenvectors
//
// Uses C++ templates to eliminate f32/f64 duplication.

#include "dtype_traits.cuh"

// ============================================================================
// General Eigenvalue Decomposition
// ============================================================================

template<typename T>
__device__ void eig_general_impl(
    T* __restrict__ t,              // [n, n] working buffer (becomes Schur form)
    T* __restrict__ z,              // [n, n] Schur vectors
    T* __restrict__ eval_real,      // [n] real part of eigenvalues
    T* __restrict__ eval_imag,      // [n] imaginary part of eigenvalues
    T* __restrict__ evec_real,      // [n, n] real part of eigenvectors
    T* __restrict__ evec_imag,      // [n, n] imaginary part of eigenvectors
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

    // === Schur decomposition (inline) ===

    // Hessenberg reduction
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

        for (unsigned int j = 0; j < n; j++) {
            T dot = dtype_traits<T>::zero();
            for (unsigned int i = 0; i < v_len; i++) {
                dot += v[i] * t[(k + 1 + i) * n + j];
            }
            for (unsigned int i = 0; i < v_len; i++) {
                t[(k + 1 + i) * n + j] -= dtype_traits<T>::two() * v[i] * dot;
            }
        }

        for (unsigned int i = 0; i < n; i++) {
            T dot = dtype_traits<T>::zero();
            for (unsigned int jj = 0; jj < v_len; jj++) {
                dot += t[i * n + (k + 1 + jj)] * v[jj];
            }
            for (unsigned int jj = 0; jj < v_len; jj++) {
                t[i * n + (k + 1 + jj)] -= dtype_traits<T>::two() * dot * v[jj];
            }
        }

        for (unsigned int i = 0; i < n; i++) {
            T dot = dtype_traits<T>::zero();
            for (unsigned int jj = 0; jj < v_len; jj++) {
                dot += z[i * n + (k + 1 + jj)] * v[jj];
            }
            for (unsigned int jj = 0; jj < v_len; jj++) {
                z[i * n + (k + 1 + jj)] -= dtype_traits<T>::two() * dot * v[jj];
            }
        }
    }

    // QR iteration
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

    // Clean up
    for (unsigned int i = 0; i < n - 1; i++) {
        T h_ii = dtype_traits<T>::abs(t[i * n + i]);
        T h_ip1 = dtype_traits<T>::abs(t[(i + 1) * n + (i + 1)]);
        T threshold = eps * dtype_traits<T>::max(h_ii + h_ip1, dtype_traits<T>::one());
        if (dtype_traits<T>::abs(t[(i + 1) * n + i]) <= threshold) {
            t[(i + 1) * n + i] = dtype_traits<T>::zero();
        }
    }

    for (unsigned int ii = 2; ii < n; ii++) {
        for (unsigned int jj = 0; jj < ii - 1; jj++) {
            t[ii * n + jj] = dtype_traits<T>::zero();
        }
    }

    // === Extract eigenvalues from Schur form ===
    unsigned int i = 0;
    while (i < n) {
        if (i == n - 1) {
            eval_real[i] = t[i * n + i];
            eval_imag[i] = dtype_traits<T>::zero();
            i++;
        } else {
            T subdiag = dtype_traits<T>::abs(t[(i + 1) * n + i]);
            T diag_scale = dtype_traits<T>::abs(t[i * n + i]) + dtype_traits<T>::abs(t[(i + 1) * n + (i + 1)]);
            T threshold = eps * dtype_traits<T>::max(diag_scale, dtype_traits<T>::one());

            if (subdiag > threshold) {
                T a_val = t[i * n + i];
                T b_val = t[i * n + (i + 1)];
                T c_val = t[(i + 1) * n + i];
                T d_val = t[(i + 1) * n + (i + 1)];

                T tr = a_val + d_val;
                T disc = (a_val - d_val) * (a_val - d_val) / (T)4 + b_val * c_val;

                if (disc < dtype_traits<T>::zero()) {
                    T real_part = tr / dtype_traits<T>::two();
                    T imag_part = dtype_traits<T>::sqrt(-disc);
                    eval_real[i] = real_part;
                    eval_imag[i] = imag_part;
                    eval_real[i + 1] = real_part;
                    eval_imag[i + 1] = -imag_part;
                } else {
                    T sqrt_disc = dtype_traits<T>::sqrt(disc);
                    eval_real[i] = tr / dtype_traits<T>::two() + sqrt_disc;
                    eval_imag[i] = dtype_traits<T>::zero();
                    eval_real[i + 1] = tr / dtype_traits<T>::two() - sqrt_disc;
                    eval_imag[i + 1] = dtype_traits<T>::zero();
                }
                i += 2;
            } else {
                eval_real[i] = t[i * n + i];
                eval_imag[i] = dtype_traits<T>::zero();
                i++;
            }
        }
    }

    // === Compute eigenvectors via back-substitution ===
    T y_real[256];
    T y_imag[256];

    i = 0;
    while (i < n) {
        T imag = eval_imag[i];

        if (dtype_traits<T>::abs(imag) < eps) {
            T lambda = eval_real[i];

            for (unsigned int k = 0; k < n; k++) {
                y_real[k] = dtype_traits<T>::zero();
                y_imag[k] = dtype_traits<T>::zero();
            }
            y_real[i] = dtype_traits<T>::one();

            for (int k = (int)i - 1; k >= 0; k--) {
                T diag = t[k * n + k] - lambda;
                T rhs = dtype_traits<T>::zero();
                for (unsigned int j = k + 1; j < n; j++) {
                    rhs -= t[k * n + j] * y_real[j];
                }
                if (dtype_traits<T>::abs(diag) > eps) {
                    y_real[k] = rhs / diag;
                } else {
                    y_real[k] = dtype_traits<T>::zero();
                }
            }

            T norm_sq = dtype_traits<T>::zero();
            for (unsigned int k = 0; k < n; k++) {
                norm_sq += y_real[k] * y_real[k];
            }
            T norm = dtype_traits<T>::sqrt(norm_sq);
            if (norm > eps) {
                for (unsigned int k = 0; k < n; k++) {
                    y_real[k] /= norm;
                }
            }

            for (unsigned int row = 0; row < n; row++) {
                T sum = dtype_traits<T>::zero();
                for (unsigned int k = 0; k < n; k++) {
                    sum += z[row * n + k] * y_real[k];
                }
                evec_real[row * n + i] = sum;
                evec_imag[row * n + i] = dtype_traits<T>::zero();
            }
            i++;
        } else {
            T lambda_real = eval_real[i];
            T lambda_imag = eval_imag[i];

            for (unsigned int k = 0; k < n; k++) {
                y_real[k] = dtype_traits<T>::zero();
                y_imag[k] = dtype_traits<T>::zero();
            }

            T a_val = t[i * n + i];
            T b_val = t[i * n + (i + 1)];
            y_real[i] = b_val;
            y_imag[i] = dtype_traits<T>::zero();
            y_real[i + 1] = lambda_real - a_val;
            y_imag[i + 1] = lambda_imag;

            for (int k = (int)i - 1; k >= 0; k--) {
                T diag_real = t[k * n + k] - lambda_real;
                T diag_imag = -lambda_imag;

                T rhs_real = dtype_traits<T>::zero();
                T rhs_imag = dtype_traits<T>::zero();

                for (unsigned int j = k + 1; j < n; j++) {
                    T t_kj = t[k * n + j];
                    rhs_real -= t_kj * y_real[j];
                    rhs_imag -= t_kj * y_imag[j];
                }

                T denom = diag_real * diag_real + diag_imag * diag_imag;
                if (denom > eps * eps) {
                    y_real[k] = (rhs_real * diag_real + rhs_imag * diag_imag) / denom;
                    y_imag[k] = (rhs_imag * diag_real - rhs_real * diag_imag) / denom;
                } else {
                    y_real[k] = dtype_traits<T>::zero();
                    y_imag[k] = dtype_traits<T>::zero();
                }
            }

            T norm_sq = dtype_traits<T>::zero();
            for (unsigned int k = 0; k < n; k++) {
                norm_sq += y_real[k] * y_real[k] + y_imag[k] * y_imag[k];
            }
            T norm = dtype_traits<T>::sqrt(norm_sq);
            if (norm > eps) {
                for (unsigned int k = 0; k < n; k++) {
                    y_real[k] /= norm;
                    y_imag[k] /= norm;
                }
            }

            for (unsigned int row = 0; row < n; row++) {
                T sum_real = dtype_traits<T>::zero();
                T sum_imag = dtype_traits<T>::zero();
                for (unsigned int k = 0; k < n; k++) {
                    T z_val = z[row * n + k];
                    sum_real += z_val * y_real[k];
                    sum_imag += z_val * y_imag[k];
                }
                evec_real[row * n + i] = sum_real;
                evec_imag[row * n + i] = sum_imag;
                evec_real[row * n + (i + 1)] = sum_real;
                evec_imag[row * n + (i + 1)] = -sum_imag;
            }
            i += 2;
        }
    }
}

// ============================================================================
// Extern "C" wrappers for PTX export
// ============================================================================

extern "C" {

__global__ void eig_general_f32(float* t, float* z, float* eval_real, float* eval_imag,
                                float* evec_real, float* evec_imag, unsigned int n,
                                int* converged_flag) {
    eig_general_impl<float>(t, z, eval_real, eval_imag, evec_real, evec_imag, n, converged_flag);
}

__global__ void eig_general_f64(double* t, double* z, double* eval_real, double* eval_imag,
                                double* evec_real, double* evec_imag, unsigned int n,
                                int* converged_flag) {
    eig_general_impl<double>(t, z, eval_real, eval_imag, evec_real, evec_imag, n, converged_flag);
}

// ============================================================================
// F16 (__half) Wrappers
// ============================================================================

__global__ void eig_general_f16(__half* t, __half* z, __half* eval_real, __half* eval_imag,
                                __half* evec_real, __half* evec_imag, unsigned int n,
                                int* converged_flag) {
    eig_general_impl<__half>(t, z, eval_real, eval_imag, evec_real, evec_imag, n, converged_flag);
}

// ============================================================================
// BF16 (__nv_bfloat16) Wrappers
// ============================================================================

__global__ void eig_general_bf16(__nv_bfloat16* t, __nv_bfloat16* z, __nv_bfloat16* eval_real,
                                 __nv_bfloat16* eval_imag, __nv_bfloat16* evec_real,
                                 __nv_bfloat16* evec_imag, unsigned int n,
                                 int* converged_flag) {
    eig_general_impl<__nv_bfloat16>(t, z, eval_real, eval_imag, evec_real, evec_imag, n, converged_flag);
}

} // extern "C"
