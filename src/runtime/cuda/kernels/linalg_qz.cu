// QZ Decomposition CUDA kernel
// Generalized Schur Decomposition for matrix pencil (A, B)
// Uses Francis's implicit double-shift algorithm for real arithmetic
//
// Uses C++ templates to eliminate f32/f64 duplication.

#include "dtype_traits.cuh"

// ============================================================================
// QZ Decomposition - Generalized Schur Decomposition
// For matrix pencil (A, B): A = Q @ S @ Z^T, B = Q @ T @ Z^T
// Uses Hessenberg-triangular reduction + double-shift QZ iteration
// ============================================================================

template<typename T>
__device__ void qz_decompose_impl(
    T* __restrict__ s,              // [n, n] input A, becomes quasi-triangular S
    T* __restrict__ t,              // [n, n] input B, becomes upper triangular T
    T* __restrict__ q,              // [n, n] output left orthogonal matrix Q
    T* __restrict__ z,              // [n, n] output right orthogonal matrix Z
    T* __restrict__ eig_real,       // [n] real part of generalized eigenvalues
    T* __restrict__ eig_imag,       // [n] imaginary part of generalized eigenvalues
    int* __restrict__ converged_flag,
    unsigned int n
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const T eps = dtype_traits<T>::eps();
    const int max_iters = 30 * (int)n;

    // Large value for "infinite" eigenvalues
    const T large_val = (T)1e30;

    // Initialize Q and Z as identity
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            q[i * n + j] = (i == j) ? dtype_traits<T>::one() : dtype_traits<T>::zero();
            z[i * n + j] = (i == j) ? dtype_traits<T>::one() : dtype_traits<T>::zero();
        }
    }

    // Step 1: Reduce B to upper triangular using QR
    for (unsigned int k = 0; k < n - 1; k++) {
        for (unsigned int i = n - 1; i > k; i--) {
            T a_val = t[(i - 1) * n + k];
            T b_val = t[i * n + k];

            if (dtype_traits<T>::abs(b_val) < eps) continue;

            T r = dtype_traits<T>::sqrt(a_val * a_val + b_val * b_val);
            T cs = a_val / r;
            T sn = b_val / r;

            // Apply Givens rotation to T (left multiply)
            for (unsigned int j = k; j < n; j++) {
                T t1 = t[(i - 1) * n + j];
                T t2 = t[i * n + j];
                t[(i - 1) * n + j] = cs * t1 + sn * t2;
                t[i * n + j] = -sn * t1 + cs * t2;
            }

            // Apply same rotation to S
            for (unsigned int j = 0; j < n; j++) {
                T s1 = s[(i - 1) * n + j];
                T s2 = s[i * n + j];
                s[(i - 1) * n + j] = cs * s1 + sn * s2;
                s[i * n + j] = -sn * s1 + cs * s2;
            }

            // Accumulate Q
            for (unsigned int j = 0; j < n; j++) {
                T q1 = q[j * n + (i - 1)];
                T q2 = q[j * n + i];
                q[j * n + (i - 1)] = cs * q1 + sn * q2;
                q[j * n + i] = -sn * q1 + cs * q2;
            }
        }
    }

    // Step 2: Reduce S to Hessenberg form while keeping T upper triangular
    for (unsigned int k = 0; k < n - 2; k++) {
        for (unsigned int i = n - 1; i > k + 1; i--) {
            T a_val = s[(k + 1) * n + k];
            T b_val = s[i * n + k];

            if (dtype_traits<T>::abs(b_val) < eps) continue;

            T r = dtype_traits<T>::sqrt(a_val * a_val + b_val * b_val);
            T cs = a_val / r;
            T sn = b_val / r;

            // Apply Givens rotation to S (left multiply)
            for (unsigned int j = k; j < n; j++) {
                T s1 = s[(k + 1) * n + j];
                T s2 = s[i * n + j];
                s[(k + 1) * n + j] = cs * s1 + sn * s2;
                s[i * n + j] = -sn * s1 + cs * s2;
            }

            // Apply same rotation to T
            for (unsigned int j = 0; j < n; j++) {
                T t1 = t[(k + 1) * n + j];
                T t2 = t[i * n + j];
                t[(k + 1) * n + j] = cs * t1 + sn * t2;
                t[i * n + j] = -sn * t1 + cs * t2;
            }

            // Accumulate Q
            for (unsigned int j = 0; j < n; j++) {
                T q1 = q[j * n + (k + 1)];
                T q2 = q[j * n + i];
                q[j * n + (k + 1)] = cs * q1 + sn * q2;
                q[j * n + i] = -sn * q1 + cs * q2;
            }

            // Restore T to upper triangular (right multiply)
            if (dtype_traits<T>::abs(t[i * n + (i - 1)]) > eps) {
                T ta = t[(i - 1) * n + (i - 1)];
                T tb = t[i * n + (i - 1)];
                T tr = dtype_traits<T>::sqrt(ta * ta + tb * tb);
                T tcs = ta / tr;
                T tsn = tb / tr;

                for (unsigned int row = 0; row < n; row++) {
                    T s1 = s[row * n + (i - 1)];
                    T s2 = s[row * n + i];
                    s[row * n + (i - 1)] = tcs * s1 + tsn * s2;
                    s[row * n + i] = -tsn * s1 + tcs * s2;
                }

                for (unsigned int row = 0; row <= i; row++) {
                    T t1 = t[row * n + (i - 1)];
                    T t2 = t[row * n + i];
                    t[row * n + (i - 1)] = tcs * t1 + tsn * t2;
                    t[row * n + i] = -tsn * t1 + tcs * t2;
                }

                for (unsigned int row = 0; row < n; row++) {
                    T z1 = z[row * n + (i - 1)];
                    T z2 = z[row * n + i];
                    z[row * n + (i - 1)] = tcs * z1 + tsn * z2;
                    z[row * n + i] = -tsn * z1 + tcs * z2;
                }
            }
        }
    }

    // Step 3: Double-shift QZ iteration (Francis's implicit algorithm)
    // Uses implicit double shift for real arithmetic on complex eigenvalue pairs
    unsigned int ihi = n;

    for (int iter = 0; iter < max_iters; iter++) {
        // Deflation: check for converged eigenvalues at the bottom
        while (ihi > 1) {
            unsigned int i = ihi - 1;
            T h_ii = dtype_traits<T>::abs(s[(i - 1) * n + (i - 1)]);
            T h_ip1 = dtype_traits<T>::abs(s[i * n + i]);
            T threshold = eps * dtype_traits<T>::max(h_ii + h_ip1, dtype_traits<T>::one());

            if (dtype_traits<T>::abs(s[i * n + (i - 1)]) <= threshold) {
                s[i * n + (i - 1)] = dtype_traits<T>::zero();
                ihi--;
            } else {
                break;
            }
        }

        if (ihi <= 1) {
            *converged_flag = 1;
            break;
        }

        // Find ilo: start of active unreduced block
        unsigned int ilo = 0;
        for (unsigned int i = ihi - 1; i >= 1; i--) {
            T h_ii = dtype_traits<T>::abs(s[(i - 1) * n + (i - 1)]);
            T h_ip1 = dtype_traits<T>::abs(s[i * n + i]);
            T threshold = eps * dtype_traits<T>::max(h_ii + h_ip1, dtype_traits<T>::one());

            if (dtype_traits<T>::abs(s[i * n + (i - 1)]) <= threshold) {
                s[i * n + (i - 1)] = dtype_traits<T>::zero();
                ilo = i;
                break;
            }
        }

        // If block size is 1 or 2, we're done with this eigenvalue/pair
        if (ihi - ilo <= 2) {
            ihi = ilo;
            continue;
        }

        // Perform implicit double-shift QZ step on active block [ilo, ihi)
        // Compute double shift from trailing 2x2 block of H*inv(R)
        unsigned int m = ihi - 1;

        T h_mm = s[(m - 1) * n + (m - 1)];
        T h_m1m = s[m * n + (m - 1)];
        T h_mm1 = s[(m - 1) * n + m];
        T h_m1m1 = s[m * n + m];

        T r_mm = t[(m - 1) * n + (m - 1)];
        T r_mm1 = t[(m - 1) * n + m];
        T r_m1m1 = t[m * n + m];

        // Compute trace and det of trailing 2x2 of H*inv(R)
        T s1_shift, s2_shift;
        if (dtype_traits<T>::abs(r_mm) > eps && dtype_traits<T>::abs(r_m1m1) > eps) {
            T inv_r_mm = dtype_traits<T>::one() / r_mm;
            T inv_r_m1m1 = dtype_traits<T>::one() / r_m1m1;

            T m00 = h_mm * inv_r_mm;
            T m01 = (h_mm1 - h_mm * r_mm1 * inv_r_mm) * inv_r_m1m1;
            T m10 = h_m1m * inv_r_mm;
            T m11 = (h_m1m1 - h_m1m * r_mm1 * inv_r_mm) * inv_r_m1m1;

            s1_shift = m00 + m11;  // trace
            s2_shift = m00 * m11 - m01 * m10;  // det
        } else {
            s1_shift = h_mm + h_m1m1;
            s2_shift = h_mm * h_m1m1 - h_mm1 * h_m1m;
        }

        // First column of (H - s1*R)(H - s2*R) implicitly
        T h00 = s[ilo * n + ilo];
        T h10 = s[(ilo + 1) * n + ilo];
        T h20 = (ilo + 2 < n) ? s[(ilo + 2) * n + ilo] : dtype_traits<T>::zero();
        T h01 = s[ilo * n + (ilo + 1)];
        T h11 = s[(ilo + 1) * n + (ilo + 1)];

        T r00 = t[ilo * n + ilo];
        T r01 = t[ilo * n + (ilo + 1)];
        T r11 = t[(ilo + 1) * n + (ilo + 1)];

        T v0 = h00 * h00 + h01 * h10 - s1_shift * h00 * r00 + s2_shift * r00 * r00;
        T v1 = h10 * (h00 + h11 - s1_shift * r00 - s1_shift * r11);
        T v2 = h10 * h20;

        // Householder to introduce bulge
        T v_norm = dtype_traits<T>::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
        if (v_norm < eps) continue;

        T beta = (v0 >= dtype_traits<T>::zero()) ? -v_norm : v_norm;
        T v0_h = v0 - beta;
        T tau = -v0_h / beta;

        T h_norm = dtype_traits<T>::sqrt(v0_h * v0_h + v1 * v1 + v2 * v2);
        if (h_norm < eps) continue;

        T u0 = v0_h / h_norm;
        T u1 = v1 / h_norm;
        T u2 = v2 / h_norm;

        // Apply initial Householder from the left to S and T
        unsigned int p_end = (ilo + 3 < ihi) ? (ilo + 3) : ihi;

        // Left apply to S: S[ilo:p_end, :] = (I - tau*u*u^T) * S[ilo:p_end, :]
        for (unsigned int j = 0; j < n; j++) {
            T dot = u0 * s[ilo * n + j];
            if (ilo + 1 < p_end) dot = dot + u1 * s[(ilo + 1) * n + j];
            if (ilo + 2 < p_end) dot = dot + u2 * s[(ilo + 2) * n + j];

            T factor = tau * dot;
            s[ilo * n + j] = s[ilo * n + j] - factor * u0;
            if (ilo + 1 < p_end) s[(ilo + 1) * n + j] = s[(ilo + 1) * n + j] - factor * u1;
            if (ilo + 2 < p_end) s[(ilo + 2) * n + j] = s[(ilo + 2) * n + j] - factor * u2;
        }

        // Left apply to T
        for (unsigned int j = ilo; j < n; j++) {
            T dot = u0 * t[ilo * n + j];
            if (ilo + 1 < p_end) dot = dot + u1 * t[(ilo + 1) * n + j];
            if (ilo + 2 < p_end) dot = dot + u2 * t[(ilo + 2) * n + j];

            T factor = tau * dot;
            t[ilo * n + j] = t[ilo * n + j] - factor * u0;
            if (ilo + 1 < p_end) t[(ilo + 1) * n + j] = t[(ilo + 1) * n + j] - factor * u1;
            if (ilo + 2 < p_end) t[(ilo + 2) * n + j] = t[(ilo + 2) * n + j] - factor * u2;
        }

        // Right apply to Q: Q[:, ilo:p_end] = Q[:, ilo:p_end] * (I - tau*u*u^T)
        for (unsigned int i = 0; i < n; i++) {
            T dot = u0 * q[i * n + ilo];
            if (ilo + 1 < p_end) dot = dot + u1 * q[i * n + (ilo + 1)];
            if (ilo + 2 < p_end) dot = dot + u2 * q[i * n + (ilo + 2)];

            T factor = tau * dot;
            q[i * n + ilo] = q[i * n + ilo] - factor * u0;
            if (ilo + 1 < p_end) q[i * n + (ilo + 1)] = q[i * n + (ilo + 1)] - factor * u1;
            if (ilo + 2 < p_end) q[i * n + (ilo + 2)] = q[i * n + (ilo + 2)] - factor * u2;
        }

        // Chase the bulge down
        for (unsigned int k = ilo; k < ihi - 2; k++) {
            unsigned int p_size = (k + 3 < ihi) ? 3 : 2;

            // Restore T to upper triangular with column Givens rotations
            for (unsigned int i = k + 1; i < k + p_size && i < ihi; i++) {
                T r1 = t[k * n + k];
                T r2 = t[i * n + k];

                if (dtype_traits<T>::abs(r2) < eps) continue;

                T rr = dtype_traits<T>::sqrt(r1 * r1 + r2 * r2);
                T c = r1 / rr;
                T sn = r2 / rr;

                // Column rotation on T
                for (unsigned int row = 0; row < ihi; row++) {
                    T t1 = t[row * n + k];
                    T t2 = t[row * n + i];
                    t[row * n + k] = c * t1 + sn * t2;
                    t[row * n + i] = -sn * t1 + c * t2;
                }

                // Same on S
                for (unsigned int row = 0; row < ihi; row++) {
                    T s1 = s[row * n + k];
                    T s2 = s[row * n + i];
                    s[row * n + k] = c * s1 + sn * s2;
                    s[row * n + i] = -sn * s1 + c * s2;
                }

                // Accumulate into Z
                for (unsigned int row = 0; row < n; row++) {
                    T z1 = z[row * n + k];
                    T z2 = z[row * n + i];
                    z[row * n + k] = c * z1 + sn * z2;
                    z[row * n + i] = -sn * z1 + c * z2;
                }
            }

            // Zero out elements below subdiagonal in column k of S
            if (k + 2 < ihi) {
                T w0 = s[(k + 1) * n + k];
                T w1 = s[(k + 2) * n + k];
                T w2 = (k + 3 < ihi) ? s[(k + 3) * n + k] : dtype_traits<T>::zero();

                unsigned int w_size = (k + 3 < ihi) ? 3 : 2;
                T w_norm = (w_size == 3) ?
                    dtype_traits<T>::sqrt(w0 * w0 + w1 * w1 + w2 * w2) :
                    dtype_traits<T>::sqrt(w0 * w0 + w1 * w1);

                if (w_norm > eps) {
                    T beta_w = (w0 >= dtype_traits<T>::zero()) ? -w_norm : w_norm;
                    T w0_h = w0 - beta_w;
                    T tau_w = -w0_h / beta_w;

                    T h_norm_w = (w_size == 3) ?
                        dtype_traits<T>::sqrt(w0_h * w0_h + w1 * w1 + w2 * w2) :
                        dtype_traits<T>::sqrt(w0_h * w0_h + w1 * w1);

                    if (h_norm_w > eps) {
                        T uu0 = w0_h / h_norm_w;
                        T uu1 = w1 / h_norm_w;
                        T uu2 = (w_size == 3) ? w2 / h_norm_w : dtype_traits<T>::zero();

                        unsigned int p_start = k + 1;
                        unsigned int p_end_w = (k + 1 + w_size < ihi) ? (k + 1 + w_size) : ihi;

                        // Left apply to S
                        for (unsigned int j = k; j < n; j++) {
                            T dot = uu0 * s[p_start * n + j];
                            if (p_start + 1 < p_end_w) dot = dot + uu1 * s[(p_start + 1) * n + j];
                            if (p_start + 2 < p_end_w && w_size == 3) dot = dot + uu2 * s[(p_start + 2) * n + j];

                            T factor = tau_w * dot;
                            s[p_start * n + j] = s[p_start * n + j] - factor * uu0;
                            if (p_start + 1 < p_end_w) s[(p_start + 1) * n + j] = s[(p_start + 1) * n + j] - factor * uu1;
                            if (p_start + 2 < p_end_w && w_size == 3) s[(p_start + 2) * n + j] = s[(p_start + 2) * n + j] - factor * uu2;
                        }

                        // Left apply to T
                        for (unsigned int j = k + 1; j < n; j++) {
                            T dot = uu0 * t[p_start * n + j];
                            if (p_start + 1 < p_end_w) dot = dot + uu1 * t[(p_start + 1) * n + j];
                            if (p_start + 2 < p_end_w && w_size == 3) dot = dot + uu2 * t[(p_start + 2) * n + j];

                            T factor = tau_w * dot;
                            t[p_start * n + j] = t[p_start * n + j] - factor * uu0;
                            if (p_start + 1 < p_end_w) t[(p_start + 1) * n + j] = t[(p_start + 1) * n + j] - factor * uu1;
                            if (p_start + 2 < p_end_w && w_size == 3) t[(p_start + 2) * n + j] = t[(p_start + 2) * n + j] - factor * uu2;
                        }

                        // Right apply to Q
                        for (unsigned int i = 0; i < n; i++) {
                            T dot = uu0 * q[i * n + p_start];
                            if (p_start + 1 < p_end_w) dot = dot + uu1 * q[i * n + (p_start + 1)];
                            if (p_start + 2 < p_end_w && w_size == 3) dot = dot + uu2 * q[i * n + (p_start + 2)];

                            T factor = tau_w * dot;
                            q[i * n + p_start] = q[i * n + p_start] - factor * uu0;
                            if (p_start + 1 < p_end_w) q[i * n + (p_start + 1)] = q[i * n + (p_start + 1)] - factor * uu1;
                            if (p_start + 2 < p_end_w && w_size == 3) q[i * n + (p_start + 2)] = q[i * n + (p_start + 2)] - factor * uu2;
                        }
                    }
                }
            }
        }
    }

    // Final cleanup of small subdiagonals
    for (unsigned int i = 1; i < n; i++) {
        T h_ii = dtype_traits<T>::abs(s[(i - 1) * n + (i - 1)]);
        T h_ip1 = dtype_traits<T>::abs(s[i * n + i]);
        T threshold = eps * dtype_traits<T>::max(h_ii + h_ip1, dtype_traits<T>::one());

        if (dtype_traits<T>::abs(s[i * n + (i - 1)]) <= threshold) {
            s[i * n + (i - 1)] = dtype_traits<T>::zero();
        }
    }

    *converged_flag = 1;

    // Extract generalized eigenvalues: lambda = S[i,i] / T[i,i]
    unsigned int i = 0;
    while (i < n) {
        if (i == n - 1 || dtype_traits<T>::abs(s[(i + 1) * n + i]) < eps) {
            // Real eigenvalue
            if (dtype_traits<T>::abs(t[i * n + i]) > eps) {
                eig_real[i] = s[i * n + i] / t[i * n + i];
            } else {
                eig_real[i] = (s[i * n + i] >= dtype_traits<T>::zero()) ? large_val : -large_val;
            }
            eig_imag[i] = dtype_traits<T>::zero();
            i++;
        } else {
            // Complex conjugate pair from 2x2 block
            T a = s[i * n + i];
            T b = s[i * n + (i + 1)];
            T c = s[(i + 1) * n + i];
            T d = s[(i + 1) * n + (i + 1)];

            T trace = a + d;
            T det = a * d - b * c;
            T disc = trace * trace - (T)4 * det;

            if (disc < dtype_traits<T>::zero()) {
                T real_part = trace / dtype_traits<T>::two();
                T imag_part = dtype_traits<T>::sqrt(-disc) / dtype_traits<T>::two();

                T t_scale = t[i * n + i];
                if (dtype_traits<T>::abs(t_scale) > eps) {
                    real_part /= t_scale;
                    imag_part /= t_scale;
                }

                eig_real[i] = real_part;
                eig_imag[i] = imag_part;
                eig_real[i + 1] = real_part;
                eig_imag[i + 1] = -imag_part;
            } else {
                T sqrt_disc = dtype_traits<T>::sqrt(disc);
                eig_real[i] = (trace + sqrt_disc) / dtype_traits<T>::two();
                eig_real[i + 1] = (trace - sqrt_disc) / dtype_traits<T>::two();
                eig_imag[i] = dtype_traits<T>::zero();
                eig_imag[i + 1] = dtype_traits<T>::zero();
            }
            i += 2;
        }
    }
}

// ============================================================================
// Extern "C" wrappers for PTX export
// ============================================================================

extern "C" {

__global__ void qz_decompose_f32(float* s, float* t, float* q, float* z, float* eig_real,
                                 float* eig_imag, int* converged_flag, unsigned int n) {
    qz_decompose_impl<float>(s, t, q, z, eig_real, eig_imag, converged_flag, n);
}

__global__ void qz_decompose_f64(double* s, double* t, double* q, double* z, double* eig_real,
                                 double* eig_imag, int* converged_flag, unsigned int n) {
    qz_decompose_impl<double>(s, t, q, z, eig_real, eig_imag, converged_flag, n);
}

// ============================================================================
// F16 (__half) Wrappers
// ============================================================================

__global__ void qz_decompose_f16(__half* s, __half* t, __half* q, __half* z, __half* eig_real,
                                 __half* eig_imag, int* converged_flag, unsigned int n) {
    qz_decompose_impl<__half>(s, t, q, z, eig_real, eig_imag, converged_flag, n);
}

// ============================================================================
// BF16 (__nv_bfloat16) Wrappers
// ============================================================================

__global__ void qz_decompose_bf16(__nv_bfloat16* s, __nv_bfloat16* t, __nv_bfloat16* q,
                                  __nv_bfloat16* z, __nv_bfloat16* eig_real,
                                  __nv_bfloat16* eig_imag, int* converged_flag, unsigned int n) {
    qz_decompose_impl<__nv_bfloat16>(s, t, q, z, eig_real, eig_imag, converged_flag, n);
}

} // extern "C"
