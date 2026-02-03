// Matrix function CUDA kernels for quasi-triangular matrices
// Includes: eigenvalue validation, diagonal block functions (exp, log, sqrt), Parlett recurrence
//
// These kernels support Schur-based matrix function computation without
// GPU→CPU→GPU data transfers.
//
// Uses C++ templates to eliminate f32/f64 duplication.

#include "dtype_traits.cuh"
#include <math.h>

// ============================================================================
// Type-specific math functions
// ============================================================================

template<typename T> struct math_funcs;

template<> struct math_funcs<float> {
    static __device__ __forceinline__ float exp(float x) { return expf(x); }
    static __device__ __forceinline__ float log(float x) { return logf(x); }
    static __device__ __forceinline__ float cos(float x) { return cosf(x); }
    static __device__ __forceinline__ float sin(float x) { return sinf(x); }
    static __device__ __forceinline__ float atan2(float y, float x) { return atan2f(y, x); }
};

template<> struct math_funcs<double> {
    static __device__ __forceinline__ double exp(double x) { return ::exp(x); }
    static __device__ __forceinline__ double log(double x) { return ::log(x); }
    static __device__ __forceinline__ double cos(double x) { return ::cos(x); }
    static __device__ __forceinline__ double sin(double x) { return ::sin(x); }
    static __device__ __forceinline__ double atan2(double y, double x) { return ::atan2(y, x); }
};

// ============================================================================
// Validate Eigenvalues - Check for non-positive real eigenvalues in Schur form
// ============================================================================

template<typename T>
__device__ void validate_eigenvalues_impl(
    const T* __restrict__ t,     // [n, n] quasi-triangular Schur matrix T
    T* __restrict__ result,       // [2] output: [has_error, problematic_value]
    unsigned int n,
    T eps,
    int mode                      // 0: log (check <= eps), 1: sqrt (check < -eps)
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    result[0] = dtype_traits<T>::zero();  // no error
    result[1] = dtype_traits<T>::zero();

    unsigned int i = 0;
    while (i < n) {
        // Check if this is a 2x2 block (has subdiagonal element)
        bool is_2x2 = (i + 1 < n) && (dtype_traits<T>::abs(t[(i + 1) * n + i]) > eps);

        if (is_2x2) {
            // 2x2 block: eigenvalues are complex conjugate pair
            // Real part is (a + d) / 2 where the block is [[a, b], [c, d]]
            T a = t[i * n + i];
            T d = t[(i + 1) * n + (i + 1)];
            T real_part = (a + d) / dtype_traits<T>::two();

            if (mode == 0) {
                // logm: check if real part <= eps
                if (real_part <= eps) {
                    result[0] = dtype_traits<T>::one();
                    result[1] = real_part;
                    return;
                }
            } else {
                // sqrtm: check if real part < -eps
                if (real_part < -eps) {
                    result[0] = dtype_traits<T>::one();
                    result[1] = real_part;
                    return;
                }
            }
            i += 2;
        } else {
            // 1x1 block: eigenvalue is the diagonal element
            T eigenvalue = t[i * n + i];

            if (mode == 0) {
                // logm: check if eigenvalue <= eps
                if (eigenvalue <= eps) {
                    result[0] = dtype_traits<T>::one();
                    result[1] = eigenvalue;
                    return;
                }
            } else {
                // sqrtm: check if eigenvalue < -eps
                if (eigenvalue < -eps) {
                    result[0] = dtype_traits<T>::one();
                    result[1] = eigenvalue;
                    return;
                }
            }
            i += 1;
        }
    }
}

// ============================================================================
// Diagonal Block Functions - Apply f to 1x1 and 2x2 diagonal blocks
// ============================================================================

// Helper: exp of 2x2 block [[a, b], [c, d]] where c != 0 (complex eigenvalues)
template<typename T>
__device__ void exp_2x2_block(T a, T b, T c, T d, T* f11, T* f12, T* f21, T* f22) {
    // Eigenvalues: lambda = (a+d)/2 +/- i*sqrt((a-d)^2/4 + bc) when bc < 0
    T tr = a + d;
    T det = a * d - b * c;
    T disc = tr * tr - (T)4 * det;

    if (disc >= dtype_traits<T>::zero()) {
        // Real eigenvalues (shouldn't happen for proper 2x2 complex block)
        T s = dtype_traits<T>::sqrt(disc);
        T l1 = (tr + s) / dtype_traits<T>::two();
        T l2 = (tr - s) / dtype_traits<T>::two();
        T exp1 = math_funcs<T>::exp(l1);
        T exp2 = math_funcs<T>::exp(l2);

        if (dtype_traits<T>::abs(l1 - l2) < dtype_traits<T>::eps()) {
            // Repeated eigenvalue
            *f11 = exp1;
            *f12 = exp1 * b;
            *f21 = exp1 * c;
            *f22 = exp1;
        } else {
            // Distinct eigenvalues
            T denom = l1 - l2;
            *f11 = (exp1 * (a - l2) - exp2 * (a - l1)) / denom;
            *f12 = (exp1 - exp2) * b / denom;
            *f21 = (exp1 - exp2) * c / denom;
            *f22 = (exp1 * (d - l2) - exp2 * (d - l1)) / denom;
        }
    } else {
        // Complex eigenvalues: lambda = alpha +/- i*beta
        T alpha = tr / dtype_traits<T>::two();
        T beta = dtype_traits<T>::sqrt(-disc) / dtype_traits<T>::two();
        T exp_alpha = math_funcs<T>::exp(alpha);
        T cos_beta = math_funcs<T>::cos(beta);
        T sin_beta = math_funcs<T>::sin(beta);

        // exp(A) = exp(alpha) * [cos(beta)*I + sin(beta)/beta * (A - alpha*I)]
        T factor = exp_alpha * sin_beta / beta;
        *f11 = exp_alpha * cos_beta + factor * (a - alpha);
        *f12 = factor * b;
        *f21 = factor * c;
        *f22 = exp_alpha * cos_beta + factor * (d - alpha);
    }
}

// Helper: log of 2x2 block
template<typename T>
__device__ void log_2x2_block(T a, T b, T c, T d, T* f11, T* f12, T* f21, T* f22) {
    T tr = a + d;
    T det = a * d - b * c;
    T disc = tr * tr - (T)4 * det;

    if (disc >= dtype_traits<T>::zero()) {
        // Real eigenvalues
        T s = dtype_traits<T>::sqrt(disc);
        T l1 = (tr + s) / dtype_traits<T>::two();
        T l2 = (tr - s) / dtype_traits<T>::two();

        if (l1 <= dtype_traits<T>::zero() || l2 <= dtype_traits<T>::zero()) {
            // Non-positive eigenvalue - this should be caught by validation
            *f11 = dtype_traits<T>::zero();
            *f12 = dtype_traits<T>::zero();
            *f21 = dtype_traits<T>::zero();
            *f22 = dtype_traits<T>::zero();
            return;
        }

        T log1 = math_funcs<T>::log(l1);
        T log2 = math_funcs<T>::log(l2);

        if (dtype_traits<T>::abs(l1 - l2) < dtype_traits<T>::eps()) {
            *f11 = log1;
            *f12 = b / l1;
            *f21 = c / l1;
            *f22 = log1;
        } else {
            T denom = l1 - l2;
            *f11 = (log1 * (a - l2) - log2 * (a - l1)) / denom;
            *f12 = (log1 - log2) * b / denom;
            *f21 = (log1 - log2) * c / denom;
            *f22 = (log1 * (d - l2) - log2 * (d - l1)) / denom;
        }
    } else {
        // Complex eigenvalues: lambda = alpha +/- i*beta
        T alpha = tr / dtype_traits<T>::two();
        T beta = dtype_traits<T>::sqrt(-disc) / dtype_traits<T>::two();
        T r = dtype_traits<T>::sqrt(alpha * alpha + beta * beta);  // |lambda|
        T theta = math_funcs<T>::atan2(beta, alpha);
        T log_r = math_funcs<T>::log(r);

        // log(A) = log(r)*I + theta/beta * (A - alpha*I)
        T factor = theta / beta;
        *f11 = log_r + factor * (a - alpha);
        *f12 = factor * b;
        *f21 = factor * c;
        *f22 = log_r + factor * (d - alpha);
    }
}

// Helper: sqrt of 2x2 block
template<typename T>
__device__ void sqrt_2x2_block(T a, T b, T c, T d, T* f11, T* f12, T* f21, T* f22) {
    T tr = a + d;
    T det = a * d - b * c;
    T disc = tr * tr - (T)4 * det;

    if (disc >= dtype_traits<T>::zero()) {
        // Real eigenvalues
        T s = dtype_traits<T>::sqrt(disc);
        T l1 = (tr + s) / dtype_traits<T>::two();
        T l2 = (tr - s) / dtype_traits<T>::two();

        if (l1 < dtype_traits<T>::zero() || l2 < dtype_traits<T>::zero()) {
            *f11 = dtype_traits<T>::zero();
            *f12 = dtype_traits<T>::zero();
            *f21 = dtype_traits<T>::zero();
            *f22 = dtype_traits<T>::zero();
            return;
        }

        T sqrt1 = dtype_traits<T>::sqrt(l1);
        T sqrt2 = dtype_traits<T>::sqrt(l2);

        if (dtype_traits<T>::abs(l1 - l2) < dtype_traits<T>::eps()) {
            T deriv = dtype_traits<T>::one() / (dtype_traits<T>::two() * sqrt1);
            *f11 = sqrt1;
            *f12 = deriv * b;
            *f21 = deriv * c;
            *f22 = sqrt1;
        } else {
            T denom = l1 - l2;
            *f11 = (sqrt1 * (a - l2) - sqrt2 * (a - l1)) / denom;
            *f12 = (sqrt1 - sqrt2) * b / denom;
            *f21 = (sqrt1 - sqrt2) * c / denom;
            *f22 = (sqrt1 * (d - l2) - sqrt2 * (d - l1)) / denom;
        }
    } else {
        // Complex eigenvalues: principal square root
        T alpha = tr / dtype_traits<T>::two();
        T beta = dtype_traits<T>::sqrt(-disc) / dtype_traits<T>::two();
        T r = dtype_traits<T>::sqrt(alpha * alpha + beta * beta);
        T theta = math_funcs<T>::atan2(beta, alpha);

        T sqrt_r = dtype_traits<T>::sqrt(r);
        T half_theta = theta / dtype_traits<T>::two();
        T new_alpha = sqrt_r * math_funcs<T>::cos(half_theta);
        T new_beta = sqrt_r * math_funcs<T>::sin(half_theta);

        // sqrt(A) = new_alpha*I + new_beta/beta * (A - alpha*I)
        T factor = new_beta / beta;
        *f11 = new_alpha + factor * (a - alpha);
        *f12 = factor * b;
        *f21 = factor * c;
        *f22 = new_alpha + factor * (d - alpha);
    }
}

template<typename T>
__device__ void diagonal_exp_impl(
    const T* __restrict__ t,     // [n, n] input Schur matrix T
    T* __restrict__ f,           // [n, n] output f(T)
    unsigned int n,
    T eps
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Zero out the output matrix first
    for (unsigned int i = 0; i < n * n; i++) {
        f[i] = dtype_traits<T>::zero();
    }

    unsigned int i = 0;
    while (i < n) {
        bool is_2x2 = (i + 1 < n) && (dtype_traits<T>::abs(t[(i + 1) * n + i]) > eps);

        if (is_2x2) {
            T a = t[i * n + i];
            T b = t[i * n + (i + 1)];
            T c = t[(i + 1) * n + i];
            T d = t[(i + 1) * n + (i + 1)];

            exp_2x2_block(a, b, c, d,
                &f[i * n + i], &f[i * n + (i + 1)],
                &f[(i + 1) * n + i], &f[(i + 1) * n + (i + 1)]);

            i += 2;
        } else {
            f[i * n + i] = math_funcs<T>::exp(t[i * n + i]);
            i += 1;
        }
    }
}

template<typename T>
__device__ void diagonal_log_impl(
    const T* __restrict__ t,
    T* __restrict__ f,
    unsigned int n,
    T eps
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (unsigned int i = 0; i < n * n; i++) {
        f[i] = dtype_traits<T>::zero();
    }

    unsigned int i = 0;
    while (i < n) {
        bool is_2x2 = (i + 1 < n) && (dtype_traits<T>::abs(t[(i + 1) * n + i]) > eps);

        if (is_2x2) {
            T a = t[i * n + i];
            T b = t[i * n + (i + 1)];
            T c = t[(i + 1) * n + i];
            T d = t[(i + 1) * n + (i + 1)];

            log_2x2_block(a, b, c, d,
                &f[i * n + i], &f[i * n + (i + 1)],
                &f[(i + 1) * n + i], &f[(i + 1) * n + (i + 1)]);

            i += 2;
        } else {
            T val = t[i * n + i];
            f[i * n + i] = (val > dtype_traits<T>::zero()) ? math_funcs<T>::log(val) : dtype_traits<T>::zero();
            i += 1;
        }
    }
}

template<typename T>
__device__ void diagonal_sqrt_impl(
    const T* __restrict__ t,
    T* __restrict__ f,
    unsigned int n,
    T eps
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (unsigned int i = 0; i < n * n; i++) {
        f[i] = dtype_traits<T>::zero();
    }

    unsigned int i = 0;
    while (i < n) {
        bool is_2x2 = (i + 1 < n) && (dtype_traits<T>::abs(t[(i + 1) * n + i]) > eps);

        if (is_2x2) {
            T a = t[i * n + i];
            T b = t[i * n + (i + 1)];
            T c = t[(i + 1) * n + i];
            T d = t[(i + 1) * n + (i + 1)];

            sqrt_2x2_block(a, b, c, d,
                &f[i * n + i], &f[i * n + (i + 1)],
                &f[(i + 1) * n + i], &f[(i + 1) * n + (i + 1)]);

            i += 2;
        } else {
            T val = t[i * n + i];
            f[i * n + i] = (val >= dtype_traits<T>::zero()) ? dtype_traits<T>::sqrt(val) : dtype_traits<T>::zero();
            i += 1;
        }
    }
}

// ============================================================================
// Parlett Column - Compute off-diagonal elements using Parlett's recurrence
// ============================================================================

template<typename T>
__device__ void parlett_column_impl(
    const T* __restrict__ t,     // [n, n] input Schur matrix T
    T* __restrict__ f,           // [n, n] output (diagonal already filled)
    unsigned int n,
    unsigned int col,            // Column to compute (1..n-1)
    T eps
) {
    // Each thread handles one row i < col
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= col) return;

    // Skip if row is second element of a 2x2 block (handled with first)
    if (row > 0 && dtype_traits<T>::abs(t[row * n + (row - 1)]) > eps) {
        return;
    }

    // Skip if col is second element of a 2x2 block (handled with first)
    if (col > 0 && dtype_traits<T>::abs(t[col * n + (col - 1)]) > eps) {
        return;
    }

    // Compute F[row, col] using Parlett's recurrence
    T t_ij = t[row * n + col];
    T sum = dtype_traits<T>::zero();

    // Sum over k between row and col: F[row,k] * T[k,col] - T[row,k] * F[k,col]
    for (unsigned int k = row + 1; k < col; k++) {
        sum += f[row * n + k] * t[k * n + col] - t[row * n + k] * f[k * n + col];
    }

    T f_ii = f[row * n + row];
    T f_jj = f[col * n + col];
    T t_ii = t[row * n + row];
    T t_jj = t[col * n + col];

    T denom = t_jj - t_ii;

    if (dtype_traits<T>::abs(denom) > eps) {
        // Normal case: distinct diagonal elements
        f[row * n + col] = (t_ij * (f_jj - f_ii) + sum) / denom;
    } else {
        // Nearly equal diagonal elements - use derivative approximation
        // For repeated eigenvalues, f'(lambda) * (T - lambda*I) contribution
        f[row * n + col] = t_ij * (f_jj + f_ii) / dtype_traits<T>::two() + sum;
    }
}

// ============================================================================
// Extern "C" wrappers for PTX export
// ============================================================================

extern "C" {

// Eigenvalue validation
__global__ void validate_eigenvalues_log_f32(const float* t, float* result, unsigned int n, float eps) {
    validate_eigenvalues_impl<float>(t, result, n, eps, 0);
}

__global__ void validate_eigenvalues_log_f64(const double* t, double* result, unsigned int n, double eps) {
    validate_eigenvalues_impl<double>(t, result, n, eps, 0);
}

__global__ void validate_eigenvalues_sqrt_f32(const float* t, float* result, unsigned int n, float eps) {
    validate_eigenvalues_impl<float>(t, result, n, eps, 1);
}

__global__ void validate_eigenvalues_sqrt_f64(const double* t, double* result, unsigned int n, double eps) {
    validate_eigenvalues_impl<double>(t, result, n, eps, 1);
}

// Diagonal exp
__global__ void diagonal_exp_f32(const float* t, float* f, unsigned int n, float eps) {
    diagonal_exp_impl<float>(t, f, n, eps);
}

__global__ void diagonal_exp_f64(const double* t, double* f, unsigned int n, double eps) {
    diagonal_exp_impl<double>(t, f, n, eps);
}

// Diagonal log
__global__ void diagonal_log_f32(const float* t, float* f, unsigned int n, float eps) {
    diagonal_log_impl<float>(t, f, n, eps);
}

__global__ void diagonal_log_f64(const double* t, double* f, unsigned int n, double eps) {
    diagonal_log_impl<double>(t, f, n, eps);
}

// Diagonal sqrt
__global__ void diagonal_sqrt_f32(const float* t, float* f, unsigned int n, float eps) {
    diagonal_sqrt_impl<float>(t, f, n, eps);
}

__global__ void diagonal_sqrt_f64(const double* t, double* f, unsigned int n, double eps) {
    diagonal_sqrt_impl<double>(t, f, n, eps);
}

// Parlett column computation
__global__ void parlett_column_f32(const float* t, float* f, unsigned int n, unsigned int col, float eps) {
    parlett_column_impl<float>(t, f, n, col, eps);
}

__global__ void parlett_column_f64(const double* t, double* f, unsigned int n, unsigned int col, double eps) {
    parlett_column_impl<double>(t, f, n, col, eps);
}

} // extern "C"
