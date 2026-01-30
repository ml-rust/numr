// Matrix decomposition CUDA kernels
// Includes: LU (Doolittle), Cholesky (Banachiewicz), QR (Householder)
//
// Uses C++ templates to eliminate f32/f64 duplication.

#include "dtype_traits.cuh"

// ============================================================================
// LU Decomposition with partial pivoting (Doolittle algorithm)
// Single-thread implementation - modifies matrix in-place
// ============================================================================

template<typename T>
__device__ void lu_decompose_impl(T* __restrict__ lu,
                                  long long* __restrict__ pivots,
                                  int* __restrict__ num_swaps,
                                  unsigned int m, unsigned int n,
                                  int* __restrict__ singular_flag) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int k = (m < n) ? m : n;
    int swaps = 0;
    const T eps = dtype_traits<T>::small_eps();

    for (unsigned int col = 0; col < k; col++) {
        // Find pivot (max absolute value in column col, rows col to m-1)
        unsigned int pivot_row = col;
        T max_val = dtype_traits<T>::abs(lu[col * n + col]);

        for (unsigned int row = col + 1; row < m; row++) {
            T val = dtype_traits<T>::abs(lu[row * n + col]);
            if (val > max_val) {
                max_val = val;
                pivot_row = row;
            }
        }

        pivots[col] = (long long)pivot_row;

        // Swap rows if needed
        if (pivot_row != col) {
            for (unsigned int j = 0; j < n; j++) {
                T tmp = lu[col * n + j];
                lu[col * n + j] = lu[pivot_row * n + j];
                lu[pivot_row * n + j] = tmp;
            }
            swaps++;
        }

        // Check for singularity
        T pivot = lu[col * n + col];
        if (dtype_traits<T>::abs(pivot) < eps) {
            *singular_flag = 1;
            return;
        }

        // Compute multipliers (L column)
        for (unsigned int row = col + 1; row < m; row++) {
            lu[row * n + col] /= pivot;
        }

        // Update trailing submatrix
        for (unsigned int row = col + 1; row < m; row++) {
            T multiplier = lu[row * n + col];
            for (unsigned int j = col + 1; j < n; j++) {
                lu[row * n + j] -= multiplier * lu[col * n + j];
            }
        }
    }

    *num_swaps = swaps;
}

// ============================================================================
// Cholesky Decomposition (Cholesky-Banachiewicz algorithm)
// A = L @ L^T where L is lower triangular
// ============================================================================

template<typename T>
__device__ void cholesky_decompose_impl(T* __restrict__ l,
                                        unsigned int n,
                                        int* __restrict__ not_positive_definite) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (unsigned int i = 0; i < n; i++) {
        // Compute diagonal element
        T sum_sq = dtype_traits<T>::zero();
        for (unsigned int k = 0; k < i; k++) {
            T val = l[i * n + k];
            sum_sq += val * val;
        }

        T diag = l[i * n + i] - sum_sq;
        if (diag <= dtype_traits<T>::zero()) {
            *not_positive_definite = 1;
            return;
        }
        l[i * n + i] = dtype_traits<T>::sqrt(diag);

        // Compute off-diagonal elements in column i
        for (unsigned int j = i + 1; j < n; j++) {
            T sum_prod = dtype_traits<T>::zero();
            for (unsigned int k = 0; k < i; k++) {
                sum_prod += l[j * n + k] * l[i * n + k];
            }
            l[j * n + i] = (l[j * n + i] - sum_prod) / l[i * n + i];
        }

        // Zero out upper triangle
        for (unsigned int j = i + 1; j < n; j++) {
            l[i * n + j] = dtype_traits<T>::zero();
        }
    }
}

// ============================================================================
// QR Decomposition using Householder reflections
// A = Q @ R where Q is orthogonal, R is upper triangular
// ============================================================================

template<typename T>
__device__ void qr_decompose_impl(T* __restrict__ q,
                                  T* __restrict__ r,
                                  T* __restrict__ workspace,
                                  unsigned int m, unsigned int n,
                                  int thin) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int k = (m < n) ? m : n;
    unsigned int q_cols = thin ? k : m;
    const T small_eps = dtype_traits<T>::small_eps();

    // Initialize Q to identity
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < q_cols; j++) {
            q[i * q_cols + j] = (i == j) ? dtype_traits<T>::one() : dtype_traits<T>::zero();
        }
    }

    T* v = workspace;

    for (unsigned int col = 0; col < k; col++) {
        // Compute norm of column below diagonal
        T norm_sq = dtype_traits<T>::zero();
        for (unsigned int i = col; i < m; i++) {
            T val = r[i * n + col];
            norm_sq += val * val;
        }
        T norm = dtype_traits<T>::sqrt(norm_sq);

        if (norm < small_eps) continue;

        // Compute Householder vector
        T alpha = (r[col * n + col] >= dtype_traits<T>::zero()) ? -norm : norm;

        T v0 = r[col * n + col] - alpha;
        T v_norm_sq = v0 * v0;
        for (unsigned int i = col + 1; i < m; i++) {
            v_norm_sq += r[i * n + col] * r[i * n + col];
        }
        T v_norm = dtype_traits<T>::sqrt(v_norm_sq);

        if (v_norm < small_eps) continue;

        // Normalize v
        v[0] = v0 / v_norm;
        for (unsigned int i = col + 1; i < m; i++) {
            v[i - col] = r[i * n + col] / v_norm;
        }
        unsigned int v_len = m - col;

        // Apply reflection to R
        for (unsigned int j = col; j < n; j++) {
            T dot = dtype_traits<T>::zero();
            for (unsigned int i = 0; i < v_len; i++) {
                dot += v[i] * r[(col + i) * n + j];
            }
            for (unsigned int i = 0; i < v_len; i++) {
                r[(col + i) * n + j] -= dtype_traits<T>::two() * v[i] * dot;
            }
        }

        // Apply reflection to Q
        for (unsigned int i = 0; i < m; i++) {
            T dot = dtype_traits<T>::zero();
            unsigned int max_jj = v_len;
            if (col + max_jj > q_cols) max_jj = q_cols - col;
            for (unsigned int jj = 0; jj < max_jj; jj++) {
                dot += q[i * q_cols + col + jj] * v[jj];
            }
            for (unsigned int jj = 0; jj < max_jj; jj++) {
                q[i * q_cols + col + jj] -= dtype_traits<T>::two() * dot * v[jj];
            }
        }
    }
}

// ============================================================================
// Extern "C" wrappers for PTX export
// ============================================================================

extern "C" {

__global__ void lu_decompose_f32(float* lu, long long* pivots, int* num_swaps,
                                 unsigned int m, unsigned int n, int* singular_flag) {
    lu_decompose_impl<float>(lu, pivots, num_swaps, m, n, singular_flag);
}

__global__ void lu_decompose_f64(double* lu, long long* pivots, int* num_swaps,
                                 unsigned int m, unsigned int n, int* singular_flag) {
    lu_decompose_impl<double>(lu, pivots, num_swaps, m, n, singular_flag);
}

__global__ void cholesky_decompose_f32(float* l, unsigned int n, int* not_positive_definite) {
    cholesky_decompose_impl<float>(l, n, not_positive_definite);
}

__global__ void cholesky_decompose_f64(double* l, unsigned int n, int* not_positive_definite) {
    cholesky_decompose_impl<double>(l, n, not_positive_definite);
}

__global__ void qr_decompose_f32(float* q, float* r, float* workspace,
                                 unsigned int m, unsigned int n, int thin) {
    qr_decompose_impl<float>(q, r, workspace, m, n, thin);
}

__global__ void qr_decompose_f64(double* q, double* r, double* workspace,
                                 unsigned int m, unsigned int n, int thin) {
    qr_decompose_impl<double>(q, r, workspace, m, n, thin);
}

// ============================================================================
// F16 (__half) Wrappers
// ============================================================================

__global__ void lu_decompose_f16(__half* lu, long long* pivots, int* num_swaps,
                                 unsigned int m, unsigned int n, int* singular_flag) {
    lu_decompose_impl<__half>(lu, pivots, num_swaps, m, n, singular_flag);
}

__global__ void cholesky_decompose_f16(__half* l, unsigned int n, int* not_positive_definite) {
    cholesky_decompose_impl<__half>(l, n, not_positive_definite);
}

__global__ void qr_decompose_f16(__half* q, __half* r, __half* workspace,
                                 unsigned int m, unsigned int n, int thin) {
    qr_decompose_impl<__half>(q, r, workspace, m, n, thin);
}

// ============================================================================
// BF16 (__nv_bfloat16) Wrappers
// ============================================================================

__global__ void lu_decompose_bf16(__nv_bfloat16* lu, long long* pivots, int* num_swaps,
                                  unsigned int m, unsigned int n, int* singular_flag) {
    lu_decompose_impl<__nv_bfloat16>(lu, pivots, num_swaps, m, n, singular_flag);
}

__global__ void cholesky_decompose_bf16(__nv_bfloat16* l, unsigned int n, int* not_positive_definite) {
    cholesky_decompose_impl<__nv_bfloat16>(l, n, not_positive_definite);
}

__global__ void qr_decompose_bf16(__nv_bfloat16* q, __nv_bfloat16* r, __nv_bfloat16* workspace,
                                  unsigned int m, unsigned int n, int thin) {
    qr_decompose_impl<__nv_bfloat16>(q, r, workspace, m, n, thin);
}

} // extern "C"
