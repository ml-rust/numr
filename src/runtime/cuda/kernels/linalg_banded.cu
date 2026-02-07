// Banded linear system solvers CUDA kernels
// Includes: thomas_solve (tridiagonal), banded_lu_solve (general banded)
//
// Uses C++ templates to eliminate f32/f64 duplication.

#include "dtype_traits.cuh"

// ============================================================================
// Thomas Algorithm for Tridiagonal Systems (kl=1, ku=1)
// ============================================================================
//
// Solves Ax = b where A is tridiagonal stored in banded format:
// - ab[ku * n + j] = main diagonal A[j, j]
// - ab[(ku-1) * n + j + 1] = upper diagonal A[j, j+1]
// - ab[(ku+1) * n + j - 1] = lower diagonal A[j+1, j]
//
// Uses work buffer for c' and d' temporary arrays.

template<typename T>
__device__ void thomas_solve_impl(const T* __restrict__ ab,
                                   const T* __restrict__ b,
                                   T* __restrict__ x,
                                   T* __restrict__ work,
                                   unsigned int n,
                                   unsigned int ku) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if (n == 0) return;
    if (n == 1) {
        x[0] = b[0] / ab[ku * n];
        return;
    }

    // work[0..n] = c' (modified upper diagonal)
    // work[n..2n] = d' (modified rhs)
    T* c = work;
    T* d = work + n;

    // Forward elimination
    T main_0 = ab[ku * n + 0];
    T upper_0 = (ku > 0 && n > 1) ? ab[(ku - 1) * n + 1] : dtype_traits<T>::zero();

    T m0_inv = dtype_traits<T>::one() / main_0;
    c[0] = upper_0 * m0_inv;
    d[0] = b[0] * m0_inv;

    for (unsigned int i = 1; i < n; i++) {
        T main_diag = ab[ku * n + i];
        T lower_diag = ab[(ku + 1) * n + i - 1];
        T upper_diag = (ku > 0 && i + 1 < n) ? ab[(ku - 1) * n + i + 1] : dtype_traits<T>::zero();

        T denom = main_diag - lower_diag * c[i - 1];
        T denom_inv = dtype_traits<T>::one() / denom;

        c[i] = (i < n - 1) ? (upper_diag * denom_inv) : dtype_traits<T>::zero();
        d[i] = (b[i] - lower_diag * d[i - 1]) * denom_inv;
    }

    // Back substitution: x[i] = d'[i] - c'[i] * x[i+1]
    x[n - 1] = d[n - 1];
    for (int i = (int)n - 2; i >= 0; i--) {
        x[i] = d[i] - c[i] * x[i + 1];
    }
}

// ============================================================================
// Banded LU Solver with Partial Pivoting
// ============================================================================
//
// Solves Ax = b where A is general banded matrix stored in band format.
// work buffer layout: [work_rows * n] where work_rows = 2*kl + ku + 1
// Original ab[r, j] is copied to work[(kl + r) * n + j].
// Element A[i, j] lives at work[(kl + ku + i - j) * n + j].

template<typename T>
__device__ void banded_lu_solve_impl(const T* __restrict__ ab,
                                      const T* __restrict__ b,
                                      T* __restrict__ x,
                                      T* __restrict__ work,
                                      unsigned int n,
                                      unsigned int kl,
                                      unsigned int ku) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if (n == 0) return;

    unsigned int band_rows = kl + ku + 1;
    unsigned int work_rows = 2 * kl + ku + 1;

    // Copy band data into working storage: work[(kl+r)*n+j] = ab[r*n+j]
    for (unsigned int r = 0; r < band_rows; r++) {
        for (unsigned int j = 0; j < n; j++) {
            work[(kl + r) * n + j] = ab[r * n + j];
        }
    }

    // Copy b to x (used as rhs, modified in-place)
    for (unsigned int i = 0; i < n; i++) {
        x[i] = b[i];
    }

    // LU factorization with partial pivoting
    for (unsigned int k = 0; k < n; k++) {
        unsigned int max_row = k + kl + 1;
        if (max_row > n) max_row = n;

        // Find pivot in column k
        unsigned int pivot_row = k;
        T pivot_val = dtype_traits<T>::zero();

        for (unsigned int i = k; i < max_row; i++) {
            unsigned int row_idx = kl + ku + i - k;
            T val = work[row_idx * n + k];
            T abs_val = (val < dtype_traits<T>::zero()) ? (dtype_traits<T>::zero() - val) : val;
            if (abs_val > pivot_val) {
                pivot_val = abs_val;
                pivot_row = i;
            }
        }

        if (pivot_val == dtype_traits<T>::zero()) continue;  // singular column

        // Swap rows if needed
        if (pivot_row != k) {
            unsigned int j_start = (k > ku) ? (k - ku) : 0;
            unsigned int j_end = k + kl + ku + 1;
            if (j_end > n) j_end = n;

            for (unsigned int j = j_start; j < j_end; j++) {
                int idx_k = (int)(kl + ku + k) - (int)j;
                int idx_p = (int)(kl + ku + pivot_row) - (int)j;
                if (idx_k >= 0 && (unsigned int)idx_k < work_rows &&
                    idx_p >= 0 && (unsigned int)idx_p < work_rows) {
                    unsigned int a_pos = (unsigned int)idx_k * n + j;
                    unsigned int b_pos = (unsigned int)idx_p * n + j;
                    T tmp = work[a_pos];
                    work[a_pos] = work[b_pos];
                    work[b_pos] = tmp;
                }
            }
            // Swap rhs
            T tmp_rhs = x[k];
            x[k] = x[pivot_row];
            x[pivot_row] = tmp_rhs;
        }

        // Eliminate below pivot
        T diag_val = work[(kl + ku) * n + k];

        for (unsigned int i = k + 1; i < max_row; i++) {
            unsigned int sub_row = kl + ku + i - k;
            T factor = work[sub_row * n + k] / diag_val;
            work[sub_row * n + k] = factor;  // Store L factor

            // Update remaining elements in row i
            unsigned int col_end = k + ku + 1;
            if (col_end > n) col_end = n;
            for (unsigned int j = k + 1; j < col_end; j++) {
                unsigned int row_i_j = kl + ku + i - j;
                unsigned int row_k_j = kl + ku + k - j;
                if (row_i_j < work_rows && row_k_j < work_rows) {
                    work[row_i_j * n + j] -= factor * work[row_k_j * n + j];
                }
            }

            // Update rhs
            x[i] -= factor * x[k];
        }
    }

    // Back substitution
    for (int k = (int)n - 1; k >= 0; k--) {
        unsigned int col_end = (unsigned int)k + ku + 1;
        if (col_end > n) col_end = n;

        for (unsigned int j = (unsigned int)k + 1; j < col_end; j++) {
            unsigned int row_idx = kl + ku + (unsigned int)k - j;
            if (row_idx < work_rows) {
                x[k] -= work[row_idx * n + j] * x[j];
            }
        }
        T diag = work[(kl + ku) * n + (unsigned int)k];
        x[k] /= diag;
    }
}

// ============================================================================
// General Banded Solver (chooses thomas vs LU based on bandwidth)
// ============================================================================

template<typename T>
__device__ void banded_solve_impl(const T* __restrict__ ab,
                                   const T* __restrict__ b,
                                   T* __restrict__ x,
                                   T* __restrict__ work,
                                   unsigned int n,
                                   unsigned int kl,
                                   unsigned int ku) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if (kl == 1 && ku == 1) {
        // Thomas algorithm - uses work[0..2n] for c' and d'
        thomas_solve_impl<T>(ab, b, x, work, n, ku);
    } else {
        banded_lu_solve_impl<T>(ab, b, x, work, n, kl, ku);
    }
}

// ============================================================================
// Extern "C" wrappers for PTX export
// ============================================================================

extern "C" {

__global__ void banded_solve_f32(const float* ab, const float* b, float* x,
                                 float* work, unsigned int n, unsigned int kl,
                                 unsigned int ku) {
    banded_solve_impl<float>(ab, b, x, work, n, kl, ku);
}

__global__ void banded_solve_f64(const double* ab, const double* b, double* x,
                                 double* work, unsigned int n, unsigned int kl,
                                 unsigned int ku) {
    banded_solve_impl<double>(ab, b, x, work, n, kl, ku);
}

__global__ void banded_solve_f16(const __half* ab, const __half* b, __half* x,
                                 __half* work, unsigned int n, unsigned int kl,
                                 unsigned int ku) {
    banded_solve_impl<__half>(ab, b, x, work, n, kl, ku);
}

__global__ void banded_solve_bf16(const __nv_bfloat16* ab, const __nv_bfloat16* b,
                                  __nv_bfloat16* x, __nv_bfloat16* work,
                                  unsigned int n, unsigned int kl, unsigned int ku) {
    banded_solve_impl<__nv_bfloat16>(ab, b, x, work, n, kl, ku);
}

} // extern "C"
