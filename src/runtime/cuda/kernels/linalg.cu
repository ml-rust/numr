// Linear Algebra CUDA kernels
// Native implementations without cuSOLVER dependency
// Supports: LU, Cholesky, QR decomposition, triangular solves, inverse, det, trace, diag
// Types: f32, f64

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" {

// ============================================================================
// Trace - Sum of diagonal elements (parallel reduction)
// ============================================================================

__global__ void trace_f32(const float* __restrict__ a, float* __restrict__ out,
                          unsigned int n, unsigned int stride) {
    extern __shared__ float sdata_f32[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one diagonal element (if in bounds)
    float val = 0.0f;
    if (i < n) {
        val = a[i * stride + i];
    }
    sdata_f32[tid] = val;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_f32[tid] += sdata_f32[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        atomicAdd(out, sdata_f32[0]);
    }
}

__global__ void trace_f64(const double* __restrict__ a, double* __restrict__ out,
                          unsigned int n, unsigned int stride) {
    extern __shared__ double sdata_f64[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (i < n) {
        val = a[i * stride + i];
    }
    sdata_f64[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_f64[tid] += sdata_f64[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // atomicAdd for double requires compute capability 6.0+
        atomicAdd(out, sdata_f64[0]);
    }
}

// ============================================================================
// Diag - Extract diagonal elements
// ============================================================================

__global__ void diag_f32(const float* __restrict__ a, float* __restrict__ out,
                         unsigned int min_dim, unsigned int n_cols) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < min_dim) {
        out[i] = a[i * n_cols + i];
    }
}

__global__ void diag_f64(const double* __restrict__ a, double* __restrict__ out,
                         unsigned int min_dim, unsigned int n_cols) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < min_dim) {
        out[i] = a[i * n_cols + i];
    }
}

// ============================================================================
// Diagflat - Create diagonal matrix from vector
// ============================================================================

__global__ void diagflat_f32(const float* __restrict__ diag, float* __restrict__ out,
                              unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * n;
    if (idx < total) {
        unsigned int row = idx / n;
        unsigned int col = idx % n;
        out[idx] = (row == col) ? diag[row] : 0.0f;
    }
}

__global__ void diagflat_f64(const double* __restrict__ diag, double* __restrict__ out,
                              unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * n;
    if (idx < total) {
        unsigned int row = idx / n;
        unsigned int col = idx % n;
        out[idx] = (row == col) ? diag[row] : 0.0;
    }
}

// ============================================================================
// Forward substitution: Solve Lx = b where L is lower triangular
// Single-thread implementation for correctness (sequential algorithm)
// ============================================================================

__global__ void forward_sub_f32(const float* __restrict__ l,
                                 const float* __restrict__ b,
                                 float* __restrict__ x,
                                 unsigned int n, int unit_diag) {
    // Sequential algorithm - single thread
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (unsigned int i = 0; i < n; i++) {
        float sum = b[i];
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

__global__ void forward_sub_f64(const double* __restrict__ l,
                                 const double* __restrict__ b,
                                 double* __restrict__ x,
                                 unsigned int n, int unit_diag) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (unsigned int i = 0; i < n; i++) {
        double sum = b[i];
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

__global__ void backward_sub_f32(const float* __restrict__ u,
                                  const float* __restrict__ b,
                                  float* __restrict__ x,
                                  unsigned int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int i = (int)n - 1; i >= 0; i--) {
        float sum = b[i];
        for (unsigned int j = i + 1; j < n; j++) {
            sum -= u[i * n + j] * x[j];
        }
        x[i] = sum / u[i * n + i];
    }
}

__global__ void backward_sub_f64(const double* __restrict__ u,
                                  const double* __restrict__ b,
                                  double* __restrict__ x,
                                  unsigned int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int i = (int)n - 1; i >= 0; i--) {
        double sum = b[i];
        for (unsigned int j = i + 1; j < n; j++) {
            sum -= u[i * n + j] * x[j];
        }
        x[i] = sum / u[i * n + i];
    }
}

// ============================================================================
// LU Decomposition with partial pivoting (Doolittle algorithm)
// Single-thread implementation - modifies matrix in-place
// ============================================================================

__global__ void lu_decompose_f32(float* __restrict__ lu,
                                  long long* __restrict__ pivots,
                                  int* __restrict__ num_swaps,
                                  unsigned int m, unsigned int n,
                                  int* __restrict__ singular_flag) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int k = (m < n) ? m : n;
    int swaps = 0;
    const float eps = 1e-10f;

    for (unsigned int col = 0; col < k; col++) {
        // Find pivot (max absolute value in column col, rows col to m-1)
        unsigned int pivot_row = col;
        float max_val = fabsf(lu[col * n + col]);

        for (unsigned int row = col + 1; row < m; row++) {
            float val = fabsf(lu[row * n + col]);
            if (val > max_val) {
                max_val = val;
                pivot_row = row;
            }
        }

        pivots[col] = (long long)pivot_row;

        // Swap rows if needed
        if (pivot_row != col) {
            for (unsigned int j = 0; j < n; j++) {
                float tmp = lu[col * n + j];
                lu[col * n + j] = lu[pivot_row * n + j];
                lu[pivot_row * n + j] = tmp;
            }
            swaps++;
        }

        // Check for singularity
        float pivot = lu[col * n + col];
        if (fabsf(pivot) < eps) {
            *singular_flag = 1;
            return;
        }

        // Compute multipliers (L column)
        for (unsigned int row = col + 1; row < m; row++) {
            lu[row * n + col] /= pivot;
        }

        // Update trailing submatrix
        for (unsigned int row = col + 1; row < m; row++) {
            float multiplier = lu[row * n + col];
            for (unsigned int j = col + 1; j < n; j++) {
                lu[row * n + j] -= multiplier * lu[col * n + j];
            }
        }
    }

    *num_swaps = swaps;
}

__global__ void lu_decompose_f64(double* __restrict__ lu,
                                  long long* __restrict__ pivots,
                                  int* __restrict__ num_swaps,
                                  unsigned int m, unsigned int n,
                                  int* __restrict__ singular_flag) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int k = (m < n) ? m : n;
    int swaps = 0;
    const double eps = 1e-15;

    for (unsigned int col = 0; col < k; col++) {
        unsigned int pivot_row = col;
        double max_val = fabs(lu[col * n + col]);

        for (unsigned int row = col + 1; row < m; row++) {
            double val = fabs(lu[row * n + col]);
            if (val > max_val) {
                max_val = val;
                pivot_row = row;
            }
        }

        pivots[col] = (long long)pivot_row;

        if (pivot_row != col) {
            for (unsigned int j = 0; j < n; j++) {
                double tmp = lu[col * n + j];
                lu[col * n + j] = lu[pivot_row * n + j];
                lu[pivot_row * n + j] = tmp;
            }
            swaps++;
        }

        double pivot = lu[col * n + col];
        if (fabs(pivot) < eps) {
            *singular_flag = 1;
            return;
        }

        for (unsigned int row = col + 1; row < m; row++) {
            lu[row * n + col] /= pivot;
        }

        for (unsigned int row = col + 1; row < m; row++) {
            double multiplier = lu[row * n + col];
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

__global__ void cholesky_decompose_f32(float* __restrict__ l,
                                        unsigned int n,
                                        int* __restrict__ not_positive_definite) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (unsigned int i = 0; i < n; i++) {
        // Compute diagonal element
        float sum_sq = 0.0f;
        for (unsigned int k = 0; k < i; k++) {
            float val = l[i * n + k];
            sum_sq += val * val;
        }

        float diag = l[i * n + i] - sum_sq;
        if (diag <= 0.0f) {
            *not_positive_definite = 1;
            return;
        }
        l[i * n + i] = sqrtf(diag);

        // Compute off-diagonal elements in column i
        for (unsigned int j = i + 1; j < n; j++) {
            float sum_prod = 0.0f;
            for (unsigned int k = 0; k < i; k++) {
                sum_prod += l[j * n + k] * l[i * n + k];
            }
            l[j * n + i] = (l[j * n + i] - sum_prod) / l[i * n + i];
        }

        // Zero out upper triangle
        for (unsigned int j = i + 1; j < n; j++) {
            l[i * n + j] = 0.0f;
        }
    }
}

__global__ void cholesky_decompose_f64(double* __restrict__ l,
                                        unsigned int n,
                                        int* __restrict__ not_positive_definite) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (unsigned int i = 0; i < n; i++) {
        double sum_sq = 0.0;
        for (unsigned int k = 0; k < i; k++) {
            double val = l[i * n + k];
            sum_sq += val * val;
        }

        double diag = l[i * n + i] - sum_sq;
        if (diag <= 0.0) {
            *not_positive_definite = 1;
            return;
        }
        l[i * n + i] = sqrt(diag);

        for (unsigned int j = i + 1; j < n; j++) {
            double sum_prod = 0.0;
            for (unsigned int k = 0; k < i; k++) {
                sum_prod += l[j * n + k] * l[i * n + k];
            }
            l[j * n + i] = (l[j * n + i] - sum_prod) / l[i * n + i];
        }

        for (unsigned int j = i + 1; j < n; j++) {
            l[i * n + j] = 0.0;
        }
    }
}

// ============================================================================
// QR Decomposition using Householder reflections
// A = Q @ R where Q is orthogonal, R is upper triangular
// Uses workspace buffer for Householder vector (no size limit)
// ============================================================================

__global__ void qr_decompose_f32(float* __restrict__ q,
                                  float* __restrict__ r,
                                  float* __restrict__ workspace,
                                  unsigned int m, unsigned int n,
                                  int thin) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int k = (m < n) ? m : n;
    unsigned int q_cols = thin ? k : m;

    // Initialize Q to identity
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < q_cols; j++) {
            q[i * q_cols + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Use workspace for Householder vector v (no size limit)
    float* v = workspace;

    for (unsigned int col = 0; col < k; col++) {
        // Compute norm of column below diagonal
        float norm_sq = 0.0f;
        for (unsigned int i = col; i < m; i++) {
            float val = r[i * n + col];
            norm_sq += val * val;
        }
        float norm = sqrtf(norm_sq);

        if (norm < 1e-10f) continue;

        // Compute Householder vector
        float alpha = (r[col * n + col] >= 0.0f) ? -norm : norm;

        // v = x - alpha * e1
        float v0 = r[col * n + col] - alpha;
        float v_norm_sq = v0 * v0;
        for (unsigned int i = col + 1; i < m; i++) {
            v_norm_sq += r[i * n + col] * r[i * n + col];
        }
        float v_norm = sqrtf(v_norm_sq);

        if (v_norm < 1e-10f) continue;

        // Normalize v
        v[0] = v0 / v_norm;
        for (unsigned int i = col + 1; i < m; i++) {
            v[i - col] = r[i * n + col] / v_norm;
        }
        unsigned int v_len = m - col;

        // Apply reflection to R: R[col:m, col:n] -= 2 * v * (v^T @ R[col:m, col:n])
        for (unsigned int j = col; j < n; j++) {
            float dot = 0.0f;
            for (unsigned int i = 0; i < v_len; i++) {
                dot += v[i] * r[(col + i) * n + j];
            }
            for (unsigned int i = 0; i < v_len; i++) {
                r[(col + i) * n + j] -= 2.0f * v[i] * dot;
            }
        }

        // Apply reflection to Q: Q[:, col:m] -= 2 * Q[:, col:m] @ v @ v^T
        for (unsigned int i = 0; i < m; i++) {
            float dot = 0.0f;
            unsigned int max_jj = v_len;
            if (col + max_jj > q_cols) max_jj = q_cols - col;
            for (unsigned int jj = 0; jj < max_jj; jj++) {
                dot += q[i * q_cols + col + jj] * v[jj];
            }
            for (unsigned int jj = 0; jj < max_jj; jj++) {
                q[i * q_cols + col + jj] -= 2.0f * dot * v[jj];
            }
        }
    }
}

__global__ void qr_decompose_f64(double* __restrict__ q,
                                  double* __restrict__ r,
                                  double* __restrict__ workspace,
                                  unsigned int m, unsigned int n,
                                  int thin) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int k = (m < n) ? m : n;
    unsigned int q_cols = thin ? k : m;

    // Initialize Q to identity
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < q_cols; j++) {
            q[i * q_cols + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Use workspace for Householder vector v
    double* v = workspace;

    for (unsigned int col = 0; col < k; col++) {
        double norm_sq = 0.0;
        for (unsigned int i = col; i < m; i++) {
            double val = r[i * n + col];
            norm_sq += val * val;
        }
        double norm = sqrt(norm_sq);

        if (norm < 1e-15) continue;

        double alpha = (r[col * n + col] >= 0.0) ? -norm : norm;

        double v0 = r[col * n + col] - alpha;
        double v_norm_sq = v0 * v0;
        for (unsigned int i = col + 1; i < m; i++) {
            v_norm_sq += r[i * n + col] * r[i * n + col];
        }
        double v_norm = sqrt(v_norm_sq);

        if (v_norm < 1e-15) continue;

        v[0] = v0 / v_norm;
        for (unsigned int i = col + 1; i < m; i++) {
            v[i - col] = r[i * n + col] / v_norm;
        }
        unsigned int v_len = m - col;

        for (unsigned int j = col; j < n; j++) {
            double dot = 0.0;
            for (unsigned int i = 0; i < v_len; i++) {
                dot += v[i] * r[(col + i) * n + j];
            }
            for (unsigned int i = 0; i < v_len; i++) {
                r[(col + i) * n + j] -= 2.0 * v[i] * dot;
            }
        }

        for (unsigned int i = 0; i < m; i++) {
            double dot = 0.0;
            unsigned int max_jj = v_len;
            if (col + max_jj > q_cols) max_jj = q_cols - col;
            for (unsigned int jj = 0; jj < max_jj; jj++) {
                dot += q[i * q_cols + col + jj] * v[jj];
            }
            for (unsigned int jj = 0; jj < max_jj; jj++) {
                q[i * q_cols + col + jj] -= 2.0 * dot * v[jj];
            }
        }
    }
}

// ============================================================================
// Determinant via LU - compute product of diagonal elements
// ============================================================================

__global__ void det_from_lu_f32(const float* __restrict__ lu,
                                 float* __restrict__ det,
                                 unsigned int n, int num_swaps) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float result = (num_swaps % 2 == 0) ? 1.0f : -1.0f;
    for (unsigned int i = 0; i < n; i++) {
        result *= lu[i * n + i];
    }
    *det = result;
}

__global__ void det_from_lu_f64(const double* __restrict__ lu,
                                 double* __restrict__ det,
                                 unsigned int n, int num_swaps) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    double result = (num_swaps % 2 == 0) ? 1.0 : -1.0;
    for (unsigned int i = 0; i < n; i++) {
        result *= lu[i * n + i];
    }
    *det = result;
}

// ============================================================================
// Apply LU permutation to vector
// ============================================================================

__global__ void apply_lu_permutation_f32(const float* __restrict__ in,
                                          float* __restrict__ out,
                                          const long long* __restrict__ pivots,
                                          unsigned int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Copy input to output
    for (unsigned int i = 0; i < n; i++) {
        out[i] = in[i];
    }

    // Apply pivots
    for (unsigned int i = 0; i < n; i++) {
        unsigned int pivot = (unsigned int)pivots[i];
        if (pivot != i) {
            float tmp = out[i];
            out[i] = out[pivot];
            out[pivot] = tmp;
        }
    }
}

__global__ void apply_lu_permutation_f64(const double* __restrict__ in,
                                          double* __restrict__ out,
                                          const long long* __restrict__ pivots,
                                          unsigned int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (unsigned int i = 0; i < n; i++) {
        out[i] = in[i];
    }

    for (unsigned int i = 0; i < n; i++) {
        unsigned int pivot = (unsigned int)pivots[i];
        if (pivot != i) {
            double tmp = out[i];
            out[i] = out[pivot];
            out[pivot] = tmp;
        }
    }
}

// ============================================================================
// Matrix copy kernel
// ============================================================================

__global__ void matrix_copy_f32(const float* __restrict__ src,
                                 float* __restrict__ dst,
                                 unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

__global__ void matrix_copy_f64(const double* __restrict__ src,
                                 double* __restrict__ dst,
                                 unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// ============================================================================
// Scatter column - write vector to column of matrix (GPU-only inverse)
// ============================================================================

__global__ void scatter_column_f32(const float* __restrict__ vec,
                                    float* __restrict__ matrix,
                                    unsigned int n, unsigned int col) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // matrix[i, col] = vec[i]  (row-major: matrix[i * n + col])
        matrix[i * n + col] = vec[i];
    }
}

__global__ void scatter_column_f64(const double* __restrict__ vec,
                                    double* __restrict__ matrix,
                                    unsigned int n, unsigned int col) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        matrix[i * n + col] = vec[i];
    }
}

// ============================================================================
// Count above threshold - for matrix_rank (parallel reduction)
// ============================================================================

__global__ void count_above_threshold_f32(const float* __restrict__ values,
                                           unsigned int* __restrict__ count,
                                           unsigned int n, float threshold) {
    extern __shared__ unsigned int scount[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread counts 1 if abs(value) > threshold
    unsigned int local_count = 0;
    if (i < n && fabsf(values[i]) > threshold) {
        local_count = 1;
    }
    scount[tid] = local_count;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            scount[tid] += scount[tid + s];
        }
        __syncthreads();
    }

    // Atomic add to global count
    if (tid == 0) {
        atomicAdd(count, scount[0]);
    }
}

__global__ void count_above_threshold_f64(const double* __restrict__ values,
                                           unsigned int* __restrict__ count,
                                           unsigned int n, double threshold) {
    extern __shared__ unsigned int scount_64[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int local_count = 0;
    if (i < n && fabs(values[i]) > threshold) {
        local_count = 1;
    }
    scount_64[tid] = local_count;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            scount_64[tid] += scount_64[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(count, scount_64[0]);
    }
}

// ============================================================================
// Max absolute value (for computing tolerance in matrix_rank)
// ============================================================================

__global__ void max_abs_f32(const float* __restrict__ values,
                             float* __restrict__ max_val,
                             unsigned int n) {
    extern __shared__ float smax_f32[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_max = 0.0f;
    if (i < n) {
        local_max = fabsf(values[i]);
    }
    smax_f32[tid] = local_max;
    __syncthreads();

    // Parallel max reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smax_f32[tid + s] > smax_f32[tid]) {
                smax_f32[tid] = smax_f32[tid + s];
            }
        }
        __syncthreads();
    }

    // Atomic max (using atomicCAS for float)
    if (tid == 0) {
        float block_max = smax_f32[0];
        unsigned int* address_as_uint = (unsigned int*)max_val;
        unsigned int old = *address_as_uint, assumed;
        do {
            assumed = old;
            float old_val = __uint_as_float(assumed);
            float new_val = (block_max > old_val) ? block_max : old_val;
            old = atomicCAS(address_as_uint, assumed, __float_as_uint(new_val));
        } while (assumed != old);
    }
}

__global__ void max_abs_f64(const double* __restrict__ values,
                             double* __restrict__ max_val,
                             unsigned int n) {
    extern __shared__ double smax_f64[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_max = 0.0;
    if (i < n) {
        local_max = fabs(values[i]);
    }
    smax_f64[tid] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smax_f64[tid + s] > smax_f64[tid]) {
                smax_f64[tid] = smax_f64[tid + s];
            }
        }
        __syncthreads();
    }

    // Atomic max for double (using atomicCAS)
    if (tid == 0) {
        double block_max = smax_f64[0];
        unsigned long long* address_as_ull = (unsigned long long*)max_val;
        unsigned long long old = *address_as_ull, assumed;
        do {
            assumed = old;
            double old_val = __longlong_as_double(assumed);
            double new_val = (block_max > old_val) ? block_max : old_val;
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_val));
        } while (assumed != old);
    }
}

// ============================================================================
// Create identity matrix on GPU (for inverse initialization)
// ============================================================================

__global__ void create_identity_f32(float* __restrict__ out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * n;
    if (idx < total) {
        unsigned int row = idx / n;
        unsigned int col = idx % n;
        out[idx] = (row == col) ? 1.0f : 0.0f;
    }
}

__global__ void create_identity_f64(double* __restrict__ out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * n;
    if (idx < total) {
        unsigned int row = idx / n;
        unsigned int col = idx % n;
        out[idx] = (row == col) ? 1.0 : 0.0;
    }
}

// ============================================================================
// Extract column from matrix (for multi-RHS solve)
// ============================================================================

__global__ void extract_column_f32(const float* __restrict__ matrix,
                                    float* __restrict__ col_out,
                                    unsigned int m, unsigned int n_cols,
                                    unsigned int col) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        col_out[i] = matrix[i * n_cols + col];
    }
}

__global__ void extract_column_f64(const double* __restrict__ matrix,
                                    double* __restrict__ col_out,
                                    unsigned int m, unsigned int n_cols,
                                    unsigned int col) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        col_out[i] = matrix[i * n_cols + col];
    }
}

// ============================================================================
// SVD Decomposition using One-Sided Jacobi algorithm
// Single-thread implementation for backend parity with CPU
// ============================================================================

__global__ void svd_jacobi_f32(float* __restrict__ b,      // [work_m, work_n] input, becomes U columns after normalization
                                float* __restrict__ v,      // [work_n, work_n] accumulates V (output as V^T)
                                float* __restrict__ s,      // [work_n] singular values
                                unsigned int work_m, unsigned int work_n,
                                int* __restrict__ converged_flag) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const float eps = 1.192092896e-07f;  // FLT_EPSILON
    const float tol = (float)work_n * eps;
    const int max_sweeps = 30;

    // Initialize V as identity
    for (unsigned int i = 0; i < work_n; i++) {
        for (unsigned int j = 0; j < work_n; j++) {
            v[i * work_n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // One-Sided Jacobi iterations
    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        float off_diag_sum = 0.0f;

        // Process all column pairs (p, q) where p < q
        for (unsigned int p = 0; p < work_n; p++) {
            for (unsigned int q = p + 1; q < work_n; q++) {
                // Compute Gram matrix elements
                float a_pp = 0.0f, a_qq = 0.0f, a_pq = 0.0f;
                for (unsigned int i = 0; i < work_m; i++) {
                    float bp = b[i * work_n + p];
                    float bq = b[i * work_n + q];
                    a_pp += bp * bp;
                    a_qq += bq * bq;
                    a_pq += bp * bq;
                }

                off_diag_sum += a_pq * a_pq;

                // Skip if off-diagonal is essentially zero
                if (fabsf(a_pq) < tol * sqrtf(a_pp * a_qq)) {
                    continue;
                }

                // Compute Jacobi rotation parameters
                float tau_num = a_qq - a_pp;
                float tau_den = 2.0f * a_pq;

                float c, s_val;
                if (fabsf(tau_den) < 1e-30f) {
                    c = 1.0f;
                    s_val = 0.0f;
                } else {
                    float tau = tau_num / tau_den;
                    float t;
                    if (tau >= 0.0f) {
                        t = 1.0f / (tau + sqrtf(1.0f + tau * tau));
                    } else {
                        t = -1.0f / (-tau + sqrtf(1.0f + tau * tau));
                    }
                    c = 1.0f / sqrtf(1.0f + t * t);
                    s_val = t * c;
                }

                // Apply rotation to B columns
                for (unsigned int i = 0; i < work_m; i++) {
                    float bp = b[i * work_n + p];
                    float bq = b[i * work_n + q];
                    b[i * work_n + p] = c * bp - s_val * bq;
                    b[i * work_n + q] = s_val * bp + c * bq;
                }

                // Apply rotation to V columns
                for (unsigned int i = 0; i < work_n; i++) {
                    float vp = v[i * work_n + p];
                    float vq = v[i * work_n + q];
                    v[i * work_n + p] = c * vp - s_val * vq;
                    v[i * work_n + q] = s_val * vp + c * vq;
                }
            }
        }

        // Check convergence
        if (sqrtf(off_diag_sum) < tol) {
            *converged_flag = 1;
            break;
        }
    }

    // Extract singular values (column norms of B)
    for (unsigned int j = 0; j < work_n; j++) {
        float norm_sq = 0.0f;
        for (unsigned int i = 0; i < work_m; i++) {
            float val = b[i * work_n + j];
            norm_sq += val * val;
        }
        s[j] = sqrtf(norm_sq);

        // Normalize B column to get U column
        float norm = s[j];
        if (norm > eps) {
            for (unsigned int i = 0; i < work_m; i++) {
                b[i * work_n + j] /= norm;
            }
        }
    }
}

__global__ void svd_jacobi_f64(double* __restrict__ b,      // [work_m, work_n] input, becomes U columns
                                double* __restrict__ v,      // [work_n, work_n] accumulates V
                                double* __restrict__ s,      // [work_n] singular values
                                unsigned int work_m, unsigned int work_n,
                                int* __restrict__ converged_flag) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const double eps = 2.220446049250313e-16;  // DBL_EPSILON
    const double tol = (double)work_n * eps;
    const int max_sweeps = 30;

    // Initialize V as identity
    for (unsigned int i = 0; i < work_n; i++) {
        for (unsigned int j = 0; j < work_n; j++) {
            v[i * work_n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // One-Sided Jacobi iterations
    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        double off_diag_sum = 0.0;

        for (unsigned int p = 0; p < work_n; p++) {
            for (unsigned int q = p + 1; q < work_n; q++) {
                double a_pp = 0.0, a_qq = 0.0, a_pq = 0.0;
                for (unsigned int i = 0; i < work_m; i++) {
                    double bp = b[i * work_n + p];
                    double bq = b[i * work_n + q];
                    a_pp += bp * bp;
                    a_qq += bq * bq;
                    a_pq += bp * bq;
                }

                off_diag_sum += a_pq * a_pq;

                if (fabs(a_pq) < tol * sqrt(a_pp * a_qq)) {
                    continue;
                }

                double tau_num = a_qq - a_pp;
                double tau_den = 2.0 * a_pq;

                double c, s_val;
                if (fabs(tau_den) < 1e-300) {
                    c = 1.0;
                    s_val = 0.0;
                } else {
                    double tau = tau_num / tau_den;
                    double t;
                    if (tau >= 0.0) {
                        t = 1.0 / (tau + sqrt(1.0 + tau * tau));
                    } else {
                        t = -1.0 / (-tau + sqrt(1.0 + tau * tau));
                    }
                    c = 1.0 / sqrt(1.0 + t * t);
                    s_val = t * c;
                }

                for (unsigned int i = 0; i < work_m; i++) {
                    double bp = b[i * work_n + p];
                    double bq = b[i * work_n + q];
                    b[i * work_n + p] = c * bp - s_val * bq;
                    b[i * work_n + q] = s_val * bp + c * bq;
                }

                for (unsigned int i = 0; i < work_n; i++) {
                    double vp = v[i * work_n + p];
                    double vq = v[i * work_n + q];
                    v[i * work_n + p] = c * vp - s_val * vq;
                    v[i * work_n + q] = s_val * vp + c * vq;
                }
            }
        }

        if (sqrt(off_diag_sum) < tol) {
            *converged_flag = 1;
            break;
        }
    }

    for (unsigned int j = 0; j < work_n; j++) {
        double norm_sq = 0.0;
        for (unsigned int i = 0; i < work_m; i++) {
            double val = b[i * work_n + j];
            norm_sq += val * val;
        }
        s[j] = sqrt(norm_sq);

        double norm = s[j];
        if (norm > eps) {
            for (unsigned int i = 0; i < work_m; i++) {
                b[i * work_n + j] /= norm;
            }
        }
    }
}

// ============================================================================
// Matrix Transpose - Optimized with shared memory tiling
// ============================================================================
// Uses 32x32 tiles with shared memory to achieve coalesced memory access
// for both reads and writes, avoiding strided access patterns.

#define TILE_DIM 32
#define BLOCK_ROWS 8

// F32 transpose: out[j,i] = in[i,j]
// in: [rows, cols] row-major -> out: [cols, rows] row-major
__global__ void transpose_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    unsigned int rows,
    unsigned int cols
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    unsigned int x = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile from input with coalesced reads
    for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
        }
    }

    __syncthreads();

    // Write tile to output with coalesced writes (transposed indices)
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // Swap x,y for output
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && (y + j) < cols) {
            out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// F64 transpose: out[j,i] = in[i,j]
__global__ void transpose_f64(
    const double* __restrict__ in,
    double* __restrict__ out,
    unsigned int rows,
    unsigned int cols
) {
    __shared__ double tile[TILE_DIM][TILE_DIM + 1];

    unsigned int x = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile from input with coalesced reads
    for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
        }
    }

    __syncthreads();

    // Write tile to output with coalesced writes (transposed indices)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && (y + j) < cols) {
            out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ============================================================================
// Eigendecomposition for Symmetric Matrices using Jacobi Algorithm
// ============================================================================

// Jacobi eigenvalue algorithm for symmetric matrices (F32)
// Input: a [n x n] symmetric matrix (in-place working buffer)
// Output: eigenvalues [n], eigenvectors [n x n]
__global__ void eig_jacobi_symmetric_f32(
    float* __restrict__ a,           // [n, n] input matrix, becomes diagonal
    float* __restrict__ eigenvectors,// [n, n] output eigenvectors
    float* __restrict__ eigenvalues, // [n] output eigenvalues
    unsigned int n,
    int* __restrict__ converged_flag
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const float eps = 1.192092896e-07f;  // FLT_EPSILON
    const float tol = (float)n * eps;
    const int max_sweeps = 30;

    // Initialize eigenvector matrix as identity
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            eigenvectors[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Symmetrize input (use lower triangle)
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < i; j++) {
            float val = a[i * n + j];
            a[j * n + i] = val;
        }
    }

    // Jacobi iterations
    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        // Find maximum off-diagonal element
        float max_off_diag = 0.0f;
        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int j = i + 1; j < n; j++) {
                float val = fabsf(a[i * n + j]);
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
                float a_pq = a[p * n + q];

                // Skip if already essentially zero
                if (fabsf(a_pq) < tol) {
                    continue;
                }

                float a_pp = a[p * n + p];
                float a_qq = a[q * n + q];

                // Compute Jacobi rotation parameters
                float tau_num = a_qq - a_pp;
                float tau_den = 2.0f * a_pq;

                float c, s;
                if (fabsf(tau_den) < 1e-30f) {
                    c = 1.0f;
                    s = 0.0f;
                } else {
                    float tau = tau_num / tau_den;
                    float t;
                    if (tau >= 0.0f) {
                        t = 1.0f / (tau + sqrtf(1.0f + tau * tau));
                    } else {
                        t = -1.0f / (-tau + sqrtf(1.0f + tau * tau));
                    }
                    c = 1.0f / sqrtf(1.0f + t * t);
                    s = t * c;
                }

                // Apply Jacobi rotation: A' = J^T @ A @ J
                // Update rows and columns p and q
                for (unsigned int k = 0; k < n; k++) {
                    if (k != p && k != q) {
                        float a_kp = a[k * n + p];
                        float a_kq = a[k * n + q];

                        float new_kp = c * a_kp - s * a_kq;
                        float new_kq = s * a_kp + c * a_kq;

                        a[k * n + p] = new_kp;
                        a[p * n + k] = new_kp;
                        a[k * n + q] = new_kq;
                        a[q * n + k] = new_kq;
                    }
                }

                // Update diagonal elements
                float c2 = c * c;
                float s2 = s * s;
                float cs2 = 2.0f * c * s;

                float new_pp = c2 * a_pp - cs2 * a_pq + s2 * a_qq;
                float new_qq = s2 * a_pp + cs2 * a_pq + c2 * a_qq;

                a[p * n + p] = new_pp;
                a[q * n + q] = new_qq;
                a[p * n + q] = 0.0f;
                a[q * n + p] = 0.0f;

                // Update eigenvector matrix: V = V @ J
                for (unsigned int i = 0; i < n; i++) {
                    float v_ip = eigenvectors[i * n + p];
                    float v_iq = eigenvectors[i * n + q];

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

// Jacobi eigenvalue algorithm for symmetric matrices (F64)
__global__ void eig_jacobi_symmetric_f64(
    double* __restrict__ a,           // [n, n] input matrix, becomes diagonal
    double* __restrict__ eigenvectors,// [n, n] output eigenvectors
    double* __restrict__ eigenvalues, // [n] output eigenvalues
    unsigned int n,
    int* __restrict__ converged_flag
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const double eps = 2.220446049250313e-16;  // DBL_EPSILON
    const double tol = (double)n * eps;
    const int max_sweeps = 30;

    // Initialize eigenvector matrix as identity
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            eigenvectors[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Symmetrize input (use lower triangle)
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < i; j++) {
            double val = a[i * n + j];
            a[j * n + i] = val;
        }
    }

    // Jacobi iterations
    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        // Find maximum off-diagonal element
        double max_off_diag = 0.0;
        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int j = i + 1; j < n; j++) {
                double val = fabs(a[i * n + j]);
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
                double a_pq = a[p * n + q];

                // Skip if already essentially zero
                if (fabs(a_pq) < tol) {
                    continue;
                }

                double a_pp = a[p * n + p];
                double a_qq = a[q * n + q];

                // Compute Jacobi rotation parameters
                double tau_num = a_qq - a_pp;
                double tau_den = 2.0 * a_pq;

                double c, s;
                if (fabs(tau_den) < 1e-300) {
                    c = 1.0;
                    s = 0.0;
                } else {
                    double tau = tau_num / tau_den;
                    double t;
                    if (tau >= 0.0) {
                        t = 1.0 / (tau + sqrt(1.0 + tau * tau));
                    } else {
                        t = -1.0 / (-tau + sqrt(1.0 + tau * tau));
                    }
                    c = 1.0 / sqrt(1.0 + t * t);
                    s = t * c;
                }

                // Apply Jacobi rotation: A' = J^T @ A @ J
                for (unsigned int k = 0; k < n; k++) {
                    if (k != p && k != q) {
                        double a_kp = a[k * n + p];
                        double a_kq = a[k * n + q];

                        double new_kp = c * a_kp - s * a_kq;
                        double new_kq = s * a_kp + c * a_kq;

                        a[k * n + p] = new_kp;
                        a[p * n + k] = new_kp;
                        a[k * n + q] = new_kq;
                        a[q * n + k] = new_kq;
                    }
                }

                // Update diagonal elements
                double c2 = c * c;
                double s2 = s * s;
                double cs2 = 2.0 * c * s;

                double new_pp = c2 * a_pp - cs2 * a_pq + s2 * a_qq;
                double new_qq = s2 * a_pp + cs2 * a_pq + c2 * a_qq;

                a[p * n + p] = new_pp;
                a[q * n + q] = new_qq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Update eigenvector matrix: V = V @ J
                for (unsigned int i = 0; i < n; i++) {
                    double v_ip = eigenvectors[i * n + p];
                    double v_iq = eigenvectors[i * n + q];

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
// Schur Decomposition - Hessenberg reduction + QR iteration
// For general (non-symmetric) matrices: A = Z @ T @ Z^T
// T is quasi-upper-triangular (real Schur form), Z is orthogonal
// ============================================================================

// Schur decomposition for general matrices (F32)
// Input: a [n x n] matrix (in-place working buffer, becomes T)
// Output: z [n x n] orthogonal matrix, t [n x n] quasi-upper-triangular
__global__ void schur_decompose_f32(
    float* __restrict__ t,           // [n, n] input matrix, becomes quasi-triangular T
    float* __restrict__ z,           // [n, n] output orthogonal matrix Z
    unsigned int n,
    int* __restrict__ converged_flag
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const float eps = 1.192092896e-07f;  // FLT_EPSILON
    const int max_sweeps = 30 * (int)n;

    // Initialize Z as identity
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            z[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Step 1: Hessenberg reduction using Householder reflections
    // Reduces A to upper Hessenberg form (zeros below first subdiagonal)
    for (unsigned int k = 0; k < n - 2; k++) {
        // Compute Householder vector for column k, rows k+1 to n-1
        float norm_sq = 0.0f;
        for (unsigned int i = k + 1; i < n; i++) {
            float val = t[i * n + k];
            norm_sq += val * val;
        }

        if (norm_sq < eps) continue;

        float norm = sqrtf(norm_sq);
        float x0 = t[(k + 1) * n + k];
        float alpha = (x0 >= 0.0f) ? -norm : norm;

        // v = x - alpha * e1, then normalize
        float v0 = x0 - alpha;
        float v_norm_sq = v0 * v0;
        for (unsigned int i = k + 2; i < n; i++) {
            float val = t[i * n + k];
            v_norm_sq += val * val;
        }

        if (v_norm_sq < eps) continue;

        float v_norm = sqrtf(v_norm_sq);
        float v[256];  // Max matrix size 256x256
        v[0] = v0 / v_norm;
        for (unsigned int i = k + 2; i < n; i++) {
            v[i - k - 1] = t[i * n + k] / v_norm;
        }
        unsigned int v_len = n - k - 1;

        // Left multiply: T = (I - 2vv^T) @ T
        for (unsigned int j = 0; j < n; j++) {
            float dot = 0.0f;
            for (unsigned int i = 0; i < v_len; i++) {
                dot += v[i] * t[(k + 1 + i) * n + j];
            }
            for (unsigned int i = 0; i < v_len; i++) {
                t[(k + 1 + i) * n + j] -= 2.0f * v[i] * dot;
            }
        }

        // Right multiply: T = T @ (I - 2vv^T)
        for (unsigned int i = 0; i < n; i++) {
            float dot = 0.0f;
            for (unsigned int j = 0; j < v_len; j++) {
                dot += t[i * n + (k + 1 + j)] * v[j];
            }
            for (unsigned int j = 0; j < v_len; j++) {
                t[i * n + (k + 1 + j)] -= 2.0f * dot * v[j];
            }
        }

        // Accumulate Z: Z = Z @ (I - 2vv^T)
        for (unsigned int i = 0; i < n; i++) {
            float dot = 0.0f;
            for (unsigned int j = 0; j < v_len; j++) {
                dot += z[i * n + (k + 1 + j)] * v[j];
            }
            for (unsigned int j = 0; j < v_len; j++) {
                z[i * n + (k + 1 + j)] -= 2.0f * dot * v[j];
            }
        }
    }

    // Step 2: QR iteration with Wilkinson shift
    for (int iter = 0; iter < max_sweeps; iter++) {
        // Check convergence (all subdiagonals essentially zero)
        int converged = 1;
        for (unsigned int i = 0; i < n - 1; i++) {
            float h_ii = fabsf(t[i * n + i]);
            float h_ip1 = fabsf(t[(i + 1) * n + (i + 1)]);
            float threshold = eps * fmaxf(h_ii + h_ip1, 1.0f);
            if (fabsf(t[(i + 1) * n + i]) > threshold) {
                converged = 0;
                break;
            }
        }

        if (converged) {
            *converged_flag = 1;
            break;
        }

        // Compute Wilkinson shift from bottom 2x2 block
        float a = t[(n - 2) * n + (n - 2)];
        float b = t[(n - 2) * n + (n - 1)];
        float c = t[(n - 1) * n + (n - 2)];
        float d = t[(n - 1) * n + (n - 1)];

        float trace = a + d;
        float det = a * d - b * c;
        float disc = trace * trace - 4.0f * det;

        float shift;
        if (disc >= 0.0f) {
            float sqrt_disc = sqrtf(disc);
            float lambda1 = (trace + sqrt_disc) / 2.0f;
            float lambda2 = (trace - sqrt_disc) / 2.0f;
            shift = (fabsf(lambda1 - d) < fabsf(lambda2 - d)) ? lambda1 : lambda2;
        } else {
            shift = trace / 2.0f;
        }

        // Apply shift
        for (unsigned int i = 0; i < n; i++) {
            t[i * n + i] -= shift;
        }

        // QR step using Givens rotations
        for (unsigned int i = 0; i < n - 1; i++) {
            float a_val = t[i * n + i];
            float b_val = t[(i + 1) * n + i];

            if (fabsf(b_val) < eps) continue;

            float r = sqrtf(a_val * a_val + b_val * b_val);
            float cs = a_val / r;
            float sn = -b_val / r;

            // Left multiply (Q^T @ T)
            for (unsigned int j = 0; j < n; j++) {
                float t1 = t[i * n + j];
                float t2 = t[(i + 1) * n + j];
                t[i * n + j] = cs * t1 - sn * t2;
                t[(i + 1) * n + j] = sn * t1 + cs * t2;
            }

            // Right multiply (T @ Q)
            for (unsigned int k = 0; k < n; k++) {
                float t1 = t[k * n + i];
                float t2 = t[k * n + (i + 1)];
                t[k * n + i] = cs * t1 - sn * t2;
                t[k * n + (i + 1)] = sn * t1 + cs * t2;
            }

            // Accumulate Z
            for (unsigned int k = 0; k < n; k++) {
                float z1 = z[k * n + i];
                float z2 = z[k * n + (i + 1)];
                z[k * n + i] = cs * z1 - sn * z2;
                z[k * n + (i + 1)] = sn * z1 + cs * z2;
            }
        }

        // Remove shift
        for (unsigned int i = 0; i < n; i++) {
            t[i * n + i] += shift;
        }
    }

    // Clean up small subdiagonals
    for (unsigned int i = 0; i < n - 1; i++) {
        float h_ii = fabsf(t[i * n + i]);
        float h_ip1 = fabsf(t[(i + 1) * n + (i + 1)]);
        float threshold = eps * fmaxf(h_ii + h_ip1, 1.0f);
        if (fabsf(t[(i + 1) * n + i]) <= threshold) {
            t[(i + 1) * n + i] = 0.0f;
        }
    }

    // Clear strictly lower triangular (except first subdiagonal for 2x2 blocks)
    for (unsigned int i = 2; i < n; i++) {
        for (unsigned int j = 0; j < i - 1; j++) {
            t[i * n + j] = 0.0f;
        }
    }
}

// Schur decomposition for general matrices (F64)
__global__ void schur_decompose_f64(
    double* __restrict__ t,
    double* __restrict__ z,
    unsigned int n,
    int* __restrict__ converged_flag
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const double eps = 2.220446049250313e-16;  // DBL_EPSILON
    const int max_sweeps = 30 * (int)n;

    // Initialize Z as identity
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            z[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Step 1: Hessenberg reduction
    for (unsigned int k = 0; k < n - 2; k++) {
        double norm_sq = 0.0;
        for (unsigned int i = k + 1; i < n; i++) {
            double val = t[i * n + k];
            norm_sq += val * val;
        }

        if (norm_sq < eps) continue;

        double norm = sqrt(norm_sq);
        double x0 = t[(k + 1) * n + k];
        double alpha = (x0 >= 0.0) ? -norm : norm;

        double v0 = x0 - alpha;
        double v_norm_sq = v0 * v0;
        for (unsigned int i = k + 2; i < n; i++) {
            double val = t[i * n + k];
            v_norm_sq += val * val;
        }

        if (v_norm_sq < eps) continue;

        double v_norm = sqrt(v_norm_sq);
        double v[256];
        v[0] = v0 / v_norm;
        for (unsigned int i = k + 2; i < n; i++) {
            v[i - k - 1] = t[i * n + k] / v_norm;
        }
        unsigned int v_len = n - k - 1;

        // Left multiply
        for (unsigned int j = 0; j < n; j++) {
            double dot = 0.0;
            for (unsigned int i = 0; i < v_len; i++) {
                dot += v[i] * t[(k + 1 + i) * n + j];
            }
            for (unsigned int i = 0; i < v_len; i++) {
                t[(k + 1 + i) * n + j] -= 2.0 * v[i] * dot;
            }
        }

        // Right multiply
        for (unsigned int i = 0; i < n; i++) {
            double dot = 0.0;
            for (unsigned int j = 0; j < v_len; j++) {
                dot += t[i * n + (k + 1 + j)] * v[j];
            }
            for (unsigned int j = 0; j < v_len; j++) {
                t[i * n + (k + 1 + j)] -= 2.0 * dot * v[j];
            }
        }

        // Accumulate Z
        for (unsigned int i = 0; i < n; i++) {
            double dot = 0.0;
            for (unsigned int j = 0; j < v_len; j++) {
                dot += z[i * n + (k + 1 + j)] * v[j];
            }
            for (unsigned int j = 0; j < v_len; j++) {
                z[i * n + (k + 1 + j)] -= 2.0 * dot * v[j];
            }
        }
    }

    // Step 2: QR iteration with Wilkinson shift
    for (int iter = 0; iter < max_sweeps; iter++) {
        int converged = 1;
        for (unsigned int i = 0; i < n - 1; i++) {
            double h_ii = fabs(t[i * n + i]);
            double h_ip1 = fabs(t[(i + 1) * n + (i + 1)]);
            double threshold = eps * fmax(h_ii + h_ip1, 1.0);
            if (fabs(t[(i + 1) * n + i]) > threshold) {
                converged = 0;
                break;
            }
        }

        if (converged) {
            *converged_flag = 1;
            break;
        }

        double a = t[(n - 2) * n + (n - 2)];
        double b = t[(n - 2) * n + (n - 1)];
        double c = t[(n - 1) * n + (n - 2)];
        double d = t[(n - 1) * n + (n - 1)];

        double trace = a + d;
        double det = a * d - b * c;
        double disc = trace * trace - 4.0 * det;

        double shift;
        if (disc >= 0.0) {
            double sqrt_disc = sqrt(disc);
            double lambda1 = (trace + sqrt_disc) / 2.0;
            double lambda2 = (trace - sqrt_disc) / 2.0;
            shift = (fabs(lambda1 - d) < fabs(lambda2 - d)) ? lambda1 : lambda2;
        } else {
            shift = trace / 2.0;
        }

        for (unsigned int i = 0; i < n; i++) {
            t[i * n + i] -= shift;
        }

        for (unsigned int i = 0; i < n - 1; i++) {
            double a_val = t[i * n + i];
            double b_val = t[(i + 1) * n + i];

            if (fabs(b_val) < eps) continue;

            double r = sqrt(a_val * a_val + b_val * b_val);
            double cs = a_val / r;
            double sn = -b_val / r;

            for (unsigned int j = 0; j < n; j++) {
                double t1 = t[i * n + j];
                double t2 = t[(i + 1) * n + j];
                t[i * n + j] = cs * t1 - sn * t2;
                t[(i + 1) * n + j] = sn * t1 + cs * t2;
            }

            for (unsigned int k = 0; k < n; k++) {
                double t1 = t[k * n + i];
                double t2 = t[k * n + (i + 1)];
                t[k * n + i] = cs * t1 - sn * t2;
                t[k * n + (i + 1)] = sn * t1 + cs * t2;
            }

            for (unsigned int k = 0; k < n; k++) {
                double z1 = z[k * n + i];
                double z2 = z[k * n + (i + 1)];
                z[k * n + i] = cs * z1 - sn * z2;
                z[k * n + (i + 1)] = sn * z1 + cs * z2;
            }
        }

        for (unsigned int i = 0; i < n; i++) {
            t[i * n + i] += shift;
        }
    }

    // Clean up
    for (unsigned int i = 0; i < n - 1; i++) {
        double h_ii = fabs(t[i * n + i]);
        double h_ip1 = fabs(t[(i + 1) * n + (i + 1)]);
        double threshold = eps * fmax(h_ii + h_ip1, 1.0);
        if (fabs(t[(i + 1) * n + i]) <= threshold) {
            t[(i + 1) * n + i] = 0.0;
        }
    }

    for (unsigned int i = 2; i < n; i++) {
        for (unsigned int j = 0; j < i - 1; j++) {
            t[i * n + j] = 0.0;
        }
    }
}

// ============================================================================
// General Eigenvalue Decomposition - for non-symmetric matrices
// Uses Schur decomposition + back-substitution for eigenvectors
// Returns real and imaginary parts of eigenvalues and eigenvectors
// ============================================================================

// General eigenvalue decomposition (F32)
__global__ void eig_general_f32(
    float* __restrict__ t,              // [n, n] working buffer (becomes Schur form)
    float* __restrict__ z,              // [n, n] Schur vectors
    float* __restrict__ eval_real,      // [n] real part of eigenvalues
    float* __restrict__ eval_imag,      // [n] imaginary part of eigenvalues
    float* __restrict__ evec_real,      // [n, n] real part of eigenvectors
    float* __restrict__ evec_imag,      // [n, n] imaginary part of eigenvectors
    unsigned int n,
    int* __restrict__ converged_flag
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const float eps = 1.192092896e-07f;
    const int max_sweeps = 30 * (int)n;

    // Initialize Z as identity
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            z[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // === Schur decomposition (inline to avoid function call overhead) ===

    // Hessenberg reduction
    for (unsigned int k = 0; k < n - 2; k++) {
        float norm_sq = 0.0f;
        for (unsigned int i = k + 1; i < n; i++) {
            float val = t[i * n + k];
            norm_sq += val * val;
        }

        if (norm_sq < eps) continue;

        float norm = sqrtf(norm_sq);
        float x0 = t[(k + 1) * n + k];
        float alpha = (x0 >= 0.0f) ? -norm : norm;

        float v0 = x0 - alpha;
        float v_norm_sq = v0 * v0;
        for (unsigned int i = k + 2; i < n; i++) {
            float val = t[i * n + k];
            v_norm_sq += val * val;
        }

        if (v_norm_sq < eps) continue;

        float v_norm = sqrtf(v_norm_sq);
        float v[256];
        v[0] = v0 / v_norm;
        for (unsigned int i = k + 2; i < n; i++) {
            v[i - k - 1] = t[i * n + k] / v_norm;
        }
        unsigned int v_len = n - k - 1;

        for (unsigned int j = 0; j < n; j++) {
            float dot = 0.0f;
            for (unsigned int i = 0; i < v_len; i++) {
                dot += v[i] * t[(k + 1 + i) * n + j];
            }
            for (unsigned int i = 0; i < v_len; i++) {
                t[(k + 1 + i) * n + j] -= 2.0f * v[i] * dot;
            }
        }

        for (unsigned int i = 0; i < n; i++) {
            float dot = 0.0f;
            for (unsigned int jj = 0; jj < v_len; jj++) {
                dot += t[i * n + (k + 1 + jj)] * v[jj];
            }
            for (unsigned int jj = 0; jj < v_len; jj++) {
                t[i * n + (k + 1 + jj)] -= 2.0f * dot * v[jj];
            }
        }

        for (unsigned int i = 0; i < n; i++) {
            float dot = 0.0f;
            for (unsigned int jj = 0; jj < v_len; jj++) {
                dot += z[i * n + (k + 1 + jj)] * v[jj];
            }
            for (unsigned int jj = 0; jj < v_len; jj++) {
                z[i * n + (k + 1 + jj)] -= 2.0f * dot * v[jj];
            }
        }
    }

    // QR iteration
    for (int iter = 0; iter < max_sweeps; iter++) {
        int converged = 1;
        for (unsigned int i = 0; i < n - 1; i++) {
            float h_ii = fabsf(t[i * n + i]);
            float h_ip1 = fabsf(t[(i + 1) * n + (i + 1)]);
            float threshold = eps * fmaxf(h_ii + h_ip1, 1.0f);
            if (fabsf(t[(i + 1) * n + i]) > threshold) {
                converged = 0;
                break;
            }
        }

        if (converged) {
            *converged_flag = 1;
            break;
        }

        float a = t[(n - 2) * n + (n - 2)];
        float b = t[(n - 2) * n + (n - 1)];
        float c = t[(n - 1) * n + (n - 2)];
        float d = t[(n - 1) * n + (n - 1)];

        float trace = a + d;
        float det = a * d - b * c;
        float disc = trace * trace - 4.0f * det;

        float shift;
        if (disc >= 0.0f) {
            float sqrt_disc = sqrtf(disc);
            float lambda1 = (trace + sqrt_disc) / 2.0f;
            float lambda2 = (trace - sqrt_disc) / 2.0f;
            shift = (fabsf(lambda1 - d) < fabsf(lambda2 - d)) ? lambda1 : lambda2;
        } else {
            shift = trace / 2.0f;
        }

        for (unsigned int i = 0; i < n; i++) {
            t[i * n + i] -= shift;
        }

        for (unsigned int i = 0; i < n - 1; i++) {
            float a_val = t[i * n + i];
            float b_val = t[(i + 1) * n + i];

            if (fabsf(b_val) < eps) continue;

            float r = sqrtf(a_val * a_val + b_val * b_val);
            float cs = a_val / r;
            float sn = -b_val / r;

            for (unsigned int j = 0; j < n; j++) {
                float t1 = t[i * n + j];
                float t2 = t[(i + 1) * n + j];
                t[i * n + j] = cs * t1 - sn * t2;
                t[(i + 1) * n + j] = sn * t1 + cs * t2;
            }

            for (unsigned int kk = 0; kk < n; kk++) {
                float t1 = t[kk * n + i];
                float t2 = t[kk * n + (i + 1)];
                t[kk * n + i] = cs * t1 - sn * t2;
                t[kk * n + (i + 1)] = sn * t1 + cs * t2;
            }

            for (unsigned int kk = 0; kk < n; kk++) {
                float z1 = z[kk * n + i];
                float z2 = z[kk * n + (i + 1)];
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
        float h_ii = fabsf(t[i * n + i]);
        float h_ip1 = fabsf(t[(i + 1) * n + (i + 1)]);
        float threshold = eps * fmaxf(h_ii + h_ip1, 1.0f);
        if (fabsf(t[(i + 1) * n + i]) <= threshold) {
            t[(i + 1) * n + i] = 0.0f;
        }
    }

    for (unsigned int ii = 2; ii < n; ii++) {
        for (unsigned int jj = 0; jj < ii - 1; jj++) {
            t[ii * n + jj] = 0.0f;
        }
    }

    // === Extract eigenvalues from Schur form ===
    unsigned int i = 0;
    while (i < n) {
        if (i == n - 1) {
            eval_real[i] = t[i * n + i];
            eval_imag[i] = 0.0f;
            i++;
        } else {
            float subdiag = fabsf(t[(i + 1) * n + i]);
            float diag_scale = fabsf(t[i * n + i]) + fabsf(t[(i + 1) * n + (i + 1)]);
            float threshold = eps * fmaxf(diag_scale, 1.0f);

            if (subdiag > threshold) {
                // 2x2 block - complex conjugate pair
                float a_val = t[i * n + i];
                float b_val = t[i * n + (i + 1)];
                float c_val = t[(i + 1) * n + i];
                float d_val = t[(i + 1) * n + (i + 1)];

                float trace = a_val + d_val;
                float disc = (a_val - d_val) * (a_val - d_val) / 4.0f + b_val * c_val;

                if (disc < 0.0f) {
                    float real_part = trace / 2.0f;
                    float imag_part = sqrtf(-disc);
                    eval_real[i] = real_part;
                    eval_imag[i] = imag_part;
                    eval_real[i + 1] = real_part;
                    eval_imag[i + 1] = -imag_part;
                } else {
                    float sqrt_disc = sqrtf(disc);
                    eval_real[i] = trace / 2.0f + sqrt_disc;
                    eval_imag[i] = 0.0f;
                    eval_real[i + 1] = trace / 2.0f - sqrt_disc;
                    eval_imag[i + 1] = 0.0f;
                }
                i += 2;
            } else {
                eval_real[i] = t[i * n + i];
                eval_imag[i] = 0.0f;
                i++;
            }
        }
    }

    // === Compute eigenvectors via back-substitution ===
    // Working buffers for Schur eigenvectors
    float y_real[256];
    float y_imag[256];

    i = 0;
    while (i < n) {
        float imag = eval_imag[i];

        if (fabsf(imag) < eps) {
            // Real eigenvalue - back-substitution for (T - λI)y = 0
            float lambda = eval_real[i];

            for (unsigned int k = 0; k < n; k++) {
                y_real[k] = 0.0f;
                y_imag[k] = 0.0f;
            }
            y_real[i] = 1.0f;

            for (int k = (int)i - 1; k >= 0; k--) {
                float diag = t[k * n + k] - lambda;
                float rhs = 0.0f;
                for (unsigned int j = k + 1; j < n; j++) {
                    rhs -= t[k * n + j] * y_real[j];
                }
                if (fabsf(diag) > eps) {
                    y_real[k] = rhs / diag;
                } else {
                    y_real[k] = 0.0f;
                }
            }

            // Normalize
            float norm_sq = 0.0f;
            for (unsigned int k = 0; k < n; k++) {
                norm_sq += y_real[k] * y_real[k];
            }
            float norm = sqrtf(norm_sq);
            if (norm > eps) {
                for (unsigned int k = 0; k < n; k++) {
                    y_real[k] /= norm;
                }
            }

            // Transform by Z: evec = Z @ y
            for (unsigned int row = 0; row < n; row++) {
                float sum = 0.0f;
                for (unsigned int k = 0; k < n; k++) {
                    sum += z[row * n + k] * y_real[k];
                }
                evec_real[row * n + i] = sum;
                evec_imag[row * n + i] = 0.0f;
            }
            i++;
        } else {
            // Complex eigenvalue - solve for complex eigenvector
            float lambda_real = eval_real[i];
            float lambda_imag = eval_imag[i];

            for (unsigned int k = 0; k < n; k++) {
                y_real[k] = 0.0f;
                y_imag[k] = 0.0f;
            }

            // Initial vector from 2x2 block
            float a_val = t[i * n + i];
            float b_val = t[i * n + (i + 1)];
            y_real[i] = b_val;
            y_imag[i] = 0.0f;
            y_real[i + 1] = lambda_real - a_val;
            y_imag[i + 1] = lambda_imag;

            // Back-substitute
            for (int k = (int)i - 1; k >= 0; k--) {
                float diag_real = t[k * n + k] - lambda_real;
                float diag_imag = -lambda_imag;

                float rhs_real = 0.0f;
                float rhs_imag = 0.0f;

                for (unsigned int j = k + 1; j < n; j++) {
                    float t_kj = t[k * n + j];
                    rhs_real -= t_kj * y_real[j];
                    rhs_imag -= t_kj * y_imag[j];
                }

                float denom = diag_real * diag_real + diag_imag * diag_imag;
                if (denom > eps * eps) {
                    y_real[k] = (rhs_real * diag_real + rhs_imag * diag_imag) / denom;
                    y_imag[k] = (rhs_imag * diag_real - rhs_real * diag_imag) / denom;
                } else {
                    y_real[k] = 0.0f;
                    y_imag[k] = 0.0f;
                }
            }

            // Normalize
            float norm_sq = 0.0f;
            for (unsigned int k = 0; k < n; k++) {
                norm_sq += y_real[k] * y_real[k] + y_imag[k] * y_imag[k];
            }
            float norm = sqrtf(norm_sq);
            if (norm > eps) {
                for (unsigned int k = 0; k < n; k++) {
                    y_real[k] /= norm;
                    y_imag[k] /= norm;
                }
            }

            // Transform by Z
            for (unsigned int row = 0; row < n; row++) {
                float sum_real = 0.0f;
                float sum_imag = 0.0f;
                for (unsigned int k = 0; k < n; k++) {
                    float z_val = z[row * n + k];
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

// General eigenvalue decomposition (F64)
__global__ void eig_general_f64(
    double* __restrict__ t,
    double* __restrict__ z,
    double* __restrict__ eval_real,
    double* __restrict__ eval_imag,
    double* __restrict__ evec_real,
    double* __restrict__ evec_imag,
    unsigned int n,
    int* __restrict__ converged_flag
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const double eps = 2.220446049250313e-16;
    const int max_sweeps = 30 * (int)n;

    // Initialize Z as identity
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            z[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Hessenberg reduction
    for (unsigned int k = 0; k < n - 2; k++) {
        double norm_sq = 0.0;
        for (unsigned int i = k + 1; i < n; i++) {
            double val = t[i * n + k];
            norm_sq += val * val;
        }

        if (norm_sq < eps) continue;

        double norm = sqrt(norm_sq);
        double x0 = t[(k + 1) * n + k];
        double alpha = (x0 >= 0.0) ? -norm : norm;

        double v0 = x0 - alpha;
        double v_norm_sq = v0 * v0;
        for (unsigned int i = k + 2; i < n; i++) {
            double val = t[i * n + k];
            v_norm_sq += val * val;
        }

        if (v_norm_sq < eps) continue;

        double v_norm = sqrt(v_norm_sq);
        double v[256];
        v[0] = v0 / v_norm;
        for (unsigned int i = k + 2; i < n; i++) {
            v[i - k - 1] = t[i * n + k] / v_norm;
        }
        unsigned int v_len = n - k - 1;

        for (unsigned int j = 0; j < n; j++) {
            double dot = 0.0;
            for (unsigned int i = 0; i < v_len; i++) {
                dot += v[i] * t[(k + 1 + i) * n + j];
            }
            for (unsigned int i = 0; i < v_len; i++) {
                t[(k + 1 + i) * n + j] -= 2.0 * v[i] * dot;
            }
        }

        for (unsigned int i = 0; i < n; i++) {
            double dot = 0.0;
            for (unsigned int jj = 0; jj < v_len; jj++) {
                dot += t[i * n + (k + 1 + jj)] * v[jj];
            }
            for (unsigned int jj = 0; jj < v_len; jj++) {
                t[i * n + (k + 1 + jj)] -= 2.0 * dot * v[jj];
            }
        }

        for (unsigned int i = 0; i < n; i++) {
            double dot = 0.0;
            for (unsigned int jj = 0; jj < v_len; jj++) {
                dot += z[i * n + (k + 1 + jj)] * v[jj];
            }
            for (unsigned int jj = 0; jj < v_len; jj++) {
                z[i * n + (k + 1 + jj)] -= 2.0 * dot * v[jj];
            }
        }
    }

    // QR iteration
    for (int iter = 0; iter < max_sweeps; iter++) {
        int converged = 1;
        for (unsigned int i = 0; i < n - 1; i++) {
            double h_ii = fabs(t[i * n + i]);
            double h_ip1 = fabs(t[(i + 1) * n + (i + 1)]);
            double threshold = eps * fmax(h_ii + h_ip1, 1.0);
            if (fabs(t[(i + 1) * n + i]) > threshold) {
                converged = 0;
                break;
            }
        }

        if (converged) {
            *converged_flag = 1;
            break;
        }

        double a = t[(n - 2) * n + (n - 2)];
        double b = t[(n - 2) * n + (n - 1)];
        double c = t[(n - 1) * n + (n - 2)];
        double d = t[(n - 1) * n + (n - 1)];

        double trace = a + d;
        double det = a * d - b * c;
        double disc = trace * trace - 4.0 * det;

        double shift;
        if (disc >= 0.0) {
            double sqrt_disc = sqrt(disc);
            double lambda1 = (trace + sqrt_disc) / 2.0;
            double lambda2 = (trace - sqrt_disc) / 2.0;
            shift = (fabs(lambda1 - d) < fabs(lambda2 - d)) ? lambda1 : lambda2;
        } else {
            shift = trace / 2.0;
        }

        for (unsigned int i = 0; i < n; i++) {
            t[i * n + i] -= shift;
        }

        for (unsigned int i = 0; i < n - 1; i++) {
            double a_val = t[i * n + i];
            double b_val = t[(i + 1) * n + i];

            if (fabs(b_val) < eps) continue;

            double r = sqrt(a_val * a_val + b_val * b_val);
            double cs = a_val / r;
            double sn = -b_val / r;

            for (unsigned int j = 0; j < n; j++) {
                double t1 = t[i * n + j];
                double t2 = t[(i + 1) * n + j];
                t[i * n + j] = cs * t1 - sn * t2;
                t[(i + 1) * n + j] = sn * t1 + cs * t2;
            }

            for (unsigned int kk = 0; kk < n; kk++) {
                double t1 = t[kk * n + i];
                double t2 = t[kk * n + (i + 1)];
                t[kk * n + i] = cs * t1 - sn * t2;
                t[kk * n + (i + 1)] = sn * t1 + cs * t2;
            }

            for (unsigned int kk = 0; kk < n; kk++) {
                double z1 = z[kk * n + i];
                double z2 = z[kk * n + (i + 1)];
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
        double h_ii = fabs(t[i * n + i]);
        double h_ip1 = fabs(t[(i + 1) * n + (i + 1)]);
        double threshold = eps * fmax(h_ii + h_ip1, 1.0);
        if (fabs(t[(i + 1) * n + i]) <= threshold) {
            t[(i + 1) * n + i] = 0.0;
        }
    }

    for (unsigned int ii = 2; ii < n; ii++) {
        for (unsigned int jj = 0; jj < ii - 1; jj++) {
            t[ii * n + jj] = 0.0;
        }
    }

    // Extract eigenvalues
    unsigned int i = 0;
    while (i < n) {
        if (i == n - 1) {
            eval_real[i] = t[i * n + i];
            eval_imag[i] = 0.0;
            i++;
        } else {
            double subdiag = fabs(t[(i + 1) * n + i]);
            double diag_scale = fabs(t[i * n + i]) + fabs(t[(i + 1) * n + (i + 1)]);
            double threshold = eps * fmax(diag_scale, 1.0);

            if (subdiag > threshold) {
                double a_val = t[i * n + i];
                double b_val = t[i * n + (i + 1)];
                double c_val = t[(i + 1) * n + i];
                double d_val = t[(i + 1) * n + (i + 1)];

                double trace = a_val + d_val;
                double disc = (a_val - d_val) * (a_val - d_val) / 4.0 + b_val * c_val;

                if (disc < 0.0) {
                    double real_part = trace / 2.0;
                    double imag_part = sqrt(-disc);
                    eval_real[i] = real_part;
                    eval_imag[i] = imag_part;
                    eval_real[i + 1] = real_part;
                    eval_imag[i + 1] = -imag_part;
                } else {
                    double sqrt_disc = sqrt(disc);
                    eval_real[i] = trace / 2.0 + sqrt_disc;
                    eval_imag[i] = 0.0;
                    eval_real[i + 1] = trace / 2.0 - sqrt_disc;
                    eval_imag[i + 1] = 0.0;
                }
                i += 2;
            } else {
                eval_real[i] = t[i * n + i];
                eval_imag[i] = 0.0;
                i++;
            }
        }
    }

    // Compute eigenvectors
    double y_real[256];
    double y_imag[256];

    i = 0;
    while (i < n) {
        double imag = eval_imag[i];

        if (fabs(imag) < eps) {
            double lambda = eval_real[i];

            for (unsigned int k = 0; k < n; k++) {
                y_real[k] = 0.0;
                y_imag[k] = 0.0;
            }
            y_real[i] = 1.0;

            for (int k = (int)i - 1; k >= 0; k--) {
                double diag = t[k * n + k] - lambda;
                double rhs = 0.0;
                for (unsigned int j = k + 1; j < n; j++) {
                    rhs -= t[k * n + j] * y_real[j];
                }
                if (fabs(diag) > eps) {
                    y_real[k] = rhs / diag;
                } else {
                    y_real[k] = 0.0;
                }
            }

            double norm_sq = 0.0;
            for (unsigned int k = 0; k < n; k++) {
                norm_sq += y_real[k] * y_real[k];
            }
            double norm = sqrt(norm_sq);
            if (norm > eps) {
                for (unsigned int k = 0; k < n; k++) {
                    y_real[k] /= norm;
                }
            }

            for (unsigned int row = 0; row < n; row++) {
                double sum = 0.0;
                for (unsigned int k = 0; k < n; k++) {
                    sum += z[row * n + k] * y_real[k];
                }
                evec_real[row * n + i] = sum;
                evec_imag[row * n + i] = 0.0;
            }
            i++;
        } else {
            double lambda_real = eval_real[i];
            double lambda_imag = eval_imag[i];

            for (unsigned int k = 0; k < n; k++) {
                y_real[k] = 0.0;
                y_imag[k] = 0.0;
            }

            double a_val = t[i * n + i];
            double b_val = t[i * n + (i + 1)];
            y_real[i] = b_val;
            y_imag[i] = 0.0;
            y_real[i + 1] = lambda_real - a_val;
            y_imag[i + 1] = lambda_imag;

            for (int k = (int)i - 1; k >= 0; k--) {
                double diag_real = t[k * n + k] - lambda_real;
                double diag_imag = -lambda_imag;

                double rhs_real = 0.0;
                double rhs_imag = 0.0;

                for (unsigned int j = k + 1; j < n; j++) {
                    double t_kj = t[k * n + j];
                    rhs_real -= t_kj * y_real[j];
                    rhs_imag -= t_kj * y_imag[j];
                }

                double denom = diag_real * diag_real + diag_imag * diag_imag;
                if (denom > eps * eps) {
                    y_real[k] = (rhs_real * diag_real + rhs_imag * diag_imag) / denom;
                    y_imag[k] = (rhs_imag * diag_real - rhs_real * diag_imag) / denom;
                } else {
                    y_real[k] = 0.0;
                    y_imag[k] = 0.0;
                }
            }

            double norm_sq = 0.0;
            for (unsigned int k = 0; k < n; k++) {
                norm_sq += y_real[k] * y_real[k] + y_imag[k] * y_imag[k];
            }
            double norm = sqrt(norm_sq);
            if (norm > eps) {
                for (unsigned int k = 0; k < n; k++) {
                    y_real[k] /= norm;
                    y_imag[k] /= norm;
                }
            }

            for (unsigned int row = 0; row < n; row++) {
                double sum_real = 0.0;
                double sum_imag = 0.0;
                for (unsigned int k = 0; k < n; k++) {
                    double z_val = z[row * n + k];
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

} // extern "C"
