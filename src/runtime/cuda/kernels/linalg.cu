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

} // extern "C"
