// Level-scheduled sparse linear algebra CUDA kernels
//
// These kernels implement level-scheduled parallel algorithms for sparse
// triangular operations. Rows within the same level are processed in parallel.
//
// Algorithms:
// - Sparse triangular solve (forward and backward substitution)
// - ILU(0) factorization with level scheduling
// - IC(0) factorization with level scheduling

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ============================================================================
// Extern "C" Kernels for Rust FFI (explicit implementations for each dtype)
// ============================================================================

extern "C" {

// ============================================================================
// Sparse Triangular Solve - Lower (Forward Substitution)
// ============================================================================

__global__ void sparse_trsv_lower_level_f32(
    const int* level_rows,       // Rows to process in this level
    int level_size,              // Number of rows in this level
    const int* row_ptrs,
    const int* col_indices,
    const float* values,
    const float* b,              // Right-hand side
    float* x,                    // Solution (updated in-place)
    int n,
    bool unit_diagonal
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int row = level_rows[tid];
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    float sum = b[row];
    float diag = 1.0f;

    for (int idx = start; idx < end; idx++) {
        int col = col_indices[idx];
        if (col < row) {
            sum -= values[idx] * x[col];
        } else if (col == row && !unit_diagonal) {
            diag = values[idx];
        }
    }

    if (!unit_diagonal) {
        sum /= diag;
    }

    x[row] = sum;
}

__global__ void sparse_trsv_lower_level_f64(
    const int* level_rows,
    int level_size,
    const int* row_ptrs,
    const int* col_indices,
    const double* values,
    const double* b,
    double* x,
    int n,
    bool unit_diagonal
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int row = level_rows[tid];
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    double sum = b[row];
    double diag = 1.0;

    for (int idx = start; idx < end; idx++) {
        int col = col_indices[idx];
        if (col < row) {
            sum -= values[idx] * x[col];
        } else if (col == row && !unit_diagonal) {
            diag = values[idx];
        }
    }

    if (!unit_diagonal) {
        sum /= diag;
    }

    x[row] = sum;
}

// ============================================================================
// Sparse Triangular Solve - Upper (Backward Substitution)
// ============================================================================

__global__ void sparse_trsv_upper_level_f32(
    const int* level_rows,
    int level_size,
    const int* row_ptrs,
    const int* col_indices,
    const float* values,
    const float* b,
    float* x,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int row = level_rows[tid];
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    float sum = b[row];
    float diag = 1.0f;

    for (int idx = start; idx < end; idx++) {
        int col = col_indices[idx];
        if (col > row) {
            sum -= values[idx] * x[col];
        } else if (col == row) {
            diag = values[idx];
        }
    }

    x[row] = sum / diag;
}

__global__ void sparse_trsv_upper_level_f64(
    const int* level_rows,
    int level_size,
    const int* row_ptrs,
    const int* col_indices,
    const double* values,
    const double* b,
    double* x,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int row = level_rows[tid];
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    double sum = b[row];
    double diag = 1.0;

    for (int idx = start; idx < end; idx++) {
        int col = col_indices[idx];
        if (col > row) {
            sum -= values[idx] * x[col];
        } else if (col == row) {
            diag = values[idx];
        }
    }

    x[row] = sum / diag;
}

// ============================================================================
// ILU(0) Level Kernels
// ============================================================================

__global__ void ilu0_level_f32(
    const int* level_rows,
    int level_size,
    const int* row_ptrs,
    const int* col_indices,
    float* values,
    const int* diag_indices,
    int n,
    float diagonal_shift
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int i = level_rows[tid];
    int row_start = row_ptrs[i];
    int row_end = row_ptrs[i + 1];

    // Process columns k < i (for L factor)
    for (int idx_ik = row_start; idx_ik < row_end; idx_ik++) {
        int k = col_indices[idx_ik];
        if (k >= i) break;

        // Get diagonal U[k,k]
        int diag_k = diag_indices[k];
        float diag_val = values[diag_k];

        // Handle zero pivot
        if (fabsf(diag_val) < 1e-15f) {
            if (diagonal_shift > 0) {
                values[diag_k] = diagonal_shift;
                diag_val = diagonal_shift;
            }
        }

        // L[i,k] = A[i,k] / U[k,k]
        float l_ik = values[idx_ik] / diag_val;
        values[idx_ik] = l_ik;

        // Update row i for columns j > k
        int k_start = row_ptrs[k];
        int k_end = row_ptrs[k + 1];

        for (int idx_kj = k_start; idx_kj < k_end; idx_kj++) {
            int j = col_indices[idx_kj];
            if (j <= k) continue;

            // Find A[i,j] if it exists
            for (int idx_ij = row_start; idx_ij < row_end; idx_ij++) {
                if (col_indices[idx_ij] == j) {
                    values[idx_ij] -= l_ik * values[idx_kj];
                    break;
                }
                if (col_indices[idx_ij] > j) break;
            }
        }
    }
}

__global__ void ilu0_level_f64(
    const int* level_rows,
    int level_size,
    const int* row_ptrs,
    const int* col_indices,
    double* values,
    const int* diag_indices,
    int n,
    double diagonal_shift
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int i = level_rows[tid];
    int row_start = row_ptrs[i];
    int row_end = row_ptrs[i + 1];

    // Process columns k < i (for L factor)
    for (int idx_ik = row_start; idx_ik < row_end; idx_ik++) {
        int k = col_indices[idx_ik];
        if (k >= i) break;

        // Get diagonal U[k,k]
        int diag_k = diag_indices[k];
        double diag_val = values[diag_k];

        // Handle zero pivot
        if (fabs(diag_val) < 1e-15) {
            if (diagonal_shift > 0) {
                values[diag_k] = diagonal_shift;
                diag_val = diagonal_shift;
            }
        }

        // L[i,k] = A[i,k] / U[k,k]
        double l_ik = values[idx_ik] / diag_val;
        values[idx_ik] = l_ik;

        // Update row i for columns j > k
        int k_start = row_ptrs[k];
        int k_end = row_ptrs[k + 1];

        for (int idx_kj = k_start; idx_kj < k_end; idx_kj++) {
            int j = col_indices[idx_kj];
            if (j <= k) continue;

            // Find A[i,j] if it exists
            for (int idx_ij = row_start; idx_ij < row_end; idx_ij++) {
                if (col_indices[idx_ij] == j) {
                    values[idx_ij] -= l_ik * values[idx_kj];
                    break;
                }
                if (col_indices[idx_ij] > j) break;
            }
        }
    }
}

// ============================================================================
// IC(0) Level Kernels
// ============================================================================

__global__ void ic0_level_f32(
    const int* level_rows,
    int level_size,
    const int* row_ptrs,
    const int* col_indices,
    float* values,
    const int* diag_indices,
    int n,
    float diagonal_shift
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int i = level_rows[tid];
    int i_start = row_ptrs[i];
    int i_end = row_ptrs[i + 1];

    // Process off-diagonal entries in row i (columns k < i)
    for (int idx_ik = i_start; idx_ik < i_end; idx_ik++) {
        int k = col_indices[idx_ik];
        if (k >= i) break;

        int k_start = row_ptrs[k];
        int k_end = row_ptrs[k + 1];

        // Compute inner product contribution
        float sum = values[idx_ik];

        for (int idx_kj = k_start; idx_kj < k_end; idx_kj++) {
            int j = col_indices[idx_kj];
            if (j >= k) break;

            // Check if L[i,j] exists
            for (int idx_ij = i_start; idx_ij < i_end; idx_ij++) {
                if (col_indices[idx_ij] == j) {
                    sum -= values[idx_ij] * values[idx_kj];
                    break;
                }
                if (col_indices[idx_ij] > j) break;
            }
        }

        // Divide by L[k,k]
        int diag_k = diag_indices[k];
        values[idx_ik] = sum / values[diag_k];
    }

    // Compute diagonal L[i,i]
    int diag_i = diag_indices[i];
    float sum = values[diag_i] + diagonal_shift;

    for (int idx_ij = i_start; idx_ij < i_end; idx_ij++) {
        int j = col_indices[idx_ij];
        if (j >= i) break;
        sum -= values[idx_ij] * values[idx_ij];
    }

    if (sum <= 0) {
        sum = diagonal_shift > 0 ? diagonal_shift : 1e-10f;
    }

    values[diag_i] = sqrtf(sum);
}

__global__ void ic0_level_f64(
    const int* level_rows,
    int level_size,
    const int* row_ptrs,
    const int* col_indices,
    double* values,
    const int* diag_indices,
    int n,
    double diagonal_shift
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int i = level_rows[tid];
    int i_start = row_ptrs[i];
    int i_end = row_ptrs[i + 1];

    // Process off-diagonal entries in row i (columns k < i)
    for (int idx_ik = i_start; idx_ik < i_end; idx_ik++) {
        int k = col_indices[idx_ik];
        if (k >= i) break;

        int k_start = row_ptrs[k];
        int k_end = row_ptrs[k + 1];

        // Compute inner product contribution
        double sum = values[idx_ik];

        for (int idx_kj = k_start; idx_kj < k_end; idx_kj++) {
            int j = col_indices[idx_kj];
            if (j >= k) break;

            // Check if L[i,j] exists
            for (int idx_ij = i_start; idx_ij < i_end; idx_ij++) {
                if (col_indices[idx_ij] == j) {
                    sum -= values[idx_ij] * values[idx_kj];
                    break;
                }
                if (col_indices[idx_ij] > j) break;
            }
        }

        // Divide by L[k,k]
        int diag_k = diag_indices[k];
        values[idx_ik] = sum / values[diag_k];
    }

    // Compute diagonal L[i,i]
    int diag_i = diag_indices[i];
    double sum = values[diag_i] + diagonal_shift;

    for (int idx_ij = i_start; idx_ij < i_end; idx_ij++) {
        int j = col_indices[idx_ij];
        if (j >= i) break;
        sum -= values[idx_ij] * values[idx_ij];
    }

    if (sum <= 0) {
        sum = diagonal_shift > 0 ? diagonal_shift : 1e-10;
    }

    values[diag_i] = sqrt(sum);
}

// ============================================================================
// Utility Kernels
// ============================================================================

__global__ void find_diag_indices(
    const int* row_ptrs,
    const int* col_indices,
    int* diag_indices,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    diag_indices[row] = -1;  // Default: no diagonal found

    for (int idx = start; idx < end; idx++) {
        if (col_indices[idx] == row) {
            diag_indices[row] = idx;
            break;
        }
    }
}

__global__ void copy_f32(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

__global__ void copy_f64(const double* src, double* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

} // extern "C"
