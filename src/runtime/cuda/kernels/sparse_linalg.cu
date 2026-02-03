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
// Level-Scheduled Sparse Triangular Solve
// ============================================================================

// Forward substitution kernel for lower triangular solve: L·x = b
// Processes all rows in a single level in parallel
template<typename T>
__global__ void sparse_trsv_lower_level_kernel(
    const int* level_rows,       // Rows to process in this level
    int level_size,              // Number of rows in this level
    const int* row_ptrs,
    const int* col_indices,
    const T* values,
    const T* b,                  // Right-hand side
    T* x,                        // Solution (updated in-place)
    int n,
    bool unit_diagonal
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int row = level_rows[tid];
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    T sum = b[row];
    T diag = static_cast<T>(1);

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

// Backward substitution kernel for upper triangular solve: U·x = b
template<typename T>
__global__ void sparse_trsv_upper_level_kernel(
    const int* level_rows,       // Rows to process in this level
    int level_size,
    const int* row_ptrs,
    const int* col_indices,
    const T* values,
    const T* b,
    T* x,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int row = level_rows[tid];
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    T sum = b[row];
    T diag = static_cast<T>(1);

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
// Level-Scheduled ILU(0) Factorization
// ============================================================================

// ILU(0) level kernel: processes all rows in a single level
// Each row computes L[i,k] and updates U entries
template<typename T>
__global__ void ilu0_level_kernel(
    const int* level_rows,       // Rows to process in this level
    int level_size,
    const int* row_ptrs,
    const int* col_indices,
    T* values,                   // In-place factorization
    const int* diag_indices,     // Precomputed diagonal positions
    int n,
    T diagonal_shift
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
        T diag_val = values[diag_k];

        // Handle zero pivot
        if (fabs(diag_val) < 1e-15) {
            if (diagonal_shift > 0) {
                values[diag_k] = diagonal_shift;
                diag_val = diagonal_shift;
            }
            // Note: proper error handling would need atomic flag
        }

        // L[i,k] = A[i,k] / U[k,k]
        T l_ik = values[idx_ik] / diag_val;
        values[idx_ik] = l_ik;

        // Update row i for columns j > k
        int k_start = row_ptrs[k];
        int k_end = row_ptrs[k + 1];

        for (int idx_kj = k_start; idx_kj < k_end; idx_kj++) {
            int j = col_indices[idx_kj];
            if (j <= k) continue;

            // Find A[i,j] if it exists (zero fill-in constraint)
            // Linear search within row i
            for (int idx_ij = row_start; idx_ij < row_end; idx_ij++) {
                if (col_indices[idx_ij] == j) {
                    // A[i,j] = A[i,j] - L[i,k] * U[k,j]
                    values[idx_ij] -= l_ik * values[idx_kj];
                    break;
                }
                if (col_indices[idx_ij] > j) break;
            }
        }
    }
}

// ============================================================================
// Level-Scheduled IC(0) Factorization
// ============================================================================

// IC(0) level kernel: processes all rows in a single level
template<typename T>
__global__ void ic0_level_kernel(
    const int* level_rows,
    int level_size,
    const int* row_ptrs,
    const int* col_indices,
    T* values,
    const int* diag_indices,
    int n,
    T diagonal_shift
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
        T sum = values[idx_ik];

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
    T sum = values[diag_i] + diagonal_shift;

    for (int idx_ij = i_start; idx_ij < i_end; idx_ij++) {
        int j = col_indices[idx_ij];
        if (j >= i) break;
        sum -= values[idx_ij] * values[idx_ij];
    }

    if (sum <= 0) {
        sum = diagonal_shift > 0 ? diagonal_shift : static_cast<T>(1e-10);
    }

    values[diag_i] = sqrt(sum);
}

// ============================================================================
// Utility Kernels
// ============================================================================

// Find diagonal indices in CSR matrix
__global__ void find_diag_indices_kernel(
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

// Copy b to x (initialization for triangular solve)
template<typename T>
__global__ void copy_kernel(
    const T* src,
    T* dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// ============================================================================
// Extern "C" Wrappers for Rust FFI
// ============================================================================

// Block size for sparse linalg kernels
#define SPARSE_LINALG_BLOCK_SIZE 256

extern "C" {

// Sparse triangular solve - lower (forward substitution)
void sparse_trsv_lower_level_f32(
    const int* level_rows, int level_size,
    const int* row_ptrs, const int* col_indices, const float* values,
    const float* b, float* x, int n, bool unit_diagonal
) {
    int blocks = (level_size + SPARSE_LINALG_BLOCK_SIZE - 1) / SPARSE_LINALG_BLOCK_SIZE;
    if (blocks > 0) {
        sparse_trsv_lower_level_kernel<float><<<blocks, SPARSE_LINALG_BLOCK_SIZE>>>(
            level_rows, level_size, row_ptrs, col_indices, values, b, x, n, unit_diagonal
        );
    }
}

void sparse_trsv_lower_level_f64(
    const int* level_rows, int level_size,
    const int* row_ptrs, const int* col_indices, const double* values,
    const double* b, double* x, int n, bool unit_diagonal
) {
    int blocks = (level_size + SPARSE_LINALG_BLOCK_SIZE - 1) / SPARSE_LINALG_BLOCK_SIZE;
    if (blocks > 0) {
        sparse_trsv_lower_level_kernel<double><<<blocks, SPARSE_LINALG_BLOCK_SIZE>>>(
            level_rows, level_size, row_ptrs, col_indices, values, b, x, n, unit_diagonal
        );
    }
}

// Sparse triangular solve - upper (backward substitution)
void sparse_trsv_upper_level_f32(
    const int* level_rows, int level_size,
    const int* row_ptrs, const int* col_indices, const float* values,
    const float* b, float* x, int n
) {
    int blocks = (level_size + SPARSE_LINALG_BLOCK_SIZE - 1) / SPARSE_LINALG_BLOCK_SIZE;
    if (blocks > 0) {
        sparse_trsv_upper_level_kernel<float><<<blocks, SPARSE_LINALG_BLOCK_SIZE>>>(
            level_rows, level_size, row_ptrs, col_indices, values, b, x, n
        );
    }
}

void sparse_trsv_upper_level_f64(
    const int* level_rows, int level_size,
    const int* row_ptrs, const int* col_indices, const double* values,
    const double* b, double* x, int n
) {
    int blocks = (level_size + SPARSE_LINALG_BLOCK_SIZE - 1) / SPARSE_LINALG_BLOCK_SIZE;
    if (blocks > 0) {
        sparse_trsv_upper_level_kernel<double><<<blocks, SPARSE_LINALG_BLOCK_SIZE>>>(
            level_rows, level_size, row_ptrs, col_indices, values, b, x, n
        );
    }
}

// ILU(0) level kernel
void ilu0_level_f32(
    const int* level_rows, int level_size,
    const int* row_ptrs, const int* col_indices, float* values,
    const int* diag_indices, int n, float diagonal_shift
) {
    int blocks = (level_size + SPARSE_LINALG_BLOCK_SIZE - 1) / SPARSE_LINALG_BLOCK_SIZE;
    if (blocks > 0) {
        ilu0_level_kernel<float><<<blocks, SPARSE_LINALG_BLOCK_SIZE>>>(
            level_rows, level_size, row_ptrs, col_indices, values,
            diag_indices, n, diagonal_shift
        );
    }
}

void ilu0_level_f64(
    const int* level_rows, int level_size,
    const int* row_ptrs, const int* col_indices, double* values,
    const int* diag_indices, int n, double diagonal_shift
) {
    int blocks = (level_size + SPARSE_LINALG_BLOCK_SIZE - 1) / SPARSE_LINALG_BLOCK_SIZE;
    if (blocks > 0) {
        ilu0_level_kernel<double><<<blocks, SPARSE_LINALG_BLOCK_SIZE>>>(
            level_rows, level_size, row_ptrs, col_indices, values,
            diag_indices, n, diagonal_shift
        );
    }
}

// IC(0) level kernel
void ic0_level_f32(
    const int* level_rows, int level_size,
    const int* row_ptrs, const int* col_indices, float* values,
    const int* diag_indices, int n, float diagonal_shift
) {
    int blocks = (level_size + SPARSE_LINALG_BLOCK_SIZE - 1) / SPARSE_LINALG_BLOCK_SIZE;
    if (blocks > 0) {
        ic0_level_kernel<float><<<blocks, SPARSE_LINALG_BLOCK_SIZE>>>(
            level_rows, level_size, row_ptrs, col_indices, values,
            diag_indices, n, diagonal_shift
        );
    }
}

void ic0_level_f64(
    const int* level_rows, int level_size,
    const int* row_ptrs, const int* col_indices, double* values,
    const int* diag_indices, int n, double diagonal_shift
) {
    int blocks = (level_size + SPARSE_LINALG_BLOCK_SIZE - 1) / SPARSE_LINALG_BLOCK_SIZE;
    if (blocks > 0) {
        ic0_level_kernel<double><<<blocks, SPARSE_LINALG_BLOCK_SIZE>>>(
            level_rows, level_size, row_ptrs, col_indices, values,
            diag_indices, n, diagonal_shift
        );
    }
}

// Utility kernels
void find_diag_indices(
    const int* row_ptrs, const int* col_indices, int* diag_indices, int n
) {
    int blocks = (n + SPARSE_LINALG_BLOCK_SIZE - 1) / SPARSE_LINALG_BLOCK_SIZE;
    if (blocks > 0) {
        find_diag_indices_kernel<<<blocks, SPARSE_LINALG_BLOCK_SIZE>>>(
            row_ptrs, col_indices, diag_indices, n
        );
    }
}

void copy_f32(const float* src, float* dst, int n) {
    int blocks = (n + SPARSE_LINALG_BLOCK_SIZE - 1) / SPARSE_LINALG_BLOCK_SIZE;
    if (blocks > 0) {
        copy_kernel<float><<<blocks, SPARSE_LINALG_BLOCK_SIZE>>>(src, dst, n);
    }
}

void copy_f64(const double* src, double* dst, int n) {
    int blocks = (n + SPARSE_LINALG_BLOCK_SIZE - 1) / SPARSE_LINALG_BLOCK_SIZE;
    if (blocks > 0) {
        copy_kernel<double><<<blocks, SPARSE_LINALG_BLOCK_SIZE>>>(src, dst, n);
    }
}

} // extern "C"
