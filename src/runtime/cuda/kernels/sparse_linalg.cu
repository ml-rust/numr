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
// Multi-RHS Sparse Triangular Solve
// Supports b and x with shape [n, nrhs] (row-major: b[row * nrhs + col])
// Each thread processes one (row, rhs_column) pair
// ============================================================================

__global__ void sparse_trsv_lower_level_multi_rhs_f32(
    const int* level_rows,       // Rows to process in this level
    int level_size,              // Number of rows in this level
    int nrhs,                    // Number of right-hand sides
    const int* row_ptrs,
    const int* col_indices,
    const float* values,
    const float* b,              // [n, nrhs] row-major
    float* x,                    // [n, nrhs] row-major
    int n,
    bool unit_diagonal
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_work = level_size * nrhs;
    if (tid >= total_work) return;

    int row_idx = tid / nrhs;
    int rhs_col = tid % nrhs;
    int row = level_rows[row_idx];

    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    float sum = b[row * nrhs + rhs_col];
    float diag = 1.0f;

    for (int idx = start; idx < end; idx++) {
        int col = col_indices[idx];
        if (col < row) {
            sum -= values[idx] * x[col * nrhs + rhs_col];
        } else if (col == row && !unit_diagonal) {
            diag = values[idx];
        }
    }

    if (!unit_diagonal) {
        sum /= diag;
    }

    x[row * nrhs + rhs_col] = sum;
}

__global__ void sparse_trsv_lower_level_multi_rhs_f64(
    const int* level_rows,
    int level_size,
    int nrhs,
    const int* row_ptrs,
    const int* col_indices,
    const double* values,
    const double* b,
    double* x,
    int n,
    bool unit_diagonal
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_work = level_size * nrhs;
    if (tid >= total_work) return;

    int row_idx = tid / nrhs;
    int rhs_col = tid % nrhs;
    int row = level_rows[row_idx];

    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    double sum = b[row * nrhs + rhs_col];
    double diag = 1.0;

    for (int idx = start; idx < end; idx++) {
        int col = col_indices[idx];
        if (col < row) {
            sum -= values[idx] * x[col * nrhs + rhs_col];
        } else if (col == row && !unit_diagonal) {
            diag = values[idx];
        }
    }

    if (!unit_diagonal) {
        sum /= diag;
    }

    x[row * nrhs + rhs_col] = sum;
}

__global__ void sparse_trsv_upper_level_multi_rhs_f32(
    const int* level_rows,
    int level_size,
    int nrhs,
    const int* row_ptrs,
    const int* col_indices,
    const float* values,
    const float* b,
    float* x,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_work = level_size * nrhs;
    if (tid >= total_work) return;

    int row_idx = tid / nrhs;
    int rhs_col = tid % nrhs;
    int row = level_rows[row_idx];

    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    float sum = b[row * nrhs + rhs_col];
    float diag = 1.0f;

    for (int idx = start; idx < end; idx++) {
        int col = col_indices[idx];
        if (col > row) {
            sum -= values[idx] * x[col * nrhs + rhs_col];
        } else if (col == row) {
            diag = values[idx];
        }
    }

    x[row * nrhs + rhs_col] = sum / diag;
}

__global__ void sparse_trsv_upper_level_multi_rhs_f64(
    const int* level_rows,
    int level_size,
    int nrhs,
    const int* row_ptrs,
    const int* col_indices,
    const double* values,
    const double* b,
    double* x,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_work = level_size * nrhs;
    if (tid >= total_work) return;

    int row_idx = tid / nrhs;
    int rhs_col = tid % nrhs;
    int row = level_rows[row_idx];

    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    double sum = b[row * nrhs + rhs_col];
    double diag = 1.0;

    for (int idx = start; idx < end; idx++) {
        int col = col_indices[idx];
        if (col > row) {
            sum -= values[idx] * x[col * nrhs + rhs_col];
        } else if (col == row) {
            diag = values[idx];
        }
    }

    x[row * nrhs + rhs_col] = sum / diag;
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

// ============================================================================
// LU Split Kernels - Scatter values from factored matrix to L and U
// ============================================================================

// Scatter values from source array to L and U arrays based on column comparison with row
// l_map[i] = destination index in l_values for source index i (or -1 if not in L)
// u_map[i] = destination index in u_values for source index i (or -1 if not in U)
__global__ void split_lu_scatter_f32(
    const float* src_values,
    float* l_values,
    float* u_values,
    const int* l_map,
    const int* u_map,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    float val = src_values[idx];
    int l_dest = l_map[idx];
    int u_dest = u_map[idx];

    if (l_dest >= 0) {
        l_values[l_dest] = val;
    }
    if (u_dest >= 0) {
        u_values[u_dest] = val;
    }
}

__global__ void split_lu_scatter_f64(
    const double* src_values,
    double* l_values,
    double* u_values,
    const int* l_map,
    const int* u_map,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    double val = src_values[idx];
    int l_dest = l_map[idx];
    int u_dest = u_map[idx];

    if (l_dest >= 0) {
        l_values[l_dest] = val;
    }
    if (u_dest >= 0) {
        u_values[u_dest] = val;
    }
}

// ============================================================================
// Lower Triangle Extraction Kernel - For IC(0) decomposition
// ============================================================================

// Scatter values from source array to lower triangular output
// lower_map[i] = destination index in output for source index i (or -1 if not in lower)
__global__ void extract_lower_scatter_f32(
    const float* src_values,
    float* dst_values,
    const int* lower_map,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    int dest = lower_map[idx];
    if (dest >= 0) {
        dst_values[dest] = src_values[idx];
    }
}

__global__ void extract_lower_scatter_f64(
    const double* src_values,
    double* dst_values,
    const int* lower_map,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    int dest = lower_map[idx];
    if (dest >= 0) {
        dst_values[dest] = src_values[idx];
    }
}

// ============================================================================
// Sparse LU Kernels - Scatter, AXPY, Pivot, Gather
// ============================================================================

// Scatter sparse column into dense work vector
// work[row_indices[i]] = values[i]
__global__ void sparse_scatter_f32(
    const float* values,
    const int* row_indices,
    float* work,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    int row = row_indices[idx];
    work[row] = values[idx];
}

__global__ void sparse_scatter_f64(
    const double* values,
    const int* row_indices,
    double* work,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    int row = row_indices[idx];
    work[row] = values[idx];
}

// Sparse AXPY: work[row_indices[i]] -= scale * values[i]
// Uses atomic operations for thread safety
__global__ void sparse_axpy_f32(
    float scale,
    const float* values,
    const int* row_indices,
    float* work,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    int row = row_indices[idx];
    atomicAdd(&work[row], -scale * values[idx]);
}

__global__ void sparse_axpy_f64(
    double scale,
    const double* values,
    const int* row_indices,
    double* work,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    int row = row_indices[idx];
    // Note: atomicAdd for double requires compute capability >= 6.0
    atomicAdd(&work[row], -scale * values[idx]);
}

// Find pivot - parallel reduction to find max absolute value
// Returns index of max in output[0], max value in output[1]
__global__ void sparse_find_pivot_f32(
    const float* work,
    int start,
    int end,
    int* out_idx,
    float* out_val
) {
    extern __shared__ char shared_mem[];
    float* sdata = (float*)shared_mem;
    int* sidx = (int*)(shared_mem + blockDim.x * sizeof(float));

    int tid = threadIdx.x;
    int i = start + blockIdx.x * blockDim.x + tid;

    // Load into shared memory
    if (i < end) {
        sdata[tid] = fabsf(work[i]);
        sidx[tid] = i;
    } else {
        sdata[tid] = 0.0f;
        sidx[tid] = start;
    }
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        atomicMax(out_idx, sidx[0]); // This is a simplification - proper implementation needed
        // For actual use, need multi-block reduction with proper max tracking
    }
}

__global__ void sparse_find_pivot_f64(
    const double* work,
    int start,
    int end,
    int* out_idx,
    double* out_val
) {
    extern __shared__ char shared_mem[];
    double* sdata = (double*)shared_mem;
    int* sidx = (int*)(shared_mem + blockDim.x * sizeof(double));

    int tid = threadIdx.x;
    int i = start + blockIdx.x * blockDim.x + tid;

    if (i < end) {
        sdata[tid] = fabs(work[i]);
        sidx[tid] = i;
    } else {
        sdata[tid] = 0.0;
        sidx[tid] = start;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Simplified - proper multi-block reduction needed
        out_idx[blockIdx.x] = sidx[0];
        out_val[blockIdx.x] = sdata[0];
    }
}

// Gather from work vector to output and clear
// output[i] = work[row_indices[i]], then work[row_indices[i]] = 0
__global__ void sparse_gather_clear_f32(
    float* work,
    const int* row_indices,
    float* output,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    int row = row_indices[idx];
    output[idx] = work[row];
    work[row] = 0.0f;
}

__global__ void sparse_gather_clear_f64(
    double* work,
    const int* row_indices,
    double* output,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    int row = row_indices[idx];
    output[idx] = work[row];
    work[row] = 0.0;
}

// Divide work vector elements by pivot
// work[row_indices[i]] /= pivot
__global__ void sparse_divide_pivot_f32(
    float* work,
    const int* row_indices,
    float inv_pivot,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    int row = row_indices[idx];
    work[row] *= inv_pivot;
}

__global__ void sparse_divide_pivot_f64(
    double* work,
    const int* row_indices,
    double inv_pivot,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    int row = row_indices[idx];
    work[row] *= inv_pivot;
}

// Clear work vector at specific indices
__global__ void sparse_clear_f32(
    float* work,
    const int* row_indices,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    int row = row_indices[idx];
    work[row] = 0.0f;
}

__global__ void sparse_clear_f64(
    double* work,
    const int* row_indices,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    int row = row_indices[idx];
    work[row] = 0.0;
}

// ============================================================================
// CSC Triangular Solve Kernels (for LU solve with CSC factors)
// ============================================================================

// CSC Lower Triangular Solve - Forward Substitution
// Processes columns in level order. For each column j:
// x[j] = b[j] / L[j,j], then b[i] -= L[i,j] * x[j] for i > j
//
// Uses level scheduling: columns in the same level can be processed in parallel.
// level_cols contains column indices to process in this level.
// diag_ptr[j] = index of diagonal L[j,j] in values array
__global__ void sparse_trsv_csc_lower_level_f32(
    const int* level_cols,       // Columns to process in this level
    int level_size,              // Number of columns in this level
    const int* col_ptrs,         // CSC column pointers
    const int* row_indices,      // CSC row indices
    const float* l_values,       // L values
    const int* diag_ptr,         // Index of diagonal for each column
    float* b,                    // RHS (modified in place, becomes x)
    int n,
    bool unit_diagonal
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int col = level_cols[tid];
    int col_start = col_ptrs[col];
    int col_end = col_ptrs[col + 1];

    // Get diagonal value
    float diag = 1.0f;
    if (!unit_diagonal) {
        int diag_idx = diag_ptr[col];
        if (diag_idx >= 0) {
            diag = l_values[diag_idx];
        }
    }

    // x[col] = b[col] / L[col,col]
    float x_col = b[col];
    if (!unit_diagonal && fabsf(diag) > 1e-15f) {
        x_col /= diag;
    }
    b[col] = x_col;

    // Update b[row] for rows below diagonal: b[row] -= L[row,col] * x[col]
    for (int idx = col_start; idx < col_end; idx++) {
        int row = row_indices[idx];
        if (row > col) {
            atomicAdd(&b[row], -l_values[idx] * x_col);
        }
    }
}

__global__ void sparse_trsv_csc_lower_level_f64(
    const int* level_cols,
    int level_size,
    const int* col_ptrs,
    const int* row_indices,
    const double* l_values,
    const int* diag_ptr,
    double* b,
    int n,
    bool unit_diagonal
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int col = level_cols[tid];
    int col_start = col_ptrs[col];
    int col_end = col_ptrs[col + 1];

    double diag = 1.0;
    if (!unit_diagonal) {
        int diag_idx = diag_ptr[col];
        if (diag_idx >= 0) {
            diag = l_values[diag_idx];
        }
    }

    double x_col = b[col];
    if (!unit_diagonal && fabs(diag) > 1e-15) {
        x_col /= diag;
    }
    b[col] = x_col;

    for (int idx = col_start; idx < col_end; idx++) {
        int row = row_indices[idx];
        if (row > col) {
            atomicAdd(&b[row], -l_values[idx] * x_col);
        }
    }
}

// CSC Upper Triangular Solve - Backward Substitution
// Processes columns in reverse level order. For each column j:
// x[j] = b[j] / U[j,j], then b[i] -= U[i,j] * x[j] for i < j
__global__ void sparse_trsv_csc_upper_level_f32(
    const int* level_cols,       // Columns to process in this level
    int level_size,              // Number of columns in this level
    const int* col_ptrs,         // CSC column pointers
    const int* row_indices,      // CSC row indices
    const float* u_values,       // U values
    const int* diag_ptr,         // Index of diagonal for each column
    float* b,                    // RHS (modified in place, becomes x)
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int col = level_cols[tid];
    int col_start = col_ptrs[col];
    int col_end = col_ptrs[col + 1];

    // Get diagonal value
    int diag_idx = diag_ptr[col];
    float diag = 1.0f;
    if (diag_idx >= 0) {
        diag = u_values[diag_idx];
    }

    // x[col] = b[col] / U[col,col]
    float x_col = b[col];
    if (fabsf(diag) > 1e-15f) {
        x_col /= diag;
    }
    b[col] = x_col;

    // Update b[row] for rows above diagonal: b[row] -= U[row,col] * x[col]
    for (int idx = col_start; idx < col_end; idx++) {
        int row = row_indices[idx];
        if (row < col) {
            atomicAdd(&b[row], -u_values[idx] * x_col);
        }
    }
}

__global__ void sparse_trsv_csc_upper_level_f64(
    const int* level_cols,
    int level_size,
    const int* col_ptrs,
    const int* row_indices,
    const double* u_values,
    const int* diag_ptr,
    double* b,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int col = level_cols[tid];
    int col_start = col_ptrs[col];
    int col_end = col_ptrs[col + 1];

    int diag_idx = diag_ptr[col];
    double diag = 1.0;
    if (diag_idx >= 0) {
        diag = u_values[diag_idx];
    }

    double x_col = b[col];
    if (fabs(diag) > 1e-15) {
        x_col /= diag;
    }
    b[col] = x_col;

    for (int idx = col_start; idx < col_end; idx++) {
        int row = row_indices[idx];
        if (row < col) {
            atomicAdd(&b[row], -u_values[idx] * x_col);
        }
    }
}

// Find diagonal indices in CSC matrix (diagonal is at column j, row j)
__global__ void find_diag_indices_csc(
    const int* col_ptrs,
    const int* row_indices,
    int* diag_ptr,
    int n
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    int start = col_ptrs[col];
    int end = col_ptrs[col + 1];

    diag_ptr[col] = -1;  // Default: no diagonal found

    for (int idx = start; idx < end; idx++) {
        if (row_indices[idx] == col) {
            diag_ptr[col] = idx;
            break;
        }
    }
}

// Apply row permutation: y[i] = b[perm[i]]
__global__ void apply_row_perm_f32(
    const float* b,
    const int* perm,
    float* y,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    y[i] = b[perm[i]];
}

__global__ void apply_row_perm_f64(
    const double* b,
    const int* perm,
    double* y,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    y[i] = b[perm[i]];
}

} // extern "C"
