// GPU-native level computation and structural analysis kernels
// for sparse linear algebra (eliminates GPU↔CPU transfers)

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ============================================================================
// Type Casting Kernels
// ============================================================================

extern "C" {

/// Cast i64 values to i32 (for structure indices)
__global__ void cast_i64_to_i32(
    const long long* input,
    int* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (int)input[idx];
    }
}

// ============================================================================
// Level Computation Kernels (Iterative BFS)
// ============================================================================

/// Compute level schedule for lower triangular matrix (iterative)
/// Each thread processes one row: level[i] = max(level[j] for j in col_indices[i] where j < i) + 1
/// Returns changed flag in changed[0]
__global__ void compute_levels_lower_iter(
    const int* row_ptrs,
    const int* col_indices,
    int* levels,
    int* changed,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    int max_dep_level = -1;
    for (int idx = start; idx < end; idx++) {
        int col = col_indices[idx];
        if (col < row) {
            max_dep_level = max(max_dep_level, levels[col]);
        }
    }

    int new_level = max_dep_level + 1;
    if (levels[row] < new_level) {
        levels[row] = new_level;
        atomicExch(changed, 1);
    }
}

/// Compute level schedule for upper triangular matrix (iterative, backward)
/// Each thread processes one row: level[i] = max(level[j] for j in col_indices[i] where j > i) + 1
__global__ void compute_levels_upper_iter(
    const int* row_ptrs,
    const int* col_indices,
    int* levels,
    int* changed,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    int max_dep_level = -1;
    for (int idx = start; idx < end; idx++) {
        int col = col_indices[idx];
        if (col > row) {
            max_dep_level = max(max_dep_level, levels[col]);
        }
    }

    int new_level = max_dep_level + 1;
    if (levels[row] < new_level) {
        levels[row] = new_level;
        atomicExch(changed, 1);
    }
}

// ============================================================================
// Reduction Kernels
// ============================================================================

/// Reduce max over i32 array (max reduction)
/// blockIdx.x = 0, results stored in result[0]
__global__ void reduce_max_i32(
    const int* data,
    int* result,
    int n
) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? data[idx] : -1;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = sdata[0];
    }
}

// ============================================================================
// Histogram and Prefix Sum Helper Kernels
// ============================================================================

/// Count occurrences of each level
__global__ void histogram_levels(
    const int* levels,
    int* counts,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int level = levels[idx];
        atomicAdd(&counts[level], 1);
    }
}

/// Scatter rows by level into level_rows array
/// Requires level_ptrs (prefix sum of counts) to be pre-computed on CPU
__global__ void scatter_by_level(
    const int* levels,
    const int* level_ptrs,
    int* level_rows,
    int* level_counters,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int level = levels[row];
    int pos = level_ptrs[level] + atomicAdd(&level_counters[level], 1);
    level_rows[pos] = row;
}

// ============================================================================
// LU Splitting Kernels (Structure Analysis)
// ============================================================================

/// Classify each entry as L or U based on column < row or >= row
__global__ void classify_lu_entries(
    const int* row_ptrs,
    const int* col_indices,
    int* l_flags,
    int* u_flags,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    for (int idx = start; idx < end; idx++) {
        int col = col_indices[idx];
        if (col < row) {
            l_flags[idx] = 1;
            u_flags[idx] = 0;
        } else {
            l_flags[idx] = 0;
            u_flags[idx] = 1;
        }
    }
}

/// Count L and U entries per row
__global__ void count_lu_per_row(
    const int* row_ptrs,
    const int* col_indices,
    int* l_counts,
    int* u_counts,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    int l_count = 0, u_count = 0;
    for (int idx = start; idx < end; idx++) {
        int col = col_indices[idx];
        if (col < row) {
            l_count++;
        } else {
            u_count++;
        }
    }

    l_counts[row] = l_count;
    u_counts[row] = u_count;
}

/// Build mapping arrays for L/U split
/// Assumes l_row_ptrs and u_row_ptrs are pre-computed on CPU (via prefix sum)
__global__ void build_lu_maps(
    const int* row_ptrs,
    const int* col_indices,
    const int* l_row_ptrs,
    const int* u_row_ptrs,
    int* l_col_indices,
    int* u_col_indices,
    int* l_map,
    int* u_map,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    int l_idx = l_row_ptrs[row];
    int u_idx = u_row_ptrs[row];

    for (int idx = start; idx < end; idx++) {
        int col = col_indices[idx];
        if (col < row) {
            l_col_indices[l_idx] = col;
            l_map[idx] = l_idx;
            u_map[idx] = -1;
            l_idx++;
        } else {
            u_col_indices[u_idx] = col;
            u_map[idx] = u_idx;
            l_map[idx] = -1;
            u_idx++;
        }
    }
}

// ============================================================================
// Lower Triangle Extraction (for IC)
// ============================================================================

/// Count lower triangle entries per row (j <= i)
__global__ void count_lower_per_row(
    const int* row_ptrs,
    const int* col_indices,
    int* lower_counts,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    int count = 0;
    for (int idx = start; idx < end; idx++) {
        int col = col_indices[idx];
        if (col <= row) {
            count++;
        }
    }

    lower_counts[row] = count;
}

/// Build mapping for lower triangle extraction
/// Assumes lower_row_ptrs is pre-computed via prefix sum
__global__ void build_lower_map(
    const int* row_ptrs,
    const int* col_indices,
    const int* lower_row_ptrs,
    int* lower_col_indices,
    int* lower_map,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    int lower_idx = lower_row_ptrs[row];

    for (int idx = start; idx < end; idx++) {
        int col = col_indices[idx];
        if (col <= row) {
            lower_col_indices[lower_idx] = col;
            lower_map[idx] = lower_idx;
            lower_idx++;
        } else {
            lower_map[idx] = -1;
        }
    }
}

} // extern "C"
