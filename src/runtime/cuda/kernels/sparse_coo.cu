//! COO Sparse Matrix Element-wise Operations with GPU Sort
//!
//! Implements GPU-native COO element-wise operations using a sort-merge approach:
//! 1. Compute composite keys (row * ncols + col) for position identification
//! 2. Concatenate entries from both matrices
//! 3. Sort by composite key on GPU
//! 4. Merge duplicates using atomic operations
//!
//! # Algorithm Complexity
//!
//! - Time: O(n log n) for sort
//! - Space: O(n) for temporary buffers
//!
//! # Dependencies
//!
//! Requires CUB library (included with CUDA Toolkit 11.0+) and Thrust

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

// Block size for element-wise kernels
#define BLOCK_SIZE 256

// ============================================================================
// Key Computation Kernels
// ============================================================================

/// Compute composite keys from (row, col) pairs
/// key = row * ncols + col (linearized index in row-major order)
extern "C" __global__ void coo_compute_keys_i64(
    const long long* row_indices,
    const long long* col_indices,
    long long* keys_out,
    long long ncols,
    unsigned int nnz
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    keys_out[idx] = row_indices[idx] * ncols + col_indices[idx];
}

/// Extract (row, col) from composite keys
/// row = key / ncols, col = key % ncols
extern "C" __global__ void coo_extract_indices_i64(
    const long long* keys,
    long long* row_indices_out,
    long long* col_indices_out,
    long long ncols,
    unsigned int nnz
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    long long key = keys[idx];
    row_indices_out[idx] = key / ncols;
    col_indices_out[idx] = key % ncols;
}

// ============================================================================
// Concatenation Kernels
// ============================================================================

/// Concatenate two key arrays: [keys_a, keys_b]
extern "C" __global__ void coo_concat_keys_i64(
    const long long* keys_a,
    const long long* keys_b,
    long long* keys_out,
    unsigned int nnz_a,
    unsigned int nnz_b
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = nnz_a + nnz_b;
    if (idx >= total) return;

    if (idx < nnz_a) {
        keys_out[idx] = keys_a[idx];
    } else {
        keys_out[idx] = keys_b[idx - nnz_a];
    }
}

/// Concatenate two value arrays with source flags - f32
extern "C" __global__ void coo_concat_values_with_source_f32(
    const float* values_a,
    const float* values_b,
    float* values_out,
    int* source_out,
    unsigned int nnz_a,
    unsigned int nnz_b
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = nnz_a + nnz_b;
    if (idx >= total) return;

    if (idx < nnz_a) {
        values_out[idx] = values_a[idx];
        source_out[idx] = 0;  // From A
    } else {
        values_out[idx] = values_b[idx - nnz_a];
        source_out[idx] = 1;  // From B
    }
}

/// Concatenate two value arrays with source flags - f64
extern "C" __global__ void coo_concat_values_with_source_f64(
    const double* values_a,
    const double* values_b,
    double* values_out,
    int* source_out,
    unsigned int nnz_a,
    unsigned int nnz_b
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = nnz_a + nnz_b;
    if (idx >= total) return;

    if (idx < nnz_a) {
        values_out[idx] = values_a[idx];
        source_out[idx] = 0;  // From A
    } else {
        values_out[idx] = values_b[idx - nnz_a];
        source_out[idx] = 1;  // From B
    }
}

// ============================================================================
// Duplicate Detection and Counting
// ============================================================================

/// Mark unique positions (where key differs from previous)
/// unique_flags[i] = 1 if keys[i] != keys[i-1] (or i == 0)
extern "C" __global__ void coo_mark_unique_i64(
    const long long* keys,
    int* unique_flags,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (idx == 0) {
        unique_flags[idx] = 1;  // First element is always unique
    } else {
        unique_flags[idx] = (keys[idx] != keys[idx - 1]) ? 1 : 0;
    }
}

// ============================================================================
// Merge Operations (Union Semantics - Add/Sub)
// ============================================================================

/// Merge sorted entries with union semantics (add operation) - f32
extern "C" __global__ void coo_merge_add_f32(
    const long long* sorted_keys,
    const float* sorted_values,
    const int* source_flags,  // 0 = from A, 1 = from B
    const int* unique_positions,  // Exclusive scan of unique_flags
    long long* out_keys,
    float* out_values,
    unsigned int n,
    unsigned int num_unique
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int out_pos = unique_positions[idx];
    long long key = sorted_keys[idx];
    float value = sorted_values[idx];

    // Check if this is the first occurrence of this key
    bool is_first = (idx == 0) || (sorted_keys[idx] != sorted_keys[idx - 1]);

    if (is_first) {
        // Initialize output position
        out_keys[out_pos] = key;
        out_values[out_pos] = value;
    } else {
        // Add to existing value (duplicate key)
        atomicAdd(&out_values[out_pos], value);
    }
}

/// Merge sorted entries with union semantics (add operation) - f64
extern "C" __global__ void coo_merge_add_f64(
    const long long* sorted_keys,
    const double* sorted_values,
    const int* source_flags,
    const int* unique_positions,
    long long* out_keys,
    double* out_values,
    unsigned int n,
    unsigned int num_unique
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int out_pos = unique_positions[idx];
    long long key = sorted_keys[idx];
    double value = sorted_values[idx];

    bool is_first = (idx == 0) || (sorted_keys[idx] != sorted_keys[idx - 1]);

    if (is_first) {
        out_keys[out_pos] = key;
        out_values[out_pos] = value;
    } else {
        atomicAdd(&out_values[out_pos], value);
    }
}

/// Merge sorted entries with union semantics (subtract operation) - f32
/// For unique from B: result = -value
extern "C" __global__ void coo_merge_sub_f32(
    const long long* sorted_keys,
    const float* sorted_values,
    const int* source_flags,
    const int* unique_positions,
    long long* out_keys,
    float* out_values,
    unsigned int n,
    unsigned int num_unique
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int out_pos = unique_positions[idx];
    long long key = sorted_keys[idx];
    float value = sorted_values[idx];
    int source = source_flags[idx];

    // For subtraction: negate values from B
    float contrib = (source == 0) ? value : -value;

    bool is_first = (idx == 0) || (sorted_keys[idx] != sorted_keys[idx - 1]);

    if (is_first) {
        out_keys[out_pos] = key;
        out_values[out_pos] = contrib;
    } else {
        atomicAdd(&out_values[out_pos], contrib);
    }
}

/// Merge sorted entries with union semantics (subtract operation) - f64
extern "C" __global__ void coo_merge_sub_f64(
    const long long* sorted_keys,
    const double* sorted_values,
    const int* source_flags,
    const int* unique_positions,
    long long* out_keys,
    double* out_values,
    unsigned int n,
    unsigned int num_unique
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int out_pos = unique_positions[idx];
    long long key = sorted_keys[idx];
    double value = sorted_values[idx];
    int source = source_flags[idx];

    double contrib = (source == 0) ? value : -value;

    bool is_first = (idx == 0) || (sorted_keys[idx] != sorted_keys[idx - 1]);

    if (is_first) {
        out_keys[out_pos] = key;
        out_values[out_pos] = contrib;
    } else {
        atomicAdd(&out_values[out_pos], contrib);
    }
}

// ============================================================================
// Intersection Detection (for Mul/Div)
// ============================================================================

/// Mark positions that have both A and B entries for the same key
/// intersection_flags[i] = 1 if this is an A entry followed by B entry with same key
extern "C" __global__ void coo_count_intersections_i64(
    const long long* sorted_keys,
    const int* source_flags,
    int* intersection_flags,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Initialize to 0
    intersection_flags[idx] = 0;

    // Mark if this A entry has a following B entry with same key
    if (source_flags[idx] == 0 && idx + 1 < n) {
        if (sorted_keys[idx] == sorted_keys[idx + 1] && source_flags[idx + 1] == 1) {
            intersection_flags[idx] = 1;
        }
    }
}

// ============================================================================
// Merge Operations (Intersection Semantics - Mul/Div)
// ============================================================================

/// Merge sorted entries with intersection semantics (multiply operation) - f32
/// Only keep entries where both A and B have values
extern "C" __global__ void coo_merge_mul_f32(
    const long long* sorted_keys,
    const float* sorted_values,
    const int* source_flags,
    const int* intersection_flags,  // 1 if this A entry has matching B
    const int* output_positions,    // Exclusive scan of intersection_flags
    long long* out_keys,
    float* out_values,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Only process A entries that have matching B entries
    if (intersection_flags[idx] == 0) return;

    // This is an A entry with a matching B entry at idx+1
    int out_pos = output_positions[idx];
    long long key = sorted_keys[idx];
    float val_a = sorted_values[idx];
    float val_b = sorted_values[idx + 1];

    out_keys[out_pos] = key;
    out_values[out_pos] = val_a * val_b;
}

/// Merge sorted entries with intersection semantics (multiply operation) - f64
extern "C" __global__ void coo_merge_mul_f64(
    const long long* sorted_keys,
    const double* sorted_values,
    const int* source_flags,
    const int* intersection_flags,
    const int* output_positions,
    long long* out_keys,
    double* out_values,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (intersection_flags[idx] == 0) return;

    int out_pos = output_positions[idx];
    long long key = sorted_keys[idx];
    double val_a = sorted_values[idx];
    double val_b = sorted_values[idx + 1];

    out_keys[out_pos] = key;
    out_values[out_pos] = val_a * val_b;
}

/// Merge sorted entries with intersection semantics (divide operation) - f32
extern "C" __global__ void coo_merge_div_f32(
    const long long* sorted_keys,
    const float* sorted_values,
    const int* source_flags,
    const int* intersection_flags,
    const int* output_positions,
    long long* out_keys,
    float* out_values,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (intersection_flags[idx] == 0) return;

    int out_pos = output_positions[idx];
    long long key = sorted_keys[idx];
    float val_a = sorted_values[idx];
    float val_b = sorted_values[idx + 1];

    out_keys[out_pos] = key;
    out_values[out_pos] = val_a / val_b;
}

/// Merge sorted entries with intersection semantics (divide operation) - f64
extern "C" __global__ void coo_merge_div_f64(
    const long long* sorted_keys,
    const double* sorted_values,
    const int* source_flags,
    const int* intersection_flags,
    const int* output_positions,
    long long* out_keys,
    double* out_values,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (intersection_flags[idx] == 0) return;

    int out_pos = output_positions[idx];
    long long key = sorted_keys[idx];
    double val_a = sorted_values[idx];
    double val_b = sorted_values[idx + 1];

    out_keys[out_pos] = key;
    out_values[out_pos] = val_a / val_b;
}

// ============================================================================
// GPU Sort Wrappers using Thrust
// ============================================================================

/// Sort (i64 keys, i32 indices) using Thrust stable_sort_by_key - FULLY ON GPU
/// This wrapper allows calling from Rust via standard kernel launch
extern "C" __global__ void thrust_sort_pairs_i64_i32_kernel(
    long long* keys,
    int* indices,
    unsigned int n
) {
    // Only thread 0 performs the sort to avoid conflicts
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        thrust::device_ptr<long long> keys_ptr(keys);
        thrust::device_ptr<int> indices_ptr(indices);

        // Use stable_sort_by_key to sort indices based on keys
        thrust::stable_sort_by_key(
            thrust::device,
            keys_ptr,
            keys_ptr + n,
            indices_ptr
        );
    }
}

// ============================================================================
// Index Initialization and Gathering Kernels
// ============================================================================

/// Initialize indices array: [0, 1, 2, ..., n-1]
extern "C" __global__ void coo_init_indices_i32(
    int* indices,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    indices[idx] = idx;
}

/// Gather f32 values using indices (permutation)
extern "C" __global__ void coo_gather_f32(
    const float* values_in,
    const int* indices,
    float* values_out,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    values_out[idx] = values_in[indices[idx]];
}

/// Gather f64 values using indices
extern "C" __global__ void coo_gather_f64(
    const double* values_in,
    const int* indices,
    double* values_out,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    values_out[idx] = values_in[indices[idx]];
}

/// Gather i32 values using indices (for sources array)
extern "C" __global__ void coo_gather_i32(
    const int* values_in,
    const int* indices,
    int* values_out,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    values_out[idx] = values_in[indices[idx]];
}

/// Gather i64 values using indices (for row/col indices)
extern "C" __global__ void coo_gather_i64(
    const long long* values_in,
    const int* indices,
    long long* values_out,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    values_out[idx] = values_in[indices[idx]];
}

// ============================================================================
// Merge Duplicates Kernels
// ============================================================================

/// Merge duplicates with add operation (f32)
extern "C" __global__ void coo_merge_duplicates_add_f32(
    const long long* sorted_keys,
    const float* sorted_values,
    const int* sorted_sources,
    const int* unique_flags,
    const int* output_positions,  // Exclusive scan of unique_flags
    long long* out_keys,
    float* out_values,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int out_pos = output_positions[idx];
    long long key = sorted_keys[idx];
    float value = sorted_values[idx];
    int source = sorted_sources[idx];

    if (unique_flags[idx] == 1) {
        // First occurrence of this key
        out_keys[out_pos] = key;
        out_values[out_pos] = value;
    } else {
        // Duplicate key - add to existing value
        // Use atomic add since multiple threads might write to same position
        atomicAdd(&out_values[out_pos], value);
    }
}

/// Merge duplicates with add operation (f64)
extern "C" __global__ void coo_merge_duplicates_add_f64(
    const long long* sorted_keys,
    const double* sorted_values,
    const int* sorted_sources,
    const int* unique_flags,
    const int* output_positions,
    long long* out_keys,
    double* out_values,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int out_pos = output_positions[idx];
    long long key = sorted_keys[idx];
    double value = sorted_values[idx];

    if (unique_flags[idx] == 1) {
        out_keys[out_pos] = key;
        out_values[out_pos] = value;
    } else {
        atomicAdd(&out_values[out_pos], value);
    }
}

/// Merge duplicates with sub operation (f32)
/// For subtraction: process sources to determine sign
extern "C" __global__ void coo_merge_duplicates_sub_f32(
    const long long* sorted_keys,
    const float* sorted_values,
    const int* sorted_sources,  // 0 = from A (positive), 1 = from B (negative)
    const int* unique_flags,
    const int* output_positions,
    long long* out_keys,
    float* out_values,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int out_pos = output_positions[idx];
    long long key = sorted_keys[idx];
    float value = sorted_values[idx];
    int source = sorted_sources[idx];

    // Apply sign based on source: A - B means negate values from B
    float signed_value = (source == 0) ? value : -value;

    if (unique_flags[idx] == 1) {
        out_keys[out_pos] = key;
        out_values[out_pos] = signed_value;
    } else {
        atomicAdd(&out_values[out_pos], signed_value);
    }
}

/// Merge duplicates with sub operation (f64)
extern "C" __global__ void coo_merge_duplicates_sub_f64(
    const long long* sorted_keys,
    const double* sorted_values,
    const int* sorted_sources,
    const int* unique_flags,
    const int* output_positions,
    long long* out_keys,
    double* out_values,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int out_pos = output_positions[idx];
    double value = sorted_values[idx];
    int source = sorted_sources[idx];

    double signed_value = (source == 0) ? value : -value;

    if (unique_flags[idx] == 1) {
        out_keys[out_pos] = sorted_keys[idx];
        out_values[out_pos] = signed_value;
    } else {
        atomicAdd(&out_values[out_pos], signed_value);
    }
}

/// Merge duplicates with mul operation (f32)
/// For multiplication: only keep positions where BOTH matrices have values
/// This requires special handling - we need pairs from both sources
extern "C" __global__ void coo_merge_duplicates_mul_f32(
    const long long* sorted_keys,
    const float* sorted_values,
    const int* sorted_sources,
    const int* unique_flags,
    const int* output_positions,
    long long* out_keys,
    float* out_values,
    int* out_valid,  // Flag to mark valid outputs (1 if we have pair, 0 otherwise)
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n - 1) return;  // Check pairs

    int out_pos = output_positions[idx];
    long long key = sorted_keys[idx];

    // Check if next element has same key and different source
    if (sorted_keys[idx + 1] == key && sorted_sources[idx] != sorted_sources[idx + 1]) {
        // We have a pair - multiply
        float val_a = (sorted_sources[idx] == 0) ? sorted_values[idx] : sorted_values[idx + 1];
        float val_b = (sorted_sources[idx] == 1) ? sorted_values[idx] : sorted_values[idx + 1];

        if (unique_flags[idx] == 1) {
            out_keys[out_pos] = key;
            out_values[out_pos] = val_a * val_b;
            out_valid[out_pos] = 1;
        }
    }
}

/// Merge duplicates with mul operation (f64)
extern "C" __global__ void coo_merge_duplicates_mul_f64(
    const long long* sorted_keys,
    const double* sorted_values,
    const int* sorted_sources,
    const int* unique_flags,
    const int* output_positions,
    long long* out_keys,
    double* out_values,
    int* out_valid,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n - 1) return;

    int out_pos = output_positions[idx];
    long long key = sorted_keys[idx];

    if (sorted_keys[idx + 1] == key && sorted_sources[idx] != sorted_sources[idx + 1]) {
        double val_a = (sorted_sources[idx] == 0) ? sorted_values[idx] : sorted_values[idx + 1];
        double val_b = (sorted_sources[idx] == 1) ? sorted_values[idx] : sorted_values[idx + 1];

        if (unique_flags[idx] == 1) {
            out_keys[out_pos] = key;
            out_values[out_pos] = val_a * val_b;
            out_valid[out_pos] = 1;
        }
    }
}

/// Merge duplicates with div operation (f32)
extern "C" __global__ void coo_merge_duplicates_div_f32(
    const long long* sorted_keys,
    const float* sorted_values,
    const int* sorted_sources,
    const int* unique_flags,
    const int* output_positions,
    long long* out_keys,
    float* out_values,
    int* out_valid,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n - 1) return;

    int out_pos = output_positions[idx];
    long long key = sorted_keys[idx];

    if (sorted_keys[idx + 1] == key && sorted_sources[idx] != sorted_sources[idx + 1]) {
        float val_a = (sorted_sources[idx] == 0) ? sorted_values[idx] : sorted_values[idx + 1];
        float val_b = (sorted_sources[idx] == 1) ? sorted_values[idx] : sorted_values[idx + 1];

        if (unique_flags[idx] == 1) {
            out_keys[out_pos] = key;
            out_values[out_pos] = val_a / val_b;
            out_valid[out_pos] = 1;
        }
    }
}

/// Merge duplicates with div operation (f64)
extern "C" __global__ void coo_merge_duplicates_div_f64(
    const long long* sorted_keys,
    const double* sorted_values,
    const int* sorted_sources,
    const int* unique_flags,
    const int* output_positions,
    long long* out_keys,
    double* out_values,
    int* out_valid,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n - 1) return;

    int out_pos = output_positions[idx];
    long long key = sorted_keys[idx];

    if (sorted_keys[idx + 1] == key && sorted_sources[idx] != sorted_sources[idx + 1]) {
        double val_a = (sorted_sources[idx] == 0) ? sorted_values[idx] : sorted_values[idx + 1];
        double val_b = (sorted_sources[idx] == 1) ? sorted_values[idx] : sorted_values[idx + 1];

        if (unique_flags[idx] == 1) {
            out_keys[out_pos] = key;
            out_values[out_pos] = val_a / val_b;
            out_valid[out_pos] = 1;
        }
    }
}

// ============================================================================
// Zero Filtering Kernels
// ============================================================================

/// Mark non-zero values
extern "C" __global__ void coo_mark_nonzero_f32(
    const float* values,
    int* nonzero_flags,
    float threshold,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float abs_val = fabsf(values[idx]);
    nonzero_flags[idx] = (abs_val >= threshold) ? 1 : 0;
}

/// Mark non-zero values - f64
extern "C" __global__ void coo_mark_nonzero_f64(
    const double* values,
    int* nonzero_flags,
    double threshold,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double abs_val = fabs(values[idx]);
    nonzero_flags[idx] = (abs_val >= threshold) ? 1 : 0;
}

/// Compact arrays based on flags - f32
extern "C" __global__ void coo_compact_f32(
    const long long* keys_in,
    const float* values_in,
    const int* flags,
    const int* positions,  // Exclusive scan of flags
    long long* keys_out,
    float* values_out,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (flags[idx] == 1) {
        int out_pos = positions[idx];
        keys_out[out_pos] = keys_in[idx];
        values_out[out_pos] = values_in[idx];
    }
}

/// Compact arrays based on flags - f64
extern "C" __global__ void coo_compact_f64(
    const long long* keys_in,
    const double* values_in,
    const int* flags,
    const int* positions,
    long long* keys_out,
    double* values_out,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (flags[idx] == 1) {
        int out_pos = positions[idx];
        keys_out[out_pos] = keys_in[idx];
        values_out[out_pos] = values_in[idx];
    }
}

// ============================================================================
// Prefix Sum (Scan) Wrappers
// ============================================================================

/// Exclusive prefix sum using CUB
extern "C" int cub_exclusive_scan_i32(
    void* temp_storage,
    size_t temp_bytes,
    const int* input,
    int* output,
    unsigned int n,
    cudaStream_t stream
) {
    size_t storage_bytes = temp_bytes;

    cub::DeviceScan::ExclusiveSum(
        temp_storage,
        storage_bytes,
        input,
        output,
        n,
        stream
    );

    return (temp_storage == nullptr) ? (int)storage_bytes : 0;
}

/// Get required temp storage size for CUB exclusive scan
extern "C" size_t cub_exclusive_scan_temp_size_i32(unsigned int n) {
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        temp_storage_bytes,
        (int*)nullptr,
        (int*)nullptr,
        n
    );
    return temp_storage_bytes;
}

/// Sum reduction using CUB
extern "C" int cub_reduce_sum_i32(
    void* temp_storage,
    size_t temp_bytes,
    const int* input,
    int* output,
    unsigned int n,
    cudaStream_t stream
) {
    size_t storage_bytes = temp_bytes;

    cub::DeviceReduce::Sum(
        temp_storage,
        storage_bytes,
        input,
        output,
        n,
        stream
    );

    return (temp_storage == nullptr) ? (int)storage_bytes : 0;
}

/// Get required temp storage size for CUB sum reduction
extern "C" size_t cub_reduce_sum_temp_size_i32(unsigned int n) {
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(
        nullptr,
        temp_storage_bytes,
        (int*)nullptr,
        (int*)nullptr,
        n
    );
    return temp_storage_bytes;
}
