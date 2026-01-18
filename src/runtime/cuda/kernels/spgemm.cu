// ESC (Exact Symbolic Computation) + Hash Accumulation SpGEMM
// Two-phase algorithm for sparse matrix-matrix multiplication
//
// Phase 1: Symbolic - count unique columns per output row
// Phase 2: Numeric - compute values with fixed-size accumulation
//
// This is the SAME algorithm used on CPU for backend parity

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Phase 1: Symbolic - Count unique columns per row (device helper, not templated)
// ============================================================================

__device__ __inline__ void spgemm_symbolic_phase_impl(
    const long long* a_row_ptrs,
    const long long* a_col_indices,
    const long long* b_row_ptrs,
    const long long* b_col_indices,
    int* row_nnz,
    unsigned int m,
    unsigned int n,
    unsigned int i
) {
    if (i >= m) return;

    // Use dynamic shared memory
    extern __shared__ unsigned char seen_cols[];
    unsigned int bytes_per_thread = (n + 7) / 8;
    unsigned char* my_seen = &seen_cols[threadIdx.x * bytes_per_thread];

    long long a_start = a_row_ptrs[i];
    long long a_end = a_row_ptrs[i + 1];

    // Initialize bitmap to zero
    for (unsigned int j = 0; j < (n + 7) / 8; j++) {
        my_seen[j] = 0;
    }

    // Scan through A[i,:] * B and mark columns
    int count = 0;
    for (long long a_idx = a_start; a_idx < a_end; a_idx++) {
        long long k = a_col_indices[a_idx];
        long long b_start = b_row_ptrs[k];
        long long b_end = b_row_ptrs[k + 1];

        for (long long b_idx = b_start; b_idx < b_end; b_idx++) {
            unsigned int j = b_col_indices[b_idx];
            unsigned int byte_idx = j / 8;
            unsigned int bit_idx = j % 8;

            // Check if not seen before
            if (!(my_seen[byte_idx] & (1 << bit_idx))) {
                my_seen[byte_idx] |= (1 << bit_idx);
                count++;
            }
        }
    }

    row_nnz[i] = count;
}

// ============================================================================
// Phase 2: Numeric - Compute values (device helpers, not templated)
// ============================================================================

template<typename T>
__device__ __inline__ void spgemm_numeric_phase_impl(
    const long long* a_row_ptrs,
    const long long* a_col_indices,
    const T* a_values,
    const long long* b_row_ptrs,
    const long long* b_col_indices,
    const T* b_values,
    const long long* c_row_ptrs,
    long long* c_col_indices,
    T* c_values,
    unsigned int m,
    unsigned int i,
    T threshold
) {
    if (i >= m) return;

    long long a_start = a_row_ptrs[i];
    long long a_end = a_row_ptrs[i + 1];
    long long out_start = c_row_ptrs[i];
    long long out_end = c_row_ptrs[i + 1];
    int row_capacity = out_end - out_start;

    if (row_capacity == 0) return;

    // Use shared memory for accumulation
    extern __shared__ char shared_mem[];
    long long* shared_cols = (long long*)shared_mem;
    T* shared_vals = (T*)&shared_cols[row_capacity];

    // Initialize shared memory (parallel across threads in block)
    for (int j = threadIdx.x; j < row_capacity; j += blockDim.x) {
        shared_cols[j] = -1;  // -1 means empty
        shared_vals[j] = T(0);
    }
    __syncthreads();

    // Each thread processes a subset of A[i,:] * B
    for (long long a_idx = a_start + threadIdx.x; a_idx < a_end; a_idx += blockDim.x) {
        long long k = a_col_indices[a_idx];
        T a_val = a_values[a_idx];

        long long b_start = b_row_ptrs[k];
        long long b_end = b_row_ptrs[k + 1];

        for (long long b_idx = b_start; b_idx < b_end; b_idx++) {
            long long j = b_col_indices[b_idx];
            T b_val = b_values[b_idx];
            T product = a_val * b_val;

            // Linear probing hash insert (simple but works)
            unsigned int hash = j % row_capacity;
            while (true) {
                long long old_col = atomicCAS((unsigned long long*)&shared_cols[hash],
                                              (unsigned long long)-1,
                                              (unsigned long long)j);
                if (old_col == -1 || old_col == j) {
                    // Successfully inserted or found existing
                    // Atomic add to accumulator
                    atomicAdd(&shared_vals[hash], product);
                    break;
                }
                // Collision - try next slot
                hash = (hash + 1) % row_capacity;
            }
        }
    }
    __syncthreads();

    // Compact and sort results (single thread)
    if (threadIdx.x == 0) {
        // Collect non-empty entries
        int write_idx = 0;
        for (int j = 0; j < row_capacity; j++) {
            if (shared_cols[j] != -1) {
                T val = shared_vals[j];
                T zero = T(0);
                T abs_val = val >= zero ? val : -val;

                // Apply threshold filter
                if (abs_val >= threshold) {
                    shared_cols[write_idx] = shared_cols[j];
                    shared_vals[write_idx] = val;
                    write_idx++;
                }
            }
        }

        // Bubble sort (simple, works for small rows)
        for (int a = 0; a < write_idx - 1; a++) {
            for (int b = 0; b < write_idx - a - 1; b++) {
                if (shared_cols[b] > shared_cols[b + 1]) {
                    // Swap
                    long long tmp_col = shared_cols[b];
                    T tmp_val = shared_vals[b];
                    shared_cols[b] = shared_cols[b + 1];
                    shared_vals[b] = shared_vals[b + 1];
                    shared_cols[b + 1] = tmp_col;
                    shared_vals[b + 1] = tmp_val;
                }
            }
        }

        // Write to output
        for (int j = 0; j < write_idx; j++) {
            c_col_indices[out_start + j] = shared_cols[j];
            c_values[out_start + j] = shared_vals[j];
        }
    }
}

// ============================================================================
// Extern C wrappers for Rust FFI
// ============================================================================

extern "C" {

// Phase 1: Symbolic wrapper
__global__ void spgemm_symbolic_phase(
    const long long* a_row_ptrs,
    const long long* a_col_indices,
    const long long* b_row_ptrs,
    const long long* b_col_indices,
    int* row_nnz,
    unsigned int m,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    spgemm_symbolic_phase_impl(a_row_ptrs, a_col_indices, b_row_ptrs, b_col_indices,
                               row_nnz, m, n, i);
}

// Phase 2: Numeric wrappers

__global__ void spgemm_numeric_phase_f32(
    const long long* a_row_ptrs, const long long* a_col_indices, const float* a_values,
    const long long* b_row_ptrs, const long long* b_col_indices, const float* b_values,
    const long long* c_row_ptrs, long long* c_col_indices, float* c_values,
    unsigned int m, unsigned int n, float threshold
) {
    unsigned int i = blockIdx.x;
    spgemm_numeric_phase_impl<float>(a_row_ptrs, a_col_indices, a_values,
                                      b_row_ptrs, b_col_indices, b_values,
                                      c_row_ptrs, c_col_indices, c_values,
                                      m, i, threshold);
}

__global__ void spgemm_numeric_phase_f64(
    const long long* a_row_ptrs, const long long* a_col_indices, const double* a_values,
    const long long* b_row_ptrs, const long long* b_col_indices, const double* b_values,
    const long long* c_row_ptrs, long long* c_col_indices, double* c_values,
    unsigned int m, unsigned int n, double threshold
) {
    unsigned int i = blockIdx.x;
    spgemm_numeric_phase_impl<double>(a_row_ptrs, a_col_indices, a_values,
                                       b_row_ptrs, b_col_indices, b_values,
                                       c_row_ptrs, c_col_indices, c_values,
                                       m, i, threshold);
}

} // extern "C"
