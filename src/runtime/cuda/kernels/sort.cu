// Sorting and search CUDA kernels
// Supports: sort, argsort, topk, unique, nonzero, searchsorted
// Types: f32, f64, i32, i64, u32, u64
//
// Uses bitonic sort for GPU-friendly parallel sorting within shared memory blocks.
// For dimension-wise operations, uses the (outer, sort, inner) decomposition pattern.

#include <cuda_runtime.h>
#include "dtype_traits.cuh"

// ============================================================================
// Comparison helpers for sorting
// ============================================================================

template<typename T>
__device__ __forceinline__ bool compare_ascending(T a, T b) {
    return a < b;
}

template<typename T>
__device__ __forceinline__ bool compare_descending(T a, T b) {
    return a > b;
}

// ============================================================================
// Bitonic sort primitives (for small arrays in shared memory)
// ============================================================================

// Bitonic compare-and-swap
template<typename T>
__device__ __forceinline__ void bitonic_cas(T& a, T& b, bool ascending) {
    if (ascending ? (a > b) : (a < b)) {
        T tmp = a;
        a = b;
        b = tmp;
    }
}

// Bitonic compare-and-swap with indices
template<typename T>
__device__ __forceinline__ void bitonic_cas_indexed(T& a_val, long long& a_idx,
                                                     T& b_val, long long& b_idx,
                                                     bool ascending) {
    bool swap = ascending ? (a_val > b_val) : (a_val < b_val);
    // For equal values, maintain stability by comparing indices
    if (a_val == b_val) {
        swap = ascending ? (a_idx > b_idx) : (a_idx < b_idx);
    }
    if (swap) {
        T tmp_val = a_val;
        a_val = b_val;
        b_val = tmp_val;
        long long tmp_idx = a_idx;
        a_idx = b_idx;
        b_idx = tmp_idx;
    }
}

// ============================================================================
// Templated sort implementations
// ============================================================================

// Sort along dimension - uses block-level sorting
// Each block handles one (outer, inner) slice
template<typename T>
__device__ void sort_dim_impl(
    const T* input, T* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending, bool output_indices
) {
    // Calculate padded size FIRST (needed for shared memory layout)
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    // Shared memory for sort buffer (max 1024 elements per block)
    // Layout: [n values of type T][n indices of type long long]
    extern __shared__ char shared_mem[];
    T* shared_vals = (T*)shared_mem;
    long long* shared_idx = (long long*)(shared_vals + n);  // Place after padded values

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    // Load data into shared memory
    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    // Pad with max/min values
    T pad_val = descending ?
        (sizeof(T) == 8 ? (T)-1e308 : (T)-1e38f) :
        (sizeof(T) == 8 ? (T)1e308 : (T)1e38f);
    for (unsigned int i = tid + sort_size; i < n; i += blockDim.x) {
        shared_vals[i] = pad_val;
        shared_idx[i] = sort_size; // Invalid index
    }
    __syncthreads();

    // Bitonic sort network
    for (unsigned int k = 2; k <= n; k *= 2) {
        for (unsigned int j = k / 2; j > 0; j /= 2) {
            for (unsigned int i = tid; i < n / 2; i += blockDim.x) {
                unsigned int ij = (i / j) * 2 * j + (i % j);
                unsigned int ij_pair = ij + j;

                bool ascending_local = ((ij / k) % 2 == 0) != descending;

                if (ij_pair < n) {
                    bitonic_cas_indexed(
                        shared_vals[ij], shared_idx[ij],
                        shared_vals[ij_pair], shared_idx[ij_pair],
                        ascending_local
                    );
                }
            }
            __syncthreads();
        }
    }

    // Write results back
    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        output[out_idx] = shared_vals[i];
        if (output_indices && indices != nullptr) {
            indices[out_idx] = shared_idx[i];
        }
    }
}

// Top-K along dimension
// Uses partial sort - only keep k smallest/largest
template<typename T>
__device__ void topk_dim_impl(
    const T* input, T* out_values, long long* out_indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    unsigned int k, bool largest, bool sorted
) {
    // Calculate padded size FIRST for correct shared memory layout
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    extern __shared__ char shared_mem[];
    T* shared_vals = (T*)shared_mem;
    long long* shared_idx = (long long*)(shared_vals + n);  // After padded values

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    // Load data
    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    // For small k, use selection approach
    // For larger k, do full sort then take top k

    // Full bitonic sort for simplicity (can optimize for partial sort later)

    T pad_val = largest ?
        (sizeof(T) == 8 ? (T)-1e308 : (T)-1e38f) :
        (sizeof(T) == 8 ? (T)1e308 : (T)1e38f);
    for (unsigned int i = tid + sort_size; i < n; i += blockDim.x) {
        shared_vals[i] = pad_val;
        shared_idx[i] = sort_size;
    }
    __syncthreads();

    bool descending = largest;

    for (unsigned int kk = 2; kk <= n; kk *= 2) {
        for (unsigned int j = kk / 2; j > 0; j /= 2) {
            for (unsigned int i = tid; i < n / 2; i += blockDim.x) {
                unsigned int ij = (i / j) * 2 * j + (i % j);
                unsigned int ij_pair = ij + j;

                bool ascending_local = ((ij / kk) % 2 == 0) != descending;

                if (ij_pair < n) {
                    bitonic_cas_indexed(
                        shared_vals[ij], shared_idx[ij],
                        shared_vals[ij_pair], shared_idx[ij_pair],
                        ascending_local
                    );
                }
            }
            __syncthreads();
        }
    }

    // If not sorted, re-sort by original index (to preserve input order)
    if (!sorted) {
        // Simple bubble sort for top-k elements (k is usually small)
        for (unsigned int i = 0; i < k && tid == 0; i++) {
            for (unsigned int j = i + 1; j < k; j++) {
                if (shared_idx[i] > shared_idx[j]) {
                    T tmp_val = shared_vals[i];
                    shared_vals[i] = shared_vals[j];
                    shared_vals[j] = tmp_val;
                    long long tmp_idx = shared_idx[i];
                    shared_idx[i] = shared_idx[j];
                    shared_idx[j] = tmp_idx;
                }
            }
        }
        __syncthreads();
    }

    // Write top-k results
    for (unsigned int i = tid; i < k; i += blockDim.x) {
        unsigned int out_idx = outer_idx * k * inner_size + i * inner_size + inner_idx;
        out_values[out_idx] = shared_vals[i];
        out_indices[out_idx] = shared_idx[i];
    }
}

// Count nonzero elements
template<typename T>
__device__ void count_nonzero_impl(
    const T* input, unsigned int* count, unsigned int n
) {
    __shared__ unsigned int block_count;

    if (threadIdx.x == 0) block_count = 0;
    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_count = 0;

    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (input[i] != (T)0) {
            local_count++;
        }
    }

    atomicAdd(&block_count, local_count);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(count, block_count);
    }
}

// Gather nonzero indices
template<typename T>
__device__ void gather_nonzero_impl(
    const T* input, long long* flat_indices, unsigned int* counter, unsigned int n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (input[i] != (T)0) {
            unsigned int pos = atomicAdd(counter, 1);
            flat_indices[pos] = i;
        }
    }
}

// Binary search (searchsorted)
template<typename T>
__device__ void searchsorted_impl(
    const T* sorted_seq, const T* values,
    long long* output, unsigned int seq_len, unsigned int num_values, bool right
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = tid; i < num_values; i += blockDim.x * gridDim.x) {
        T val = values[i];

        // Binary search
        unsigned int lo = 0;
        unsigned int hi = seq_len;

        while (lo < hi) {
            unsigned int mid = lo + (hi - lo) / 2;
            T mid_val = sorted_seq[mid];

            bool go_left;
            if (right) {
                go_left = !(mid_val > val);  // mid_val <= val
            } else {
                go_left = mid_val < val;
            }

            if (go_left) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        output[i] = lo;
    }
}

// Count unique elements in sorted array
template<typename T>
__device__ void count_unique_impl(
    const T* sorted_input, unsigned int* count, unsigned int n
) {
    __shared__ unsigned int block_count;

    if (threadIdx.x == 0) block_count = 0;
    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_count = 0;

    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (i == 0 || sorted_input[i] != sorted_input[i - 1]) {
            local_count++;
        }
    }

    atomicAdd(&block_count, local_count);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(count, block_count);
    }
}

// Extract unique elements from sorted array
template<typename T>
__device__ void extract_unique_impl(
    const T* sorted_input, T* unique_output,
    unsigned int* counter, unsigned int n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (i == 0 || sorted_input[i] != sorted_input[i - 1]) {
            unsigned int pos = atomicAdd(counter, 1);
            unique_output[pos] = sorted_input[i];
        }
    }
}

// Bincount: count occurrences of each index in the input array
// Used for counting unique element occurrences in unique_with_counts
__device__ void bincount_impl(
    const long long* indices, long long* counts,
    unsigned int n, unsigned int num_bins
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x) {
        long long idx = indices[i];
        if (idx >= 0 && idx < (long long)num_bins) {
            atomicAdd((unsigned long long*)&counts[idx], 1ULL);
        }
    }
}

// ============================================================================
// extern "C" wrapper kernels for Rust FFI
// ============================================================================

extern "C" {

// Convert flat index to multi-dimensional index
__global__ void flat_to_multi_index(
    const long long* flat_indices, long long* multi_indices,
    unsigned int nnz, unsigned int ndim,
    const unsigned int* shape
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= nnz) return;

    long long flat_idx = flat_indices[tid];

    // Compute multi-index (row-major order)
    for (int d = ndim - 1; d >= 0; d--) {
        multi_indices[tid * ndim + d] = flat_idx % shape[d];
        flat_idx /= shape[d];
    }
}

// ----------------------------------------------------------------------------
// Sort with indices - F32
// ----------------------------------------------------------------------------
__global__ void sort_f32(
    const float* input, float* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<float>(input, output, indices, outer_size, sort_size, inner_size, descending, true);
}

__global__ void sort_values_only_f32(
    const float* input, float* output,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<float>(input, output, nullptr, outer_size, sort_size, inner_size, descending, false);
}

__global__ void argsort_f32(
    const float* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    // Calculate padded size FIRST for correct shared memory layout
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    extern __shared__ char shared_mem[];
    float* shared_vals = (float*)shared_mem;
    long long* shared_idx = (long long*)(shared_vals + n);  // After padded values

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    // Load data
    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    float pad_val = descending ? -1e38f : 1e38f;
    for (unsigned int i = tid + sort_size; i < n; i += blockDim.x) {
        shared_vals[i] = pad_val;
        shared_idx[i] = sort_size;
    }
    __syncthreads();

    for (unsigned int k = 2; k <= n; k *= 2) {
        for (unsigned int j = k / 2; j > 0; j /= 2) {
            for (unsigned int i = tid; i < n / 2; i += blockDim.x) {
                unsigned int ij = (i / j) * 2 * j + (i % j);
                unsigned int ij_pair = ij + j;
                bool ascending_local = ((ij / k) % 2 == 0) != descending;

                if (ij_pair < n) {
                    bitonic_cas_indexed(shared_vals[ij], shared_idx[ij],
                                       shared_vals[ij_pair], shared_idx[ij_pair],
                                       ascending_local);
                }
            }
            __syncthreads();
        }
    }

    // Write indices only
    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        indices[out_idx] = shared_idx[i];
    }
}

// ----------------------------------------------------------------------------
// Sort with indices - F64
// ----------------------------------------------------------------------------
__global__ void sort_f64(
    const double* input, double* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<double>(input, output, indices, outer_size, sort_size, inner_size, descending, true);
}

__global__ void sort_values_only_f64(
    const double* input, double* output,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<double>(input, output, nullptr, outer_size, sort_size, inner_size, descending, false);
}

__global__ void argsort_f64(
    const double* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    // Calculate padded size FIRST for correct shared memory layout
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    extern __shared__ char shared_mem[];
    double* shared_vals = (double*)shared_mem;
    long long* shared_idx = (long long*)(shared_vals + n);  // After padded values

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    double pad_val = descending ? -1e308 : 1e308;
    for (unsigned int i = tid + sort_size; i < n; i += blockDim.x) {
        shared_vals[i] = pad_val;
        shared_idx[i] = sort_size;
    }
    __syncthreads();

    for (unsigned int k = 2; k <= n; k *= 2) {
        for (unsigned int j = k / 2; j > 0; j /= 2) {
            for (unsigned int i = tid; i < n / 2; i += blockDim.x) {
                unsigned int ij = (i / j) * 2 * j + (i % j);
                unsigned int ij_pair = ij + j;
                bool ascending_local = ((ij / k) % 2 == 0) != descending;

                if (ij_pair < n) {
                    bitonic_cas_indexed(shared_vals[ij], shared_idx[ij],
                                       shared_vals[ij_pair], shared_idx[ij_pair],
                                       ascending_local);
                }
            }
            __syncthreads();
        }
    }

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        indices[out_idx] = shared_idx[i];
    }
}

// ----------------------------------------------------------------------------
// Sort with indices - I32
// ----------------------------------------------------------------------------
__global__ void sort_i32(
    const int* input, int* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<int>(input, output, indices, outer_size, sort_size, inner_size, descending, true);
}

__global__ void sort_values_only_i32(
    const int* input, int* output,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<int>(input, output, nullptr, outer_size, sort_size, inner_size, descending, false);
}

__global__ void argsort_i32(
    const int* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    // Calculate padded size FIRST for correct shared memory layout
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    extern __shared__ char shared_mem[];
    int* shared_vals = (int*)shared_mem;
    long long* shared_idx = (long long*)(shared_vals + n);  // After padded values

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    int pad_val = descending ? INT_MIN : INT_MAX;
    for (unsigned int i = tid + sort_size; i < n; i += blockDim.x) {
        shared_vals[i] = pad_val;
        shared_idx[i] = sort_size;
    }
    __syncthreads();

    for (unsigned int k = 2; k <= n; k *= 2) {
        for (unsigned int j = k / 2; j > 0; j /= 2) {
            for (unsigned int i = tid; i < n / 2; i += blockDim.x) {
                unsigned int ij = (i / j) * 2 * j + (i % j);
                unsigned int ij_pair = ij + j;
                bool ascending_local = ((ij / k) % 2 == 0) != descending;

                if (ij_pair < n) {
                    bitonic_cas_indexed(shared_vals[ij], shared_idx[ij],
                                       shared_vals[ij_pair], shared_idx[ij_pair],
                                       ascending_local);
                }
            }
            __syncthreads();
        }
    }

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        indices[out_idx] = shared_idx[i];
    }
}

// ----------------------------------------------------------------------------
// Sort with indices - I64
// ----------------------------------------------------------------------------
__global__ void sort_i64(
    const long long* input, long long* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<long long>(input, output, indices, outer_size, sort_size, inner_size, descending, true);
}

__global__ void sort_values_only_i64(
    const long long* input, long long* output,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<long long>(input, output, nullptr, outer_size, sort_size, inner_size, descending, false);
}

__global__ void argsort_i64(
    const long long* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    // Calculate padded size FIRST for correct shared memory layout
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    extern __shared__ char shared_mem[];
    long long* shared_vals = (long long*)shared_mem;
    long long* shared_idx = (long long*)(shared_vals + n);  // After padded values

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    long long pad_val = descending ? LLONG_MIN : LLONG_MAX;
    for (unsigned int i = tid + sort_size; i < n; i += blockDim.x) {
        shared_vals[i] = pad_val;
        shared_idx[i] = sort_size;
    }
    __syncthreads();

    for (unsigned int k = 2; k <= n; k *= 2) {
        for (unsigned int j = k / 2; j > 0; j /= 2) {
            for (unsigned int i = tid; i < n / 2; i += blockDim.x) {
                unsigned int ij = (i / j) * 2 * j + (i % j);
                unsigned int ij_pair = ij + j;
                bool ascending_local = ((ij / k) % 2 == 0) != descending;

                if (ij_pair < n) {
                    bitonic_cas_indexed(shared_vals[ij], shared_idx[ij],
                                       shared_vals[ij_pair], shared_idx[ij_pair],
                                       ascending_local);
                }
            }
            __syncthreads();
        }
    }

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        indices[out_idx] = shared_idx[i];
    }
}

// ----------------------------------------------------------------------------
// Sort with indices - U32
// ----------------------------------------------------------------------------
__global__ void sort_u32(
    const unsigned int* input, unsigned int* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<unsigned int>(input, output, indices, outer_size, sort_size, inner_size, descending, true);
}

__global__ void sort_values_only_u32(
    const unsigned int* input, unsigned int* output,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<unsigned int>(input, output, nullptr, outer_size, sort_size, inner_size, descending, false);
}

__global__ void argsort_u32(
    const unsigned int* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    // Calculate padded size FIRST for correct shared memory layout
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    extern __shared__ char shared_mem[];
    unsigned int* shared_vals = (unsigned int*)shared_mem;
    long long* shared_idx = (long long*)(shared_vals + n);  // After padded values

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    unsigned int pad_val = descending ? 0u : UINT_MAX;
    for (unsigned int i = tid + sort_size; i < n; i += blockDim.x) {
        shared_vals[i] = pad_val;
        shared_idx[i] = sort_size;
    }
    __syncthreads();

    for (unsigned int k = 2; k <= n; k *= 2) {
        for (unsigned int j = k / 2; j > 0; j /= 2) {
            for (unsigned int i = tid; i < n / 2; i += blockDim.x) {
                unsigned int ij = (i / j) * 2 * j + (i % j);
                unsigned int ij_pair = ij + j;
                bool ascending_local = ((ij / k) % 2 == 0) != descending;

                if (ij_pair < n) {
                    bitonic_cas_indexed(shared_vals[ij], shared_idx[ij],
                                       shared_vals[ij_pair], shared_idx[ij_pair],
                                       ascending_local);
                }
            }
            __syncthreads();
        }
    }

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        indices[out_idx] = shared_idx[i];
    }
}

// ----------------------------------------------------------------------------
// Sort with indices - U64
// ----------------------------------------------------------------------------
__global__ void sort_u64(
    const unsigned long long* input, unsigned long long* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<unsigned long long>(input, output, indices, outer_size, sort_size, inner_size, descending, true);
}

__global__ void sort_values_only_u64(
    const unsigned long long* input, unsigned long long* output,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<unsigned long long>(input, output, nullptr, outer_size, sort_size, inner_size, descending, false);
}

__global__ void argsort_u64(
    const unsigned long long* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    // Calculate padded size FIRST for correct shared memory layout
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    extern __shared__ char shared_mem[];
    unsigned long long* shared_vals = (unsigned long long*)shared_mem;
    long long* shared_idx = (long long*)(shared_vals + n);  // After padded values

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    unsigned long long pad_val = descending ? 0ull : ULLONG_MAX;
    for (unsigned int i = tid + sort_size; i < n; i += blockDim.x) {
        shared_vals[i] = pad_val;
        shared_idx[i] = sort_size;
    }
    __syncthreads();

    for (unsigned int k = 2; k <= n; k *= 2) {
        for (unsigned int j = k / 2; j > 0; j /= 2) {
            for (unsigned int i = tid; i < n / 2; i += blockDim.x) {
                unsigned int ij = (i / j) * 2 * j + (i % j);
                unsigned int ij_pair = ij + j;
                bool ascending_local = ((ij / k) % 2 == 0) != descending;

                if (ij_pair < n) {
                    bitonic_cas_indexed(shared_vals[ij], shared_idx[ij],
                                       shared_vals[ij_pair], shared_idx[ij_pair],
                                       ascending_local);
                }
            }
            __syncthreads();
        }
    }

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        indices[out_idx] = shared_idx[i];
    }
}

// ----------------------------------------------------------------------------
// Top-K kernels
// ----------------------------------------------------------------------------
__global__ void topk_f32(
    const float* input, float* out_values, long long* out_indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    unsigned int k, bool largest, bool sorted
) {
    topk_dim_impl<float>(input, out_values, out_indices, outer_size, sort_size, inner_size, k, largest, sorted);
}

__global__ void topk_f64(
    const double* input, double* out_values, long long* out_indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    unsigned int k, bool largest, bool sorted
) {
    topk_dim_impl<double>(input, out_values, out_indices, outer_size, sort_size, inner_size, k, largest, sorted);
}

__global__ void topk_i32(
    const int* input, int* out_values, long long* out_indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    unsigned int k, bool largest, bool sorted
) {
    topk_dim_impl<int>(input, out_values, out_indices, outer_size, sort_size, inner_size, k, largest, sorted);
}

__global__ void topk_i64(
    const long long* input, long long* out_values, long long* out_indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    unsigned int k, bool largest, bool sorted
) {
    topk_dim_impl<long long>(input, out_values, out_indices, outer_size, sort_size, inner_size, k, largest, sorted);
}

// ----------------------------------------------------------------------------
// Nonzero count kernels
// ----------------------------------------------------------------------------
__global__ void count_nonzero_f32(const float* input, unsigned int* count, unsigned int n) {
    count_nonzero_impl<float>(input, count, n);
}

__global__ void count_nonzero_f64(const double* input, unsigned int* count, unsigned int n) {
    count_nonzero_impl<double>(input, count, n);
}

__global__ void count_nonzero_i32(const int* input, unsigned int* count, unsigned int n) {
    count_nonzero_impl<int>(input, count, n);
}

__global__ void count_nonzero_i64(const long long* input, unsigned int* count, unsigned int n) {
    count_nonzero_impl<long long>(input, count, n);
}

// ----------------------------------------------------------------------------
// Nonzero gather kernels
// ----------------------------------------------------------------------------
__global__ void gather_nonzero_f32(const float* input, long long* indices, unsigned int* counter, unsigned int n) {
    gather_nonzero_impl<float>(input, indices, counter, n);
}

__global__ void gather_nonzero_f64(const double* input, long long* indices, unsigned int* counter, unsigned int n) {
    gather_nonzero_impl<double>(input, indices, counter, n);
}

__global__ void gather_nonzero_i32(const int* input, long long* indices, unsigned int* counter, unsigned int n) {
    gather_nonzero_impl<int>(input, indices, counter, n);
}

__global__ void gather_nonzero_i64(const long long* input, long long* indices, unsigned int* counter, unsigned int n) {
    gather_nonzero_impl<long long>(input, indices, counter, n);
}

// ----------------------------------------------------------------------------
// Searchsorted kernels
// ----------------------------------------------------------------------------
__global__ void searchsorted_f32(const float* seq, const float* values, long long* output,
                                  unsigned int seq_len, unsigned int num_values, bool right) {
    searchsorted_impl<float>(seq, values, output, seq_len, num_values, right);
}

__global__ void searchsorted_f64(const double* seq, const double* values, long long* output,
                                  unsigned int seq_len, unsigned int num_values, bool right) {
    searchsorted_impl<double>(seq, values, output, seq_len, num_values, right);
}

__global__ void searchsorted_i32(const int* seq, const int* values, long long* output,
                                  unsigned int seq_len, unsigned int num_values, bool right) {
    searchsorted_impl<int>(seq, values, output, seq_len, num_values, right);
}

__global__ void searchsorted_i64(const long long* seq, const long long* values, long long* output,
                                  unsigned int seq_len, unsigned int num_values, bool right) {
    searchsorted_impl<long long>(seq, values, output, seq_len, num_values, right);
}

// ----------------------------------------------------------------------------
// Unique count kernels
// ----------------------------------------------------------------------------
__global__ void count_unique_f32(const float* input, unsigned int* count, unsigned int n) {
    count_unique_impl<float>(input, count, n);
}

__global__ void count_unique_f64(const double* input, unsigned int* count, unsigned int n) {
    count_unique_impl<double>(input, count, n);
}

__global__ void count_unique_i32(const int* input, unsigned int* count, unsigned int n) {
    count_unique_impl<int>(input, count, n);
}

__global__ void count_unique_i64(const long long* input, unsigned int* count, unsigned int n) {
    count_unique_impl<long long>(input, count, n);
}

// ----------------------------------------------------------------------------
// Unique extract kernels
// ----------------------------------------------------------------------------
__global__ void extract_unique_f32(const float* input, float* output, unsigned int* counter, unsigned int n) {
    extract_unique_impl<float>(input, output, counter, n);
}

__global__ void extract_unique_f64(const double* input, double* output, unsigned int* counter, unsigned int n) {
    extract_unique_impl<double>(input, output, counter, n);
}

__global__ void extract_unique_i32(const int* input, int* output, unsigned int* counter, unsigned int n) {
    extract_unique_impl<int>(input, output, counter, n);
}

__global__ void extract_unique_i64(const long long* input, long long* output, unsigned int* counter, unsigned int n) {
    extract_unique_impl<long long>(input, output, counter, n);
}

// ----------------------------------------------------------------------------
// Bincount kernel - counts occurrences of indices
// ----------------------------------------------------------------------------
__global__ void bincount(const long long* indices, long long* counts,
                         unsigned int n, unsigned int num_bins) {
    bincount_impl(indices, counts, n, num_bins);
}

} // extern "C"
