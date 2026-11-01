// Sparse format conversion kernels. Indices are int64.
//
// expand_ptrs_i64:                   compressed pointers -> one index per entry
// build_ptrs_from_sorted_indices_i64: sorted indices -> compressed pointers
//
// CSR/CSC -> COO expands the pointers. COO -> CSR/CSC sorts the entries on
// the GPU (sparse_coo.cu), then builds the pointers from the sorted indices.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

/// indices_out[j] = b for every j in ptrs[b]..ptrs[b + 1].
__global__ void expand_ptrs_i64(
    const int64_t* ptrs,
    int64_t* indices_out,
    unsigned int n_rows_or_cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows_or_cols) return;

    int64_t start = ptrs[idx];
    int64_t end = ptrs[idx + 1];
    for (int64_t pos = start; pos < end; pos++) {
        indices_out[pos] = idx;
    }
}

/// ptrs_out[b] = number of sorted indices below b, for b in 0..=n_rows_or_cols.
/// One thread per pointer, each a binary search, so empty rows or columns
/// need no special case.
__global__ void build_ptrs_from_sorted_indices_i64(
    const int64_t* sorted_indices,
    int64_t* ptrs_out,
    unsigned int nnz,
    unsigned int n_rows_or_cols
) {
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b > n_rows_or_cols) return;

    unsigned int lo = 0;
    unsigned int hi = nnz;
    while (lo < hi) {
        unsigned int mid = lo + (hi - lo) / 2;
        if (sorted_indices[mid] < (int64_t)b) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    ptrs_out[b] = lo;
}

} // extern "C"
