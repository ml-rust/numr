// COO sort kernels for the sparse format conversions.
//
// COO -> CSR/CSC sorts entries by a composite key on the GPU:
// 1. coo_compute_keys_i64: key = primary * extent + secondary
// 2. coo_init_indices_i32 + thrust_sort_pairs_i64_i32_kernel: stable sort of
//    the entry ids by key, so equal keys keep their input order
// 3. coo_gather_w*: gather indices and values through the sorted ids
//
// The gathers move raw words of 1, 2, 4 or 8 bytes, so one kernel serves
// every value dtype of that width.

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

/// key = primary * extent + secondary (row-major for (row, col) with
/// extent = ncols, column-major for (col, row) with extent = nrows).
extern "C" __global__ void coo_compute_keys_i64(
    const long long* primary,
    const long long* secondary,
    long long* keys_out,
    long long extent,
    unsigned int nnz
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    keys_out[idx] = primary[idx] * extent + secondary[idx];
}

/// Stable sort of (i64 key, i32 id) pairs by key with Thrust.
/// Launched with one thread. Equal keys keep their input order.
extern "C" __global__ void thrust_sort_pairs_i64_i32_kernel(
    long long* keys,
    int* indices,
    unsigned int n
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        thrust::device_ptr<long long> keys_ptr(keys);
        thrust::device_ptr<int> indices_ptr(indices);
        thrust::stable_sort_by_key(
            thrust::device,
            keys_ptr,
            keys_ptr + n,
            indices_ptr
        );
    }
}

/// indices = [0, 1, ..., n - 1]
extern "C" __global__ void coo_init_indices_i32(
    int* indices,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    indices[idx] = idx;
}

#define COO_GATHER(NAME, TYPE)                                              \
extern "C" __global__ void NAME(                                            \
    const TYPE* values_in,                                                  \
    const int* indices,                                                     \
    TYPE* values_out,                                                       \
    unsigned int n                                                          \
) {                                                                         \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
    if (idx >= n) return;                                                   \
    values_out[idx] = values_in[indices[idx]];                              \
}

// values_out[i] = values_in[indices[i]] for elements of 1, 2, 4 and 8 bytes.
COO_GATHER(coo_gather_w1, unsigned char)
COO_GATHER(coo_gather_w2, unsigned short)
COO_GATHER(coo_gather_w4, unsigned int)
COO_GATHER(coo_gather_w8, unsigned long long)
