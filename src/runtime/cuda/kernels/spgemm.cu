// SpGEMM C = A * B over CSR operands with int64 indices.
//
// 1. spgemm_row_bound:    bound[i] = sum of nnz(B[k, :]) over the entries A[i, k]
// 2. exclusive scan of the bounds (host side) gives each row a scratch slice
// 3. spgemm_accumulate_*: hash every product of row i into its scratch slice,
//    then one thread per row sorts the slice by column, drops entries with
//    |value| <= threshold (as the CPU backend does) and writes kept[i]
// 4. exclusive scan of kept (host side) gives the output row pointers
// 5. spgemm_gather_*:     copy each row's kept entries into the output
//
// Scratch lives in global memory, so no row size or column count limit
// applies. Accumulation runs in float for F16 and BF16, in the value type
// otherwise.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void spgemm_row_bound(
    const long long* a_row_ptrs,
    const long long* a_col_indices,
    const long long* b_row_ptrs,
    int* bound,
    unsigned int m
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    long long total = 0;
    for (long long a = a_row_ptrs[i]; a < a_row_ptrs[i + 1]; a++) {
        long long k = a_col_indices[a];
        total += b_row_ptrs[k + 1] - b_row_ptrs[k];
    }
    bound[i] = (int)total;
}

template<typename T, typename Acc>
__device__ __forceinline__ void spgemm_accumulate_impl(
    const long long* a_row_ptrs, const long long* a_col_indices, const T* a_values,
    const long long* b_row_ptrs, const long long* b_col_indices, const T* b_values,
    const int* slot_ptrs, long long* slot_cols, Acc* slot_vals, int* kept,
    unsigned int m, Acc threshold
) {
    unsigned int i = blockIdx.x;
    if (i >= m) return;

    int start = slot_ptrs[i];
    int capacity = slot_ptrs[i + 1] - start;
    long long* cols = slot_cols + start;
    Acc* vals = slot_vals + start;

    for (int j = threadIdx.x; j < capacity; j += blockDim.x) {
        cols[j] = -1;
        vals[j] = Acc(0);
    }
    __syncthreads();

    // Hash every product A[i, k] * B[k, j] into the row's slice.
    long long a_end = a_row_ptrs[i + 1];
    for (long long a = a_row_ptrs[i] + threadIdx.x; a < a_end; a += blockDim.x) {
        long long k = a_col_indices[a];
        Acc a_val = Acc(a_values[a]);
        for (long long b = b_row_ptrs[k]; b < b_row_ptrs[k + 1]; b++) {
            long long col = b_col_indices[b];
            Acc product = a_val * Acc(b_values[b]);
            int slot = (int)(col % capacity);
            while (true) {
                long long old = (long long)atomicCAS(
                    (unsigned long long*)&cols[slot],
                    (unsigned long long)-1LL,
                    (unsigned long long)col);
                if (old == -1 || old == col) {
                    atomicAdd(&vals[slot], product);
                    break;
                }
                slot = (slot + 1) % capacity;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    // Keep |value| > threshold, then insertion-sort the kept entries by column.
    int count = 0;
    for (int j = 0; j < capacity; j++) {
        if (cols[j] == -1) continue;
        Acc v = vals[j];
        Acc magnitude = v >= Acc(0) ? v : -v;
        if (!(magnitude > threshold)) continue;
        long long c = cols[j];
        int pos = count;
        while (pos > 0 && cols[pos - 1] > c) {
            cols[pos] = cols[pos - 1];
            vals[pos] = vals[pos - 1];
            pos--;
        }
        cols[pos] = c;
        vals[pos] = v;
        count++;
    }
    kept[i] = count;
}

template<typename T, typename Acc>
__device__ __forceinline__ void spgemm_gather_impl(
    const int* slot_ptrs, const long long* slot_cols, const Acc* slot_vals,
    const int* out_ptrs, long long* out_cols, T* out_vals, unsigned int m
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    int src = slot_ptrs[i];
    int dst = out_ptrs[i];
    int count = out_ptrs[i + 1] - dst;
    for (int j = 0; j < count; j++) {
        out_cols[dst + j] = slot_cols[src + j];
        out_vals[dst + j] = T(slot_vals[src + j]);
    }
}

#define SPGEMM_KERNELS(T, ACC, SUFFIX)                                                     \
extern "C" __global__ void spgemm_accumulate_##SUFFIX(                                     \
    const long long* a_row_ptrs, const long long* a_col_indices, const T* a_values,        \
    const long long* b_row_ptrs, const long long* b_col_indices, const T* b_values,        \
    const int* slot_ptrs, long long* slot_cols, ACC* slot_vals, int* kept,                 \
    unsigned int m, ACC threshold                                                          \
) {                                                                                        \
    spgemm_accumulate_impl<T, ACC>(a_row_ptrs, a_col_indices, a_values, b_row_ptrs,       \
        b_col_indices, b_values, slot_ptrs, slot_cols, slot_vals, kept, m, threshold);     \
}                                                                                          \
extern "C" __global__ void spgemm_gather_##SUFFIX(                                         \
    const int* slot_ptrs, const long long* slot_cols, const ACC* slot_vals,                \
    const int* out_ptrs, long long* out_cols, T* out_vals, unsigned int m                  \
) {                                                                                        \
    spgemm_gather_impl<T, ACC>(slot_ptrs, slot_cols, slot_vals, out_ptrs, out_cols,        \
        out_vals, m);                                                                      \
}

SPGEMM_KERNELS(float, float, f32)
SPGEMM_KERNELS(double, double, f64)
SPGEMM_KERNELS(__half, float, f16)
SPGEMM_KERNELS(__nv_bfloat16, float, bf16)
