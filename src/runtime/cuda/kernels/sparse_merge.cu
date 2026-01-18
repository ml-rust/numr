// Sparse matrix element-wise merge kernels (CSR format)
// Two-pass algorithm:
// 1. Count output size per row
// 2. Compute merged output (after exclusive scan)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Pass 1: Count output size per row
// ============================================================================

/// Count non-zeros in merged row (union of indices)
__global__ void csr_merge_count_kernel(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    int* row_counts,  // Output: nnz per row
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];

    int i = start_a, j = start_b, count = 0;

    // Merge-count (union semantics)
    while (i < end_a || j < end_b) {
        int col_a = (i < end_a) ? col_indices_a[i] : INT_MAX;
        int col_b = (j < end_b) ? col_indices_b[j] : INT_MAX;

        if (col_a < col_b) {
            count++; i++;
        } else if (col_a > col_b) {
            count++; j++;
        } else {
            count++; i++; j++;
        }
    }

    row_counts[row] = count;
}

// ============================================================================
// Pass 2: Compute merged output
// ============================================================================

// Addition: C = A + B
template<typename T>
__global__ void csr_add_compute_kernel(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const T* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const T* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    T* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int col_a = (i < end_a) ? col_indices_a[i] : INT_MAX;
        int col_b = (j < end_b) ? col_indices_b[j] : INT_MAX;

        if (col_a < col_b) {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (col_a > col_b) {
            out_col_indices[out_idx] = col_b;
            out_values[out_idx] = values_b[j];
            j++; out_idx++;
        } else {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i] + values_b[j];
            i++; j++; out_idx++;
        }
    }
}

// Specialization for F16 (accumulate in F32)
template<>
__global__ void csr_add_compute_kernel<__half>(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const __half* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const __half* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    __half* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int col_a = (i < end_a) ? col_indices_a[i] : INT_MAX;
        int col_b = (j < end_b) ? col_indices_b[j] : INT_MAX;

        if (col_a < col_b) {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (col_a > col_b) {
            out_col_indices[out_idx] = col_b;
            out_values[out_idx] = values_b[j];
            j++; out_idx++;
        } else {
            out_col_indices[out_idx] = col_a;
            float val_a = __half2float(values_a[i]);
            float val_b = __half2float(values_b[j]);
            out_values[out_idx] = __float2half(val_a + val_b);
            i++; j++; out_idx++;
        }
    }
}

// Specialization for BF16 (accumulate in F32)
template<>
__global__ void csr_add_compute_kernel<__nv_bfloat16>(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const __nv_bfloat16* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const __nv_bfloat16* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    __nv_bfloat16* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int col_a = (i < end_a) ? col_indices_a[i] : INT_MAX;
        int col_b = (j < end_b) ? col_indices_b[j] : INT_MAX;

        if (col_a < col_b) {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (col_a > col_b) {
            out_col_indices[out_idx] = col_b;
            out_values[out_idx] = values_b[j];
            j++; out_idx++;
        } else {
            out_col_indices[out_idx] = col_a;
            float val_a = __bfloat162float(values_a[i]);
            float val_b = __bfloat162float(values_b[j]);
            out_values[out_idx] = __float2bfloat16(val_a + val_b);
            i++; j++; out_idx++;
        }
    }
}

// Subtraction: C = A - B
template<typename T>
__global__ void csr_sub_compute_kernel(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const T* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const T* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    T* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int col_a = (i < end_a) ? col_indices_a[i] : INT_MAX;
        int col_b = (j < end_b) ? col_indices_b[j] : INT_MAX;

        if (col_a < col_b) {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (col_a > col_b) {
            out_col_indices[out_idx] = col_b;
            out_values[out_idx] = -values_b[j];
            j++; out_idx++;
        } else {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i] - values_b[j];
            i++; j++; out_idx++;
        }
    }
}

// Specialization for F16
template<>
__global__ void csr_sub_compute_kernel<__half>(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const __half* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const __half* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    __half* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int col_a = (i < end_a) ? col_indices_a[i] : INT_MAX;
        int col_b = (j < end_b) ? col_indices_b[j] : INT_MAX;

        if (col_a < col_b) {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (col_a > col_b) {
            out_col_indices[out_idx] = col_b;
            out_values[out_idx] = __hneg(values_b[j]);
            j++; out_idx++;
        } else {
            out_col_indices[out_idx] = col_a;
            float val_a = __half2float(values_a[i]);
            float val_b = __half2float(values_b[j]);
            out_values[out_idx] = __float2half(val_a - val_b);
            i++; j++; out_idx++;
        }
    }
}

// Specialization for BF16
template<>
__global__ void csr_sub_compute_kernel<__nv_bfloat16>(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const __nv_bfloat16* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const __nv_bfloat16* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    __nv_bfloat16* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int col_a = (i < end_a) ? col_indices_a[i] : INT_MAX;
        int col_b = (j < end_b) ? col_indices_b[j] : INT_MAX;

        if (col_a < col_b) {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (col_a > col_b) {
            out_col_indices[out_idx] = col_b;
            out_values[out_idx] = __hneg(values_b[j]);
            j++; out_idx++;
        } else {
            out_col_indices[out_idx] = col_a;
            float val_a = __bfloat162float(values_a[i]);
            float val_b = __bfloat162float(values_b[j]);
            out_values[out_idx] = __float2bfloat16(val_a - val_b);
            i++; j++; out_idx++;
        }
    }
}

// Element-wise multiplication: C = A .* B (intersection semantics)
template<typename T>
__global__ void csr_mul_compute_kernel(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const T* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const T* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    T* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    // Only output where both have values (intersection)
    while (i < end_a && j < end_b) {
        int col_a = col_indices_a[i];
        int col_b = col_indices_b[j];

        if (col_a < col_b) {
            i++;
        } else if (col_a > col_b) {
            j++;
        } else {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i] * values_b[j];
            i++; j++; out_idx++;
        }
    }
}

// Specialization for F16
template<>
__global__ void csr_mul_compute_kernel<__half>(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const __half* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const __half* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    __half* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    while (i < end_a && j < end_b) {
        int col_a = col_indices_a[i];
        int col_b = col_indices_b[j];

        if (col_a < col_b) {
            i++;
        } else if (col_a > col_b) {
            j++;
        } else {
            out_col_indices[out_idx] = col_a;
            float val_a = __half2float(values_a[i]);
            float val_b = __half2float(values_b[j]);
            out_values[out_idx] = __float2half(val_a * val_b);
            i++; j++; out_idx++;
        }
    }
}

// Specialization for BF16
template<>
__global__ void csr_mul_compute_kernel<__nv_bfloat16>(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const __nv_bfloat16* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const __nv_bfloat16* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    __nv_bfloat16* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    while (i < end_a && j < end_b) {
        int col_a = col_indices_a[i];
        int col_b = col_indices_b[j];

        if (col_a < col_b) {
            i++;
        } else if (col_a > col_b) {
            j++;
        } else {
            out_col_indices[out_idx] = col_a;
            float val_a = __bfloat162float(values_a[i]);
            float val_b = __bfloat162float(values_b[j]);
            out_values[out_idx] = __float2bfloat16(val_a * val_b);
            i++; j++; out_idx++;
        }
    }
}

// Count kernel for element-wise multiplication (intersection semantics)
__global__ void csr_mul_count_kernel(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    int* row_counts,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];

    int i = start_a, j = start_b, count = 0;

    // Only count where both have values (intersection)
    while (i < end_a && j < end_b) {
        int col_a = col_indices_a[i];
        int col_b = col_indices_b[j];

        if (col_a < col_b) {
            i++;
        } else if (col_a > col_b) {
            j++;
        } else {
            count++;
            i++; j++;
        }
    }

    row_counts[row] = count;
}

// ============================================================================
// Extern "C" wrapper kernels for Rust FFI
// ============================================================================

extern "C" {

// Count kernels (dtype-agnostic)
__global__ void csr_merge_count(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    int* row_counts,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];

    int i = start_a, j = start_b, count = 0;

    // Merge-count (union semantics)
    while (i < end_a || j < end_b) {
        int col_a = (i < end_a) ? col_indices_a[i] : INT_MAX;
        int col_b = (j < end_b) ? col_indices_b[j] : INT_MAX;

        if (col_a < col_b) {
            count++; i++;
        } else if (col_a > col_b) {
            count++; j++;
        } else {
            count++; i++; j++;
        }
    }

    row_counts[row] = count;
}

__global__ void csr_mul_count(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    int* row_counts,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];

    int i = start_a, j = start_b, count = 0;

    // Only count where both have values (intersection)
    while (i < end_a && j < end_b) {
        int col_a = col_indices_a[i];
        int col_b = col_indices_b[j];

        if (col_a < col_b) {
            i++;
        } else if (col_a > col_b) {
            j++;
        } else {
            count++;
            i++; j++;
        }
    }

    row_counts[row] = count;
}


// ============================================================================
// Add Compute Kernels (Inlined)
// ============================================================================

__global__ void csr_add_compute_f32(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const float* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const float* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    float* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int col_a = (i < end_a) ? col_indices_a[i] : INT_MAX;
        int col_b = (j < end_b) ? col_indices_b[j] : INT_MAX;

        if (col_a < col_b) {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (col_a > col_b) {
            out_col_indices[out_idx] = col_b;
            out_values[out_idx] = values_b[j];
            j++; out_idx++;
        } else {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i] + values_b[j];
            i++; j++; out_idx++;
        }
    }
}

__global__ void csr_add_compute_f64(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const double* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const double* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    double* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int col_a = (i < end_a) ? col_indices_a[i] : INT_MAX;
        int col_b = (j < end_b) ? col_indices_b[j] : INT_MAX;

        if (col_a < col_b) {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (col_a > col_b) {
            out_col_indices[out_idx] = col_b;
            out_values[out_idx] = values_b[j];
            j++; out_idx++;
        } else {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i] + values_b[j];
            i++; j++; out_idx++;
        }
    }
}

__global__ void csr_add_compute_f16(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const __half* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const __half* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    __half* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int col_a = (i < end_a) ? col_indices_a[i] : INT_MAX;
        int col_b = (j < end_b) ? col_indices_b[j] : INT_MAX;

        if (col_a < col_b) {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (col_a > col_b) {
            out_col_indices[out_idx] = col_b;
            out_values[out_idx] = values_b[j];
            j++; out_idx++;
        } else {
            out_col_indices[out_idx] = col_a;
            // Accumulate in F32 for numerical stability
            float sum = __half2float(values_a[i]) + __half2float(values_b[j]);
            out_values[out_idx] = __float2half(sum);
            i++; j++; out_idx++;
        }
    }
}

__global__ void csr_add_compute_bf16(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const __nv_bfloat16* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const __nv_bfloat16* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    __nv_bfloat16* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int col_a = (i < end_a) ? col_indices_a[i] : INT_MAX;
        int col_b = (j < end_b) ? col_indices_b[j] : INT_MAX;

        if (col_a < col_b) {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (col_a > col_b) {
            out_col_indices[out_idx] = col_b;
            out_values[out_idx] = values_b[j];
            j++; out_idx++;
        } else {
            out_col_indices[out_idx] = col_a;
            // Accumulate in F32 for numerical stability
            float sum = __bfloat162float(values_a[i]) + __bfloat162float(values_b[j]);
            out_values[out_idx] = __float2bfloat16(sum);
            i++; j++; out_idx++;
        }
    }
}

// ============================================================================
// Sub Compute Kernels (Inlined)
// ============================================================================

__global__ void csr_sub_compute_f32(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const float* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const float* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    float* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int col_a = (i < end_a) ? col_indices_a[i] : INT_MAX;
        int col_b = (j < end_b) ? col_indices_b[j] : INT_MAX;

        if (col_a < col_b) {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (col_a > col_b) {
            out_col_indices[out_idx] = col_b;
            out_values[out_idx] = -values_b[j];
            j++; out_idx++;
        } else {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i] - values_b[j];
            i++; j++; out_idx++;
        }
    }
}

__global__ void csr_sub_compute_f64(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const double* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const double* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    double* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int col_a = (i < end_a) ? col_indices_a[i] : INT_MAX;
        int col_b = (j < end_b) ? col_indices_b[j] : INT_MAX;

        if (col_a < col_b) {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (col_a > col_b) {
            out_col_indices[out_idx] = col_b;
            out_values[out_idx] = -values_b[j];
            j++; out_idx++;
        } else {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i] - values_b[j];
            i++; j++; out_idx++;
        }
    }
}

__global__ void csr_sub_compute_f16(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const __half* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const __half* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    __half* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int col_a = (i < end_a) ? col_indices_a[i] : INT_MAX;
        int col_b = (j < end_b) ? col_indices_b[j] : INT_MAX;

        if (col_a < col_b) {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (col_a > col_b) {
            out_col_indices[out_idx] = col_b;
            out_values[out_idx] = __float2half(-__half2float(values_b[j]));
            j++; out_idx++;
        } else {
            out_col_indices[out_idx] = col_a;
            // Accumulate in F32 for numerical stability
            float diff = __half2float(values_a[i]) - __half2float(values_b[j]);
            out_values[out_idx] = __float2half(diff);
            i++; j++; out_idx++;
        }
    }
}

__global__ void csr_sub_compute_bf16(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const __nv_bfloat16* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const __nv_bfloat16* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    __nv_bfloat16* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int col_a = (i < end_a) ? col_indices_a[i] : INT_MAX;
        int col_b = (j < end_b) ? col_indices_b[j] : INT_MAX;

        if (col_a < col_b) {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (col_a > col_b) {
            out_col_indices[out_idx] = col_b;
            out_values[out_idx] = __float2bfloat16(-__bfloat162float(values_b[j]));
            j++; out_idx++;
        } else {
            out_col_indices[out_idx] = col_a;
            // Accumulate in F32 for numerical stability
            float diff = __bfloat162float(values_a[i]) - __bfloat162float(values_b[j]);
            out_values[out_idx] = __float2bfloat16(diff);
            i++; j++; out_idx++;
        }
    }
}

// ============================================================================
// Mul Compute Kernels (Inlined - Intersection Semantics)
// ============================================================================

__global__ void csr_mul_compute_f32(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const float* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const float* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    float* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    // Only output where both have values
    while (i < end_a && j < end_b) {
        int col_a = col_indices_a[i];
        int col_b = col_indices_b[j];

        if (col_a < col_b) {
            i++;
        } else if (col_a > col_b) {
            j++;
        } else {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i] * values_b[j];
            i++; j++; out_idx++;
        }
    }
}

__global__ void csr_mul_compute_f64(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const double* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const double* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    double* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    // Only output where both have values
    while (i < end_a && j < end_b) {
        int col_a = col_indices_a[i];
        int col_b = col_indices_b[j];

        if (col_a < col_b) {
            i++;
        } else if (col_a > col_b) {
            j++;
        } else {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i] * values_b[j];
            i++; j++; out_idx++;
        }
    }
}

__global__ void csr_mul_compute_f16(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const __half* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const __half* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    __half* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    // Only output where both have values
    while (i < end_a && j < end_b) {
        int col_a = col_indices_a[i];
        int col_b = col_indices_b[j];

        if (col_a < col_b) {
            i++;
        } else if (col_a > col_b) {
            j++;
        } else {
            out_col_indices[out_idx] = col_a;
            // Accumulate in F32 for numerical stability
            float prod = __half2float(values_a[i]) * __half2float(values_b[j]);
            out_values[out_idx] = __float2half(prod);
            i++; j++; out_idx++;
        }
    }
}

__global__ void csr_mul_compute_bf16(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const __nv_bfloat16* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const __nv_bfloat16* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    __nv_bfloat16* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    // Only output where both have values
    while (i < end_a && j < end_b) {
        int col_a = col_indices_a[i];
        int col_b = col_indices_b[j];

        if (col_a < col_b) {
            i++;
        } else if (col_a > col_b) {
            j++;
        } else {
            out_col_indices[out_idx] = col_a;
            // Accumulate in F32 for numerical stability
            float prod = __bfloat162float(values_a[i]) * __bfloat162float(values_b[j]);
            out_values[out_idx] = __float2bfloat16(prod);
            i++; j++; out_idx++;
        }
    }
}

// ============================================================================
// Div Compute Kernels (Inlined - Intersection Semantics)
// Division only occurs where both matrices have values (like mul)
// ============================================================================

__global__ void csr_div_compute_f32(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const float* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const float* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    float* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    // Only output where both have values
    while (i < end_a && j < end_b) {
        int col_a = col_indices_a[i];
        int col_b = col_indices_b[j];

        if (col_a < col_b) {
            i++;
        } else if (col_a > col_b) {
            j++;
        } else {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i] / values_b[j];
            i++; j++; out_idx++;
        }
    }
}

__global__ void csr_div_compute_f64(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const double* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const double* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    double* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    // Only output where both have values
    while (i < end_a && j < end_b) {
        int col_a = col_indices_a[i];
        int col_b = col_indices_b[j];

        if (col_a < col_b) {
            i++;
        } else if (col_a > col_b) {
            j++;
        } else {
            out_col_indices[out_idx] = col_a;
            out_values[out_idx] = values_a[i] / values_b[j];
            i++; j++; out_idx++;
        }
    }
}

__global__ void csr_div_compute_f16(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const __half* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const __half* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    __half* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    // Only output where both have values
    while (i < end_a && j < end_b) {
        int col_a = col_indices_a[i];
        int col_b = col_indices_b[j];

        if (col_a < col_b) {
            i++;
        } else if (col_a > col_b) {
            j++;
        } else {
            out_col_indices[out_idx] = col_a;
            // Compute in F32 for numerical stability
            float quot = __half2float(values_a[i]) / __half2float(values_b[j]);
            out_values[out_idx] = __float2half(quot);
            i++; j++; out_idx++;
        }
    }
}

__global__ void csr_div_compute_bf16(
    const int* row_ptrs_a,
    const int* col_indices_a,
    const __nv_bfloat16* values_a,
    const int* row_ptrs_b,
    const int* col_indices_b,
    const __nv_bfloat16* values_b,
    const int* out_row_ptrs,
    int* out_col_indices,
    __nv_bfloat16* out_values,
    int nrows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    int start_a = row_ptrs_a[row];
    int end_a = row_ptrs_a[row + 1];
    int start_b = row_ptrs_b[row];
    int end_b = row_ptrs_b[row + 1];
    int out_idx = out_row_ptrs[row];

    int i = start_a, j = start_b;

    // Only output where both have values
    while (i < end_a && j < end_b) {
        int col_a = col_indices_a[i];
        int col_b = col_indices_b[j];

        if (col_a < col_b) {
            i++;
        } else if (col_a > col_b) {
            j++;
        } else {
            out_col_indices[out_idx] = col_a;
            // Compute in F32 for numerical stability
            float quot = __bfloat162float(values_a[i]) / __bfloat162float(values_b[j]);
            out_values[out_idx] = __float2bfloat16(quot);
            i++; j++; out_idx++;
        }
    }
}

// ============================================================================
// CSC Operations - Column-oriented sparse merge
// Uses same two-pass algorithm but iterates over columns instead of rows
// ============================================================================

// Count kernel for CSC merge (union semantics for add/sub)
__global__ void csc_merge_count(
    const int* col_ptrs_a,
    const int* row_indices_a,
    const int* col_ptrs_b,
    const int* row_indices_b,
    int* col_counts,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col];
    int end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col];
    int end_b = col_ptrs_b[col + 1];

    int i = start_a, j = start_b, count = 0;

    // Merge-count (union semantics)
    while (i < end_a || j < end_b) {
        int row_a = (i < end_a) ? row_indices_a[i] : INT_MAX;
        int row_b = (j < end_b) ? row_indices_b[j] : INT_MAX;

        if (row_a < row_b) {
            count++; i++;
        } else if (row_a > row_b) {
            count++; j++;
        } else {
            count++; i++; j++;
        }
    }

    col_counts[col] = count;
}

// Count kernel for CSC intersection (mul/div)
__global__ void csc_intersect_count(
    const int* col_ptrs_a,
    const int* row_indices_a,
    const int* col_ptrs_b,
    const int* row_indices_b,
    int* col_counts,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col];
    int end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col];
    int end_b = col_ptrs_b[col + 1];

    int i = start_a, j = start_b, count = 0;

    // Intersection semantics
    while (i < end_a && j < end_b) {
        int row_a = row_indices_a[i];
        int row_b = row_indices_b[j];

        if (row_a < row_b) {
            i++;
        } else if (row_a > row_b) {
            j++;
        } else {
            count++;
            i++; j++;
        }
    }

    col_counts[col] = count;
}

// CSC Add Compute Kernels
__global__ void csc_add_compute_f32(
    const int* col_ptrs_a, const int* row_indices_a, const float* values_a,
    const int* col_ptrs_b, const int* row_indices_b, const float* values_b,
    const int* out_col_ptrs, int* out_row_indices, float* out_values,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col], end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col], end_b = col_ptrs_b[col + 1];
    int out_idx = out_col_ptrs[col];
    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int row_a = (i < end_a) ? row_indices_a[i] : INT_MAX;
        int row_b = (j < end_b) ? row_indices_b[j] : INT_MAX;

        if (row_a < row_b) {
            out_row_indices[out_idx] = row_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (row_a > row_b) {
            out_row_indices[out_idx] = row_b;
            out_values[out_idx] = values_b[j];
            j++; out_idx++;
        } else {
            out_row_indices[out_idx] = row_a;
            out_values[out_idx] = values_a[i] + values_b[j];
            i++; j++; out_idx++;
        }
    }
}

__global__ void csc_add_compute_f64(
    const int* col_ptrs_a, const int* row_indices_a, const double* values_a,
    const int* col_ptrs_b, const int* row_indices_b, const double* values_b,
    const int* out_col_ptrs, int* out_row_indices, double* out_values,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col], end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col], end_b = col_ptrs_b[col + 1];
    int out_idx = out_col_ptrs[col];
    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int row_a = (i < end_a) ? row_indices_a[i] : INT_MAX;
        int row_b = (j < end_b) ? row_indices_b[j] : INT_MAX;

        if (row_a < row_b) {
            out_row_indices[out_idx] = row_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (row_a > row_b) {
            out_row_indices[out_idx] = row_b;
            out_values[out_idx] = values_b[j];
            j++; out_idx++;
        } else {
            out_row_indices[out_idx] = row_a;
            out_values[out_idx] = values_a[i] + values_b[j];
            i++; j++; out_idx++;
        }
    }
}

__global__ void csc_add_compute_f16(
    const int* col_ptrs_a, const int* row_indices_a, const __half* values_a,
    const int* col_ptrs_b, const int* row_indices_b, const __half* values_b,
    const int* out_col_ptrs, int* out_row_indices, __half* out_values,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col], end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col], end_b = col_ptrs_b[col + 1];
    int out_idx = out_col_ptrs[col];
    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int row_a = (i < end_a) ? row_indices_a[i] : INT_MAX;
        int row_b = (j < end_b) ? row_indices_b[j] : INT_MAX;

        if (row_a < row_b) {
            out_row_indices[out_idx] = row_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (row_a > row_b) {
            out_row_indices[out_idx] = row_b;
            out_values[out_idx] = values_b[j];
            j++; out_idx++;
        } else {
            out_row_indices[out_idx] = row_a;
            float sum = __half2float(values_a[i]) + __half2float(values_b[j]);
            out_values[out_idx] = __float2half(sum);
            i++; j++; out_idx++;
        }
    }
}

__global__ void csc_add_compute_bf16(
    const int* col_ptrs_a, const int* row_indices_a, const __nv_bfloat16* values_a,
    const int* col_ptrs_b, const int* row_indices_b, const __nv_bfloat16* values_b,
    const int* out_col_ptrs, int* out_row_indices, __nv_bfloat16* out_values,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col], end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col], end_b = col_ptrs_b[col + 1];
    int out_idx = out_col_ptrs[col];
    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int row_a = (i < end_a) ? row_indices_a[i] : INT_MAX;
        int row_b = (j < end_b) ? row_indices_b[j] : INT_MAX;

        if (row_a < row_b) {
            out_row_indices[out_idx] = row_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (row_a > row_b) {
            out_row_indices[out_idx] = row_b;
            out_values[out_idx] = values_b[j];
            j++; out_idx++;
        } else {
            out_row_indices[out_idx] = row_a;
            float sum = __bfloat162float(values_a[i]) + __bfloat162float(values_b[j]);
            out_values[out_idx] = __float2bfloat16(sum);
            i++; j++; out_idx++;
        }
    }
}

// CSC Sub Compute Kernels
__global__ void csc_sub_compute_f32(
    const int* col_ptrs_a, const int* row_indices_a, const float* values_a,
    const int* col_ptrs_b, const int* row_indices_b, const float* values_b,
    const int* out_col_ptrs, int* out_row_indices, float* out_values,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col], end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col], end_b = col_ptrs_b[col + 1];
    int out_idx = out_col_ptrs[col];
    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int row_a = (i < end_a) ? row_indices_a[i] : INT_MAX;
        int row_b = (j < end_b) ? row_indices_b[j] : INT_MAX;

        if (row_a < row_b) {
            out_row_indices[out_idx] = row_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (row_a > row_b) {
            out_row_indices[out_idx] = row_b;
            out_values[out_idx] = -values_b[j];
            j++; out_idx++;
        } else {
            out_row_indices[out_idx] = row_a;
            out_values[out_idx] = values_a[i] - values_b[j];
            i++; j++; out_idx++;
        }
    }
}

__global__ void csc_sub_compute_f64(
    const int* col_ptrs_a, const int* row_indices_a, const double* values_a,
    const int* col_ptrs_b, const int* row_indices_b, const double* values_b,
    const int* out_col_ptrs, int* out_row_indices, double* out_values,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col], end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col], end_b = col_ptrs_b[col + 1];
    int out_idx = out_col_ptrs[col];
    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int row_a = (i < end_a) ? row_indices_a[i] : INT_MAX;
        int row_b = (j < end_b) ? row_indices_b[j] : INT_MAX;

        if (row_a < row_b) {
            out_row_indices[out_idx] = row_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (row_a > row_b) {
            out_row_indices[out_idx] = row_b;
            out_values[out_idx] = -values_b[j];
            j++; out_idx++;
        } else {
            out_row_indices[out_idx] = row_a;
            out_values[out_idx] = values_a[i] - values_b[j];
            i++; j++; out_idx++;
        }
    }
}

__global__ void csc_sub_compute_f16(
    const int* col_ptrs_a, const int* row_indices_a, const __half* values_a,
    const int* col_ptrs_b, const int* row_indices_b, const __half* values_b,
    const int* out_col_ptrs, int* out_row_indices, __half* out_values,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col], end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col], end_b = col_ptrs_b[col + 1];
    int out_idx = out_col_ptrs[col];
    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int row_a = (i < end_a) ? row_indices_a[i] : INT_MAX;
        int row_b = (j < end_b) ? row_indices_b[j] : INT_MAX;

        if (row_a < row_b) {
            out_row_indices[out_idx] = row_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (row_a > row_b) {
            out_row_indices[out_idx] = row_b;
            out_values[out_idx] = __float2half(-__half2float(values_b[j]));
            j++; out_idx++;
        } else {
            out_row_indices[out_idx] = row_a;
            float diff = __half2float(values_a[i]) - __half2float(values_b[j]);
            out_values[out_idx] = __float2half(diff);
            i++; j++; out_idx++;
        }
    }
}

__global__ void csc_sub_compute_bf16(
    const int* col_ptrs_a, const int* row_indices_a, const __nv_bfloat16* values_a,
    const int* col_ptrs_b, const int* row_indices_b, const __nv_bfloat16* values_b,
    const int* out_col_ptrs, int* out_row_indices, __nv_bfloat16* out_values,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col], end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col], end_b = col_ptrs_b[col + 1];
    int out_idx = out_col_ptrs[col];
    int i = start_a, j = start_b;

    while (i < end_a || j < end_b) {
        int row_a = (i < end_a) ? row_indices_a[i] : INT_MAX;
        int row_b = (j < end_b) ? row_indices_b[j] : INT_MAX;

        if (row_a < row_b) {
            out_row_indices[out_idx] = row_a;
            out_values[out_idx] = values_a[i];
            i++; out_idx++;
        } else if (row_a > row_b) {
            out_row_indices[out_idx] = row_b;
            out_values[out_idx] = __float2bfloat16(-__bfloat162float(values_b[j]));
            j++; out_idx++;
        } else {
            out_row_indices[out_idx] = row_a;
            float diff = __bfloat162float(values_a[i]) - __bfloat162float(values_b[j]);
            out_values[out_idx] = __float2bfloat16(diff);
            i++; j++; out_idx++;
        }
    }
}

// CSC Mul Compute Kernels (Intersection Semantics)
__global__ void csc_mul_compute_f32(
    const int* col_ptrs_a, const int* row_indices_a, const float* values_a,
    const int* col_ptrs_b, const int* row_indices_b, const float* values_b,
    const int* out_col_ptrs, int* out_row_indices, float* out_values,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col], end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col], end_b = col_ptrs_b[col + 1];
    int out_idx = out_col_ptrs[col];
    int i = start_a, j = start_b;

    while (i < end_a && j < end_b) {
        int row_a = row_indices_a[i];
        int row_b = row_indices_b[j];

        if (row_a < row_b) { i++; }
        else if (row_a > row_b) { j++; }
        else {
            out_row_indices[out_idx] = row_a;
            out_values[out_idx] = values_a[i] * values_b[j];
            i++; j++; out_idx++;
        }
    }
}

__global__ void csc_mul_compute_f64(
    const int* col_ptrs_a, const int* row_indices_a, const double* values_a,
    const int* col_ptrs_b, const int* row_indices_b, const double* values_b,
    const int* out_col_ptrs, int* out_row_indices, double* out_values,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col], end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col], end_b = col_ptrs_b[col + 1];
    int out_idx = out_col_ptrs[col];
    int i = start_a, j = start_b;

    while (i < end_a && j < end_b) {
        int row_a = row_indices_a[i];
        int row_b = row_indices_b[j];

        if (row_a < row_b) { i++; }
        else if (row_a > row_b) { j++; }
        else {
            out_row_indices[out_idx] = row_a;
            out_values[out_idx] = values_a[i] * values_b[j];
            i++; j++; out_idx++;
        }
    }
}

__global__ void csc_mul_compute_f16(
    const int* col_ptrs_a, const int* row_indices_a, const __half* values_a,
    const int* col_ptrs_b, const int* row_indices_b, const __half* values_b,
    const int* out_col_ptrs, int* out_row_indices, __half* out_values,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col], end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col], end_b = col_ptrs_b[col + 1];
    int out_idx = out_col_ptrs[col];
    int i = start_a, j = start_b;

    while (i < end_a && j < end_b) {
        int row_a = row_indices_a[i];
        int row_b = row_indices_b[j];

        if (row_a < row_b) { i++; }
        else if (row_a > row_b) { j++; }
        else {
            out_row_indices[out_idx] = row_a;
            float prod = __half2float(values_a[i]) * __half2float(values_b[j]);
            out_values[out_idx] = __float2half(prod);
            i++; j++; out_idx++;
        }
    }
}

__global__ void csc_mul_compute_bf16(
    const int* col_ptrs_a, const int* row_indices_a, const __nv_bfloat16* values_a,
    const int* col_ptrs_b, const int* row_indices_b, const __nv_bfloat16* values_b,
    const int* out_col_ptrs, int* out_row_indices, __nv_bfloat16* out_values,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col], end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col], end_b = col_ptrs_b[col + 1];
    int out_idx = out_col_ptrs[col];
    int i = start_a, j = start_b;

    while (i < end_a && j < end_b) {
        int row_a = row_indices_a[i];
        int row_b = row_indices_b[j];

        if (row_a < row_b) { i++; }
        else if (row_a > row_b) { j++; }
        else {
            out_row_indices[out_idx] = row_a;
            float prod = __bfloat162float(values_a[i]) * __bfloat162float(values_b[j]);
            out_values[out_idx] = __float2bfloat16(prod);
            i++; j++; out_idx++;
        }
    }
}

// CSC Div Compute Kernels (Intersection Semantics)
__global__ void csc_div_compute_f32(
    const int* col_ptrs_a, const int* row_indices_a, const float* values_a,
    const int* col_ptrs_b, const int* row_indices_b, const float* values_b,
    const int* out_col_ptrs, int* out_row_indices, float* out_values,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col], end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col], end_b = col_ptrs_b[col + 1];
    int out_idx = out_col_ptrs[col];
    int i = start_a, j = start_b;

    while (i < end_a && j < end_b) {
        int row_a = row_indices_a[i];
        int row_b = row_indices_b[j];

        if (row_a < row_b) { i++; }
        else if (row_a > row_b) { j++; }
        else {
            out_row_indices[out_idx] = row_a;
            out_values[out_idx] = values_a[i] / values_b[j];
            i++; j++; out_idx++;
        }
    }
}

__global__ void csc_div_compute_f64(
    const int* col_ptrs_a, const int* row_indices_a, const double* values_a,
    const int* col_ptrs_b, const int* row_indices_b, const double* values_b,
    const int* out_col_ptrs, int* out_row_indices, double* out_values,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col], end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col], end_b = col_ptrs_b[col + 1];
    int out_idx = out_col_ptrs[col];
    int i = start_a, j = start_b;

    while (i < end_a && j < end_b) {
        int row_a = row_indices_a[i];
        int row_b = row_indices_b[j];

        if (row_a < row_b) { i++; }
        else if (row_a > row_b) { j++; }
        else {
            out_row_indices[out_idx] = row_a;
            out_values[out_idx] = values_a[i] / values_b[j];
            i++; j++; out_idx++;
        }
    }
}

__global__ void csc_div_compute_f16(
    const int* col_ptrs_a, const int* row_indices_a, const __half* values_a,
    const int* col_ptrs_b, const int* row_indices_b, const __half* values_b,
    const int* out_col_ptrs, int* out_row_indices, __half* out_values,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col], end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col], end_b = col_ptrs_b[col + 1];
    int out_idx = out_col_ptrs[col];
    int i = start_a, j = start_b;

    while (i < end_a && j < end_b) {
        int row_a = row_indices_a[i];
        int row_b = row_indices_b[j];

        if (row_a < row_b) { i++; }
        else if (row_a > row_b) { j++; }
        else {
            out_row_indices[out_idx] = row_a;
            float quot = __half2float(values_a[i]) / __half2float(values_b[j]);
            out_values[out_idx] = __float2half(quot);
            i++; j++; out_idx++;
        }
    }
}

__global__ void csc_div_compute_bf16(
    const int* col_ptrs_a, const int* row_indices_a, const __nv_bfloat16* values_a,
    const int* col_ptrs_b, const int* row_indices_b, const __nv_bfloat16* values_b,
    const int* out_col_ptrs, int* out_row_indices, __nv_bfloat16* out_values,
    int ncols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;

    int start_a = col_ptrs_a[col], end_a = col_ptrs_a[col + 1];
    int start_b = col_ptrs_b[col], end_b = col_ptrs_b[col + 1];
    int out_idx = out_col_ptrs[col];
    int i = start_a, j = start_b;

    while (i < end_a && j < end_b) {
        int row_a = row_indices_a[i];
        int row_b = row_indices_b[j];

        if (row_a < row_b) { i++; }
        else if (row_a > row_b) { j++; }
        else {
            out_row_indices[out_idx] = row_a;
            float quot = __bfloat162float(values_a[i]) / __bfloat162float(values_b[j]);
            out_values[out_idx] = __float2bfloat16(quot);
            i++; j++; out_idx++;
        }
    }
}

}  // extern "C"
