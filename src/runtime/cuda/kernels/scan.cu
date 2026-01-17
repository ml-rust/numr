//! Parallel Prefix Sum (Scan) Operations
//!
//! Implements work-efficient exclusive scan (prefix sum) on GPU using the
//! Blelloch scan algorithm. Used for sparse operations that need to compute
//! CSR row_ptrs from per-row counts.
//!
//! # Algorithm
//!
//! For arrays larger than block size:
//! 1. Block-level scan: Each block scans its portion independently
//! 2. Extract last element of each block as "block sum"
//! 3. Scan the block sums recursively
//! 4. Add scanned block sums as offsets to each block's elements
//!
//! # References
//!
//! - GPU Gems 3, Chapter 39: "Parallel Prefix Sum (Scan) with CUDA"
//! - Blelloch, G. E. (1990). "Prefix sums and their applications"

#include <cuda_runtime.h>

// Maximum block size for scan operations
#define SCAN_BLOCK_SIZE 512

// ============================================================================
// Warp-level Scan (32 threads)
// ============================================================================

/// Perform inclusive scan within a warp using shuffle instructions.
/// This is the fastest scan primitive available on modern GPUs.
template<typename T>
__device__ __forceinline__ T warp_scan_inclusive(T val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        T temp = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x % 32 >= offset) {
            val += temp;
        }
    }
    return val;
}

/// Convert inclusive scan to exclusive scan within a warp
template<typename T>
__device__ __forceinline__ T warp_scan_exclusive(T val) {
    T inclusive = warp_scan_inclusive(val);
    // Shift right by 1 within warp
    T exclusive = __shfl_up_sync(0xffffffff, inclusive, 1);
    if (threadIdx.x % 32 == 0) {
        exclusive = 0;
    }
    return exclusive;
}

// ============================================================================
// Block-level Scan (up to 1024 threads)
// ============================================================================

/// Perform exclusive scan within a thread block using shared memory.
/// Returns the scanned value for this thread and the total sum in block_sum.
template<typename T>
__device__ T block_scan_exclusive(T val, T* shared, T& block_sum) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    // Step 1: Warp-level scan
    T warp_sum = warp_scan_inclusive(val);

    // Step 2: Last thread in each warp writes warp sum to shared memory
    __shared__ T warp_sums[32];  // Support up to 1024 threads (32 warps)
    if (lane == 31) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    // Step 3: First warp scans the warp sums
    T warp_offset = 0;
    if (warp_id == 0) {
        T warp_val = (tid < 32) ? warp_sums[tid] : 0;
        warp_val = warp_scan_inclusive(warp_val);
        warp_sums[tid] = warp_val;
        if (tid == 31) {
            block_sum = warp_val;  // Total block sum
        }
    }
    __syncthreads();

    // Step 4: Add warp offset to convert to exclusive scan
    if (warp_id > 0) {
        warp_offset = warp_sums[warp_id - 1];
    }

    // Convert inclusive to exclusive: shift right, subtract current value
    T exclusive = warp_sum - val + warp_offset;

    return exclusive;
}

// ============================================================================
// Single-Block Scan (n <= SCAN_BLOCK_SIZE)
// ============================================================================

/// Scan a small array that fits in a single block.
/// Output has size n+1, with output[0] = 0 and output[n] = total sum.
__global__ void exclusive_scan_single_block_i32(
    const int* input,
    int* output,
    unsigned int n
) {
    __shared__ int shared[SCAN_BLOCK_SIZE];

    int tid = threadIdx.x;

    // Load input (or 0 if out of bounds)
    int val = (tid < n) ? input[tid] : 0;

    int block_sum;
    int scanned = block_scan_exclusive(val, shared, block_sum);

    // Write output
    if (tid < n) {
        output[tid] = scanned;
    }

    // Last thread writes the total sum at output[n]
    if (tid == 0) {
        output[n] = block_sum;
    }
}

// ============================================================================
// Multi-Block Scan (n > SCAN_BLOCK_SIZE)
// ============================================================================

/// Step 1: Scan each block independently and store block sums
__global__ void scan_blocks_i32(
    const int* input,
    int* output,
    int* block_sums,
    unsigned int n
) {
    __shared__ int shared[SCAN_BLOCK_SIZE];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * SCAN_BLOCK_SIZE;
    int global_idx = block_offset + tid;

    // Load input
    int val = (global_idx < n) ? input[global_idx] : 0;

    int block_sum;
    int scanned = block_scan_exclusive(val, shared, block_sum);

    // Write output
    if (global_idx < n) {
        output[global_idx] = scanned;
    }

    // Last thread in block writes the block sum
    if (tid == SCAN_BLOCK_SIZE - 1 || global_idx == n - 1) {
        block_sums[blockIdx.x] = block_sum;
    }
}

/// Step 3: Add block offsets to all elements
__global__ void add_block_offsets_i32(
    int* output,
    const int* block_offsets,
    unsigned int n
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < n && blockIdx.x > 0) {
        output[global_idx] += block_offsets[blockIdx.x - 1];
    }
}

// ============================================================================
// Extern "C" Kernels (for Rust FFI)
// ============================================================================

extern "C" {

/// I32 single-block scan
__global__ void exclusive_scan_i32(
    const int* input,
    int* output,
    unsigned int n
) {
    __shared__ int shared[SCAN_BLOCK_SIZE];

    int tid = threadIdx.x;

    // Load input (or 0 if out of bounds)
    int val = (tid < n) ? input[tid] : 0;

    int block_sum;
    int scanned = block_scan_exclusive(val, shared, block_sum);

    // Write output
    if (tid < n) {
        output[tid] = scanned;
    }

    // Last thread writes the total sum at output[n]
    if (tid == 0) {
        output[n] = block_sum;
    }
}

/// I32 multi-block scan - step 1
__global__ void scan_blocks_i32_step1(
    const int* input,
    int* output,
    int* block_sums,
    unsigned int n
) {
    __shared__ int shared[SCAN_BLOCK_SIZE];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * SCAN_BLOCK_SIZE;
    int global_idx = block_offset + tid;

    // Load input
    int val = (global_idx < n) ? input[global_idx] : 0;

    int block_sum;
    int scanned = block_scan_exclusive(val, shared, block_sum);

    // Write output
    if (global_idx < n) {
        output[global_idx] = scanned;
    }

    // Last thread in block writes the block sum
    if (tid == SCAN_BLOCK_SIZE - 1 || global_idx == n - 1) {
        block_sums[blockIdx.x] = block_sum;
    }
}

/// I32 multi-block scan - step 3
__global__ void add_block_offsets_i32_step3(
    int* output,
    const int* block_offsets,
    unsigned int n
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < n && blockIdx.x > 0) {
        output[global_idx] += block_offsets[blockIdx.x - 1];
    }
}

/// I64 single-block scan
__global__ void exclusive_scan_i64(
    const long long* input,
    long long* output,
    unsigned int n
) {
    __shared__ long long shared[SCAN_BLOCK_SIZE];

    int tid = threadIdx.x;

    long long val = (tid < n) ? input[tid] : 0;

    long long block_sum;
    long long scanned = block_scan_exclusive(val, shared, block_sum);

    if (tid < n) {
        output[tid] = scanned;
    }

    if (tid == 0) {
        output[n] = block_sum;
    }
}

/// I64 multi-block scan - step 1
__global__ void scan_blocks_i64_step1(
    const long long* input,
    long long* output,
    long long* block_sums,
    unsigned int n
) {
    __shared__ long long shared[SCAN_BLOCK_SIZE];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * SCAN_BLOCK_SIZE;
    int global_idx = block_offset + tid;

    long long val = (global_idx < n) ? input[global_idx] : 0;

    long long block_sum;
    long long scanned = block_scan_exclusive(val, shared, block_sum);

    if (global_idx < n) {
        output[global_idx] = scanned;
    }

    if (tid == SCAN_BLOCK_SIZE - 1 || global_idx == n - 1) {
        block_sums[blockIdx.x] = block_sum;
    }
}

/// I64 multi-block scan - step 3
__global__ void add_block_offsets_i64_step3(
    long long* output,
    const long long* block_offsets,
    unsigned int n
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx < n && blockIdx.x > 0) {
        output[global_idx] += block_offsets[blockIdx.x - 1];
    }
}

}  // extern "C"
