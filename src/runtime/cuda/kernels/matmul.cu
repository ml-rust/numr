// Native Register-Tiled Matrix Multiplication CUDA Kernels
// Implements C = A @ B using the universal tiled GEMM algorithm:
//
// Level 1: Block Tiling (Shared Memory)
//   - BLOCK_M × BLOCK_N output tiles loaded to shared memory
// Level 2: Register Tiling (Per-Thread Micro-Tiles)
//   - Each thread computes THREAD_M × THREAD_N output elements
// Level 3: FMA Micro-Kernel
//   - Outer product of register arrays
//
// TILE SIZES ARE NOT HARDCODED - passed as kernel parameters
// Optimal values vary by device:
//   - Apple M1/M2: 32×32 or 64×64 tiles
//   - NVIDIA Desktop: 128×128 with 8×8 thread tiles
//   - Mobile/Low-end: Smaller tiles (32×32 with 4×4 thread tiles)

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ============================================================================
// Configurable Register-Tiled GEMM Kernel (F32)
// ============================================================================
// Tile configuration passed as kernel parameters:
//   block_m, block_n, block_k - Shared memory tile sizes
//   thread_m, thread_n - Per-thread register tile sizes
//
// Dynamic shared memory layout:
//   As[block_m][block_k] followed by Bs[block_k][block_n]

extern "C" __global__ void matmul_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    // Dynamic shared memory - allocated at kernel launch
    extern __shared__ float shared_mem[];
    float* As = shared_mem;                        // [block_m][block_k]
    float* Bs = shared_mem + block_m * block_k;    // [block_k][block_n]

    // Thread position within block
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    // Global starting position for this thread's micro-tile
    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    // Register accumulators - use max supported size, only use what's needed
    // Max 8×8 = 64 elements per thread
    float reg_c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        // Cooperative load of A tile [block_m × block_k]
        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = A[global_row * K + global_col];
            }
            As[load_row * block_k + load_col] = val;
        }

        // Cooperative load of B tile [block_k × block_n]
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = B[global_row * N + global_col];
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        // Register tiling: compute partial products
        for (unsigned int k = 0; k < block_k; k++) {
            // Load thread_m elements from A tile into registers
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }

            // Load thread_n elements from B tile into registers
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }

            // Outer product: thread_m × thread_n FMAs
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // Write register tile to global memory
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = reg_c[i][j];
                }
            }
        }
    }
}

// ============================================================================
// Configurable Batched GEMM (F32)
// ============================================================================

extern "C" __global__ void matmul_batched_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int stride_a = M * K;
    const unsigned int stride_b = K * N;
    const unsigned int stride_c = M * N;

    const float* A_batch = A + b * stride_a;
    const float* B_batch = B + b * stride_b;
    float* C_batch = C + b * stride_c;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = A_batch[global_row * K + global_col];
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = B_batch[global_row * N + global_col];
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C_batch[global_row * N + global_col] = reg_c[i][j];
                }
            }
        }
    }
}

// ============================================================================
// Configurable GEMM (F64)
// ============================================================================

extern "C" __global__ void matmul_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ double shared_mem_f64[];
    double* As = shared_mem_f64;
    double* Bs = shared_mem_f64 + block_m * block_k;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    double reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0;
        }
    }

    double reg_a[8];
    double reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            double val = 0.0;
            if (global_row < M && global_col < K) {
                val = A[global_row * K + global_col];
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            double val = 0.0;
            if (global_row < K && global_col < N) {
                val = B[global_row * N + global_col];
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = reg_c[i][j];
                }
            }
        }
    }
}

extern "C" __global__ void matmul_batched_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ double shared_mem_f64[];
    double* As = shared_mem_f64;
    double* Bs = shared_mem_f64 + block_m * block_k;

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int stride_a = M * K;
    const unsigned int stride_b = K * N;
    const unsigned int stride_c = M * N;

    const double* A_batch = A + b * stride_a;
    const double* B_batch = B + b * stride_b;
    double* C_batch = C + b * stride_c;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    double reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0;
        }
    }

    double reg_a[8];
    double reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            double val = 0.0;
            if (global_row < M && global_col < K) {
                val = A_batch[global_row * K + global_col];
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            double val = 0.0;
            if (global_row < K && global_col < N) {
                val = B_batch[global_row * N + global_col];
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C_batch[global_row * N + global_col] = reg_c[i][j];
                }
            }
        }
    }
}

// ============================================================================
// Configurable GEMM (F16 with F32 Accumulation)
// ============================================================================

extern "C" __global__ void matmul_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    // F16 uses F32 shared memory and accumulation for numerical stability
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = __half2float(A[global_row * K + global_col]);
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = __half2float(B[global_row * N + global_col]);
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = __float2half(reg_c[i][j]);
                }
            }
        }
    }
}

extern "C" __global__ void matmul_batched_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int stride_a = M * K;
    const unsigned int stride_b = K * N;
    const unsigned int stride_c = M * N;

    const __half* A_batch = A + b * stride_a;
    const __half* B_batch = B + b * stride_b;
    __half* C_batch = C + b * stride_c;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = __half2float(A_batch[global_row * K + global_col]);
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = __half2float(B_batch[global_row * N + global_col]);
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C_batch[global_row * N + global_col] = __float2half(reg_c[i][j]);
                }
            }
        }
    }
}

// ============================================================================
// Configurable GEMM (BF16 with F32 Accumulation)
// ============================================================================

extern "C" __global__ void matmul_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = __bfloat162float(A[global_row * K + global_col]);
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = __bfloat162float(B[global_row * N + global_col]);
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = __float2bfloat16(reg_c[i][j]);
                }
            }
        }
    }
}

extern "C" __global__ void matmul_batched_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int stride_a = M * K;
    const unsigned int stride_b = K * N;
    const unsigned int stride_c = M * N;

    const __nv_bfloat16* A_batch = A + b * stride_a;
    const __nv_bfloat16* B_batch = B + b * stride_b;
    __nv_bfloat16* C_batch = C + b * stride_c;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = __bfloat162float(A_batch[global_row * K + global_col]);
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = __bfloat162float(B_batch[global_row * N + global_col]);
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C_batch[global_row * N + global_col] = __float2bfloat16(reg_c[i][j]);
                }
            }
        }
    }
}

// ============================================================================
// Fused GEMM with Bias Addition: C = A @ B + bias
// ============================================================================
// Same algorithm as matmul kernels above, but bias[j] is added in the epilogue
// before writing to global memory. This fuses the bias addition to avoid
// an extra memory round-trip.

extern "C" __global__ void matmul_bias_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = A[global_row * K + global_col];
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = B[global_row * N + global_col];
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // EPILOGUE: Write with fused bias addition
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    // Fused bias addition: C[i][j] = matmul_result + bias[j]
                    C[global_row * N + global_col] = reg_c[i][j] + bias[global_col];
                }
            }
        }
    }
}

extern "C" __global__ void matmul_bias_batched_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int stride_a = M * K;
    const unsigned int stride_b = K * N;
    const unsigned int stride_c = M * N;

    const float* A_batch = A + b * stride_a;
    const float* B_batch = B + b * stride_b;
    float* C_batch = C + b * stride_c;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = A_batch[global_row * K + global_col];
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = B_batch[global_row * N + global_col];
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // EPILOGUE: Write with fused bias addition (same bias for all batches)
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C_batch[global_row * N + global_col] = reg_c[i][j] + bias[global_col];
                }
            }
        }
    }
}

// ============================================================================
// Fused GEMM with Bias (F64)
// ============================================================================

extern "C" __global__ void matmul_bias_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    const double* __restrict__ bias,
    double* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ double shared_mem_f64[];
    double* As = shared_mem_f64;
    double* Bs = shared_mem_f64 + block_m * block_k;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    double reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0;
        }
    }

    double reg_a[8];
    double reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            double val = 0.0;
            if (global_row < M && global_col < K) {
                val = A[global_row * K + global_col];
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            double val = 0.0;
            if (global_row < K && global_col < N) {
                val = B[global_row * N + global_col];
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // EPILOGUE: Write with fused bias addition
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = reg_c[i][j] + bias[global_col];
                }
            }
        }
    }
}

extern "C" __global__ void matmul_bias_batched_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    const double* __restrict__ bias,
    double* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ double shared_mem_f64[];
    double* As = shared_mem_f64;
    double* Bs = shared_mem_f64 + block_m * block_k;

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int stride_a = M * K;
    const unsigned int stride_b = K * N;
    const unsigned int stride_c = M * N;

    const double* A_batch = A + b * stride_a;
    const double* B_batch = B + b * stride_b;
    double* C_batch = C + b * stride_c;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    double reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0;
        }
    }

    double reg_a[8];
    double reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            double val = 0.0;
            if (global_row < M && global_col < K) {
                val = A_batch[global_row * K + global_col];
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            double val = 0.0;
            if (global_row < K && global_col < N) {
                val = B_batch[global_row * N + global_col];
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // EPILOGUE: Write with fused bias addition
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C_batch[global_row * N + global_col] = reg_c[i][j] + bias[global_col];
                }
            }
        }
    }
}

// ============================================================================
// Fused GEMM with Bias (F16 with F32 Accumulation)
// ============================================================================

extern "C" __global__ void matmul_bias_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
    __half* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = __half2float(A[global_row * K + global_col]);
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = __half2float(B[global_row * N + global_col]);
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // EPILOGUE: Write with fused bias addition (convert bias to F32 for add)
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    float bias_val = __half2float(bias[global_col]);
                    C[global_row * N + global_col] = __float2half(reg_c[i][j] + bias_val);
                }
            }
        }
    }
}

extern "C" __global__ void matmul_bias_batched_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
    __half* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int stride_a = M * K;
    const unsigned int stride_b = K * N;
    const unsigned int stride_c = M * N;

    const __half* A_batch = A + b * stride_a;
    const __half* B_batch = B + b * stride_b;
    __half* C_batch = C + b * stride_c;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = __half2float(A_batch[global_row * K + global_col]);
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = __half2float(B_batch[global_row * N + global_col]);
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // EPILOGUE: Write with fused bias addition
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    float bias_val = __half2float(bias[global_col]);
                    C_batch[global_row * N + global_col] = __float2half(reg_c[i][j] + bias_val);
                }
            }
        }
    }
}

// ============================================================================
// Fused GEMM with Bias (BF16 with F32 Accumulation)
// ============================================================================

extern "C" __global__ void matmul_bias_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = __bfloat162float(A[global_row * K + global_col]);
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = __bfloat162float(B[global_row * N + global_col]);
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // EPILOGUE: Write with fused bias addition
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    float bias_val = __bfloat162float(bias[global_col]);
                    C[global_row * N + global_col] = __float2bfloat16(reg_c[i][j] + bias_val);
                }
            }
        }
    }
}

extern "C" __global__ void matmul_bias_batched_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int stride_a = M * K;
    const unsigned int stride_b = K * N;
    const unsigned int stride_c = M * N;

    const __nv_bfloat16* A_batch = A + b * stride_a;
    const __nv_bfloat16* B_batch = B + b * stride_b;
    __nv_bfloat16* C_batch = C + b * stride_c;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = __bfloat162float(A_batch[global_row * K + global_col]);
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = __bfloat162float(B_batch[global_row * N + global_col]);
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // EPILOGUE: Write with fused bias addition
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    float bias_val = __bfloat162float(bias[global_col]);
                    C_batch[global_row * N + global_col] = __float2bfloat16(reg_c[i][j] + bias_val);
                }
            }
        }
    }
}
