// Distance computation CUDA kernels
//
// Provides efficient pairwise distance computation for various metrics.
// All kernels support F32, F64, F16, and BF16 data types.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// ============================================================================
// Type Conversion Helpers
// ============================================================================

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<__half>(__half val) {
    return __half2float(val);
}

template<>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ __half from_float<__half>(float val) {
    return __float2half(val);
}

template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) {
    return __float2bfloat16(val);
}

// ============================================================================
// Distance Metric Implementations
// ============================================================================

// Squared Euclidean distance between two vectors
template<typename T>
__device__ float sqeuclidean_dist(const T* a, const T* b, unsigned int d) {
    float sum = 0.0f;
    for (unsigned int k = 0; k < d; k++) {
        float diff = to_float(a[k]) - to_float(b[k]);
        sum += diff * diff;
    }
    return sum;
}

// Euclidean (L2) distance
template<typename T>
__device__ float euclidean_dist(const T* a, const T* b, unsigned int d) {
    return sqrtf(sqeuclidean_dist(a, b, d));
}

// Manhattan (L1) distance
template<typename T>
__device__ float manhattan_dist(const T* a, const T* b, unsigned int d) {
    float sum = 0.0f;
    for (unsigned int k = 0; k < d; k++) {
        sum += fabsf(to_float(a[k]) - to_float(b[k]));
    }
    return sum;
}

// Chebyshev (L-infinity) distance
template<typename T>
__device__ float chebyshev_dist(const T* a, const T* b, unsigned int d) {
    float max_val = 0.0f;
    for (unsigned int k = 0; k < d; k++) {
        float abs_diff = fabsf(to_float(a[k]) - to_float(b[k]));
        if (abs_diff > max_val) max_val = abs_diff;
    }
    return max_val;
}

// Minkowski (Lp) distance
template<typename T>
__device__ float minkowski_dist(const T* a, const T* b, unsigned int d, float p) {
    float sum = 0.0f;
    for (unsigned int k = 0; k < d; k++) {
        sum += powf(fabsf(to_float(a[k]) - to_float(b[k])), p);
    }
    return powf(sum, 1.0f / p);
}

// Cosine distance: 1 - cos(theta)
template<typename T>
__device__ float cosine_dist(const T* a, const T* b, unsigned int d) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (unsigned int k = 0; k < d; k++) {
        float ak = to_float(a[k]);
        float bk = to_float(b[k]);
        dot += ak * bk;
        norm_a += ak * ak;
        norm_b += bk * bk;
    }

    float denom = sqrtf(norm_a * norm_b);
    if (denom == 0.0f) return 0.0f;
    return 1.0f - dot / denom;
}

// Correlation distance: 1 - Pearson r
template<typename T>
__device__ float correlation_dist(const T* a, const T* b, unsigned int d) {
    // Compute means
    float sum_a = 0.0f;
    float sum_b = 0.0f;
    for (unsigned int k = 0; k < d; k++) {
        sum_a += to_float(a[k]);
        sum_b += to_float(b[k]);
    }
    float mean_a = sum_a / d;
    float mean_b = sum_b / d;

    // Compute correlation
    float cov = 0.0f;
    float var_a = 0.0f;
    float var_b = 0.0f;
    for (unsigned int k = 0; k < d; k++) {
        float da = to_float(a[k]) - mean_a;
        float db = to_float(b[k]) - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    float denom = sqrtf(var_a * var_b);
    if (denom == 0.0f) return 0.0f;
    return 1.0f - cov / denom;
}

// Hamming distance: fraction of differing elements
template<typename T>
__device__ float hamming_dist(const T* a, const T* b, unsigned int d) {
    float count = 0.0f;
    for (unsigned int k = 0; k < d; k++) {
        if (to_float(a[k]) != to_float(b[k])) {
            count += 1.0f;
        }
    }
    return count / d;
}

// Jaccard distance: 1 - |intersection|/|union| for binary vectors
template<typename T>
__device__ float jaccard_dist(const T* a, const T* b, unsigned int d) {
    float intersection = 0.0f;
    float union_count = 0.0f;

    for (unsigned int k = 0; k < d; k++) {
        float ak = to_float(a[k]);
        float bk = to_float(b[k]);
        bool a_nonzero = (ak != 0.0f);
        bool b_nonzero = (bk != 0.0f);

        if (a_nonzero && b_nonzero) intersection += 1.0f;
        if (a_nonzero || b_nonzero) union_count += 1.0f;
    }

    if (union_count == 0.0f) return 0.0f;
    return 1.0f - intersection / union_count;
}

// ============================================================================
// Metric Dispatch
// ============================================================================

// Distance metric enum values (must match Rust DistanceMetric)
#define METRIC_EUCLIDEAN 0
#define METRIC_SQEUCLIDEAN 1
#define METRIC_MANHATTAN 2
#define METRIC_CHEBYSHEV 3
#define METRIC_MINKOWSKI 4
#define METRIC_COSINE 5
#define METRIC_CORRELATION 6
#define METRIC_HAMMING 7
#define METRIC_JACCARD 8

template<typename T>
__device__ float compute_distance(const T* a, const T* b, unsigned int d,
                                  unsigned int metric, float p) {
    switch (metric) {
        case METRIC_EUCLIDEAN: return euclidean_dist(a, b, d);
        case METRIC_SQEUCLIDEAN: return sqeuclidean_dist(a, b, d);
        case METRIC_MANHATTAN: return manhattan_dist(a, b, d);
        case METRIC_CHEBYSHEV: return chebyshev_dist(a, b, d);
        case METRIC_MINKOWSKI: return minkowski_dist(a, b, d, p);
        case METRIC_COSINE: return cosine_dist(a, b, d);
        case METRIC_CORRELATION: return correlation_dist(a, b, d);
        case METRIC_HAMMING: return hamming_dist(a, b, d);
        case METRIC_JACCARD: return jaccard_dist(a, b, d);
        default: return 0.0f;
    }
}

// ============================================================================
// CDIST Device Function - Pairwise distances between two sets
// ============================================================================

template<typename T>
__device__ void cdist_kernel_impl(
    const T* __restrict__ x,    // (n, d)
    const T* __restrict__ y,    // (m, d)
    T* __restrict__ out,        // (n, m)
    unsigned int n,
    unsigned int m,
    unsigned int d,
    unsigned int metric,
    float p
) {
    // Each thread computes one distance
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * m;

    if (idx < total) {
        unsigned int i = idx / m;  // Row in output (index into x)
        unsigned int j = idx % m;  // Col in output (index into y)

        const T* x_row = x + i * d;
        const T* y_row = y + j * d;

        float dist = compute_distance(x_row, y_row, d, metric, p);
        out[idx] = from_float<T>(dist);
    }
}

// ============================================================================
// PDIST Device Function - Pairwise distances within one set (condensed)
// ============================================================================

template<typename T>
__device__ void pdist_kernel_impl(
    const T* __restrict__ x,    // (n, d)
    T* __restrict__ out,        // (n*(n-1)/2,)
    unsigned int n,
    unsigned int d,
    unsigned int metric,
    float p
) {
    // Each thread computes one distance from condensed index
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * (n - 1) / 2;

    if (k < total) {
        // Convert condensed index k to (i, j) where i < j
        // Using formula: k = n*i - i*(i+1)/2 + j - i - 1
        // Inverse: i = n - 2 - floor(sqrt(-8k + 4n*(n-1) - 7) / 2 - 0.5)
        //          j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2

        // Simpler approach: iterate to find i, j
        unsigned int i = 0;
        unsigned int j_start = 1;
        unsigned int count = 0;

        while (true) {
            unsigned int row_count = n - 1 - i;
            if (count + row_count > k) {
                j_start = k - count + i + 1;
                break;
            }
            count += row_count;
            i++;
        }
        unsigned int j = j_start;

        const T* x_i = x + i * d;
        const T* x_j = x + j * d;

        float dist = compute_distance(x_i, x_j, d, metric, p);
        out[k] = from_float<T>(dist);
    }
}

// ============================================================================
// Squareform Device Function - Condensed to square
// ============================================================================

template<typename T>
__device__ void squareform_kernel_impl(
    const T* __restrict__ condensed,  // (n*(n-1)/2,)
    T* __restrict__ square,           // (n, n)
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * n;

    if (idx < total) {
        unsigned int i = idx / n;
        unsigned int j = idx % n;

        if (i == j) {
            // Diagonal is zero
            square[idx] = from_float<T>(0.0f);
        } else if (i < j) {
            // Upper triangle: k = n*i - i*(i+1)/2 + j - i - 1
            unsigned int k = n * i - i * (i + 1) / 2 + j - i - 1;
            square[idx] = condensed[k];
        } else {
            // Lower triangle: mirror from upper
            unsigned int k = n * j - j * (j + 1) / 2 + i - j - 1;
            square[idx] = condensed[k];
        }
    }
}

// ============================================================================
// Squareform Inverse Device Function - Square to condensed
// ============================================================================

template<typename T>
__device__ void squareform_inverse_kernel_impl(
    const T* __restrict__ square,     // (n, n)
    T* __restrict__ condensed,        // (n*(n-1)/2,)
    unsigned int n
) {
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * (n - 1) / 2;

    if (k < total) {
        // Convert k to (i, j) where i < j
        unsigned int i = 0;
        unsigned int count = 0;

        while (true) {
            unsigned int row_count = n - 1 - i;
            if (count + row_count > k) {
                unsigned int j = k - count + i + 1;
                condensed[k] = square[i * n + j];
                break;
            }
            count += row_count;
            i++;
        }
    }
}

// ============================================================================
// Kernel Instantiations
// ============================================================================

#define INSTANTIATE_DISTANCE_KERNELS(T, suffix) \
    extern "C" __global__ void cdist_##suffix( \
        const T* x, const T* y, T* out, \
        unsigned int n, unsigned int m, unsigned int d, \
        unsigned int metric, float p) { \
        cdist_kernel_impl(x, y, out, n, m, d, metric, p); \
    } \
    extern "C" __global__ void pdist_##suffix( \
        const T* x, T* out, \
        unsigned int n, unsigned int d, \
        unsigned int metric, float p) { \
        pdist_kernel_impl(x, out, n, d, metric, p); \
    } \
    extern "C" __global__ void squareform_##suffix( \
        const T* condensed, T* square, unsigned int n) { \
        squareform_kernel_impl(condensed, square, n); \
    } \
    extern "C" __global__ void squareform_inverse_##suffix( \
        const T* square, T* condensed, unsigned int n) { \
        squareform_inverse_kernel_impl(square, condensed, n); \
    }

INSTANTIATE_DISTANCE_KERNELS(float, f32)
INSTANTIATE_DISTANCE_KERNELS(double, f64)
INSTANTIATE_DISTANCE_KERNELS(__half, f16)
INSTANTIATE_DISTANCE_KERNELS(__nv_bfloat16, bf16)
