// Distance computation CUDA kernels
//
// Provides efficient pairwise distance computation for various metrics.
// All kernels support F32, F64, F16, and BF16 data types.
//
// Precision: F32/F64 accumulate in native precision.
// F16/BF16 accumulate in F32 for accuracy.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include "dtype_traits.cuh"

// ============================================================================
// Accumulation Type Traits
// ============================================================================

// AccT: the type used for accumulation and intermediate computation.
// F32 -> float, F64 -> double, F16/BF16 -> float (compute in F32 for accuracy)
template<typename T> struct AccType { using type = T; };
template<> struct AccType<__half> { using type = float; };
template<> struct AccType<__nv_bfloat16> { using type = float; };
template<> struct AccType<numr_fp8_e4m3> { using type = float; };
template<> struct AccType<numr_fp8_e5m2> { using type = float; };

// ============================================================================
// Type Conversion Helpers (to/from AccT)
// ============================================================================

template<typename AccT, typename T>
__device__ __forceinline__ AccT to_acc(T val) {
    return static_cast<AccT>(val);
}

template<>
__device__ __forceinline__ float to_acc<float, __half>(__half val) {
    return __half2float(val);
}

template<>
__device__ __forceinline__ float to_acc<float, __nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template<typename T, typename AccT>
__device__ __forceinline__ T from_acc(AccT val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ __half from_acc<__half, float>(float val) {
    return __float2half(val);
}

template<>
__device__ __forceinline__ __nv_bfloat16 from_acc<__nv_bfloat16, float>(float val) {
    return __float2bfloat16(val);
}

template<>
__device__ __forceinline__ float to_acc<float, numr_fp8_e4m3>(numr_fp8_e4m3 val) {
    return fp8_e4m3_to_f32(val.data);
}

template<>
__device__ __forceinline__ float to_acc<float, numr_fp8_e5m2>(numr_fp8_e5m2 val) {
    return fp8_e5m2_to_f32(val.data);
}

template<>
__device__ __forceinline__ numr_fp8_e4m3 from_acc<numr_fp8_e4m3, float>(float val) {
    return numr_fp8_e4m3(f32_to_fp8_e4m3(val));
}

template<>
__device__ __forceinline__ numr_fp8_e5m2 from_acc<numr_fp8_e5m2, float>(float val) {
    return numr_fp8_e5m2(f32_to_fp8_e5m2(val));
}

// ============================================================================
// Math helpers — dispatch sqrt/fabs/pow to correct precision
// ============================================================================

__device__ __forceinline__ float  acc_sqrt(float  x) { return sqrtf(x); }
__device__ __forceinline__ double acc_sqrt(double x) { return sqrt(x); }

__device__ __forceinline__ float  acc_fabs(float  x) { return fabsf(x); }
__device__ __forceinline__ double acc_fabs(double x) { return fabs(x); }

__device__ __forceinline__ float  acc_pow(float  x, float  y) { return powf(x, y); }
__device__ __forceinline__ double acc_pow(double x, double y) { return pow(x, y); }

// ============================================================================
// Distance Metric Implementations (templated on T and AccT)
// ============================================================================

// Squared Euclidean distance
template<typename T, typename AccT>
__device__ AccT sqeuclidean_dist(const T* a, const T* b, unsigned int d) {
    AccT sum = AccT(0);
    for (unsigned int k = 0; k < d; k++) {
        AccT diff = to_acc<AccT>(a[k]) - to_acc<AccT>(b[k]);
        sum += diff * diff;
    }
    return sum;
}

// Euclidean (L2) distance
template<typename T, typename AccT>
__device__ AccT euclidean_dist(const T* a, const T* b, unsigned int d) {
    return acc_sqrt(sqeuclidean_dist<T, AccT>(a, b, d));
}

// Manhattan (L1) distance
template<typename T, typename AccT>
__device__ AccT manhattan_dist(const T* a, const T* b, unsigned int d) {
    AccT sum = AccT(0);
    for (unsigned int k = 0; k < d; k++) {
        sum += acc_fabs(to_acc<AccT>(a[k]) - to_acc<AccT>(b[k]));
    }
    return sum;
}

// Chebyshev (L-infinity) distance
template<typename T, typename AccT>
__device__ AccT chebyshev_dist(const T* a, const T* b, unsigned int d) {
    AccT max_val = AccT(0);
    for (unsigned int k = 0; k < d; k++) {
        AccT abs_diff = acc_fabs(to_acc<AccT>(a[k]) - to_acc<AccT>(b[k]));
        if (abs_diff > max_val) max_val = abs_diff;
    }
    return max_val;
}

// Minkowski (Lp) distance
template<typename T, typename AccT>
__device__ AccT minkowski_dist(const T* a, const T* b, unsigned int d, AccT p) {
    AccT sum = AccT(0);
    for (unsigned int k = 0; k < d; k++) {
        sum += acc_pow(acc_fabs(to_acc<AccT>(a[k]) - to_acc<AccT>(b[k])), p);
    }
    return acc_pow(sum, AccT(1) / p);
}

// Cosine distance: 1 - cos(theta)
template<typename T, typename AccT>
__device__ AccT cosine_dist(const T* a, const T* b, unsigned int d) {
    AccT dot = AccT(0);
    AccT norm_a = AccT(0);
    AccT norm_b = AccT(0);

    for (unsigned int k = 0; k < d; k++) {
        AccT ak = to_acc<AccT>(a[k]);
        AccT bk = to_acc<AccT>(b[k]);
        dot += ak * bk;
        norm_a += ak * ak;
        norm_b += bk * bk;
    }

    AccT denom = acc_sqrt(norm_a * norm_b);
    if (denom == AccT(0)) return AccT(0);
    return AccT(1) - dot / denom;
}

// Correlation distance: 1 - Pearson r
template<typename T, typename AccT>
__device__ AccT correlation_dist(const T* a, const T* b, unsigned int d) {
    AccT sum_a = AccT(0);
    AccT sum_b = AccT(0);
    for (unsigned int k = 0; k < d; k++) {
        sum_a += to_acc<AccT>(a[k]);
        sum_b += to_acc<AccT>(b[k]);
    }
    AccT mean_a = sum_a / AccT(d);
    AccT mean_b = sum_b / AccT(d);

    AccT cov = AccT(0);
    AccT var_a = AccT(0);
    AccT var_b = AccT(0);
    for (unsigned int k = 0; k < d; k++) {
        AccT da = to_acc<AccT>(a[k]) - mean_a;
        AccT db = to_acc<AccT>(b[k]) - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    AccT denom = acc_sqrt(var_a * var_b);
    if (denom == AccT(0)) return AccT(0);
    return AccT(1) - cov / denom;
}

// Hamming distance: fraction of differing elements
template<typename T, typename AccT>
__device__ AccT hamming_dist(const T* a, const T* b, unsigned int d) {
    AccT count = AccT(0);
    for (unsigned int k = 0; k < d; k++) {
        if (to_acc<AccT>(a[k]) != to_acc<AccT>(b[k])) {
            count += AccT(1);
        }
    }
    return count / AccT(d);
}

// Jaccard distance: 1 - |intersection|/|union| for binary vectors
template<typename T, typename AccT>
__device__ AccT jaccard_dist(const T* a, const T* b, unsigned int d) {
    AccT intersection = AccT(0);
    AccT union_count = AccT(0);

    for (unsigned int k = 0; k < d; k++) {
        AccT ak = to_acc<AccT>(a[k]);
        AccT bk = to_acc<AccT>(b[k]);
        bool a_nonzero = (ak != AccT(0));
        bool b_nonzero = (bk != AccT(0));

        if (a_nonzero && b_nonzero) intersection += AccT(1);
        if (a_nonzero || b_nonzero) union_count += AccT(1);
    }

    if (union_count == AccT(0)) return AccT(0);
    return AccT(1) - intersection / union_count;
}

// ============================================================================
// Metric Dispatch
// ============================================================================

#define METRIC_EUCLIDEAN 0
#define METRIC_SQEUCLIDEAN 1
#define METRIC_MANHATTAN 2
#define METRIC_CHEBYSHEV 3
#define METRIC_MINKOWSKI 4
#define METRIC_COSINE 5
#define METRIC_CORRELATION 6
#define METRIC_HAMMING 7
#define METRIC_JACCARD 8

template<typename T, typename AccT>
__device__ AccT compute_distance(const T* a, const T* b, unsigned int d,
                                  unsigned int metric, AccT p) {
    switch (metric) {
        case METRIC_EUCLIDEAN:    return euclidean_dist<T, AccT>(a, b, d);
        case METRIC_SQEUCLIDEAN:  return sqeuclidean_dist<T, AccT>(a, b, d);
        case METRIC_MANHATTAN:    return manhattan_dist<T, AccT>(a, b, d);
        case METRIC_CHEBYSHEV:    return chebyshev_dist<T, AccT>(a, b, d);
        case METRIC_MINKOWSKI:    return minkowski_dist<T, AccT>(a, b, d, p);
        case METRIC_COSINE:       return cosine_dist<T, AccT>(a, b, d);
        case METRIC_CORRELATION:  return correlation_dist<T, AccT>(a, b, d);
        case METRIC_HAMMING:      return hamming_dist<T, AccT>(a, b, d);
        case METRIC_JACCARD:      return jaccard_dist<T, AccT>(a, b, d);
        default: return AccT(0);
    }
}

// ============================================================================
// CDIST Kernel - Pairwise distances between two sets
// ============================================================================

template<typename T, typename AccT>
__device__ void cdist_kernel_impl(
    const T* __restrict__ x,    // (n, d)
    const T* __restrict__ y,    // (m, d)
    T* __restrict__ out,        // (n, m)
    unsigned int n,
    unsigned int m,
    unsigned int d,
    unsigned int metric,
    AccT p
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * m;

    if (idx < total) {
        unsigned int i = idx / m;
        unsigned int j = idx % m;

        const T* x_row = x + i * d;
        const T* y_row = y + j * d;

        AccT dist = compute_distance<T, AccT>(x_row, y_row, d, metric, p);
        out[idx] = from_acc<T>(dist);
    }
}

// ============================================================================
// PDIST Kernel - Pairwise distances within one set (condensed)
// ============================================================================

template<typename T, typename AccT>
__device__ void pdist_kernel_impl(
    const T* __restrict__ x,    // (n, d)
    T* __restrict__ out,        // (n*(n-1)/2,)
    unsigned int n,
    unsigned int d,
    unsigned int metric,
    AccT p
) {
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * (n - 1) / 2;

    if (k < total) {
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

        AccT dist = compute_distance<T, AccT>(x_i, x_j, d, metric, p);
        out[k] = from_acc<T>(dist);
    }
}

// ============================================================================
// Squareform Kernel - Condensed to square
// ============================================================================

template<typename T, typename AccT>
__device__ void squareform_kernel_impl(
    const T* __restrict__ condensed,
    T* __restrict__ square,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * n;

    if (idx < total) {
        unsigned int i = idx / n;
        unsigned int j = idx % n;

        if (i == j) {
            square[idx] = from_acc<T>(AccT(0));
        } else if (i < j) {
            unsigned int k = n * i - i * (i + 1) / 2 + j - i - 1;
            square[idx] = condensed[k];
        } else {
            unsigned int k = n * j - j * (j + 1) / 2 + i - j - 1;
            square[idx] = condensed[k];
        }
    }
}

// ============================================================================
// Squareform Inverse Kernel - Square to condensed
// ============================================================================

template<typename T>
__device__ void squareform_inverse_kernel_impl(
    const T* __restrict__ square,
    T* __restrict__ condensed,
    unsigned int n
) {
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n * (n - 1) / 2;

    if (k < total) {
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

// F32: accumulate in float
// F64: accumulate in double
// F16/BF16: accumulate in float

#define INSTANTIATE_DISTANCE_KERNELS(T, AccT, suffix) \
    extern "C" __global__ void cdist_##suffix( \
        const T* x, const T* y, T* out, \
        unsigned int n, unsigned int m, unsigned int d, \
        unsigned int metric, AccT p) { \
        cdist_kernel_impl<T, AccT>(x, y, out, n, m, d, metric, p); \
    } \
    extern "C" __global__ void pdist_##suffix( \
        const T* x, T* out, \
        unsigned int n, unsigned int d, \
        unsigned int metric, AccT p) { \
        pdist_kernel_impl<T, AccT>(x, out, n, d, metric, p); \
    } \
    extern "C" __global__ void squareform_##suffix( \
        const T* condensed, T* square, unsigned int n) { \
        squareform_kernel_impl<T, AccT>(condensed, square, n); \
    } \
    extern "C" __global__ void squareform_inverse_##suffix( \
        const T* square, T* condensed, unsigned int n) { \
        squareform_inverse_kernel_impl(square, condensed, n); \
    }

INSTANTIATE_DISTANCE_KERNELS(float, float, f32)
INSTANTIATE_DISTANCE_KERNELS(double, double, f64)
INSTANTIATE_DISTANCE_KERNELS(__half, float, f16)
INSTANTIATE_DISTANCE_KERNELS(__nv_bfloat16, float, bf16)
INSTANTIATE_DISTANCE_KERNELS(numr_fp8_e4m3, float, fp8_e4m3)
INSTANTIATE_DISTANCE_KERNELS(numr_fp8_e5m2, float, fp8_e5m2)
