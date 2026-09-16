// Fused Add + RMSNorm CUDA kernels (forward + backward)
//
// Forward:  pre_norm = input + residual
//           output   = pre_norm * rsqrt(mean(pre_norm^2) + eps) * weight
// Backward: d_input_residual and d_weight from grad and the saved pre_norm.
//           d_weight accumulates across rows with an atomicAdd in the storage
//           dtype, so it must be zeroed before launch.
//
// One templated body per direction, instantiated per dtype through
// DEFINE_FUSED_ADD_RMS_NORM. The accumulator is F32 for f32, f16, bf16 and
// FP8, F64 for f64.
//
// Shared memory: forward blockDim.x accumulator elements, backward
// 2 * blockDim.x (sum of squares, then the grad.weight.pre_norm dot).

#include "norm_common.cuh"

// Each block handles one row (hidden_size elements).
template <typename T, typename Acc>
__device__ __forceinline__ void fused_add_rms_norm_impl(
    const T* input, const T* residual, const T* weight,
    T* output, T* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, Acc eps, Acc* shared
) {
    typedef AccumTraits<T, Acc> AT;
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    const T* row_in = input + row * hidden_size;
    const T* row_res = residual + row * hidden_size;
    T* row_pn = pre_norm + row * hidden_size;
    T* row_out = output + row * hidden_size;

    // Phase 1: add residual + sum of squares
    Acc thread_sum = (Acc)0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        Acc pn = AT::load(row_in, (int)i) + AT::load(row_res, (int)i);
        AT::store(row_pn, (int)i, pn);
        thread_sum += pn * pn;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce<Acc>(shared);

    Acc rms_inv = numr_norm_rsqrt(shared[0] / hidden_size + eps);
    __syncthreads();

    // Phase 2: normalize and apply weight. Reads pre_norm back in the storage
    // dtype, so the rounding at the store above is what gets scaled.
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        Acc pn = AT::load(row_pn, (int)i);
        AT::store(row_out, (int)i, pn * rms_inv * AT::load(weight, (int)i));
    }
}

template <typename T, typename Acc>
__device__ __forceinline__ void fused_add_rms_norm_bwd_impl(
    const T* grad, const T* pre_norm, const T* weight,
    T* d_input_residual, T* d_weight,
    unsigned int batch_size, unsigned int hidden_size, Acc eps, Acc* shared
) {
    typedef AccumTraits<T, Acc> AT;
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    Acc* sum_sq_shared = shared;
    Acc* dot_shared = shared + blockDim.x;

    const T* row_g = grad + row * hidden_size;
    const T* row_pn = pre_norm + row * hidden_size;
    T* row_dir = d_input_residual + row * hidden_size;

    // Phase 1: sum_sq and dot = sum(grad * weight * pre_norm)
    Acc thread_sq = (Acc)0, thread_dot = (Acc)0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        Acc pn = AT::load(row_pn, (int)i);
        Acc g = AT::load(row_g, (int)i);
        Acc w = AT::load(weight, (int)i);
        thread_sq += pn * pn;
        thread_dot += g * w * pn;
    }
    sum_sq_shared[threadIdx.x] = thread_sq;
    dot_shared[threadIdx.x] = thread_dot;
    __syncthreads();

    block_sum_reduce<Acc>(sum_sq_shared);
    block_sum_reduce<Acc>(dot_shared);

    Acc mean_sq = sum_sq_shared[0] / hidden_size;
    Acc inv_rms = numr_norm_rsqrt(mean_sq + eps);
    Acc dot = dot_shared[0];
    Acc coeff = dot * inv_rms / (hidden_size * (mean_sq + eps));
    __syncthreads();

    // Phase 2: d_input_residual, and atomicAdd into d_weight
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        Acc g = AT::load(row_g, (int)i);
        Acc w = AT::load(weight, (int)i);
        Acc pn = AT::load(row_pn, (int)i);
        AT::store(row_dir, (int)i, (g * w - pn * coeff) * inv_rms);
        norm_atomic_add(&d_weight[i], g * pn * inv_rms);
    }
}

#define DEFINE_FUSED_ADD_RMS_NORM(suffix, dtype, acc) \
__global__ void fused_add_rms_norm_##suffix( \
    const dtype* input, const dtype* residual, const dtype* weight, \
    dtype* output, dtype* pre_norm, \
    unsigned int batch_size, unsigned int hidden_size, acc eps \
) { \
    extern __shared__ acc farn_smem_##acc[]; \
    fused_add_rms_norm_impl<dtype, acc>( \
        input, residual, weight, output, pre_norm, batch_size, hidden_size, eps, farn_smem_##acc); \
} \
\
__global__ void fused_add_rms_norm_bwd_##suffix( \
    const dtype* grad, const dtype* pre_norm, const dtype* weight, \
    dtype* d_input_residual, dtype* d_weight, \
    unsigned int batch_size, unsigned int hidden_size, acc eps \
) { \
    extern __shared__ acc farn_bwd_smem_##acc[]; \
    fused_add_rms_norm_bwd_impl<dtype, acc>( \
        grad, pre_norm, weight, d_input_residual, d_weight, \
        batch_size, hidden_size, eps, farn_bwd_smem_##acc); \
}

extern "C" {

NORM_DTYPES(DEFINE_FUSED_ADD_RMS_NORM)

} // extern "C"
